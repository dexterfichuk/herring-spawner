#!/usr/bin/env python3
"""SHSI-based BC coast scan for herring spawn candidates.

Computes the Sentinel-2 Herring Spawn Index (SHSI = B3²/B4) across all BC
herring habitat regions, filters to top candidates, downloads thumbnails,
and builds a review page.

Three phases:
  1. Scan all grid points — compute SHSI_mean and SHSI_max from Sentinel-2
  2. Filter — keep points with SHSI_mean > threshold, take top N
  3. Download — RGB thumbnails for top candidates only

Usage:
    python scripts/scan_shsi_candidates.py \\
        --output data/candidates_shsi \\
        --start-year 2024 \\
        --end-year 2024 \\
        --grid-spacing 0.02 \\
        --shsi-threshold 0.02 \\
        --max-cloud 80 \\
        --top-n 150 \\
        --workers 8
"""

import argparse
import html
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scripts.scan_bc_coast import (
    REGIONS,
    download_thumbnail,
    find_best_scene,
    generate_grid_points,
)


# ---------------------------------------------------------------------------
# SHSI Computation
# ---------------------------------------------------------------------------

def compute_shsi_for_point(
    ee_module: Any,
    lat: float,
    lon: float,
    scene_id: str,
    buffer_m: int = 100,
) -> dict[str, float] | None:
    """Compute SHSI (B3²/B4) statistics for a point over a buffer.

    Returns dict with shsi_mean, shsi_max, or None on failure.
    """
    try:
        scene_img = ee_module.Image(f"COPERNICUS/S2_SR_HARMONIZED/{scene_id}")

        # Cloud mask Sentinel-2
        def mask_s2_clouds(image):
            qa = image.select("QA60")
            cloud_bit_mask = 1 << 10
            cirrus_bit_mask = 1 << 11
            mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                qa.bitwiseAnd(cirrus_bit_mask).eq(0)
            )
            return image.updateMask(mask).divide(10000)

        masked = mask_s2_clouds(scene_img)

        # Raw SHSI = Green² / Red  (B3² / B4)
        green = masked.select("B3")
        red = masked.select("B4")
        shsi = green.multiply(green).divide(red).rename("SHSI")

        # Reduce over 100 m buffer
        point_geom = ee_module.Geometry.Point(lon, lat)
        buffer = point_geom.buffer(buffer_m)
        stats = shsi.reduceRegion(
            reducer=ee_module.Reducer.mean().combine(
                ee_module.Reducer.max(), "", True
            ),
            geometry=buffer,
            scale=10,
            maxPixels=1e6,
            bestEffort=True,
        ).getInfo()

        return {
            "shsi_mean": float(stats.get("SHSI_mean", 0) or 0),
            "shsi_max": float(stats.get("SHSI_max", 0) or 0),
        }
    except Exception as exc:
        print(f"    SHSI error at ({lat:.4f}, {lon:.4f}) scene {scene_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Phase 1 — scan one grid point
# ---------------------------------------------------------------------------

def scan_point(
    ee_module: Any,
    point: dict[str, Any],
    start_date: str,
    end_date: str,
    max_cloud: float,
    idx: int,
    total: int,
    start_time: float,
) -> dict[str, Any] | None:
    """Scan a single grid point: find best scene, compute SHSI.

    Returns info dict or None if no scene found or SHSI computation fails.
    """
    elapsed = time.time() - start_time
    scene_info = find_best_scene(
        ee_module, point["lat"], point["lon"], start_date, end_date, max_cloud
    )
    if scene_info is None:
        _print_progress(idx, total, point["region"], point["lat"], point["lon"],
                        "no scene", elapsed)
        return None

    shsi = compute_shsi_for_point(
        ee_module, point["lat"], point["lon"], scene_info["scene_id"]
    )
    if shsi is None:
        _print_progress(idx, total, point["region"], point["lat"], point["lon"],
                        "SHSI error", elapsed)
        return None

    result = {
        "region": point["region"],
        "lat": point["lat"],
        "lon": point["lon"],
        "date": scene_info["date"],
        "scene_id": scene_info["scene_id"],
        "cloud": scene_info["cloud"],
        "shsi_mean": shsi["shsi_mean"],
        "shsi_max": shsi["shsi_max"],
    }

    _print_progress(idx, total, point["region"], point["lat"], point["lon"],
                    f"SHSI mean={shsi['shsi_mean']:.4f}", elapsed)
    return result


# ---------------------------------------------------------------------------
# Phase 3 — download one thumbnail
# ---------------------------------------------------------------------------

def download_candidate_thumbnail(
    ee_module: Any,
    candidate: dict[str, Any],
    output_dir: Path,
    idx: int,
    total: int,
) -> dict[str, Any] | None:
    """Download thumbnail for one candidate and save to disk.

    Returns the entry dict (with thumbnail_path added) or None on failure.
    """
    try:
        thumb_bytes = download_thumbnail(
            ee_module, candidate["lat"], candidate["lon"], candidate["scene_id"]
        )
        if thumb_bytes is None:
            print(f"  [{idx + 1}/{total}] Download failed for "
                  f"{candidate['region']} ({candidate['lat']:.4f}, {candidate['lon']:.4f})")
            return None

        # Build filename
        scene_short = (candidate["scene_id"][:8]
                       if len(candidate["scene_id"]) >= 8
                       else candidate["scene_id"])
        fname = (f"{candidate['region']}_{candidate['date']}_"
                 f"shsi{candidate['shsi_mean']:.4f}_"
                 f"{candidate['lat']}_{candidate['lon']}_"
                 f"{scene_short}.png")
        fname = "".join(c if c.isalnum() or c in "._-" else "_" for c in fname)

        (output_dir / fname).write_bytes(thumb_bytes)

        entry = {**candidate, "thumbnail_path": fname}
        print(f"  [{idx + 1}/{total}] Saved {fname}")
        return entry
    except Exception as exc:
        print(f"  [{idx + 1}/{total}] Error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def _print_progress(
    idx: int,
    total: int,
    region: str,
    lat: float,
    lon: float,
    status: str,
    elapsed: float,
) -> None:
    """Print a single progress line."""
    pct = 100.0 * (idx + 1) / total
    if idx > 0 and elapsed > 0:
        rate = (idx + 1) / elapsed
        remaining_s = (total - idx - 1) / rate if rate > 0 else 0
        eta = time.strftime("%H:%M:%S", time.gmtime(remaining_s))
    else:
        eta = "?"
    loc = f"({lat:.4f}, {lon:.4f})"
    print(f"  [{idx + 1}/{total}] ({pct:.0f}%) {region} {loc} | {status} | ETA {eta}")


# ---------------------------------------------------------------------------
# Review HTML
# ---------------------------------------------------------------------------

def build_review_html(
    entries: list[dict[str, Any]], summary: dict[str, Any]
) -> str:
    """Build a self-contained review HTML page in the KNN card-grid format."""
    cards = []
    for row in sorted(
        entries,
        key=lambda item: (-float(item.get("shsi_mean", 0)), item.get("region", "")),
    ):
        cards.append(f"""
            <article class="card">
              <img src="{html.escape(row['thumbnail_path'])}" alt="candidate">
              <div class="meta"><strong>{html.escape(row['region'])}</strong> &#183; {html.escape(row['date'])}</div>
              <div class="meta">SHSI mean: {row['shsi_mean']:.4f} &#183; max: {row['shsi_max']:.4f}</div>
              <div class="meta">({row['lat']:.4f}, {row['lon']:.4f}) &#183; cloud {row['cloud']:.1f}%</div>
            </article>""")

    region_rows = "".join(
        f"<tr><td>{html.escape(region)}</td><td>{count}</td></tr>"
        for region, count in sorted(
            summary.get("region_counts", {}).items(),
            key=lambda item: (-item[1], item[0]),
        )
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SHSI Candidate Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #f5f6fa; color: #1f2937; }}
    header {{ background: linear-gradient(135deg, #111827, #0f172a); color: white; padding: 24px; }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .stat {{ background: white; border-radius: 12px; padding: 14px 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .value {{ font-size: 28px; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 14px; }}
    .card {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .card img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; }}
    .meta {{ padding: 0 14px 8px; font-size: 13px; color: #374151; }}
    details {{ padding: 0 14px 14px; font-size: 12px; color: #4b5563; }}
    table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-top: 12px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
    th {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <header>
    <h1>SHSI Candidate Review</h1>
    <p>SHSI (B3&#178;/B4) herring spawn candidates from Sentinel-2 BC coast scan.</p>
  </header>
  <main>
    <section class="stats">
      <div class="stat"><div class="label">Grid points</div><div class="value">{summary['points_scanned']}</div></div>
      <div class="stat"><div class="label">SHSI &gt; threshold</div><div class="value">{summary['above_threshold']}</div></div>
      <div class="stat"><div class="label">Top candidates</div><div class="value">{summary['candidate_count']}</div></div>
      <div class="stat"><div class="label">Regions with data</div><div class="value">{summary['regions_with_data']}</div></div>
      <div class="stat"><div class="label">Runtime</div><div class="value">{summary['elapsed_seconds']:.1f}s</div></div>
    </section>

    <h2>By region</h2>
    <table><thead><tr><th>Region</th><th>Candidates</th></tr></thead><tbody>{region_rows}</tbody></table>

    <h2>Candidates (sorted by SHSI mean)</h2>
    <section class="grid">{''.join(cards)}</section>
  </main>
</body>
</html>"""


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/candidates_shsi"),
        help="Output directory (default: data/candidates_shsi)",
    )
    parser.add_argument(
        "--start-year", type=int, default=2024,
        help="Start year (default: 2024, scans Feb 1 – Apr 30)",
    )
    parser.add_argument(
        "--end-year", type=int, default=2024,
        help="End year (default: 2024)",
    )
    parser.add_argument(
        "--grid-spacing", type=float, default=0.02,
        help="Grid spacing in degrees (default: 0.02 ≈ 2.2 km)",
    )
    parser.add_argument(
        "--shsi-threshold", type=float, default=0.02,
        help="Minimum SHSI mean to qualify as candidate (default: 0.02)",
    )
    parser.add_argument(
        "--max-cloud", type=float, default=80,
        help="Maximum cloud percentage (default: 80)",
    )
    parser.add_argument(
        "--top-n", type=int, default=150,
        help="Number of top candidates to download (default: 150)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Concurrent workers (default: 8)",
    )
    parser.add_argument(
        "--project", default="redd-fish",
        help="GEE project name (default: redd-fish)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print grid stats and exit without calling GEE",
    )
    args = parser.parse_args()

    t0 = time.time()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = f"{args.start_year}-02-01"
    end_date = f"{args.end_year}-04-30"

    # ------------------------------------------------------------------
    # 1. Generate grid points
    # ------------------------------------------------------------------
    print("=== Generating grid points ===")
    points = generate_grid_points(REGIONS, args.grid_spacing)
    print(f"  Total grid points: {len(points)} across {len(REGIONS)} regions")

    if not points:
        print("ERROR: No grid points generated. Check region definitions and spacing.")
        return 1

    if args.dry_run:
        print(f"\n=== Dry Run ===")
        print(f"  Regions:        {len(REGIONS)}")
        print(f"  Grid points:    {len(points)}")
        print(f"  Grid spacing:   {args.grid_spacing}° ({args.grid_spacing * 111:.1f} km)")
        print(f"  Date range:     {start_date} to {end_date}")
        print(f"  SHSI threshold: {args.shsi_threshold}")
        print(f"  Top N:          {args.top_n}")
        print(f"  Max cloud:      {args.max_cloud}%")
        print(f"  Workers:        {args.workers}")
        print(f"  Output:         {output_dir.resolve()}")
        return 0

    # ------------------------------------------------------------------
    # 2. Initialize GEE
    # ------------------------------------------------------------------
    print("\n=== Initializing Google Earth Engine ===")
    try:
        import ee  # noqa: F811
        ee.Initialize(project=args.project)
        print(f"  GEE initialized (project: {args.project})")
    except Exception as exc:
        print(f"ERROR: GEE initialization failed: {exc}")
        print("  Ensure you are authenticated: earthengine authenticate")
        return 1

    # ------------------------------------------------------------------
    # 3. Phase 1 — Scan all points, compute SHSI
    # ------------------------------------------------------------------
    print(f"\n=== Phase 1: Scanning {len(points)} grid points ===")
    print(f"  Date range:  {start_date} to {end_date}")
    print(f"  Max cloud:   {args.max_cloud}%")
    print(f"  Workers:     {args.workers}")
    print()

    scan_results: list[dict[str, Any]] = []
    processed = 0
    no_scene = 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    scan_point, ee, point, start_date, end_date,
                    args.max_cloud, idx, len(points), t0,
                ): idx
                for idx, point in enumerate(points)
            }

            for future in as_completed(futures):
                result = future.result()
                processed += 1
                if result is None:
                    no_scene += 1
                else:
                    scan_results.append(result)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Working with partial scan results.")
    except Exception as exc:
        print(f"\n\nError during scan: {exc}")
        import traceback
        traceback.print_exc()

    elapsed_phase1 = time.time() - t0
    print(f"\n=== Phase 1 complete: {len(scan_results)} points with data, "
          f"{no_scene} skipped in {elapsed_phase1:.1f}s ===")

    if not scan_results:
        print("ERROR: No points returned SHSI data. Cannot continue.")
        return 1

    # ------------------------------------------------------------------
    # 4. Phase 2 — Filter top candidates
    # ------------------------------------------------------------------
    print(f"\n=== Phase 2: Filtering top candidates ===")
    print(f"  SHSI threshold: {args.shsi_threshold}")

    above_threshold = [r for r in scan_results if r["shsi_mean"] > args.shsi_threshold]
    above_threshold.sort(key=lambda r: -r["shsi_mean"])
    top_candidates = above_threshold[: args.top_n]

    print(f"  Points above threshold: {len(above_threshold)}")
    print(f"  Top candidates:         {len(top_candidates)}")

    if above_threshold:
        print(f"  SHSI range: {above_threshold[-1]['shsi_mean']:.4f} – "
              f"{above_threshold[0]['shsi_mean']:.4f}")

    if not top_candidates:
        print("ERROR: No candidates above threshold. Try lowering --shsi-threshold.")
        return 1

    # ------------------------------------------------------------------
    # 5. Phase 3 — Download thumbnails for top candidates
    # ------------------------------------------------------------------
    print(f"\n=== Phase 3: Downloading {len(top_candidates)} thumbnails ===")
    print(f"  Workers: {args.workers}")
    print()

    candidate_entries: list[dict[str, Any]] = []
    download_errors = 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_candidate_thumbnail,
                    ee, cand, output_dir, idx, len(top_candidates),
                ): idx
                for idx, cand in enumerate(top_candidates)
            }

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    candidate_entries.append(result)
                else:
                    download_errors += 1
    except KeyboardInterrupt:
        print("\n\nInterrupted during download! Partial results saved.")
    except Exception as exc:
        print(f"\n\nError during download: {exc}")

    # ------------------------------------------------------------------
    # 6. Save manifest, summary, review page
    # ------------------------------------------------------------------
    print(f"\n=== Saving results ===")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(candidate_entries, indent=2), encoding="utf-8")
    print(f"  Manifest: {manifest_path}")

    region_counts: Counter[str] = Counter()
    for row in candidate_entries:
        region_counts[row.get("region", "unknown")] += 1

    elapsed = time.time() - t0
    summary = {
        "points_scanned": len(points),
        "above_threshold": len(above_threshold),
        "candidate_count": len(candidate_entries),
        "regions_with_data": len({r["region"] for r in scan_results}),
        "region_counts": dict(region_counts),
        "elapsed_seconds": round(elapsed, 1),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    review_path = output_dir / "review.html"
    review_path.write_text(build_review_html(candidate_entries, summary), encoding="utf-8")

    # ------------------------------------------------------------------
    # 7. Report
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  SHSI Scan Complete")
    print(f"  {'=' * 60}")
    print(f"  Grid points:              {len(points)}")
    print(f"  Points with data:         {len(scan_results)}")
    print(f"  Skipped (no scene/etc):   {no_scene}")
    print(f"  Above SHSI threshold:     {len(above_threshold)}")
    print(f"  Top N candidates:         {len(top_candidates)}")
    print(f"  Thumbnails downloaded:    {len(candidate_entries)}")
    print(f"  Download errors:          {download_errors}")
    print(f"  Regions with data:        {len({r['region'] for r in scan_results})}")
    print(f"  {'=' * 60}")
    print(f"  Phase 1 scan:             {elapsed_phase1:.1f}s")
    print(f"  Phase 3 download:         {(elapsed - elapsed_phase1):.1f}s")
    print(f"  Total time:               {elapsed:.1f}s")
    print(f"  {'=' * 60}")
    print(f"  Output:                   {output_dir.resolve()}")
    print(f"  Manifest:                 {manifest_path}")
    print(f"  Review page:              {review_path}")

    if candidate_entries:
        print(f"\n  Top 5 SHSI candidates:")
        for i, row in enumerate(candidate_entries[:5]):
            print(f"    {i + 1}. {row['region']} ({row['lat']:.4f}, "
                  f"{row['lon']:.4f})  SHSI={row['shsi_mean']:.4f}  "
                  f"cloud={row['cloud']:.1f}%")
        print(f"\n  Review candidates:")
        print(f"    python -m http.server 8766 --directory {output_dir.parent}")
        print(f"    Then open http://localhost:8766/{output_dir.name}/review.html")

    return 0


if __name__ == "__main__":
    sys.exit(main())
