#!/usr/bin/env python3
"""Validate whether SVM classifier detects actual spawn events vs shoreline memorization.

For the top N candidates from a scan run, find off-season (summer) Sentinel-2
thumbnails at the same coordinates, score them with the same SVM, and report
the false positive rate. High FP rate (>30%) means the model has learned
shoreline appearance. Low FP rate (<10%) means it detects actual spawn events.

Usage:
    python scripts/validate_candidates.py \\
        --candidates data/candidates_v2/manifest.json \\
        --output data/candidates_v2/validation_report.html

Output:
    - validation_report.html with stats and side-by-side comparisons
    - Off-season thumbnails cached in data/candidates_v2/offseason/
"""

import argparse
import html
import json
import os
import pickle
import sys
import time
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Configuration (must match scan_bc_coast.py exactly)
# ---------------------------------------------------------------------------
MODEL_NAME = "dinov2_vits14"
EMBED_DIM = 384

DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Off-season window: summer when there is no herring spawn on the BC coast
OFF_START = "2024-06-01"
OFF_END = "2024-07-31"
MAX_CLOUD = 50.0

# ---------------------------------------------------------------------------
# GEE helpers (mirrored from scan_bc_coast.py)
# ---------------------------------------------------------------------------

def find_best_scene(
    ee_module: Any,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_cloud: float,
) -> dict[str, Any] | None:
    """Find the single best (lowest cloud) Sentinel-2 scene for a point."""
    try:
        collection = ee_module.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        point = ee_module.Geometry.Point(lon, lat)

        scenes = (
            collection
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee_module.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        scene_ids = scenes.aggregate_array("system:index").getInfo()
        clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()

        if not scene_ids:
            return None

        best_idx = 0
        sid = scene_ids[best_idx]
        return {
            "scene_id": sid,
            "cloud": float(clouds[best_idx]),
            "date": f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}",
            "lat": lat,
            "lon": lon,
        }
    except Exception as exc:
        return None


def download_thumbnail(
    ee_module: Any,
    lat: float,
    lon: float,
    scene_id: str,
) -> bytes | None:
    """Download a 512x512 RGB thumbnail from a Sentinel-2 scene."""
    try:
        scene_img = ee_module.Image(
            f"COPERNICUS/S2_SR_HARMONIZED/{scene_id}"
        )
        rgb = scene_img.select(["B4", "B3", "B2"])
        region = ee_module.Geometry.Point(lon, lat).buffer(1280).bounds()

        url = rgb.getThumbURL({
            "min": 0,
            "max": 3000,
            "region": region,
            "dimensions": 512,
            "format": "png",
        })

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_thumbnail(
    png_bytes: bytes,
    model: torch.nn.Module,
    device: torch.device,
    svm_classifier: Any,
) -> float | None:
    """Compute SVM decision function score for a PNG thumbnail."""
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        tensor = DINO_TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(tensor)
        emb = F.normalize(emb, dim=1).cpu().numpy().flatten()

        score = float(svm_classifier.decision_function(emb.reshape(1, -1))[0])
        return score
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-candidate validation
# ---------------------------------------------------------------------------

def validate_one_candidate(
    candidate: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    svm_classifier: Any,
    ee_module: Any,
    offseason_cache: Path,
) -> dict[str, Any]:
    """Validate a single candidate by finding and scoring an off-season scene.

    Returns a dict with validation results, or None values if off-season
    scene could not be found.
    """
    lat = candidate["lat"]
    lon = candidate["lon"]
    spawn_score = candidate["score"]
    region = candidate["region"]
    spawn_thumbnail = candidate.get("thumbnail_path", "")

    # Search for off-season scene
    scene = find_best_scene(ee_module, lat, lon, OFF_START, OFF_END, MAX_CLOUD)
    if scene is None:
        return {
            "region": region,
            "lat": lat,
            "lon": lon,
            "spawn_score": spawn_score,
            "spawn_thumbnail": spawn_thumbnail,
            "off_score": None,
            "off_date": None,
            "off_scene_id": None,
            "off_thumbnail_path": None,
            "is_false_positive": None,
            "status": "no_offseason_scene",
        }

    # Check cache for off-season thumbnail
    safe_name = (
        f"{region}_{lat}_{lon}_off_{scene['date']}_{scene['scene_id'][:8]}".replace(
            ".", "_"
        ).replace("-", "_")
        + ".png"
    )
    cached_path = offseason_cache / safe_name

    if cached_path.exists():
        png_bytes = cached_path.read_bytes()
        thumb_path = str(cached_path.relative_to(cached_path.parent.parent.parent))
    else:
        png_bytes = download_thumbnail(ee_module, lat, lon, scene["scene_id"])
        if png_bytes is None:
            return {
                "region": region,
                "lat": lat,
                "lon": lon,
                "spawn_score": spawn_score,
                "spawn_thumbnail": spawn_thumbnail,
                "off_score": None,
                "off_date": scene["date"],
                "off_scene_id": scene["scene_id"],
                "off_thumbnail_path": None,
                "is_false_positive": None,
                "status": "download_error",
            }
        offseason_cache.mkdir(parents=True, exist_ok=True)
        cached_path.write_bytes(png_bytes)
        thumb_path = str(cached_path.relative_to(cached_path.parent.parent.parent))

    # Score the off-season thumbnail
    off_score = score_thumbnail(png_bytes, model, device, svm_classifier)
    if off_score is None:
        return {
            "region": region,
            "lat": lat,
            "lon": lon,
            "spawn_score": spawn_score,
            "spawn_thumbnail": spawn_thumbnail,
            "off_score": None,
            "off_date": scene["date"],
            "off_scene_id": scene["scene_id"],
            "off_thumbnail_path": thumb_path,
            "is_false_positive": None,
            "status": "scoring_error",
        }

    is_fp = bool(off_score > 0)

    return {
        "region": region,
        "lat": lat,
        "lon": lon,
        "spawn_score": spawn_score,
        "spawn_thumbnail": spawn_thumbnail,
        "off_score": off_score,
        "off_date": scene["date"],
        "off_scene_id": scene["scene_id"],
        "off_thumbnail_path": thumb_path,
        "is_false_positive": is_fp,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: list[dict[str, Any]],
    output_path: Path,
    n_requested: int,
    elapsed: float,
    svm_n_train: int | str,
    svm_n_pos: int | str,
    svm_n_neg: int | str,
    svm_full_acc: float,
    svm_cv_mean: float,
    svm_cv_std: float,
) -> None:
    """Generate the validation report HTML."""
    total = len(results)
    validated = sum(1 for r in results if r["status"] == "ok")
    fps = sum(1 for r in results if r["is_false_positive"])
    fpr = (fps / validated * 100) if validated > 0 else 0.0
    no_scene = sum(1 for r in results if r["status"] == "no_offseason_scene")
    dl_errors = sum(1 for r in results if r["status"] == "download_error")
    score_errors = sum(1 for r in results if r["status"] == "scoring_error")

    # Interpretation
    if validated > 0:
        if fpr > 30:
            interpretation = (
                "<span style='color:#dc3545;font-weight:bold;'>HIGH FALSE POSITIVE RATE</span> — "
                "The model is likely learning shoreline/water appearance, not spawn events. "
                "Consider collecting more diverse negative training samples including off-season imagery."
            )
        elif fpr < 10:
            interpretation = (
                "<span style='color:#28a745;font-weight:bold;'>LOW FALSE POSITIVE RATE</span> — "
                "The model appears to detect actual spawn events rather than shoreline appearance. "
                f"Only {fps}/{validated} off-season images triggered a spawn prediction."
            )
        else:
            interpretation = (
                "<span style='color:#ffc107;font-weight:bold;'>MODERATE FALSE POSITIVE RATE</span> — "
                f"{fpr:.1f}% of off-season images are classified as spawn. "
                "Some shoreline/water patterns may overlap with spawn signals. "
                "Review the side-by-side comparisons below to identify patterns."
            )
    else:
        interpretation = "No candidates could be validated (no off-season scenes found)."

    # Sort: false positives first (highest off_score), then by spawn score
    valid_results = [r for r in results if r["status"] == "ok"]
    valid_results.sort(
        key=lambda r: (-r["is_false_positive"], -abs(r["off_score"]) if r["off_score"] is not None else 0)
    )

    # Build rows
    rows_html = ""
    for i, r in enumerate(valid_results[:20], 1):
        spawn_score = f"{r['spawn_score']:.4f}"
        off_score = f"{r['off_score']:.4f}" if r['off_score'] is not None else "N/A"

        # Thumbnail paths
        spawn_thumb = r["spawn_thumbnail"]
        off_thumb = r["off_thumbnail_path"] or ""

        # Row highlight
        row_class = " class='fp-row'" if r["is_false_positive"] else ""

        # Score cell style
        spawn_style = "score-high" if r["spawn_score"] > 0.5 else "score-low"
        off_style = "score-fp" if r["is_false_positive"] else "score-tn"

        rows_html += f"""<tr{row_class}>
    <td>{i}</td>
    <td>{html.escape(r["region"])}</td>
    <td class="{spawn_style}">{spawn_score}</td>
    <td class="{off_style}">{off_score}</td>
    <td>{'🚫 FALSE POSITIVE' if r['is_false_positive'] else '✅ True Negative'}</td>
    <td><img src="{html.escape(spawn_thumb)}" width="256" loading="lazy"></td>
    <td><img src="{html.escape(off_thumb)}" width="256" loading="lazy"></td>
    <td style="font-size:0.8em">{html.escape(r.get("off_date", ""))}</td>
</tr>
"""

    # Failed rows (no scene, errors)
    failed = [r for r in results if r["status"] != "ok"]
    failed_rows = ""
    for i, r in enumerate(failed, 1):
        status_label = {
            "no_offseason_scene": "No off-season scene",
            "download_error": "Download error",
            "scoring_error": "Scoring error",
        }.get(r["status"], r["status"])
        failed_rows += f"""<tr class='failed-row'>
    <td>{validated + i}</td>
    <td>{html.escape(r["region"])}</td>
    <td>{r['spawn_score']:.4f}</td>
    <td colspan="2">{status_label}</td>
    <td colspan="3"><span style="color:#999">(skipped — no off-season data)</span></td>
</tr>
"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Herring Spawn Candidate Validation Report</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
h1 {{ margin-top: 0; color: #1a1a2e; }}
h2 {{ color: #16213e; margin-top: 30px; }}
.summary {{ background: white; padding: 24px; border-radius: 8px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
.stat {{ background: #f8f9fa; padding: 14px; border-radius: 6px; text-align: center; }}
.stat-label {{ font-size: 0.75em; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
.stat-value {{ font-size: 1.6em; font-weight: bold; color: #333; }}
.stat-value.danger {{ color: #dc3545; }}
.stat-value.success {{ color: #28a745; }}
.stat-value.warning {{ color: #ffc107; }}
.interpretation {{ background: #fff8e1; border-left: 4px solid #ffc107; padding: 14px 18px; margin-top: 16px; border-radius: 4px; font-size: 0.95em; line-height: 1.5; }}
.interpretation.danger {{ background: #fbe9e7; border-left-color: #dc3545; }}
.interpretation.success {{ background: #e8f5e9; border-left-color: #28a745; }}
table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
th {{ background: #1a1a2e; color: white; padding: 10px 8px; text-align: left; position: sticky; top: 0; font-size: 0.85em; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: middle; font-size: 0.9em; }}
tr:hover {{ background: #f0f7ff; }}
.fp-row {{ background: #fff5f5; }}
.fp-row:hover {{ background: #ffe0e0; }}
.failed-row {{ color: #999; }}
.failed-row td {{ font-style: italic; }}
.score-high {{ color: #28a745; font-weight: bold; }}
.score-low {{ color: #6c757d; }}
.score-fp {{ color: #dc3545; font-weight: bold; }}
.score-tn {{ color: #28a745; }}
img {{ border-radius: 4px; border: 1px solid #ddd; display: block; }}
.meta {{ color: #666; font-size: 0.85em; margin-top: 16px; padding: 12px; background: #f8f9fa; border-radius: 6px; }}
.method {{ background: #e3f2fd; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px; font-size: 0.9em; line-height: 1.5; }}
.method code {{ background: #e0e0e0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
.collapsible {{ cursor: pointer; user-select: none; }}
.collapsible::before {{ content: '▶ '; }}
.collapsible.active::before {{ content: '▼ '; }}
.failed-content {{ display: none; }}
.failed-content.show {{ display: block; }}
</style>
</head>
<body>

<h1>🐟 Herring Spawn Candidate Validation Report</h1>

<div class="method">
    <strong>Method:</strong> For the top <code>{n_requested}</code> candidates by SVM score, we search for the best (least cloudy) Sentinel-2 scene during the off-season window
    (<code>{OFF_START}</code> to <code>{OFF_END}</code>) at the same coordinates. If a scene is found, we download the RGB thumbnail and score it with
    the same SVM classifier. A <strong>false positive</strong> is when the off-season image scores above 0 (the spawn threshold).
    This tells us whether the model has learned shoreline appearance rather than actual spawn events.
</div>

<div class="summary">
    <div class="summary-grid">
        <div class="stat">
            <div class="stat-label">Candidates Requested</div>
            <div class="stat-value">{n_requested}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Validated (had off-season scene)</div>
            <div class="stat-value">{validated}</div>
        </div>
        <div class="stat">
            <div class="stat-label">False Positives</div>
            <div class="stat-value {'danger' if fps > 0 else 'success'}">{fps}</div>
        </div>
        <div class="stat">
            <div class="stat-label">False Positive Rate</div>
            <div class="stat-value {'danger' if fpr > 30 else 'warning' if fpr > 10 else 'success'}">{fpr:.1f}%</div>
        </div>
        <div class="stat">
            <div class="stat-label">No Off-Season Scene</div>
            <div class="stat-value">{no_scene}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Download / Score Errors</div>
            <div class="stat-value">{dl_errors + score_errors}</div>
        </div>
    </div>
    <div class="interpretation {'danger' if fpr > 30 else 'success' if fpr < 10 else ''}">
        {interpretation}
    </div>
</div>

<h2>📊 Top Validated Candidates — Side by Side</h2>
<p style="color:#666; font-size:0.9em;">Showing first {min(20, validated)} validated candidates, sorted by false-positive status (FP first) then spawn score. False positive rows highlighted in red.</p>

<table>
<thead>
<tr>
    <th>#</th>
    <th>Region</th>
    <th>Spawn Score</th>
    <th>Off-Season Score</th>
    <th>Verdict</th>
    <th>Spawn Season (Mar-Apr)</th>
    <th>Off-Season (Jun-Jul)</th>
    <th>Off Date</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>

{'<h2>⚠️ Skipped Candidates</h2>' + '''
<table>
<thead><tr><th>#</th><th>Region</th><th>Spawn Score</th><th colspan="2">Reason</th><th colspan="3">Details</th></tr></thead>
<tbody>
''' + failed_rows + '''
</tbody>
</table>''' if failed_rows else ''}

<div class="meta">
    <strong>Report generated:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")} |
    <strong>Validation completed in:</strong> {elapsed:.1f}s |
    <strong>SVM model:</strong> RBF kernel, trained on {svm_n_train} samples ({svm_n_pos} pos, {svm_n_neg} neg) |
    <strong>Full accuracy:</strong> {svm_full_acc:.1%} |
    <strong>CV accuracy:</strong> {svm_cv_mean:.1%} ± {svm_cv_std:.1%}
</div>

</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"\n  Report saved to: {output_path.resolve()}")


# ===================================================================
# Main
# ===================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SVM candidates with off-season negative sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--candidates",
        default="data/candidates_v2/manifest.json",
        help="Path to candidate manifest JSON (default: data/candidates_v2/manifest.json)",
    )
    parser.add_argument(
        "--output",
        default="data/candidates_v2/validation_report.html",
        help="Output HTML report path (default: data/candidates_v2/validation_report.html)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=200,
        help="Number of top candidates to validate (default: 200)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers (default: 8)",
    )
    parser.add_argument(
        "--off-start",
        default=OFF_START,
        help="Off-season start date YYYY-MM-DD (default: 2024-06-01)",
    )
    parser.add_argument(
        "--off-end",
        default=OFF_END,
        help="Off-season end date YYYY-MM-DD (default: 2024-07-31)",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=MAX_CLOUD,
        help="Maximum cloud percentage for off-season scenes (default: 50)",
    )
    parser.add_argument(
        "--svm-model",
        default="data/models/dinov2_svm.pkl",
        help="Path to trained SVM model (default: data/models/dinov2_svm.pkl)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent

    candidates_path = repo_root / args.candidates
    output_path = repo_root / args.output
    svm_model_path = repo_root / args.svm_model
    offseason_cache = output_path.parent / "offseason"

    global OFF_START, OFF_END, MAX_CLOUD
    OFF_START = args.off_start
    OFF_END = args.off_end
    MAX_CLOUD = args.max_cloud

    # ------------------------------------------------------------------
    # 1. Load candidates
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Loading candidates...")
    if not candidates_path.exists():
        print(f"  ERROR: Candidates manifest not found: {candidates_path}")
        return 1

    with open(candidates_path) as f:
        all_candidates: list[dict[str, Any]] = json.load(f)

    # Sort by score descending and take top N
    all_candidates.sort(key=lambda c: c["score"], reverse=True)
    candidates = all_candidates[: args.max_candidates]

    print(f"  Total candidates in manifest: {len(all_candidates)}")
    print(f"  Validating top {len(candidates)} by SVM score")
    print(f"  Score range: {candidates[-1]['score']:.4f} to {candidates[0]['score']:.4f}")
    print(f"  Regions: {sorted(set(c['region'] for c in candidates))}")

    # ------------------------------------------------------------------
    # 2. Load DINOv2 model
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Loading DINOv2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    try:
        model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
        model.eval()
        model = model.to(device)
    except Exception as exc:
        print(f"  ERROR: Failed to load DINOv2 model: {exc}")
        return 1
    print(f"  Model: {MODEL_NAME}")

    # ------------------------------------------------------------------
    # 3. Load SVM classifier
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Loading SVM classifier from {svm_model_path}...")
    if not svm_model_path.exists():
        print(f"  ERROR: SVM model not found: {svm_model_path}")
        return 1

    try:
        with open(svm_model_path, "rb") as f:
            svm_data = pickle.load(f)
        svm_classifier = svm_data["svm"]
        svm_n_train = svm_data.get("n_train", "?")
        svm_n_pos = svm_data.get("n_pos", "?")
        svm_n_neg = svm_data.get("n_neg", "?")
        svm_full_acc = svm_data.get("full_accuracy", 0.0) or svm_data.get("test_accuracy", 0.0)
        svm_cv_mean = svm_data.get("cv_accuracy_mean", 0.0)
        svm_cv_std = svm_data.get("cv_accuracy_std", 0.0)
        sep = svm_data.get("separation", 0.0)
        print(f"  SVM loaded: {svm_n_train} samples ({svm_n_pos} pos, {svm_n_neg} neg)")
        print(f"  Full accuracy: {svm_full_acc:.4f}, CV: {svm_cv_mean:.4f} +/- {svm_cv_std:.4f}")
        print(f"  Separation: {sep:.4f}")
    except Exception as exc:
        print(f"  ERROR: Failed to load SVM model: {exc}")
        return 1

    # ------------------------------------------------------------------
    # 4. Initialize GEE
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Initializing Google Earth Engine...")
    try:
        import ee
        ee.Initialize(project="redd-fish")
        print("  GEE initialized (project: redd-fish)")
    except Exception as exc:
        print(f"  ERROR: GEE initialization failed: {exc}")
        return 1

    # ------------------------------------------------------------------
    # 5. Validate candidates concurrently
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Validating {len(candidates)} candidates with {args.workers} workers...")
    print(f"  Off-season window: {OFF_START} to {OFF_END}")
    print(f"  Max cloud: {MAX_CLOUD}%")
    print()

    start_time = time.time()
    results: list[dict[str, Any]] = [None] * len(candidates)  # pre-allocate

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {}
            for idx, candidate in enumerate(candidates):
                future = executor.submit(
                    validate_one_candidate,
                    candidate, model, device, svm_classifier, ee, offseason_cache,
                )
                future_map[future] = idx

            done_count = 0
            for future in as_completed(future_map):
                idx = future_map[future]
                result = future.result()
                results[idx] = result
                done_count += 1

                # Progress
                status_icon = {
                    "ok": "✓",
                    "no_offseason_scene": "⊘",
                    "download_error": "✗",
                    "scoring_error": "✗",
                }.get(result["status"], "?")

                fp_str = ""
                if result["is_false_positive"] is True:
                    fp_str = " FP! "
                elif result["is_false_positive"] is False:
                    fp_str = " TN  "

                rgn = result["region"]
                ss = result["spawn_score"]
                os_ = f"{result['off_score']:.4f}" if result["off_score"] is not None else "N/A"
                print(
                    f"  [{done_count}/{len(candidates)}] {status_icon} "
                    f"{rgn:20s} spawn={ss:.4f} off={os_:>8s}{fp_str}"
                )

    except KeyboardInterrupt:
        print("\n\n  Interrupted! Partial results saved.")
        # Fill None results with placeholder
        for i, r in enumerate(results):
            if r is None:
                results[i] = {
                    "region": candidates[i]["region"],
                    "lat": candidates[i]["lat"],
                    "lon": candidates[i]["lon"],
                    "spawn_score": candidates[i]["score"],
                    "spawn_thumbnail": candidates[i].get("thumbnail_path", ""),
                    "status": "interrupted",
                    "is_false_positive": None,
                    "off_score": None,
                }

    elapsed = time.time() - start_time

    # Fill any remaining None results
    for i, r in enumerate(results):
        if r is None:
            c = candidates[i]
            results[i] = {
                "region": c["region"],
                "lat": c["lat"],
                "lon": c["lon"],
                "spawn_score": c["score"],
                "spawn_thumbnail": c.get("thumbnail_path", ""),
                "status": "error",
                "is_false_positive": None,
                "off_score": None,
            }

    # ------------------------------------------------------------------
    # 6. Generate report
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Generating validation report...")

    valid_count = sum(1 for r in results if r["status"] == "ok")
    fp_count = sum(1 for r in results if r["is_false_positive"])
    fp_rate = (fp_count / valid_count * 100) if valid_count > 0 else 0.0

    print(f"  Validated: {valid_count} / {len(candidates)}")
    print(f"  False positives: {fp_count} ({fp_rate:.1f}%)")
    print(f"  No off-season scene: {sum(1 for r in results if r['status'] == 'no_offseason_scene')}")
    print(f"  Download/score errors: {sum(1 for r in results if r['status'] in ('download_error', 'scoring_error'))}")

    generate_report(
        results, output_path, args.max_candidates, elapsed,
        svm_n_train, svm_n_pos, svm_n_neg,
        svm_full_acc, svm_cv_mean, svm_cv_std,
    )

    # Print final summary
    print(f"\n{'=' * 60}")
    print("  VALIDATION RESULTS")
    print(f"  {'=' * 60}")
    print(f"  Candidates requested:       {args.max_candidates}")
    print(f"  Validated (had summer img): {valid_count}")
    print(f"  False positives:            {fp_count} ({fp_rate:.1f}%)")
    print(f"  True negatives:             {valid_count - fp_count}")
    print(f"  No off-season scene:        {sum(1 for r in results if r['status'] == 'no_offseason_scene')}")
    print(f"  Download errors:            {sum(1 for r in results if r['status'] == 'download_error')}")
    print(f"  Scoring errors:             {sum(1 for r in results if r['status'] == 'scoring_error')}")
    print(f"  {'=' * 60}")

    if valid_count > 0:
        if fp_rate > 30:
            print("  ⚠️  HIGH FALSE POSITIVE RATE — model likely learning shoreline")
        elif fp_rate < 10:
            print("  ✅  LOW FALSE POSITIVE RATE — model detects actual spawn events")
        else:
            print("  ⚠️  MODERATE FALSE POSITIVE RATE — review side-by-side comparisons")

    print(f"\n  Report: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
