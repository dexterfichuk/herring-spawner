#!/usr/bin/env python3
"""Download Landsat 8/9 thumbnails → DINOv2 embeddings for positive spawn locations.

HLS collections are not accessible from redd-fish project, so this uses
Landsat 8/9 SR as supplementary temporal coverage (~8-day combined revisit).

To save disk: each thumbnail is downloaded at 256x256, immediately embedded
through DINOv2, and the 384-dim embedding saved as a .npy file (~1.5KB).
The PNG is discarded after embedding.

Usage:
    source .venv/bin/activate
    python scripts/download_hls_thumbnails.py \
        --manifest data/samples/training_manifest.json \
        --output data/landsat_embeddings \
        --max-cloud 50 \
        --years 2023 2024 \
        --delay 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.knn_detector import DINO_TRANSFORM, _pick_device
from scripts.temporal_detector import parse_location_from_filename, parse_observation_date

# Landsat 8/9 Surface Reflectance collections
LANDSAT_COLLECTIONS: dict[str, str] = {
    "L8": "LANDSAT/LC08/C02/T1_L2",
    "L9": "LANDSAT/LC09/C02/T1_L2",
}

MODEL_NAME = "dinov2_vits14"

# Fallback coordinates for DFO-ingress positive files lacking lat/lon in filename
KNOWN_POSITIVE_COORDS: dict[str, tuple[float, float]] = {
    "dfo-verified-breakwater-island": (49.135000, -123.683056),
    "dfo-verified-fan-island": (53.905833, -130.739444),
}


def _make_embedding_filename(
    location_key: str,
    date_str: str,
    source: str,
    cloud_pct: float,
    lat: float,
    lon: float,
    scene_date: str,
) -> str:
    """Create a parseable embedding filename.

    Follows the same convention as PNG filenames but with .npy extension,
    so that parse_location_from_filename() and parse_observation_date() work
    when we substitute '.npy' → '.png' for parsing.
    """
    cloud_int = max(0, min(99, round(cloud_pct)))
    return (
        f"{source}_{location_key}_{date_str}_score0.00_"
        f"cloud{cloud_int}_"
        f"{lat}_{lon}_{scene_date}.npy"
    )


def select_positive_locations(manifest_path: Path) -> list[dict[str, Any]]:
    """Extract ~10 positive locations, prioritizing those with poor S2 coverage.

    Skips most Strait of Georgia points (already well-covered by S2) and
    prioritizes remote/undersampled locations.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_locations: list[dict[str, Any]] = []
    seen_coords: set[tuple[float, float]] = set()

    for filename in manifest.get("positives", []):
        parsed = parse_location_from_filename(filename)
        if parsed is not None:
            lat = float(parsed["lat"])
            lon = float(parsed["lon"])
        else:
            for known_key, (klat, klon) in KNOWN_POSITIVE_COORDS.items():
                if known_key in filename:
                    lat, lon = klat, klon
                    break
            else:
                continue

        obs_date = parse_observation_date(filename)
        region = filename.split("_")[0] if "_" in filename else "unknown"

        rounded = (round(lat, 4), round(lon, 4))
        if rounded not in seen_coords:
            seen_coords.add(rounded)
            all_locations.append({
                "lat": lat,
                "lon": lon,
                "name": region,
                "date": obs_date.isoformat() if obs_date else "",
                "source_file": filename,
            })

    # Prioritize remote locations over SoG
    sog_names = {"SoG", "qualicum", "comox", "denman-island"}
    remote = [loc for loc in all_locations if loc["name"] not in sog_names]
    sog = [loc for loc in all_locations if loc["name"] in sog_names]

    # Take all remote (~8), plus a few SoG points
    selected = remote[:]
    selected.extend(sog[: max(0, 10 - len(remote))])
    # If still under 10, add more SoG
    if len(selected) < 10:
        selected.extend(sog[len(selected) - len(remote):])

    print(f"Selected {len(selected)} locations for Landsat download "
          f"(skipped {len(all_locations) - len(selected)} well-covered ones):")
    for loc in selected:
        sog_flag = " (SoG)" if loc["name"] in sog_names else ""
        print(f"  {loc['name']}: ({loc['lat']:.4f}, {loc['lon']:.4f}){sog_flag}")

    return selected


def query_landsat_scenes(
    ee_module: Any,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_cloud: float,
) -> list[dict[str, Any]]:
    """Query Landsat 8 and 9 collections for scenes at a point."""
    point = ee_module.Geometry.Point(lon, lat)
    results: list[dict[str, Any]] = []

    for source, collection_id in LANDSAT_COLLECTIONS.items():
        try:
            collection = ee_module.ImageCollection(collection_id)
            scenes = (
                collection
                .filterBounds(point)
                .filterDate(start_date, end_date)
                .filter(ee_module.Filter.lte("CLOUD_COVER", max_cloud))
                .sort("CLOUD_COVER")
            )

            scene_ids: list[str] = scenes.aggregate_array("system:index").getInfo()
            clouds: list[float] = scenes.aggregate_array("CLOUD_COVER").getInfo()
            time_starts: list[int] = scenes.aggregate_array("system:time_start").getInfo()

            for sid, cloud, ts in zip(scene_ids, clouds, time_starts):
                dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
                results.append({
                    "scene_id": sid,
                    "source": source,
                    "cloud": float(cloud),
                    "date": dt.strftime("%Y-%m-%d"),
                    "scene_date": dt.strftime("%Y%m%d"),
                })

        except Exception as exc:
            print(f"      {source} error: {exc}")

    results.sort(key=lambda r: (r["date"], r["cloud"], r["source"]))
    return results


def download_thumbnail(
    ee_module: Any,
    lat: float,
    lon: float,
    scene_id: str,
    source: str,
) -> bytes | None:
    """Download a 256x256 RGB thumbnail from a Landsat scene (compact size)."""
    try:
        collection_id = LANDSAT_COLLECTIONS[source]
        scene_img = ee_module.Image(f"{collection_id}/{scene_id}")
        rgb = scene_img.select(["SR_B4", "SR_B3", "SR_B2"])
        region = ee_module.Geometry.Point(lon, lat).buffer(1280).bounds()

        url = rgb.getThumbURL({
            "min": 0,
            "max": 3000,
            "region": region,
            "dimensions": 256,
            "format": "png",
        })

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content

    except requests.Timeout:
        return None
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            print(f"      Rate limited, waiting 10s...")
            time.sleep(10)
        return None
    except Exception:
        return None


def embed_thumbnail(
    png_bytes: bytes, model: torch.nn.Module, device: torch.device
) -> np.ndarray | None:
    """Extract DINOv2 embedding from a PNG thumbnail."""
    try:
        import io
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        tensor = DINO_TRANSFORM(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor)
        return F.normalize(emb, dim=1).cpu().numpy().flatten().astype(np.float32)
    except Exception:
        return None


def process_location(
    loc: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    ee_module: Any,
    output_dir: Path,
    years: list[int],
    max_cloud: float,
    delay: float,
) -> dict[str, Any]:
    """Download Landsat scenes for one location, embed immediately, save .npy."""
    loc_name = loc["name"]
    lat, lon = loc["lat"], loc["lon"]
    loc_output = output_dir / loc_name
    loc_output.mkdir(parents=True, exist_ok=True)

    all_scenes: list[dict[str, Any]] = []
    for year in years:
        year_scenes = query_landsat_scenes(
            ee_module, lat, lon, f"{year}-01-01", f"{year}-05-31", max_cloud
        )
        all_scenes.extend(year_scenes)

    if not all_scenes:
        return {"name": loc_name, "n_scenes": 0, "n_embedded": 0,
                "n_skipped": 0, "n_errors": 0}

    n_downloaded = 0
    n_skipped = 0
    n_errors = 0

    for scene in all_scenes:
        emb_filename = _make_embedding_filename(
            loc_name, scene["date"], scene["source"],
            scene["cloud"], lat, lon, scene["scene_date"],
        )
        emb_path = loc_output / emb_filename

        if emb_path.exists():
            n_skipped += 1
            continue

        png_bytes = download_thumbnail(
            ee_module, lat, lon, scene["scene_id"], scene["source"]
        )
        if png_bytes is None:
            n_errors += 1
            if delay > 0:
                time.sleep(delay)
            continue

        embedding = embed_thumbnail(png_bytes, model, device)
        if embedding is None:
            n_errors += 1
        else:
            np.save(emb_path, embedding)
            n_downloaded += 1

        if delay > 0:
            time.sleep(delay)

    return {
        "name": loc_name,
        "lat": lat,
        "lon": lon,
        "n_scenes": len(all_scenes),
        "n_embedded": n_downloaded,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download → embed Landsat 8/9 for positive spawn locations"
    )
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/samples/training_manifest.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/landsat_embeddings"))
    parser.add_argument("--max-cloud", type=float, default=50.0)
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024],
                        help="Years to process (default: 2023 2024)")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--ee-project", type=str, default="redd-fish")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu, mps, cuda, or auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Init GEE
    try:
        import ee
        ee.Initialize(project=args.ee_project)
        print(f"GEE ready (project: {args.ee_project})")
    except Exception as exc:
        print(f"GEE error: {exc}")
        return 1

    # Load DINOv2
    device = _pick_device() if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME).eval().to(device)
    print(f"DINOv2 {MODEL_NAME} loaded")

    # Select positive locations
    manifest_path = args.manifest
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    print(f"\nSelecting positive locations...")
    locations = select_positive_locations(manifest_path)
    print(f"\nProcessing {len(locations)} locations "
          f"(years={args.years}, cloud<{args.max_cloud}%)...")

    # Process each location
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    location_stats: list[dict[str, Any]] = []

    for idx, loc in enumerate(locations):
        print(f"\n[{idx + 1}/{len(locations)}] {loc['name']} "
              f"({loc['lat']:.4f}, {loc['lon']:.4f})")
        stats = process_location(
            loc, model, device, ee, output_dir,
            args.years, args.max_cloud, args.delay,
        )
        location_stats.append(stats)
        print(f"  {stats['n_embedded']} embedded, "
              f"{stats['n_skipped']} cached, {stats['n_errors']} errors "
              f"({stats['n_scenes']} total)")

    # Summary
    total_embedded = sum(s["n_embedded"] for s in location_stats)
    total_skipped = sum(s["n_skipped"] for s in location_stats)
    total_errors = sum(s["n_errors"] for s in location_stats)
    total_scenes = sum(s["n_scenes"] for s in location_stats)

    # Estimate disk usage
    emb_bytes = 0
    for npy in sorted(output_dir.rglob("*.npy")):
        emb_bytes += npy.stat().st_size

    summary = {
        "description": "Landsat 8/9 DINOv2 embeddings for positive spawn locations",
        "model": MODEL_NAME,
        "collections": LANDSAT_COLLECTIONS,
        "years": args.years,
        "max_cloud": args.max_cloud,
        "locations_processed": len(locations),
        "total_scenes_found": total_scenes,
        "total_embedded": total_embedded,
        "total_skipped": total_skipped,
        "total_errors": total_errors,
        "embedding_size_bytes": emb_bytes,
        "embedding_dim": 384,
        "locations": location_stats,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 50}")
    print(f"Complete: {total_embedded} embeddings from {total_scenes} scenes")
    print(f"  Embedded: {total_embedded}")
    print(f"  Skipped (cached): {total_skipped}")
    print(f"  Errors: {total_errors}")
    print(f"  Disk: {emb_bytes / 1024:.1f} KB ({emb_bytes / (1024**2):.2f} MB)")
    print(f"  Output: file://{output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
