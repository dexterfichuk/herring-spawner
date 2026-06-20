#!/usr/bin/env python3
"""Download pre-spawn baseline Sentinel-2 thumbnails for golden positive locations.

For each known spawn location, finds the best cloud-free Sentinel-2 scene
from the baseline window (spawn_date - 45 days ± 14 days) and downloads a
512x512 RGB thumbnail. Outputs baseline + spawn pairs to data/baseline_pairs/.

Usage:
    python scripts/download_baselines.py
"""
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import ee
import requests

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASELINE_DAYS_BEFORE = 45
WINDOW_DAYS = 14
MAX_CLOUD = 50
BANDS = ["B4", "B3", "B2"]
OUTPUT_DIR = PROJECT_ROOT / "data" / "baseline_pairs"
THUMBNAIL_DIM = 512
THUMBNAIL_MIN = 0
THUMBNAIL_MAX = 3000

# ---- 7 golden positives needing baseline coverage ---------------------------
LOCATIONS = [
    {"region": "Nanaimo 1",      "lat": 49.134865, "lon": -123.676603, "spawn_date": date(2024, 3, 18)},
    {"region": "Nanaimo 2",      "lat": 49.134865, "lon": -123.696603, "spawn_date": date(2024, 3, 18)},
    {"region": "Milbanke Sound 1","lat": 52.544865, "lon": -128.741984, "spawn_date": date(2024, 3, 24)},
    {"region": "Milbanke Sound 2","lat": 52.544865, "lon": -128.721984, "spawn_date": date(2024, 3, 24)},
    {"region": "Milbanke Sound 3","lat": 52.524865, "lon": -128.741984, "spawn_date": date(2024, 3, 24)},
    {"region": "Breakwater Island","lat": 49.13485, "lon": -123.677,    "spawn_date": date(2024, 3, 18)},
    {"region": "Barkley Sound",   "lat": 48.984865, "lon": -125.325486, "spawn_date": date(2024, 5, 22)},
]


def find_best_scene(lat, lon, start, end):
    """Return (scene_id, cloud_pct, date_str) for the least-cloudy S2 scene."""
    coll = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    scenes = (
        coll.filterBounds(ee.Geometry.Point(lon, lat))
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    ids = scenes.aggregate_array("system:index").getInfo()
    clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    if not ids:
        return None
    sid = ids[0]
    return {"scene_id": sid, "cloud": float(clouds[0]),
            "date": f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}"}


def download_thumb(lat, lon, scene_id):
    """Download 512x512 S2 RGB PNG thumbnail. Returns bytes or None."""
    img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{scene_id}")
    rgb = img.select(BANDS)
    region = ee.Geometry.Point(lon, lat).buffer(1280).bounds()
    url = rgb.getThumbURL({
        "min": THUMBNAIL_MIN,
        "max": THUMBNAIL_MAX,
        "region": region,
        "dimensions": THUMBNAIL_DIM,
        "format": "png",
    })
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def main():
    ee.Initialize(project="redd-fish")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for loc in LOCATIONS:
        region = loc["region"]
        lat, lon = loc["lat"], loc["lon"]
        spawn = loc["spawn_date"]
        safe = region.lower().replace(" ", "_")

        print(f"\n{'='*60}")
        print(f"  {region}  (lat={lat}, lon={lon}, spawn={spawn})")
        print(f"  {'='*60}")

        # Spawn window
        s_start = (spawn - timedelta(days=WINDOW_DAYS)).isoformat()
        s_end   = (spawn + timedelta(days=WINDOW_DAYS)).isoformat()
        print(f"  Spawn window: {s_start} → {s_end}")

        spawn_scene = find_best_scene(lat, lon, s_start, s_end)
        if not spawn_scene:
            print(f"  SKIP: No spawn scene found for {region}")
            continue
        print(f"  Spawn scene: {spawn_scene['scene_id']} (cloud={spawn_scene['cloud']:.1f}%)")

        # Baseline window
        base_center = spawn - timedelta(days=BASELINE_DAYS_BEFORE)
        b_start = (base_center - timedelta(days=WINDOW_DAYS)).isoformat()
        b_end   = (base_center + timedelta(days=WINDOW_DAYS)).isoformat()
        print(f"  Baseline window: {b_start} → {b_end}")

        baseline = find_best_scene(lat, lon, b_start, b_end)
        if not baseline:
            # Fallback: wider window
            b_start2 = (base_center - timedelta(days=28)).isoformat()
            b_end2   = (base_center + timedelta(days=28)).isoformat()
            print(f"  Baseline fallback: {b_start2} → {b_end2}")
            baseline = find_best_scene(lat, lon, b_start2, b_end2)
            if not baseline:
                print(f"  SKIP: No baseline scene found for {region}")
                continue

        print(f"  Baseline scene: {baseline['scene_id']} (cloud={baseline['cloud']:.1f}%)")

        # Download thumbnails
        spawn_bytes = download_thumb(lat, lon, spawn_scene["scene_id"])
        base_bytes = download_thumb(lat, lon, baseline["scene_id"])

        base_fname = f"{safe}_baseline_{baseline['date']}.png"
        spawn_fname = f"{safe}_spawn_{spawn_scene['date']}.png"
        (OUTPUT_DIR / base_fname).write_bytes(base_bytes)
        (OUTPUT_DIR / spawn_fname).write_bytes(spawn_bytes)
        print(f"  Saved: {base_fname}  +  {spawn_fname}")

        results.append({
            "region": region,
            "lat": lat, "lon": lon,
            "baseline_scene": baseline["scene_id"],
            "baseline_date": baseline["date"],
            "baseline_cloud": baseline["cloud"],
            "spawn_scene": spawn_scene["scene_id"],
            "spawn_date": spawn_scene["date"],
            "spawn_cloud": spawn_scene["cloud"],
            "baseline_file": base_fname,
            "spawn_file": spawn_fname,
        })

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results: {len(results)}/{len(LOCATIONS)} locations downloaded")
    print(f"  Manifest: {manifest_path}\n")


if __name__ == "__main__":
    main()
