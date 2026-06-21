#!/usr/bin/env python3
"""
Reproduction script: fetch exact Sentinel-2 thumbnails from labeled_locations.csv.

Reads the CSV of labeled spawn/no-spawn locations and re-downloads the exact
same 512x512 true-color crops from Microsoft Planetary Computer STAC.

Usage:
  .venv/bin/python3 scripts/fetch_from_csv.py labeled_locations.csv --output-dir ./data/spawn_images
  .venv/bin/python3 scripts/fetch_from_csv.py labeled_locations.csv --filter spawn --limit 10 --output-dir ./test

Output:
  {output-dir}/*.png — true-color thumbnails
  {output-dir}/reproduction_manifest.json — metadata for each downloaded image

Requirements:
  pip install pystac-client planetary-computer rasterio Pillow
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from pystac_client import Client as STACClient
from rasterio.windows import Window

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BAND_R, BAND_G, BAND_B = "B04", "B03", "B02"
WINDOW_SIZE_PX = 512


def percent_clip(data: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    if data.size == 0:
        return np.zeros_like(data, dtype=np.uint8)
    p_low, p_high = np.percentile(data[data > 0], [low, high])
    if p_high - p_low < 1e-6:
        p_low, p_high = data.min(), data.max()
    clipped = np.clip(data, p_low, p_high)
    return ((clipped - p_low) / max(p_high - p_low, 1e-6) * 255).astype(np.uint8)


def download_scene_crop(
    catalog: STACClient,
    scene_id: str,
    lon: float,
    lat: float,
    output_path: Path,
) -> bool:
    """Download a 512x512 true-color PNG centered on (lon, lat) from a specific scene."""
    import planetary_computer as pc

    try:
        item = catalog.get_collection("sentinel-2-l2a").get_item(scene_id)
    except Exception:
        return False
    if item is None:
        return False

    for b in (BAND_R, BAND_G, BAND_B):
        if b not in item.assets:
            return False

    try:
        rh = pc.sign(item.assets[BAND_R].href)
        with rasterio.open(rh) as sr:
            crs, tr = sr.crs, sr.transform
            t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            ux, uy = t.transform(lon, lat)
            col = int(round((~tr * (ux, uy))[0]))
            row = int(round((~tr * (ux, uy))[1]))
            hf = WINDOW_SIZE_PX // 2
            co, ro = col - hf, row - hf
            h, w = sr.shape
            co = max(0, min(co, w - WINDOW_SIZE_PX))
            ro = max(0, min(ro, h - WINDOW_SIZE_PX))
            win = Window(co, ro, WINDOW_SIZE_PX, WINDOW_SIZE_PX)

            gh = pc.sign(item.assets[BAND_G].href)
            bh = pc.sign(item.assets[BAND_B].href)
            red = sr.read(1, window=win).astype(np.float32) / 10000.0
            green = rasterio.open(gh).read(1, window=win).astype(np.float32) / 10000.0
            blue = rasterio.open(bh).read(1, window=win).astype(np.float32) / 10000.0

        rgb = np.stack([percent_clip(red * 255), percent_clip(green * 255), percent_clip(blue * 255)], axis=-1)
        from PIL import Image
        Image.fromarray(rgb, mode="RGB").save(str(output_path), optimize=True)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Reproduce labeled thumbnails from CSV")
    parser.add_argument("csv_path", help="Path to labeled_locations.csv")
    parser.add_argument("--output-dir", default="./spawn_images", help="Output directory")
    parser.add_argument("--filter", choices=["spawn", "no-spawn", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Max images to download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter
    if args.filter != "all":
        rows = [r for r in rows if r["label"] == args.filter]
    if args.limit:
        rows = rows[: args.limit]

    print(f"Reproducing {len(rows)} images from {args.csv_path}")
    print(f"Output: {output_dir}")

    catalog = STACClient.open(STAC_URL)
    print("Connected to Planetary Computer STAC.\n")

    manifest = []
    ok, fail = 0, 0
    for i, row in enumerate(rows):
        scene_id = row["scene_id"]
        lat, lon = float(row["lat"]), float(row["lon"])
        fname = f"{row['region']}_{row['scene_date']}_{lat}_{lon}.png"
        out_path = output_dir / fname

        print(f"[{i+1}/{len(rows)}] {row['label']} {row['region']} {row['scene_date']} — {scene_id[:50]}...", end=" ")

        if scene_id and download_scene_crop(catalog, scene_id, lon, lat, out_path):
            ok += 1
            print("OK")
            manifest.append({**row, "reproduced": True, "filename": fname})
        else:
            fail += 1
            print("FAILED — scene not found or download error")
            manifest.append({**row, "reproduced": False, "filename": None})

        time.sleep(0.2)

    with open(output_dir / "reproduction_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone: {ok} OK, {fail} failed")
    print(f"Images + manifest saved to {output_dir}")


if __name__ == "__main__":
    main()
