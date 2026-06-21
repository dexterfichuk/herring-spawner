#!/usr/bin/env python3
"""
Fetch Sentinel-2 true-color thumbnails for DFO herring spawn records.

For each post-2016 DFO spawn record with valid coordinates, queries Microsoft
Planetary Computer STAC for the best Sentinel-2 L2A scene. Crops a 512x512 px
true-color thumbnail centered on the spawn location.

Tries progressively wider time windows and cloud tolerances to maximize yield.
If nothing found, just records "no_scene" and moves on.

Outputs:
  /Volumes/Z Slim/herring-spawn-data/candidates_fresh/*.png
  manifest.json, labels.json

Usage:
  python3 scripts/fetch_candidates.py               # full run
  python3 scripts/fetch_candidates.py --limit 50    # test
  python3 scripts/fetch_candidates.py --resume      # resume from checkpoint
  python3 scripts/fetch_candidates.py --quiet       # minimal logging
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from pystac_client import Client as STACClient
from rasterio.windows import Window

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fetch_candidates")

# Paths
DFO_CSV = Path.home() / "Downloads" / "Pacific_herring_spawn_index_data_2025_EN.csv"
OUTPUT_DIR = Path("/Volumes/Z Slim/herring-spawn-data/candidates_fresh")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
BAND_R, BAND_G, BAND_B = "B04", "B03", "B02"
WINDOW_SIZE_PX = 512
SEARCH_BUFFER_DEG = 0.12


def parse_date(d: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(d.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def percent_clip(data: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    if data.size == 0:
        return np.zeros_like(data, dtype=np.uint8)
    p_low, p_high = np.percentile(data[data > 0], [low, high])
    if p_high - p_low < 1e-6:
        p_low, p_high = data.min(), data.max()
    clipped = np.clip(data, p_low, p_high)
    return ((clipped - p_low) / max(p_high - p_low, 1e-6) * 255).astype(np.uint8)


def find_best_scene(catalog: STACClient, lon: float, lat: float,
                    target_date: datetime) -> dict | None:
    """Search STAC for best S2 scene near (lon, lat) around target_date."""
    attempts = [
        (7, 50),    # tight date + good cloud
        (14, 50),   # wider date, good cloud
        (7, 80),    # tight date, more cloud
        (14, 90),   # wide date, high cloud
        (30, 90),   # very wide, high cloud
    ]
    bbox = [lon - SEARCH_BUFFER_DEG, lat - SEARCH_BUFFER_DEG,
            lon + SEARCH_BUFFER_DEG, lat + SEARCH_BUFFER_DEG]
    seen_ids = set()
    candidates = []

    for days, cloud in attempts:
        ds = target_date - timedelta(days=days)
        de = target_date + timedelta(days=days)
        dt_str = f"{ds.strftime('%Y-%m-%d')}/{de.strftime('%Y-%m-%d')}"
        try:
            search = catalog.search(
                collections=[S2_COLLECTION], bbox=bbox, datetime=dt_str,
                query={"eo:cloud_cover": {"lte": cloud}}, max_items=50)
            for item in search.items():
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    candidates.append(item)
        except Exception:
            continue
        if len(candidates) >= 5:
            break

    if not candidates:
        return None

    best, best_score = None, float("inf")
    for item in candidates:
        cc = item.properties.get("eo:cloud_cover", 100) or 100
        idt = datetime.fromisoformat(item.properties["datetime"][:19])
        dd = abs((idt - target_date).days)
        score = cc * 1.5 + dd
        if score < best_score:
            best_score = score
            best = {"item": item, "cloud_cover": cc, "days_from_target": dd,
                    "scene_id": item.id, "scene_date": item.properties["datetime"][:10]}
    return best


def download_thumbnail(scene: dict, lon: float, lat: float, output_path: Path) -> bool:
    """Download a 512x512 true-color PNG centered on (lon, lat) from a S2 L2A scene."""
    import planetary_computer as pc
    item = scene["item"]
    for b in (BAND_R, BAND_G, BAND_B):
        if b not in item.assets:
            return False
    try:
        rh = pc.sign(item.assets[BAND_R].href)
        with rasterio.open(rh) as sr:
            crs, tr = sr.crs, sr.transform
            t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            ux, uy = t.transform(lon, lat)
            c, r = int(round((~tr * (ux, uy))[0])), int(round((~tr * (ux, uy))[1]))
            hf = WINDOW_SIZE_PX // 2
            co, ro = c - hf, r - hf
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(str(output_path), optimize=True)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch S2 thumbnails for DFO spawn records")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DFO_CSV.exists():
        logger.error(f"DFO CSV not found: {DFO_CSV}")
        sys.exit(1)

    with open(DFO_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_records = list(reader)
    logger.info(f"Total DFO records: {len(all_records)}")

    records = []
    for r in all_records:
        yr, lo, la, ds_ = (r.get(k, "").strip() for k in ("Year", "Longitude", "Latitude", "StartDate"))
        if not (yr and lo and la and ds_):
            continue
        try:
            if int(yr) < 2016:
                continue
            lon, lat = float(lo), float(la)
        except (ValueError, TypeError):
            continue
        sd = parse_date(ds_)
        if sd is None:
            continue
        try:
            length = float(r.get("Length", 0)) if r.get("Length", "").strip() else 0
        except ValueError:
            length = 0
        try:
            width = float(r.get("Width", 0)) if r.get("Width", "").strip() and r["Width"].strip() != "NA" else 0
        except ValueError:
            width = 0
        records.append({
            "region": r.get("Region", "").strip(), "year": int(yr),
            "lon": lon, "lat": lat, "start_date": ds_, "start_date_dt": sd,
            "location_name": r.get("LocationName", "").strip(),
            "length_m": length, "width_m": width,
            "method": r.get("Method", "").strip(),
        })

    logger.info(f"Post-2016 records with valid coords: {len(records)}")
    if args.limit:
        records = records[:args.limit]
        logger.info(f"Limited to {len(records)}")

    existing = {}
    if args.resume and MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            for entry in json.load(f):
                existing[entry["dfo_key"]] = entry
        logger.info(f"Resuming: {len(existing)} already in manifest")

    logger.info("Connecting to Planetary Computer STAC...")
    catalog = STACClient.open(STAC_URL)
    logger.info("Connected.")

    manifest = list(existing.values())
    fetched = len(existing)
    total = len(records)
    quiet = args.quiet

    for i, rec in enumerate(records):
        key = f"{rec['lon']:.4f}_{rec['lat']:.4f}_{rec['start_date']}"
        if key in existing:
            continue
        if args.limit and fetched >= args.limit:
            break

        if not quiet or fetched % 100 == 0:
            logger.info(f"[{fetched+1}/{total}] {rec['region']} {rec['start_date']} @ {rec['lat']:.4f},{rec['lon']:.4f} — {rec['location_name']}")

        scene = find_best_scene(catalog, rec["lon"], rec["lat"], rec["start_date_dt"])

        if scene is None:
            manifest.append({"dfo_key": key, "filename": None, "region": rec["region"],
                "date": rec["start_date"], "lon": rec["lon"], "lat": rec["lat"],
                "location_name": rec["location_name"], "spawn_length_m": rec["length_m"],
                "spawn_width_m": rec["width_m"], "method": rec["method"],
                "scene_id": None, "scene_date": None, "cloud_cover": None, "days_from_spawn": None,
                "status": "no_scene"})
            fetched += 1
            continue

        fname = f"{rec['region'].replace(' ', '_')}_{scene['scene_date']}_{rec['lat']}_{rec['lon']}.png"
        ok = download_thumbnail(scene, rec["lon"], rec["lat"], OUTPUT_DIR / fname)

        if not ok:
            manifest.append({"dfo_key": key, "filename": None, "region": rec["region"],
                "date": rec["start_date"], "lon": rec["lon"], "lat": rec["lat"],
                "location_name": rec["location_name"], "spawn_length_m": rec["length_m"],
                "spawn_width_m": rec["width_m"], "method": rec["method"],
                "scene_id": scene["scene_id"], "scene_date": scene["scene_date"],
                "cloud_cover": scene["cloud_cover"], "days_from_spawn": scene["days_from_target"],
                "status": "download_failed"})
            fetched += 1
            continue

        manifest.append({"dfo_key": key, "filename": fname, "region": rec["region"],
            "date": rec["start_date"], "lon": rec["lon"], "lat": rec["lat"],
            "location_name": rec["location_name"], "spawn_length_m": rec["length_m"],
            "spawn_width_m": rec["width_m"], "method": rec["method"],
            "scene_id": scene["scene_id"], "scene_date": scene["scene_date"],
            "cloud_cover": scene["cloud_cover"], "days_from_spawn": scene["days_from_target"],
            "status": "ok"})
        fetched += 1

        if not quiet:
            logger.info(f"  ok cloud={scene['cloud_cover']:.0f}% ±{scene['days_from_target']}d")
        elif fetched % 100 == 0:
            s = sum(1 for m in manifest if m["status"] == "ok")
            logger.info(f"  [{fetched}/{total}] {s} saved")

        if fetched % 25 == 0:
            with open(MANIFEST_PATH, "w") as f:
                json.dump(manifest, f, indent=2)

        time.sleep(0.1)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    s = sum(1 for m in manifest if m["status"] == "ok")
    ns = sum(1 for m in manifest if m["status"] == "no_scene")
    dl = sum(1 for m in manifest if m["status"] == "download_failed")
    print(f"\n{'='*60}")
    print(f"  FETCH COMPLETE")
    print(f"  Total:     {len(manifest)}")
    print(f"  Saved:     {s}")
    print(f"  No scene:  {ns}")
    print(f"  DL failed: {dl}")
    print(f"  Manifest:  {MANIFEST_PATH}")
    print(f"{'='*60}")
    print(f"\n  Next: python3 scripts/label_app.py")


if __name__ == "__main__":
    main()
