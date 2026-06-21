#!/usr/bin/env python3
"""
Fetch multi-temporal Sentinel-2 sequences for labeled spawn locations.

For each labeled location, fetches ALL available Sentinel-2 scenes from
T-35d to T+21d around the DFO spawn date. Each timestep:
  1. True-color 512x512 PNG (for visualization)
  2. SHSI spectral features vector (for TempCNN training)
  3. GeoRSCLIP embedding (optional, for sequence model)

Output: /Volumes/Z Slim/herring-spawn-data/temporal_dataset/{region}_{lat}_{lon}/
  metadata.json   — location info + list of timesteps
  timesteps/
    {scene_date}/
      rgb.png         — true-color thumbnail
      features.npz    — spectral features [SHSI, G, R, NIR, B-C, Coastal]
      embedding.npz   — GeoRSCLIP 512-d vector (if --embeddings)

Usage:
  .venv/bin/python3 scripts/fetch_temporal.py --label spawn --limit 50
  .venv/bin/python3 scripts/fetch_temporal.py --embeddings  # slower but enables GRU model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from pystac_client import Client as STACClient
from rasterio.windows import Window

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = ["B02", "B03", "B04", "B08", "B01"]  # B, G, R, NIR, Coastal
BAND_ASSETS = {"B02": "B02", "B03": "B03", "B04": "B04", "B08": "B08", "B01": "B01"}
WINDOW_SIZE_PX = 512
SEARCH_BUFFER = 0.005  # tight — we want intersecting scenes, not nearby
TEMPORAL_PRE_DAYS = 35
TEMPORAL_POST_DAYS = 21
MAX_CLOUD = 90

IMAGE_DIR = Path("/Volumes/Z Slim/herring-spawn-data/candidates_fresh")
OUTPUT_DIR = Path("/Volumes/Z Slim/herring-spawn-data/temporal_dataset")


def percent_clip(data: np.ndarray, low=2.0, high=98.0) -> np.ndarray:
    if data.size == 0:
        return np.zeros_like(data, dtype=np.uint8)
    pl, ph = np.percentile(data[data > 0], [low, high])
    if ph - pl < 1e-6:
        pl, ph = data.min(), data.max()
    clipped = np.clip(data, pl, ph)
    return ((clipped - pl) / max(ph - pl, 1e-6) * 255).astype(np.uint8)


def download_timestep(catalog: STACClient, scene_id: str, lon: float, lat: float,
                      output_dir: Path) -> dict | None:
    """Download one timestep: RGB + spectral features."""
    import planetary_computer as pc

    try:
        item = catalog.get_collection("sentinel-2-l2a").get_item(scene_id)
    except Exception:
        return None
    if item is None:
        return None

    scene_date = item.properties["datetime"][:10]
    cloud = item.properties.get("eo:cloud_cover", 100)

    ts_dir = output_dir / "timesteps" / scene_date
    ts_dir.mkdir(parents=True, exist_ok=True)

    # Download RGB
    rgb_path = ts_dir / "rgb.png"
    try:
        rh = pc.sign(item.assets["B04"].href)
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

            gh = pc.sign(item.assets["B03"].href)
            bh = pc.sign(item.assets["B02"].href)
            red = sr.read(1, window=win).astype(np.float32) / 10000.0
            green = rasterio.open(gh).read(1, window=win).astype(np.float32) / 10000.0
            blue = rasterio.open(bh).read(1, window=win).astype(np.float32) / 10000.0

        rgb = np.stack([percent_clip(red * 255), percent_clip(green * 255), percent_clip(blue * 255)], axis=-1)
        Image.fromarray(rgb, mode="RGB").save(str(rgb_path), optimize=True)
    except Exception:
        return None

    # Download spectral bands for features (all 5 in the same window)
    try:
        feat_path = ts_dir / "features.npz"
        r = rasterio.open(pc.sign(item.assets["B04"].href)).read(1, window=win).astype(np.float32) / 10000.0
        g = rasterio.open(pc.sign(item.assets["B03"].href)).read(1, window=win).astype(np.float32) / 10000.0
        b = rasterio.open(pc.sign(item.assets["B02"].href)).read(1, window=win).astype(np.float32) / 10000.0
        nir = rasterio.open(pc.sign(item.assets["B08"].href)).read(1, window=win).astype(np.float32) / 10000.0
        coastal = rasterio.open(pc.sign(item.assets["B01"].href)).read(1, window=win).astype(np.float32) / 10000.0

        # Compute SHSI with water masking
        green_min, red_min, nir_max = 0.025, 0.005, 0.025
        mask = (nir < nir_max) & (g > green_min) & (r > red_min) & (b > coastal)
        shsi = np.full_like(g, -9999, dtype=np.float32)
        shsi[mask] = (g[mask] ** 2) / r[mask]
        shsi = np.clip(shsi, 0, 1.0)

        # Average over valid water pixels
        if mask.sum() < 10:
            return None

        features = np.array([
            shsi[mask].mean(),
            g[mask].mean(),
            r[mask].mean(),
            nir[mask].mean(),
            (b[mask] - coastal[mask]).mean(),
            coastal[mask].mean(),
        ], dtype=np.float32)

        np.savez_compressed(feat_path, features=features, n_valid_pixels=int(mask.sum()))

    except Exception:
        return None

    return {"scene_id": scene_id, "date": scene_date, "cloud_cover": cloud}


def main():
    parser = argparse.ArgumentParser(description="Fetch temporal sequences for labeled locations")
    parser.add_argument("--label", choices=["spawn", "no-spawn", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--embeddings", action="store_true", help="Also compute GeoRSCLIP embeddings (slow)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load labels + manifest
    with open(IMAGE_DIR / "labels.json") as f:
        labels = json.load(f)
    with open(IMAGE_DIR / "manifest.json") as f:
        manifest = json.load(f)

    lookup = {e["filename"]: e for e in manifest if e.get("filename")}

    # Build location groups: deduplicate by (lat, lon, dfo_date) to avoid re-fetching same spawn
    location_groups = {}
    for fname, label in labels.items():
        if label == "skip" or fname not in lookup:
            continue
        if args.label != "all" and label != args.label:
            continue
        entry = lookup[fname]
        key = f"{entry['region']}_{entry['lat']}_{entry['lon']}_{entry['date'][:10]}"
        if key not in location_groups:
            location_groups[key] = {
                "region": entry.get("region", ""),
                "lat": entry["lat"], "lon": entry["lon"],
                "dfo_date": entry["date"][:10],
                "dfo_location": entry.get("location_name", ""),
                "label": label,
                "best_scene_id": entry.get("scene_id"),
                "spawn_length_m": entry.get("spawn_length_m", 0),
                "spawn_width_m": entry.get("spawn_width_m", 0),
            }

    groups = list(location_groups.values())
    if args.limit:
        groups = groups[: args.limit]

    print(f"Processing {len(groups)} unique spawn locations ({args.label})")

    catalog = STACClient.open(STAC_URL)
    print("Connected to Planetary Computer STAC.\n")

    # Load GeoRSCLIP if embeddings requested
    if args.embeddings:
        import open_clip
        import torch

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        ckpt = os.path.expanduser(
            "~/.cache/huggingface/hub/models--Zilun--GeoRSCLIP/snapshots/"
            "4920188e6eba4e711ef9848cfd7cb77e874ee33f/ckpt/RS5M_ViT-B-32.pt"
        )
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        model = model.to(device).eval()
        print("GeoRSCLIP model loaded for embeddings.")
    else:
        model = None

    skipped, ok = 0, 0
    for i, loc in enumerate(groups):
        dfo_dt = datetime.strptime(loc["dfo_date"], "%Y-%m-%d")
        t_start = dfo_dt - timedelta(days=TEMPORAL_PRE_DAYS)
        t_end = dfo_dt + timedelta(days=TEMPORAL_POST_DAYS)

        loc_dir = OUTPUT_DIR / f"{loc['region']}_{loc['lat']}_{loc['lon']}"
        if loc_dir.exists():
            skipped += 1
            continue

        loc_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1}/{len(groups)}] {loc['region']} {loc['dfo_date']} @ {loc['lat']:.4f},{loc['lon']:.4f}")

        # Search STAC for all scenes in temporal window
        bbox = [loc["lon"] - SEARCH_BUFFER, loc["lat"] - SEARCH_BUFFER,
                loc["lon"] + SEARCH_BUFFER, loc["lat"] + SEARCH_BUFFER]
        dt_str = f"{t_start.strftime('%Y-%m-%d')}/{t_end.strftime('%Y-%m-%d')}"

        try:
            search = catalog.search(
                collections=["sentinel-2-l2a"], bbox=bbox, datetime=dt_str,
                query={"eo:cloud_cover": {"lte": MAX_CLOUD}}, max_items=100)
            scenes = list(search.items())
        except Exception:
            print(f"  STAC search failed")
            continue

        if not scenes:
            print(f"  No scenes in window")
            continue

        # Download each scene as a timestep
        timesteps = []
        for item in scenes:
            result = download_timestep(catalog, item.id, loc["lon"], loc["lat"], loc_dir)
            if result:
                timesteps.append(result)
            time.sleep(0.05)

        # Sort by date
        timesteps.sort(key=lambda x: x["date"])

        # Compute embeddings if requested
        if model and timesteps:
            import torch
            from open_clip import tokenizer

            for ts in timesteps:
                rgb_path = loc_dir / "timesteps" / ts["date"] / "rgb.png"
                if rgb_path.exists():
                    img = Image.open(str(rgb_path)).convert("RGB")
                    img_resized = img.resize((224, 224), Image.LANCZOS)
                    with torch.no_grad():
                        emb = model.encode_image(
                            torch.from_numpy(np.array(img_resized, dtype=np.float32) / 255.0)
                            .permute(2, 0, 1).unsqueeze(0).to(device)
                        )
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                    np.savez_compressed(
                        loc_dir / "timesteps" / ts["date"] / "embedding.npz",
                        embedding=emb.cpu().numpy().flatten()
                    )

        # Save metadata
        metadata = {**loc, "timesteps": timesteps, "n_timesteps": len(timesteps)}
        with open(loc_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        ok += 1
        n_ts = len(timesteps)
        print(f"  → {n_ts} timesteps ({t_start.date()} to {t_end.date()})")

    print(f"\nDone: {ok} locations processed")
    total_timesteps = sum(1 for d in OUTPUT_DIR.iterdir() if d.is_dir() and (d / "metadata.json").exists()
                          for _ in [0])
    # Count properly
    ts_count = 0
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir():
            m = d / "metadata.json"
            if m.exists():
                with open(m) as f:
                    md = json.load(f)
                ts_count += md.get("n_timesteps", 0)
    print(f"Total timesteps downloaded: {ts_count}")
    print(f"\nNext: .venv/bin/python3 scripts/train_tempcnn.py")


if __name__ == "__main__":
    main()
