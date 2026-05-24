#!/usr/bin/env python3
"""
Final BC coast sweep — rose-verified spawns only as training data.

Pipeline:
  1. Build training set: 5 unique rose-verified spawns (from rose_super_review.json) + 50 existing negatives
  2. Retrain SVM classifier
  3. Run full BC coast sweep (scan_bc_coast.py or improved_detector.py)
  4. Temporal validation: check candidate points for ±7 day multi-date confirmation
  5. Classify: super_positive (2+ dates), positive (1 date)
  6. Generate review page

Usage:
    source .venv/bin/activate
    python scripts/final_bc_sweep.py [--dry-run]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
POS_DIR = REPO_ROOT / "data" / "samples" / "positive"
NEG_DIR = REPO_ROOT / "data" / "samples" / "negative"
CANDIDATES_V2 = REPO_ROOT / "data" / "candidates_v2"
OUTPUT_DIR = REPO_ROOT / "data" / "candidates_final"
MODEL_PATH = REPO_ROOT / "data" / "models" / "dinov2_svm.pkl"

# Rose-verified spawn files (from rose_super_review.json classification="spawn")
ROSE_SPAWN_FILES = [
    "qualicum_2024-03-18_score0.01_49.254865_-124.497442_20240318.png",
    "tofino_2024-03-16_score0.00_49.114865_-125.806603_20240316.png",
    "tofino_2024-03-16_score0.01_49.194865_-126.026603_20240316.png",
    "nootka-sound_2024-03-16_score0.00_49.584865_-126.528503_20240316.png",
    "nootka-sound_2024-02-12_score0.00_49.564865_-126.508503_20240212.png",
]

# 13 BC coast herring habitat regions
REGIONS: list[dict[str, Any]] = [
    {"name": "qualicum", "lat": 49.35, "lon": -124.45, "radius_km": 15},
    {"name": "nanaimo", "lat": 49.15, "lon": -123.85, "radius_km": 15},
    {"name": "comox", "lat": 49.68, "lon": -124.88, "radius_km": 15},
    {"name": "denman-island", "lat": 49.55, "lon": -124.80, "radius_km": 10},
    {"name": "tofino", "lat": 49.15, "lon": -125.90, "radius_km": 15},
    {"name": "ucluelet", "lat": 48.94, "lon": -125.55, "radius_km": 10},
    {"name": "nootka-sound", "lat": 49.60, "lon": -126.60, "radius_km": 15},
    {"name": "quatsino-sound", "lat": 50.50, "lon": -128.00, "radius_km": 15},
    {"name": "spiller-channel", "lat": 52.30, "lon": -128.30, "radius_km": 15},
    {"name": "milbanke-sound", "lat": 52.50, "lon": -128.80, "radius_km": 15},
    {"name": "prince-rupert", "lat": 54.30, "lon": -130.40, "radius_km": 20},
    {"name": "haida-gwaii-south", "lat": 52.40, "lon": -131.40, "radius_km": 15},
    {"name": "masset-inlet", "lat": 53.70, "lon": -132.90, "radius_km": 15},
]

# DINOv2
MODEL_NAME = "dinov2_vits14"
EMBED_DIM = 384
DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ===================================================================
# Step 1: Prepare training data
# ===================================================================

def prepare_training_data(dry_run: bool = False) -> dict[str, Any]:
    """Clear positives and copy only rose-verified spawns. Keep negatives as-is."""
    print("\n" + "=" * 60)
    print("  Step 1: Prepare Training Data")
    print("=" * 60)

    results = {"pos_before": 0, "pos_after": 0, "neg_count": 0, "copied": []}

    # Count before
    pos_before = len(list(POS_DIR.glob("*.png")))
    neg_count = len(list(NEG_DIR.glob("*.png")))
    results["pos_before"] = pos_before
    results["neg_count"] = neg_count
    print(f"  Current positives: {pos_before}")
    print(f"  Current negatives: {neg_count}")

    if dry_run:
        print(f"  [DRY RUN] Would remove {pos_before} positives and copy {len(ROSE_SPAWN_FILES)} rose spawns")
        results["pos_after"] = len(ROSE_SPAWN_FILES)
        return results

    # Remove all current positives
    for p in POS_DIR.glob("*.png"):
        p.unlink()
    print(f"  Removed {pos_before} existing positives")

    # Copy rose-verified spawns
    copied = 0
    for fname in ROSE_SPAWN_FILES:
        src = CANDIDATES_V2 / fname
        dst = POS_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
            results["copied"].append(fname)
            print(f"  Copied: {fname}")
        else:
            print(f"  WARNING: Not found in candidates_v2: {fname}")

    pos_after = len(list(POS_DIR.glob("*.png")))
    results["pos_after"] = pos_after
    print(f"\n  Positives after cleanup: {pos_after}")
    print(f"  Negatives: {neg_count}")
    print(f"  Total training samples: {pos_after + neg_count}")

    # Save a manifest of what we're using
    manifest = {
        "description": "Training data for final BC coast sweep",
        "rose_verified_positives": ROSE_SPAWN_FILES,
        "positive_count": pos_after,
        "negative_count": neg_count,
    }
    manifest_path = REPO_ROOT / "data" / "samples" / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest saved: {manifest_path}")

    return results


# ===================================================================
# Step 2: Train classifier
# ===================================================================

def train_classifier(dry_run: bool = False) -> dict[str, Any]:
    """Train SVM classifier on the prepared training data."""
    print("\n" + "=" * 60)
    print("  Step 2: Train SVM Classifier")
    print("=" * 60)

    pos_dir = POS_DIR
    neg_dir = NEG_DIR

    pos_files = sorted(pos_dir.glob("*.png"))
    neg_files = sorted(neg_dir.glob("*.png"))

    if not pos_files:
        print("ERROR: No positive training samples!")
        return {"error": "no positives"}
    if not neg_files:
        print("ERROR: No negative training samples!")
        return {"error": "no negatives"}

    print(f"  Training on {len(pos_files)} positives + {len(neg_files)} negatives")

    if dry_run:
        print("  [DRY RUN] Would train SVM classifier")
        return {
            "dry_run": True,
            "n_pos": len(pos_files),
            "n_neg": len(neg_files),
        }

    # Use the existing train_classifier.py script
    cmd = [
        sys.executable, "scripts/train_classifier.py",
        "--positive-dir", str(pos_dir),
        "--negative-dir", str(neg_dir),
        "--output-model", str(MODEL_PATH),
        "--output-vectors", str(REPO_ROOT / "data" / "embeddings" / "rose_training_vectors.npz"),
        "--kernel", "rbf",
        "--cv-folds", "5",
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"  STDERR: {result.stderr[:500]}")

    if result.returncode != 0:
        print(f"  ERROR: Training failed with return code {result.returncode}")
        return {"error": f"training failed: {result.stderr[:500]}"}

    # Load and return stats
    summary_path = MODEL_PATH.with_suffix(".summary.json")
    if summary_path.exists():
        stats = json.loads(summary_path.read_text())
        print(f"  Model saved: {MODEL_PATH}")
        print(f"  Full accuracy: {stats.get('full_accuracy', '?'):.4f}")
        print(f"  Separation: {stats.get('separation', '?'):.4f}")
        print(f"  CV accuracy: {stats.get('cv_accuracy_mean', '?'):.4f} +/- {stats.get('cv_accuracy_std', '?'):.4f}")
        return stats

    return {"note": "trained, but summary not found"}


# ===================================================================
# Grid point generation
# ===================================================================

def generate_grid_points(spacing_deg: float = 0.02) -> list[dict[str, Any]]:
    """Generate grid points within each region's circular buffer."""
    points: list[dict[str, Any]] = []
    for region in REGIONS:
        lat, lon = region["lat"], region["lon"]
        radius_km = region["radius_km"]
        radius_deg_lat = radius_km / 111.0
        radius_deg_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
        n_steps_lat = max(1, int(2 * radius_deg_lat / spacing_deg))
        n_steps_lon = max(1, int(2 * radius_deg_lon / spacing_deg))
        region_points = 0
        for i in range(n_steps_lat + 1):
            p_lat = lat - radius_deg_lat + i * spacing_deg
            if abs(p_lat - lat) > radius_deg_lat + spacing_deg * 0.5:
                continue
            for j in range(n_steps_lon + 1):
                p_lon = lon - radius_deg_lon + j * spacing_deg
                if abs(p_lon - lon) > radius_deg_lon + spacing_deg * 0.5:
                    continue
                dlat = (p_lat - lat) * 111.0
                dlon = (p_lon - lon) * 111.0 * math.cos(math.radians(lat))
                dist_km = math.sqrt(dlat**2 + dlon**2)
                if dist_km <= radius_km:
                    points.append({
                        "region": region["name"],
                        "lat": round(p_lat, 6),
                        "lon": round(p_lon, 6),
                    })
                    region_points += 1
        print(f"  {region['name']}: {region_points} grid points")
    return points


# ===================================================================
# GEE operations
# ===================================================================

def find_best_scene(ee, lat, lon, start_date, end_date, max_cloud):
    """Find the single best Sentinel-2 scene for a point."""
    try:
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        point = ee.Geometry.Point(lon, lat)
        scenes = (
            collection.filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        scene_ids = scenes.aggregate_array("system:index").getInfo()
        clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        if not scene_ids:
            return None
        sid = scene_ids[0]
        return {
            "scene_id": sid,
            "cloud": float(clouds[0]),
            "date": f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}",
        }
    except Exception as exc:
        return None


def find_multiple_scenes(ee, lat, lon, start_date, end_date, max_cloud, max_scenes=5):
    """Find multiple Sentinel-2 scenes for a point (for temporal validation)."""
    try:
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        point = ee.Geometry.Point(lon, lat)
        scenes = (
            collection.filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        scene_ids = scenes.aggregate_array("system:index").getInfo()
        clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        if not scene_ids:
            return []
        results = []
        for i, sid in enumerate(scene_ids[:max_scenes]):
            results.append({
                "scene_id": sid,
                "cloud": float(clouds[i]),
                "date": f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}",
            })
        return results
    except Exception:
        return []


def download_thumbnail(ee, lat, lon, scene_id):
    """Download a 512x512 RGB thumbnail from Sentinel-2."""
    try:
        import requests
        scene_img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{scene_id}")
        rgb = scene_img.select(["B4", "B3", "B2"])
        region = ee.Geometry.Point(lon, lat).buffer(1280).bounds()
        url = rgb.getThumbURL({
            "min": 0, "max": 3000,
            "region": region, "dimensions": 512, "format": "png",
        })
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


# ===================================================================
# DINOv2 scoring
# ===================================================================

def load_dinov2() -> tuple[torch.nn.Module, torch.device]:
    """Load DINOv2 model."""
    print("\n=== Loading DINOv2 model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval()
    model = model.to(device)
    return model, device


def load_svm() -> Any:
    """Load trained SVM from pickle."""
    import pickle
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"SVM model not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"  SVM loaded: {data.get('n_train', '?')} train samples, "
          f"accuracy={data.get('full_accuracy', '?'):.4f}")
    return data["svm"]


def score_thumbnail(png_bytes: bytes, model: torch.nn.Module, device: torch.device,
                    svm: Any) -> float | None:
    """Score a PNG thumbnail using DINOv2 + SVM."""
    try:
        import io
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        tensor = DINO_TRANSFORM(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor)
        emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
        score = float(svm.decision_function(emb.reshape(1, -1))[0])
        return score
    except Exception:
        return None


# ===================================================================
# Step 3: Main sweep
# ===================================================================

_stats_lock = Lock()

def process_point(point, args, output_dir, model, device, svm, ee, idx, total):
    """Process a single grid point."""
    result = {"processed": 1, "candidates": 0, "no_scene": 0,
              "download_errors": 0, "low_score": 0}

    scene = find_best_scene(ee, point["lat"], point["lon"],
                           args.start, args.end, args.max_cloud)
    if scene is None:
        with _stats_lock:
            _progress(idx, total, point["region"], point["lat"], point["lon"], "no scene")
        result["no_scene"] = 1
        return result

    thumb = download_thumbnail(ee, point["lat"], point["lon"], scene["scene_id"])
    if thumb is None:
        with _stats_lock:
            _progress(idx, total, point["region"], point["lat"], point["lon"], "dl error")
        result["download_errors"] = 1
        return result

    score = score_thumbnail(thumb, model, device, svm)
    if score is None:
        with _stats_lock:
            _progress(idx, total, point["region"], point["lat"], point["lon"], "score error")
        result["download_errors"] = 1
        return result

    if score > args.threshold:
        info = {
            "region": point["region"],
            "lat": point["lat"],
            "lon": point["lon"],
            "date": scene["date"],
            "scene_id": scene["scene_id"],
            "cloud": scene["cloud"],
            "score": round(score, 4),
        }
        fname = _save_candidate(output_dir, thumb, info, score)
        with _stats_lock:
            _update_manifest(output_dir, {**info, "thumbnail_path": fname})
            _progress(idx, total, point["region"], point["lat"], point["lon"],
                      f"CANDIDATE score={score:.4f}")
        result["candidates"] = 1
    else:
        with _stats_lock:
            _progress(idx, total, point["region"], point["lat"], point["lon"],
                      f"below ({score:.4f})")
        result["low_score"] = 1

    return result


def _progress(idx, total, region, lat, lon, status):
    pct = 100.0 * (idx + 1) / total
    print(f"  [{idx + 1}/{total}] ({pct:.0f}%) {region} ({lat:.4f}, {lon:.4f}) | {status}")


def _save_candidate(output_dir, png_bytes, info, score):
    region = info["region"]
    date = info["date"]
    lat = info["lat"]
    lon = info["lon"]
    scene_id = info["scene_id"]
    scene_short = scene_id[:8] if len(scene_id) >= 8 else scene_id
    fname = f"{region}_{date}_score{score:.2f}_{lat}_{lon}_{scene_short}.png"
    fname = "".join(c if c.isalnum() or c in "._-" else "_" for c in fname)
    (output_dir / fname).write_bytes(png_bytes)
    return fname


def _update_manifest(output_dir, entry):
    manifest_path = output_dir / "manifest.json"
    entries = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text())
            if not isinstance(entries, list):
                entries = []
        except Exception:
            entries = []
    entries.append(entry)
    manifest_path.write_text(json.dumps(entries, indent=2))


def run_sweep(args, model, device, svm, ee) -> dict[str, int]:
    """Run the main BC coast sweep."""
    print("\n" + "=" * 60)
    print("  Step 3: BC Coast Sweep")
    print("=" * 60)

    spacing = args.grid_spacing
    print(f"\n=== Generating grid points (spacing={spacing}°) ===")
    points = generate_grid_points(spacing)
    print(f"  Total: {len(points)} grid points across {len(REGIONS)} regions")

    if not points:
        print("ERROR: No grid points generated")
        return {}

    if args.dry_run:
        print("\n  [DRY RUN] Would scan these points")
        return {"dry_run": True, "n_points": len(points)}

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Scanning with {args.workers} workers ===")
    print(f"  Date: {args.start} to {args.end}")
    print(f"  Cloud: {args.max_cloud}%")
    print(f"  Threshold: {args.threshold}")
    print(f"  Output: {output_dir}")

    processed = candidates = no_scene = dl_errors = low_score = 0
    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_point, point, args, output_dir,
                               model, device, svm, ee, idx, len(points)): idx
                for idx, point in enumerate(points)
            }
            for future in as_completed(futures):
                r = future.result()
                processed += r["processed"]
                candidates += r["candidates"]
                no_scene += r["no_scene"]
                dl_errors += r["download_errors"]
                low_score += r["low_score"]
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    except Exception as exc:
        print(f"\n\nError: {exc}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 60}")
    print("  Sweep Summary")
    print(f"  {'=' * 60}")
    print(f"  Total points:    {len(points)}")
    print(f"  Processed:       {processed}")
    print(f"  Candidates:      {candidates}")
    print(f"  No scene:        {no_scene}")
    print(f"  DL errors:       {dl_errors}")
    print(f"  Below threshold: {low_score}")
    print(f"  Time:            {elapsed:.1f}s ({rate:.1f} pts/s)")

    return {
        "total_points": len(points),
        "processed": processed,
        "candidates": candidates,
        "no_scene": no_scene,
        "download_errors": dl_errors,
        "low_score": low_score,
        "elapsed_s": round(elapsed, 1),
    }


# ===================================================================
# Step 4: Temporal validation
# ===================================================================

def run_temporal_validation(args, model, device, svm, ee) -> dict[str, Any]:
    """Check each candidate point for multiple dates ±7 days from the candidate date.

    For each unique lat/lon, search for ALL scenes in the ±7 day window,
    download and score them. A point is:
      - super_positive: 2+ dates with score > threshold
      - positive: 1 date with score > threshold
    """
    print("\n" + "=" * 60)
    print("  Step 4: Temporal Validation")
    print("=" * 60)

    manifest_path = OUTPUT_DIR / "manifest.json"
    if not manifest_path.exists():
        print("  No manifest found — nothing to validate")
        return {}

    candidates = json.loads(manifest_path.read_text())
    print(f"  Loaded {len(candidates)} candidate entries")

    # Group by unique lat/lon
    from collections import OrderedDict
    locations: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for c in candidates:
        key = f"{c['lat']}_{c['lon']}"
        if key not in locations:
            locations[key] = {
                "lat": c["lat"],
                "lon": c["lon"],
                "region": c["region"],
                "primary_date": c["date"],
                "primary_score": c["score"],
                "primary_scene": c["scene_id"],
                "primary_thumb": c["thumbnail_path"],
            }

    print(f"  Unique locations: {len(locations)}")

    if args.dry_run:
        print("  [DRY RUN] Would check each location for additional dates")
        return {"dry_run": True, "n_locations": len(locations)}

    validated = {"super_positive": [], "positive": []}

    # For each unique location, check additional dates ±7 days
    for idx, (key, loc) in enumerate(locations.items()):
        # Parse the primary date
        from datetime import datetime, timedelta
        primary_dt = datetime.strptime(loc["primary_date"], "%Y-%m-%d")

        # Search ±7 days from primary date
        window_start = (primary_dt - timedelta(days=7)).strftime("%Y-%m-%d")
        window_end = (primary_dt + timedelta(days=7)).strftime("%Y-%m-%d")

        scenes = find_multiple_scenes(ee, loc["lat"], loc["lon"],
                                      window_start, window_end, args.max_cloud)

        # Score each scene
        date_scores = {}
        for scene in scenes:
            thumb = download_thumbnail(ee, loc["lat"], loc["lon"], scene["scene_id"])
            if thumb is None:
                continue
            score = score_thumbnail(thumb, model, device, svm)
            if score is None:
                continue
            date_scores[scene["date"]] = {
                "date": scene["date"],
                "score": round(score, 4),
                "scene_id": scene["scene_id"],
                "cloud": scene["cloud"],
            }

        # Include the primary date's score
        primary_result = {
            "date": loc["primary_date"],
            "score": loc["primary_score"],
            "scene_id": loc["primary_scene"],
            "is_primary": True,
        }

        # Combine all dates
        all_dates = {}
        for d, info in date_scores.items():
            all_dates[d] = info
        if loc["primary_date"] not in all_dates:
            all_dates[loc["primary_date"]] = primary_result

        # Count dates above threshold
        above = [d for d, info in all_dates.items() if info["score"] > args.threshold]
        n_above = len(above)

        classification = "super_positive" if n_above >= 2 else ("positive" if n_above >= 1 else "negative")

        result = {
            "lat": loc["lat"],
            "lon": loc["lon"],
            "region": loc["region"],
            "primary_date": loc["primary_date"],
            "n_dates_total": len(all_dates),
            "n_dates_above_threshold": n_above,
            "dates_above_threshold": above,
            "classification": classification,
            "primary_score": round(loc["primary_score"], 4),
            "dates": sorted(all_dates.values(), key=lambda x: x["date"]),
            "primary_thumbnail": loc["primary_thumb"],
        }

        if classification == "super_positive":
            validated["super_positive"].append(result)
        elif classification == "positive":
            validated["positive"].append(result)

        status = f"super_pos({n_above}/{len(all_dates)})" if n_above >= 2 else \
                 f"pos({n_above}/{len(all_dates)})" if n_above >= 1 else \
                 f"neg({n_above}/{len(all_dates)})"
        print(f"  [{idx + 1}/{len(locations)}] {loc['region']} ({loc['lat']:.4f}, {loc['lon']:.4f}) | {status} | primary={loc['primary_score']:.4f}")

    # Save validation results
    validated_path = OUTPUT_DIR / "temporal_validation.json"
    validated_path.write_text(json.dumps({
        "super_positive_count": len(validated["super_positive"]),
        "positive_count": len(validated["positive"]),
        "super_positive": validated["super_positive"],
        "positive": validated["positive"],
    }, indent=2))

    print(f"\n  Temporal validation complete:")
    print(f"    Super positive (2+ dates): {len(validated['super_positive'])}")
    print(f"    Positive (1 date):         {len(validated['positive'])}")
    print(f"  Saved: {validated_path}")

    return validated


# ===================================================================
# Step 5: Review page
# ===================================================================

def generate_review_page(validated: dict[str, Any], output_dir: Path) -> str:
    """Generate HTML review page with super_positives first."""
    print("\n" + "=" * 60)
    print("  Step 5: Generate Review Page")
    print("=" * 60)

    all_entries = []
    for entry in validated.get("super_positive", []):
        all_entries.append(("super_positive", entry))
    for entry in validated.get("positive", []):
        all_entries.append(("positive", entry))

    if not all_entries:
        print("  No entries to display")
        return ""

    cards = []
    for classification, entry in all_entries:
        thumb_path = entry.get("primary_thumbnail", "")
        score = entry.get("primary_score", 0)
        n_dates = entry.get("n_dates_above_threshold", 0)
        n_total = entry.get("n_dates_total", 1)

        date_list = []
        for d in entry.get("dates", []):
            marker = "★" if d.get("is_primary") else "·"
            date_list.append(f"{marker} {d['date']}: {d['score']:.4f}")

        dates_html = "<br>".join(date_list)

        card_class = "super" if classification == "super_positive" else "pos"

        cards.append(f"""
    <div class="card {card_class}">
        <img src="{html_escape(thumb_path)}" alt="" loading="lazy">
        <div class="body">
            <div class="info"><strong>{html_escape(entry.get('region', '?'))}</strong> &middot; {entry.get('lat', 0):.4f}, {entry.get('lon', 0):.4f}</div>
            <div class="info">Primary: {entry.get('primary_date', '?')} &middot; {n_dates}/{n_total} dates above threshold</div>
            <div class="info"><strong>Classification: {classification}</strong></div>
            <div class="scores">
                <span class="badge score-badge">Score: {score:.4f}</span>
                <span class="badge date-badge">{n_dates} dates &gt; thresh</span>
            </div>
            <div class="dates">{dates_html}</div>
        </div>
    </div>""")

    cards_joined = "\n".join(cards)
    n_super = len(validated.get("super_positive", []))
    n_pos = len(validated.get("positive", []))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Final BC Coast Sweep — Candidates Review</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #0d0d1a; color: #eee; }}
.bar {{ background: #1a1a2e; padding: 14px 20px; position: sticky; top: 0; z-index: 99; border-bottom: 1px solid #333; }}
.bar h1 {{ margin: 0; font-size: 20px; }}
.bar .sub {{ font-size: 13px; color: #888; margin-top: 4px; }}
.g {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 14px; padding: 14px; }}
.card {{ background: #1a1a2e; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.5); }}
.super {{ border-left: 5px solid #00E676; }}
.pos {{ border-left: 5px solid #FFD740; }}
.card img {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; }}
.body {{ padding: 10px 14px; }}
.info {{ font-size: 12px; color: #aaa; margin-bottom: 4px; }}
.info strong {{ color: #fff; }}
.scores {{ display: flex; flex-wrap: wrap; gap: 4px; margin: 8px 0; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
.score-badge {{ background: #1b3a2b; color: #4CAF50; }}
.date-badge {{ background: #1b2a3a; color: #64B5F6; }}
.dates {{ font-size: 11px; color: #888; line-height: 1.6; padding-top: 4px; border-top: 1px solid #2a2a3e; margin-top: 6px; }}
.summary {{ background: #1a1a2e; margin: 14px; padding: 16px 20px; border-radius: 8px; font-size: 13px; }}
.summary strong {{ color: #fff; }}
.summary-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 10px; }}
.summary-stat {{ text-align: center; padding: 10px 20px; background: #0a0a18; border-radius: 8px; min-width: 120px; }}
.summary-stat .num {{ font-size: 28px; font-weight: 700; color: #fff; }}
.summary-stat .lbl {{ font-size: 11px; color: #888; text-transform: uppercase; margin-top: 2px; }}
.summary-stat.super .num {{ color: #00E676; }}
.summary-stat.pos .num {{ color: #FFD740; }}
</style>
</head>
<body>
<div class="bar">
    <h1>🐟 Final BC Coast Sweep — {n_super + n_pos} Candidates</h1>
    <div class="sub">
        Rose-verified training (5 spawns, 50 negatives) &middot; DINOv2+SVM &middot; Temporal: ±7 day multi-date confirmation
    </div>
</div>
<div class="summary">
    <strong>Summary:</strong> {n_super + n_pos} total candidates from 13 BC coast regions.
    <div class="summary-grid">
        <div class="summary-stat super"><div class="num">{n_super}</div><div class="lbl">Super Positive (2+ dates)</div></div>
        <div class="summary-stat pos"><div class="num">{n_pos}</div><div class="lbl">Positive (1 date)</div></div>
        <div class="summary-stat"><div class="num">{n_super + n_pos}</div><div class="lbl">Total Candidates</div></div>
    </div>
</div>
<div class="g">
{cards_joined}
</div>
</body>
</html>"""

    review_path = output_dir / "review.html"
    review_path.write_text(html, encoding="utf-8")
    print(f"  Review page: file://{review_path.resolve()}")
    return str(review_path)


def html_escape(s: str) -> str:
    """Simple HTML escape."""
    import html
    return html.escape(str(s), quote=True)


# ===================================================================
# Main
# ===================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Final BC coast sweep with rose-verified training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't execute GEE calls")
    parser.add_argument("--start", default="2024-02-01", help="Start date")
    parser.add_argument("--end", default="2024-05-31", help="End date")
    parser.add_argument("--max-cloud", type=float, default=50, help="Max cloud %")
    parser.add_argument("--threshold", type=float, default=0.0, help="Score threshold")
    parser.add_argument("--grid-spacing", type=float, default=0.02, help="Grid spacing in degrees (~2km)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent workers")
    parser.add_argument("--skip-sweep", action="store_true", help="Skip the GEE sweep, only do temporal validation + review")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Ensure output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare training data
    train_prep = prepare_training_data(dry_run=args.dry_run)
    if "error" in train_prep:
        print(f"  ERROR in training data prep: {train_prep['error']}")
        return 1

    # Step 2: Train classifier
    train_stats = train_classifier(dry_run=args.dry_run)
    if "error" in train_stats:
        print(f"  ERROR in training: {train_stats['error']}")
        return 1

    if args.dry_run:
        print("\n=== Dry run complete! ===")
        print(f"  Training data: {train_prep['pos_after']} pos + {train_prep['neg_count']} neg")
        print(f"  Grid: 0.02° spacing across {len(REGIONS)} regions")
        print(f"  To run for real, omit --dry-run")
        return 0

    # Load DINOv2 and SVM
    model, device = load_dinov2()
    svm = load_svm()

    # Step 3: Run sweep (unless --skip-sweep)
    if not args.skip_sweep:
        # Initialize GEE
        print("\n=== Initializing GEE ===")
        import ee
        ee.Initialize(project="redd-fish")
        print("  GEE initialized")

        sweep_results = run_sweep(args, model, device, svm, ee)
        if not sweep_results:
            print("  Sweep produced no results")
            return 1
    else:
        print("\n=== Skipping sweep (--skip-sweep) ===")
        # Check if we have existing results
        manifest_path = OUTPUT_DIR / "manifest.json"
        if manifest_path.exists():
            candidates = json.loads(manifest_path.read_text())
            print(f"  Using existing {len(candidates)} candidates from {OUTPUT_DIR}")
        else:
            print(f"  ERROR: No existing results found in {OUTPUT_DIR}")
            print(f"  Run without --skip-sweep first")
            return 1

    # Step 4: Temporal validation
    if not args.skip_sweep:
        import ee
        validated = run_temporal_validation(args, model, device, svm, ee)
    else:
        # Reload validation if it exists
        val_path = OUTPUT_DIR / "temporal_validation.json"
        if val_path.exists():
            validated = json.loads(val_path.read_text())
            print(f"  Loaded existing validation: "
                  f"{validated.get('super_positive_count', 0)} super, "
                  f"{validated.get('positive_count', 0)} positive")
        else:
            # Need to run temporal validation
            import ee
            validated = run_temporal_validation(args, model, device, svm, ee)

    if not validated:
        print("  No validated results")
        return 1

    # Step 5: Generate review page
    generate_review_page(validated, OUTPUT_DIR)

    # Final summary
    n_super = len(validated.get("super_positive", []))
    n_pos = len(validated.get("positive", []))
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Super positive (2+ dates): {n_super}")
    print(f"  Positive (1 date):         {n_pos}")
    print(f"  Total candidates:          {n_super + n_pos}")
    print(f"  Review page: file://{OUTPUT_DIR / 'review.html'}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
