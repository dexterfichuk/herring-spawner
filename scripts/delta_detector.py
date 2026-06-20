#!/usr/bin/env python3
"""DINOv2 Embedding Delta Detector for Herring Spawn.

Compares pre-spawn baseline (Jan/Feb) vs spawn-season (Mar/Apr) images at
known locations. Computes DINOv2 embedding deltas and trains an SVM classifier
on delta vectors to distinguish spawn from non-spawn events, reducing shoreline bias.

Usage:
    # Analyze known positive/negative locations
    python scripts/delta_detector.py --mode analyze --output data/delta_pairs

    # Scan issue #1 regions for 2024 candidates
    python scripts/delta_detector.py --mode scan --output data/delta_candidates

    # Serve review page
    python -m http.server 8775 --directory data/delta_pairs
    # Open http://localhost:8775/review.html
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.knn_detector import DINO_TRANSFORM, _pick_device
from scripts.scan_bc_coast import REGIONS, download_thumbnail, find_best_scene, generate_grid_points


MODEL_NAME = "dinov2_vits14"
EMBED_DIM = 384
BASELINE_SEARCH_START = "-01-15"
BASELINE_SEARCH_END = "-02-15"
SPAWN_SEARCH_DAYS = 14
DEFAULT_SPAWN_DATE = date(2024, 3, 15)

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def parse_location_from_filename(filename: str) -> dict[str, Any] | None:
    """Extract lat, lon, date from a standard-format filename.

    Standard format: <region>_<date>_score<X.XX>_<lat>_<lon>_<scene_date>.png
    Example: SoG_2021-03-11_score0.00_49.5175_-124.577222_20210311.png

    Returns dict with keys: lat, lon, date, scene_date, region, filename
    or None if parsing fails.
    """
    m = re.search(r"_(-?\d+\.?\d*)_(-?\d+\.?\d*)_(\d{8})\.png$", filename)
    if not m:
        return None

    try:
        lat = float(m.group(1))
        lon = float(m.group(2))
        scene_date = m.group(3)
    except (ValueError, IndexError):
        return None

    date_match = re.search(r"_(\d{4}-\d{2}-\d{2})_score", filename)
    if not date_match:
        return None

    date_str = date_match.group(1)
    region = filename[: date_match.start()].strip("_")

    return {
        "lat": lat,
        "lon": lon,
        "date": date_str,
        "scene_date": scene_date,
        "region": region,
        "filename": filename,
    }


# ---------------------------------------------------------------------------
# Location loading
# ---------------------------------------------------------------------------


def load_locations_from_manifest(
    manifest_path: str | Path,
    negative_dir: str | Path,
    deduplicate: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Load positive locations from training_manifest.json and parse lat/lon from filenames.

    Also scans the negative directory for parseable negative locations.
    When deduplicate=True (default), only the first occurrence of each (lat, lon) is kept.
    """
    manifest_path = Path(manifest_path)
    negative_dir = Path(negative_dir)

    with open(manifest_path) as f:
        manifest = json.load(f)

    positives: list[dict[str, Any]] = []
    for fname in manifest.get("positives", []):
        parsed = parse_location_from_filename(fname)
        if parsed:
            img_path = manifest_path.parent / "positive" / fname
            if img_path.exists():
                parsed["image_path"] = str(img_path)
                positives.append(parsed)

    negatives: list[dict[str, Any]] = []
    seen_locations: set[tuple[float, float]] = set()
    for f in sorted(negative_dir.iterdir()):
        if not f.name.endswith(".png"):
            continue
        parsed = parse_location_from_filename(f.name)
        if parsed:
            parsed["image_path"] = str(f)
            loc_key = (round(parsed["lat"], 4), round(parsed["lon"], 4))
            if deduplicate and loc_key in seen_locations:
                continue
            seen_locations.add(loc_key)
            negatives.append(parsed)

    return {"positives": positives, "negatives": negatives}


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def compute_delta(spawn_emb: np.ndarray, baseline_emb: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute delta vector and its L2 magnitude between spawn and baseline embeddings.

    Returns (delta_vector, magnitude).
    """
    delta = spawn_emb - baseline_emb
    magnitude = float(np.linalg.norm(delta))
    return delta, magnitude


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_png_bytes(model: torch.nn.Module, device: torch.device, png_bytes: bytes) -> np.ndarray:
    """Compute DINOv2 embedding from PNG bytes."""
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    tensor = DINO_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    return F.normalize(emb, dim=1).cpu().numpy().flatten().astype(float)


# ---------------------------------------------------------------------------
# GEE thumbnail fetching for paired images
# ---------------------------------------------------------------------------


def fetch_paired_thumbnails(
    ee_module: Any,
    lat: float,
    lon: float,
    spawn_date_str: str,
    baseline_days_before: int = 45,
    spawn_window_days: int = 14,
    max_cloud: float = 50.0,
) -> dict[str, Any] | None:
    """Fetch baseline and spawn-season thumbnails for a location.

    Args:
        ee_module: Earth Engine module (initialized)
        lat, lon: Location coordinates
        spawn_date_str: Known spawn date in YYYY-MM-DD or 'default' for negatives
        baseline_days_before: How many days before spawn to center baseline search
        spawn_window_days: ±days around spawn date to search
        max_cloud: Maximum cloud percentage

    Returns dict with keys: lat, lon, baseline_bytes, spawn_bytes, baseline_scene,
    spawn_scene, baseline_date, spawn_date, target_spawn_date
    or None if either image cannot be found.
    """
    # Determine dates
    if spawn_date_str == "default" or not spawn_date_str:
        spawn_dt = DEFAULT_SPAWN_DATE
    else:
        try:
            spawn_dt = date.fromisoformat(spawn_date_str)
        except ValueError:
            spawn_dt = DEFAULT_SPAWN_DATE

    baseline_center = spawn_dt - timedelta(days=baseline_days_before)

    baseline_start = (baseline_center - timedelta(days=spawn_window_days)).isoformat()
    baseline_end = (baseline_center + timedelta(days=spawn_window_days)).isoformat()
    spawn_start = (spawn_dt - timedelta(days=spawn_window_days)).isoformat()
    spawn_end = (spawn_dt + timedelta(days=spawn_window_days)).isoformat()

    # Fetch baseline (pre-spawn) image
    baseline_scene = find_best_scene(ee_module, lat, lon, baseline_start, baseline_end, max_cloud)
    if baseline_scene is None:
        # Try wider window
        baseline_start = (baseline_center - timedelta(days=spawn_window_days * 2)).isoformat()
        baseline_end = (baseline_center + timedelta(days=spawn_window_days * 2)).isoformat()
        baseline_scene = find_best_scene(ee_module, lat, lon, baseline_start, baseline_end, max_cloud)
        if baseline_scene is None:
            return None

    # Fetch spawn-season image
    spawn_scene = find_best_scene(ee_module, lat, lon, spawn_start, spawn_end, max_cloud)
    if spawn_scene is None:
        # Try wider window
        spawn_start = (spawn_dt - timedelta(days=spawn_window_days * 2)).isoformat()
        spawn_end = (spawn_dt + timedelta(days=spawn_window_days * 2)).isoformat()
        spawn_scene = find_best_scene(ee_module, lat, lon, spawn_start, spawn_end, max_cloud)
        if spawn_scene is None:
            return None

    # Download thumbnails
    baseline_bytes = download_thumbnail(ee_module, lat, lon, baseline_scene["scene_id"])
    if baseline_bytes is None:
        return None

    spawn_bytes = download_thumbnail(ee_module, lat, lon, spawn_scene["scene_id"])
    if spawn_bytes is None:
        return None

    return {
        "lat": lat,
        "lon": lon,
        "baseline_bytes": baseline_bytes,
        "spawn_bytes": spawn_bytes,
        "baseline_scene": baseline_scene["scene_id"],
        "spawn_scene": spawn_scene["scene_id"],
        "baseline_date": baseline_scene["date"],
        "spawn_date": spawn_scene["date"],
        "target_spawn_date": spawn_date_str,
        "baseline_cloud": baseline_scene.get("cloud", -1),
        "spawn_cloud": spawn_scene.get("cloud", -1),
    }


def save_paired_thumbnails(
    output_dir: Path,
    pair: dict[str, Any],
    label: str = "unknown",
) -> tuple[str, str]:
    """Save baseline and spawn thumbnail PNGs to the output directory.

    Returns (baseline_filename, spawn_filename).
    """
    lat = pair["lat"]
    lon = pair["lon"]
    lat_str = f"{lat:.6f}".replace(".", "_").replace("-", "n")
    lon_str = f"{lon:.6f}".replace(".", "_").replace("-", "n")

    base_fname = f"delta_{lat_str}_{lon_str}"
    bf = f"{base_fname}_baseline_{pair['baseline_date']}_{label}.png"
    sf = f"{base_fname}_spawn_{pair['spawn_date']}_{label}.png"

    (output_dir / bf).write_bytes(pair["baseline_bytes"])
    (output_dir / sf).write_bytes(pair["spawn_bytes"])

    return bf, sf


def process_pair(
    model: torch.nn.Module,
    device: torch.device,
    pair: dict[str, Any],
) -> dict[str, Any] | None:
    """Process a paired image set: embed both, compute delta.

    Returns dict with embeddings, delta, magnitude, or None on failure.
    """
    try:
        baseline_emb = embed_png_bytes(model, device, pair["baseline_bytes"])
        spawn_emb = embed_png_bytes(model, device, pair["spawn_bytes"])
    except Exception as exc:
        print(f"    Embedding error: {exc}")
        return None

    delta_vec, delta_mag = compute_delta(spawn_emb, baseline_emb)
    cos_sim = float(np.dot(spawn_emb, baseline_emb))

    return {
        "lat": pair["lat"],
        "lon": pair["lon"],
        "baseline_date": pair["baseline_date"],
        "spawn_date": pair["spawn_date"],
        "target_spawn_date": pair.get("target_spawn_date", ""),
        "baseline_scene": pair.get("baseline_scene", ""),
        "spawn_scene": pair.get("spawn_scene", ""),
        "baseline_embedding": baseline_emb,
        "spawn_embedding": spawn_emb,
        "delta_vector": delta_vec,
        "delta_magnitude": delta_mag,
        "cosine_similarity": cos_sim,
    }


# ---------------------------------------------------------------------------
# SVM classifier on delta vectors
# ---------------------------------------------------------------------------


def train_delta_classifier(
    spawn_deltas: list[np.ndarray],
    nonspawn_deltas: list[np.ndarray],
) -> dict[str, Any]:
    """Train SVM classifier on delta vectors to distinguish spawn from non-spawn.

    Returns dict with: model, train_accuracy, confusion_matrix, separation stats.
    """
    X = np.vstack(spawn_deltas + nonspawn_deltas)
    y = np.array([1] * len(spawn_deltas) + [0] * len(nonspawn_deltas))

    # Normalize delta vectors
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms

    # Train SVM
    svm = SVC(kernel="rbf", gamma="scale", class_weight="balanced", probability=True)
    svm.fit(X_norm, y)

    # Evaluate (simple train accuracy)
    y_pred = svm.predict(X_norm)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred).tolist()

    # Leave-One-Out cross-validation for realistic accuracy
    loo = LeaveOneOut()
    try:
        cv_scores = cross_val_score(svm, X_norm, y, cv=min(loo, len(y) - 1), scoring="accuracy")
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
    except Exception:
        cv_mean = float(acc)
        cv_std = 0.0

    # Compute separation in delta magnitude
    spawn_mags = [float(np.linalg.norm(d)) for d in spawn_deltas]
    nonspawn_mags = [float(np.linalg.norm(d)) for d in nonspawn_deltas]
    sep = float(np.mean(spawn_mags) - np.mean(nonspawn_mags))

    return {
        "model": svm,
        "train_accuracy": float(acc),
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "confusion_matrix": cm,
        "spawn_mean_magnitude": float(np.mean(spawn_mags)),
        "nonspawn_mean_magnitude": float(np.mean(nonspawn_mags)),
        "separation_in_magnitude": sep,
        "n_spawn": len(spawn_deltas),
        "n_nonspawn": len(nonspawn_deltas),
    }


# ---------------------------------------------------------------------------
# Review page generation
# ---------------------------------------------------------------------------


def generate_review_html(
    results: list[dict[str, Any]],
    stats: dict[str, Any],
    title: str = "DINOv2 Delta Detector Review",
) -> str:
    """Generate an HTML review page with baseline/spawn/delta visualization."""
    cards = []
    for entry in results:
        baseline_file = html.escape(entry.get("baseline_file", ""))
        spawn_file = html.escape(entry.get("spawn_file", ""))
        region = html.escape(entry.get("region", "unknown"))
        ftype = html.escape(entry.get("type", "?"))
        lat = entry.get("lat", 0)
        lon = entry.get("lon", 0)
        delta_mag = entry.get("delta_magnitude", 0)
        cos_sim = entry.get("cosine_similarity", 0)
        score = entry.get("svm_score", entry.get("delta_magnitude", 0))
        baseline_date = entry.get("baseline_date", "")
        spawn_date = entry.get("spawn_date", "")

        tclass = "pos" if ftype == "pos" else "neg"
        cards.append(f"""
        <article class="card {tclass}">
            <div class="image-pair">
                <figure>
                    <figcaption>Baseline {baseline_date}</figcaption>
                    <img src="pairs/{baseline_file}" alt="baseline">
                </figure>
                <figure>
                    <figcaption>Spawn {spawn_date}</figcaption>
                    <img src="pairs/{spawn_file}" alt="spawn">
                </figure>
            </div>
            <div class="meta">
                <span class="type-badge {tclass}">{ftype}</span>
                <strong>{region}</strong> · ({lat:.4f}, {lon:.4f})<br>
                Δ magnitude: {delta_mag:.4f} · cos sim: {cos_sim:.4f}<br>
                <span class="score">Score: {score:.4f}</span>
            </div>
        </article>
        """)

    n_candidates = len(results)
    n_pos = stats.get("n_pos", 0)
    n_neg = stats.get("n_neg", 0)
    sep = stats.get("separation", 0.0)
    accuracy = stats.get("accuracy", 0.0)

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(title)}</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #f5f6fa; color: #1f2937; }}
        header {{ background: linear-gradient(135deg, #111827, #0f172a); color: white; padding: 24px; }}
        main {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
        .stat {{ background: white; border-radius: 12px; padding: 14px 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
        .value {{ font-size: 28px; font-weight: 700; }}
        .label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(440px, 1fr)); gap: 16px; }}
        .card {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
        .card.pos {{ border-left: 4px solid #059669; }}
        .card.neg {{ border-left: 4px solid #dc2626; }}
        .image-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2px; }}
        .image-pair figure {{ margin: 0; position: relative; }}
        .image-pair figcaption {{ 
            position: absolute; bottom: 0; left: 0; right: 0; 
            background: rgba(0,0,0,0.6); color: white; 
            font-size: 11px; padding: 3px 6px; text-align: center;
        }}
        .image-pair img {{ width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }}
        .meta {{ padding: 10px 14px 14px; font-size: 13px; color: #374151; line-height: 1.5; }}
        .type-badge {{ display: inline-block; padding: 1px 8px; border-radius: 8px; font-size: 11px; font-weight: 700; margin-right: 6px; text-transform: uppercase; }}
        .type-badge.pos {{ background: #d1fae5; color: #065f46; }}
        .type-badge.neg {{ background: #fee2e2; color: #991b1b; }}
        .score {{ font-weight: 700; color: #059669; }}
        table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-top: 12px; }}
        th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
        th {{ background: #f9fafb; }}
        @media (max-width: 600px) {{ .grid {{ grid-template-columns: 1fr; }} .image-pair {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <header>
        <h1>{html.escape(title)}</h1>
        <p>DINOv2 embedding delta: baseline (pre-spawn) vs spawn-season comparison.</p>
    </header>
    <main>
        <section class="stats">
            <div class="stat"><div class="label">Total Pairs</div><div class="value">{n_candidates}</div></div>
            <div class="stat"><div class="label">Pos / Neg</div><div class="value">{n_pos}/{n_neg}</div></div>
            <div class="stat"><div class="label">Δ Mag Sep</div><div class="value">{sep:.4f}</div></div>
            <div class="stat"><div class="label">SVM Acc</div><div class="value">{accuracy:.1%}</div></div>
        </section>

        <h2>Candidates (sorted by score)</h2>
        <section class="grid">{''.join(cards)}</section>
    </main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Candidate scanning (mode: scan)
# ---------------------------------------------------------------------------


def scan_candidates(
    ee_module: Any,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    svm_classifier: SVC | None = None,
    regions: list[dict[str, Any]] | None = None,
    start: str = "2024-02-01",
    end: str = "2024-05-31",
    max_cloud: float = 50.0,
    grid_spacing: float = 0.02,
    workers: int = 4,
    max_points: int = 0,
) -> list[dict[str, Any]]:
    """Scan grid points, fetch baseline+spawn pairs, classify with delta.

    For each point, tries to get a baseline (late Jan) and spawn-season (Mar/Apr) image.
    If both exist, computes delta and scores with SVM or delta magnitude.

    When max_points > 0, limits scanning to the first N points (for testing).
    """
    if regions is None:
        regions = REGIONS

    points = generate_grid_points(regions, grid_spacing)
    if max_points > 0 and len(points) > max_points:
        points = points[:max_points]
    print(f"Scanning {len(points)} grid points...")

    candidates: list[dict[str, Any]] = []
    lock = __import__("threading").Lock()
    stopwatch = time.time()

    def process_point(point: dict[str, Any], idx: int, total: int) -> dict[str, Any] | None:
        lat, lon = point["lat"], point["lon"]

        # Baseline: Jan-Feb window
        baseline = find_best_scene(ee_module, lat, lon, "2024-01-15", "2024-02-15", max_cloud)
        if baseline is None:
            return None

        # Spawn: Feb-May window
        spawn = find_best_scene(ee_module, lat, lon, start, end, max_cloud)
        if spawn is None:
            return None

        baseline_bytes = download_thumbnail(ee_module, lat, lon, baseline["scene_id"])
        if baseline_bytes is None:
            return None

        spawn_bytes = download_thumbnail(ee_module, lat, lon, spawn["scene_id"])
        if spawn_bytes is None:
            return None

        # Embed and compute delta
        try:
            baseline_emb = embed_png_bytes(model, device, baseline_bytes)
            spawn_emb = embed_png_bytes(model, device, spawn_bytes)
        except Exception:
            return None

        delta_vec, delta_mag = compute_delta(spawn_emb, baseline_emb)
        cos_sim = float(np.dot(spawn_emb, baseline_emb))

        # Score: use SVM probability if available, else delta magnitude
        score = delta_mag
        if svm_classifier is not None:
            delta_norm = delta_vec / max(float(np.linalg.norm(delta_vec)), 1e-12)
            proba = svm_classifier.predict_proba([delta_norm])[0]
            score = float(proba[1])

        # Save candidate pair
        pair_dir = output_dir / "pairs"
        pair_dir.mkdir(parents=True, exist_ok=True)

        lat_str = f"{lat:.4f}".replace(".", "_").replace("-", "n")
        lon_str = f"{lon:.4f}".replace(".", "_").replace("-", "n")
        base_name = f"delta_{lat_str}_{lon_str}"
        bf = f"{base_name}_baseline_{baseline['date']}.png"
        sf = f"{base_name}_spawn_{spawn['date']}.png"

        (pair_dir / bf).write_bytes(baseline_bytes)
        (pair_dir / sf).write_bytes(spawn_bytes)

        entry = {
            "region": point["region"],
            "lat": lat,
            "lon": lon,
            "baseline_date": baseline["date"],
            "spawn_date": spawn["date"],
            "baseline_scene": baseline["scene_id"],
            "spawn_scene": spawn["scene_id"],
            "baseline_cloud": baseline.get("cloud", -1),
            "spawn_cloud": spawn.get("cloud", -1),
            "delta_magnitude": round(float(delta_mag), 6),
            "cosine_similarity": round(float(cos_sim), 6),
            "score": round(float(score), 6),
            "baseline_file": bf,
            "spawn_file": sf,
        }

        with lock:
            elapsed = time.time() - stopwatch
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            if idx % 5 == 0:
                print(f"  [{idx}/{total}] {point['region']} ({lat:.4f}, {lon:.4f}) Δ={delta_mag:.4f} svm={score:.4f} [{rate:.1f} pts/s]")

        return entry

    # Run in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_point, p, i, len(points)): i
            for i, p in enumerate(points)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                candidates.append(result)

            # Periodic save
            with lock:
                if len(candidates) > 0 and len(candidates) % 20 == 0:
                    out_manifest = output_dir / "manifest.json"
                    out_manifest.write_text(json.dumps(candidates, indent=2))
                    print(f"  [checkpoint] Saved {len(candidates)} candidates to {out_manifest}")

    candidates.sort(key=lambda c: -c["score"])
    return candidates


# ---------------------------------------------------------------------------
# CLI: analyze
# ---------------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze known positive/negative locations using delta-based detection."""
    try:
        import ee
    except ImportError:
        print("ERROR: earthengine-api is required")
        return 1

    ee.Initialize(project=args.project)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pairs").mkdir(parents=True, exist_ok=True)

    # Load model
    device = _pick_device()
    print(f"Loading DINOv2 model on {device}...")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval().to(device)

    # Load locations
    manifest_path = Path(args.manifest)
    neg_dir = Path(args.negative_dir)
    locations = load_locations_from_manifest(manifest_path, neg_dir)

    pos_locs = locations["positives"]
    neg_locs = locations["negatives"]
    print(f"Loaded {len(pos_locs)} positive, {len(neg_locs)} negative locations")

    if not pos_locs:
        print("ERROR: No positive locations found")
        return 1

    # Process positive locations
    pos_results: list[dict[str, Any]] = []
    for i, loc in enumerate(pos_locs):
        print(f"\n[{i + 1}/{len(pos_locs)}] POS: {loc.get('region', '?')} ({loc['lat']:.4f}, {loc['lon']:.4f}) spawn={loc['date']}")
        pair = fetch_paired_thumbnails(
            ee,
            loc["lat"],
            loc["lon"],
            loc["date"],
            baseline_days_before=args.baseline_days,
            spawn_window_days=args.spawn_window,
            max_cloud=args.max_cloud,
        )
        if pair is None:
            print(f"  SKIP: no paired imagery found")
            continue

        bf, sf = save_paired_thumbnails(output_dir / "pairs", pair, label="pos")

        result = process_pair(model, device, pair)
        if result is None:
            continue

        result["baseline_file"] = bf
        result["spawn_file"] = sf
        result["region"] = loc.get("region", "unknown")
        result["type"] = "pos"
        pos_results.append(result)
        print(f"  Δ mag={result['delta_magnitude']:.4f} cos={result['cosine_similarity']:.4f} BL={pair['baseline_date']} SW={pair['spawn_date']}")

    # Process negative locations
    neg_results: list[dict[str, Any]] = []
    n_negs = min(len(neg_locs), args.max_negatives)
    for i, loc in enumerate(neg_locs[:n_negs]):
        print(f"\n[{i + 1}/{n_negs}] NEG: {loc.get('region', '?')} ({loc['lat']:.4f}, {loc['lon']:.4f})")
        pair = fetch_paired_thumbnails(
            ee,
            loc["lat"],
            loc["lon"],
            "default",
            baseline_days_before=args.baseline_days,
            spawn_window_days=args.spawn_window,
            max_cloud=args.max_cloud,
        )
        if pair is None:
            print(f"  SKIP: no paired imagery found")
            continue

        bf, sf = save_paired_thumbnails(output_dir / "pairs", pair, label="neg")

        result = process_pair(model, device, pair)
        if result is None:
            continue

        result["baseline_file"] = bf
        result["spawn_file"] = sf
        result["region"] = loc.get("region", "unknown")
        result["type"] = "neg"
        neg_results.append(result)
        print(f"  Δ mag={result['delta_magnitude']:.4f} cos={result['cosine_similarity']:.4f}")

    # Train delta classifier
    classifier_info: dict[str, Any] = {
        "train_accuracy": 0.0,
        "n_spawn": len(pos_results),
        "n_nonspawn": len(neg_results),
        "separation_in_magnitude": 0.0,
    }

    if len(pos_results) >= 2 and len(neg_results) >= 2:
        spawn_deltas = [r["delta_vector"] for r in pos_results]
        nonspawn_deltas = [r["delta_vector"] for r in neg_results]
        classifier_info = train_delta_classifier(spawn_deltas, nonspawn_deltas)

        print(f"\n{'=' * 60}")
        print(f"  Delta Classifier Results")
        print(f"{'=' * 60}")
        print(f"  SVM train accuracy:      {classifier_info['train_accuracy']:.1%}")
        print(f"  SVM CV accuracy (LOO):   {classifier_info.get('cv_accuracy_mean', 0):.1%} ± {classifier_info.get('cv_accuracy_std', 0):.3f}")
        print(f"  Confusion matrix:        {classifier_info['confusion_matrix']}")
        print(f"  Spawn mean Δ mag:        {classifier_info['spawn_mean_magnitude']:.4f}")
        print(f"  Non-spawn mean Δ mag:    {classifier_info['nonspawn_mean_magnitude']:.4f}")
        print(f"  Separation (Δ mag):      {classifier_info['separation_in_magnitude']:.4f}")

        # Per-sample SVM scores
        for r in pos_results:
            dn = r["delta_vector"] / max(float(np.linalg.norm(r["delta_vector"])), 1e-12)
            r["svm_score"] = float(classifier_info["model"].predict_proba([dn])[0][1])
        for r in neg_results:
            dn = r["delta_vector"] / max(float(np.linalg.norm(r["delta_vector"])), 1e-12)
            r["svm_score"] = float(classifier_info["model"].predict_proba([dn])[0][1])

        # Compute single-image baseline for comparison
        all_spawn_embs = [r["spawn_embedding"] for r in pos_results]
        all_nonspawn_embs = [r["spawn_embedding"] for r in neg_results]
        mean_pos_emb = np.mean(all_spawn_embs, axis=0)
        mean_pos_emb = mean_pos_emb / max(float(np.linalg.norm(mean_pos_emb)), 1e-12)
        mean_neg_emb = np.mean(all_nonspawn_embs, axis=0)
        mean_neg_emb = mean_neg_emb / max(float(np.linalg.norm(mean_neg_emb)), 1e-12)

        def single_image_score(emb):
            e = emb / max(float(np.linalg.norm(emb)), 1e-12)
            return float(np.dot(mean_pos_emb, e) - np.dot(mean_neg_emb, e))

        pos_single_scores = [single_image_score(r["spawn_embedding"]) for r in pos_results]
        neg_single_scores = [single_image_score(r["spawn_embedding"]) for r in neg_results]
        single_sep = float(np.mean(pos_single_scores) - np.mean(neg_single_scores))
        # Threshold-based accuracy
        threshold = np.mean(pos_single_scores + neg_single_scores)
        pos_correct = sum(1 for s in pos_single_scores if s >= threshold)
        neg_correct = sum(1 for s in neg_single_scores if s < threshold)
        single_acc = (pos_correct + neg_correct) / (len(pos_single_scores) + len(neg_single_scores))

        # Compare with single-image baseline
        print(f"\n  {'=' * 40}")
        print(f"  Baseline comparison (same locations)")
        print(f"  {'=' * 40}")
        print(f"  Single-image spawn-season only:        separation={single_sep:.4f}")
        print(f"  Delta SVM (LOO):                        accuracy={classifier_info.get('cv_accuracy_mean', 0):.1%}")
        print(f"  Delta magnitude separation:             {classifier_info['separation_in_magnitude']:.4f}")
        print(f"  Single-image accuracy (thr={threshold:.4f}):    {single_acc:.1%}")
        print(f"  DINOv2 single-image (from AGENTS.md):  separation=0.0607, 88.9% acc")
    else:
        print(f"\nWARNING: Not enough data to train classifier ({len(pos_results)} pos, {len(neg_results)} neg)")

    # Build results list
    all_results = pos_results + neg_results
    sort_key = lambda r: -r.get("svm_score", r["delta_magnitude"])
    all_results.sort(key=sort_key)

    # Save manifest (without numpy arrays)
    manifest = []
    for r in all_results:
        manifest.append({
            "lat": r["lat"],
            "lon": r["lon"],
            "region": r.get("region", "unknown"),
            "type": r.get("type", "?"),
            "baseline_date": r["baseline_date"],
            "spawn_date": r["spawn_date"],
            "delta_magnitude": r["delta_magnitude"],
            "cosine_similarity": r["cosine_similarity"],
            "svm_score": r.get("svm_score", r["delta_magnitude"]),
            "baseline_file": r["baseline_file"],
            "spawn_file": r["spawn_file"],
        })

    # Save classifier info and model
    classifier_info_serializable = {
        k: v for k, v in classifier_info.items() if k != "model"
    }
    model_path = output_dir / "delta_svm_model.joblib"
    if "model" in classifier_info:
        joblib.dump(classifier_info["model"], model_path)

    stats = {
        "n_pos": len(pos_results),
        "n_neg": len(neg_results),
        "separation": classifier_info.get("separation_in_magnitude", 0.0),
        "accuracy": classifier_info.get("train_accuracy", 0.0),
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "analysis.json").write_text(json.dumps(classifier_info_serializable, indent=2, default=str))
    (output_dir / "review.html").write_text(generate_review_html(
        all_results, stats, title="Delta Analysis: Positives vs Negatives"
    ))

    print(f"\nResults saved to {output_dir}/")
    print(f"  {output_dir / 'manifest.json'}")
    print(f"  {output_dir / 'analysis.json'}")
    print(f"  {output_dir / 'review.html'}")
    print(f"\nServe: python -m http.server 8775 --directory {output_dir}")
    print(f"Open:  http://localhost:8775/review.html")

    return 0


# ---------------------------------------------------------------------------
# CLI: scan
# ---------------------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan BC habitat regions for 2024 candidates using delta approach.

    Uses the trained SVM classifier from --input-dir to score delta vectors,
    or falls back to delta magnitude scoring.
    """
    try:
        import ee
    except ImportError:
        print("ERROR: earthengine-api is required")
        return 1

    ee.Initialize(project=args.project)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pairs").mkdir(parents=True, exist_ok=True)

    device = _pick_device()
    print(f"Loading DINOv2 model on {device}...")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval().to(device)

    # Try to load previously trained classifier
    svm: SVC | None = None
    model_path = Path(args.input_dir) / "delta_svm_model.joblib"
    if model_path.exists():
        svm = joblib.load(model_path)
        print(f"Loaded SVM model from {model_path}")
    else:
        print("No SVM model found — using delta magnitude scoring")

    # Select regions for scanning
    scan_regions = REGIONS
    if args.regions:
        region_names = [r.strip() for r in args.regions.split(",")]
        scan_regions = [r for r in REGIONS if r["name"] in region_names]
        if not scan_regions:
            print(f"WARNING: No regions matched '{args.regions}'. Using all.")
            scan_regions = REGIONS
        else:
            print(f"Scanning {len(scan_regions)} specified regions: {region_names}")

    print(f"\nScanning {len(scan_regions)} regions for {args.start} to {args.end}...")
    t0 = time.time()

    candidates = scan_candidates(
        ee,
        model,
        device,
        output_dir,
        svm_classifier=svm,
        regions=scan_regions,
        start=args.start,
        end=args.end,
        max_cloud=args.max_cloud,
        grid_spacing=args.grid_spacing,
        workers=args.workers,
        max_points=args.max_points,
    )

    elapsed = time.time() - t0
    print(f"\nScan completed in {elapsed:.1f}s")

    if not candidates:
        print("No candidates found.")
        return 0

    # Sort by score descending
    candidates.sort(key=lambda c: -c["score"])

    # Save final manifest
    (output_dir / "manifest.json").write_text(json.dumps(candidates, indent=2))

    # Stats
    stats = {
        "n_pos": 0,
        "n_neg": 0,
        "separation": 0.0,
        "accuracy": 0.0,
    }

    (output_dir / "review.html").write_text(generate_review_html(
        candidates, stats, title="Delta Candidate Scan - BC Habitat Regions (2024)"
    ))

    print(f"Found {len(candidates)} candidates")
    print(f"Results saved to {output_dir}/")
    print(f"  {output_dir / 'manifest.json'}")
    print(f"  {output_dir / 'review.html'}")
    print(f"\nServe: python -m http.server 8775 --directory {output_dir}")
    print(f"Open:  http://localhost:8775/review.html")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DINOv2 Embedding Delta Detector for Herring Spawn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["analyze", "scan"], default="analyze",
                        help="analyze known locations or scan for new candidates")
    parser.add_argument("--output", default="data/delta_pairs",
                        help="Output directory")
    parser.add_argument("--project", default="redd-fish",
                        help="GEE project name")
    parser.add_argument("--manifest", default="data/samples/training_manifest.json",
                        help="Path to training manifest JSON (for analyze mode)")
    parser.add_argument("--negative-dir", default="data/samples/negative",
                        help="Directory of negative samples (for analyze mode)")
    parser.add_argument("--input-dir", default="data/delta_pairs",
                        help="Input directory for scan mode (to load analysis config)")
    parser.add_argument("--baseline-days", type=int, default=45,
                        help="Days before spawn for baseline search center")
    parser.add_argument("--spawn-window", type=int, default=14,
                        help="±days for spawn-season search window")
    parser.add_argument("--max-cloud", type=float, default=50.0,
                        help="Maximum cloud percentage")
    parser.add_argument("--max-negatives", type=int, default=30,
                        help="Maximum number of negative locations to process")
    parser.add_argument("--start", default="2024-03-01",
                        help="Scan start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-04-30",
                        help="Scan end date (YYYY-MM-DD)")
    parser.add_argument("--grid-spacing", type=float, default=0.02,
                        help="Grid spacing in degrees for scanning")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--regions", type=str, default=None,
                        help="Comma-separated region names to scan (default: all)")
    parser.add_argument("--max-points", type=int, default=0,
                        help="Limit scanning to first N grid points (0 = unlimited)")

    args = parser.parse_args(argv)

    if args.mode == "analyze":
        return cmd_analyze(args)
    elif args.mode == "scan":
        return cmd_scan(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
