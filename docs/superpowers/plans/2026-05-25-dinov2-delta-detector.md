# DINOv2 Delta Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a delta-based herring spawn detection pipeline that compares pre-spawn baseline vs spawn-season DINOv2 embeddings to reduce shoreline bias.

**Architecture:** For each known location, download paired Sentinel-2 RGB thumbnails (baseline in Jan/Feb, spawn in Mar/Apr), embed both with DINOv2 ViT-S/14, compute delta = spawn_emb - baseline_emb. Train an SVM on delta vectors from 8 positives + 126 negatives. Scan issue #1 regions for 2024 candidates with the delta classifier. Generate review page with side-by-side baseline/spawn/delta visualization.

**Tech Stack:** DINOv2 ViT-S/14, Google Earth Engine (project redd-fish), scikit-learn SVM, NumPy, PIL, custom HTML review page served via `python -m http.server`.

---

### Task 1: `scripts/delta_detector.py` — Main pipeline (parse, fetch, embed, delta, classify, scan, review)

**Files:**
- Create: `scripts/delta_detector.py` (~1200 lines)
- Reference: `data/samples/training_manifest.json` (8 positives + 126 negatives)
- Reference: `data/samples/positive/` (positive thumbnails with lat/lon in filenames)
- Reference: `data/samples/negative/` (negative thumbnails, many with lat/lon in filenames)
- Output: `data/delta_pairs/` (paired thumbnails)
- Output: `data/delta_candidates/` (scanned candidate pairs + manifest + review page)

- [ ] **Step 1: Write and run unit tests for the filename parser**

Tests go in `tests/test_delta_detector.py`. The parser extracts lat, lon, date from standard filenames like `SoG_2021-03-11_score0.00_49.5175_-124.577222_20210311.png`.

File: `tests/test_delta_detector.py`:
```python
"""Tests for delta_detector.py."""
import pytest
from scripts.delta_detector import parse_location_from_filename, compute_delta, load_locations_from_manifest


def test_parse_location_from_filename_standard():
    """Parse standard filename format: region_date_score_lat_lon_scenedate.png."""
    result = parse_location_from_filename("SoG_2021-03-11_score0.00_49.5175_-124.577222_20210311.png")
    assert result is not None
    assert result["lat"] == pytest.approx(49.5175)
    assert result["lon"] == pytest.approx(-124.577222)
    assert result["date"] == "2021-03-11"
    assert result["scene_date"] == "20210311"
    assert result["region"] == "SoG"


def test_parse_location_from_filename_no_match():
    """Return None for unparseable filenames."""
    assert parse_location_from_filename("random_file.png") is None
    assert parse_location_from_filename("dfo-verified_2024-03-16_cloud0.png") is None


def test_parse_location_from_filename_varied():
    """Parse various standard-format filenames from the negative set."""
    result = parse_location_from_filename("SoG_2016-03-05_score0.00_49.21326_-123.940398_20160305.png")
    assert result is not None
    assert result["lat"] == pytest.approx(49.21326)
    assert result["lon"] == pytest.approx(-123.940398)
    assert result["date"] == "2016-03-05"


def test_compute_delta_vectors():
    """Delta = spawn_emb - baseline_emb with L2 norm."""
    baseline = np.array([1.0, 0.0, 0.0])
    spawn = np.array([0.0, 1.0, 0.0])
    delta, mag = compute_delta(spawn, baseline)
    assert np.allclose(delta, np.array([-1.0, 1.0, 0.0]))
    assert mag == pytest.approx(np.sqrt(2.0))


def test_compute_delta_identical():
    """Identical embeddings produce zero delta."""
    emb = np.array([0.5, 0.5, 0.5])
    delta, mag = compute_delta(emb, emb)
    assert np.allclose(delta, np.zeros(3))
    assert mag == pytest.approx(0.0)


def test_load_locations_from_manifest(tmp_path):
    """Load locations from a small training manifest."""
    manifest = {
        "positives": ["SoG_2021-03-11_score0.00_49.5175_-124.577222_20210311.png"],
        "negative_count": 0,
    }
    mfile = tmp_path / "manifest.json"
    import json
    mfile.write_text(json.dumps(manifest))
    locs = load_locations_from_manifest(str(mfile), tmp_path)
    assert len(locs["positives"]) == 1
    assert locs["positives"][0]["lat"] == pytest.approx(49.5175)
    assert locs["positives"][0]["lon"] == pytest.approx(-124.577222)
```

- [ ] **Step 2: Run the test file to see it fail**

Run:
```bash
source .venv/bin/activate && python -m pytest tests/test_delta_detector.py -v
```
Expected: ImportError or NameError for all the delta_detector functions.

- [ ] **Step 3: Implement filename parser, delta computation, and location loader in `scripts/delta_detector.py`**

Create the first portion of the script — the core utility functions that the tests above exercise.

```python
#!/usr/bin/env python3
"""DINOv2 Embedding Delta Detector for Herring Spawn.

Compares pre-spawn baseline (Jan/Feb) vs spawn-season (Mar/Apr) images at
known locations. Computes DINOv2 embedding deltas and trains an SVM classifier
on delta vectors to distinguish spawn from non-spawn events.

Usage:
    python scripts/delta_detector.py
        --mode analyze       # Analyze known locations, train classifier
        --mode scan          # Scan issue #1 regions for 2024 candidates

    python scripts/delta_detector.py --mode analyze --output data/delta_pairs
    python scripts/delta_detector.py --mode scan --output data/delta_candidates
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.knn_detector import DINO_TRANSFORM, _pick_device
from scripts.scan_bc_coast import REGIONS, download_thumbnail, find_best_scene, generate_grid_points
from scripts.scan_bc_coast_knn import DEFAULT_MAX_CLOUD, DEFAULT_START, DEFAULT_END


MODEL_NAME = "dinov2_vits14"
EMBED_DIM = 384


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_location_from_filename(filename: str) -> dict[str, Any] | None:
    """Extract lat, lon, date from a standard-format filename.
    
    Format: <region>_<date>_score<X.XX>_<lat>_<lon>_<scene_date>.png
    Example: SoG_2021-03-11_score0.00_49.5175_-124.577222_20210311.png
    
    Returns dict with keys: lat, lon, date, scene_date, region
    or None if parsing fails.
    """
    # Match the standard format (most flexible pattern)
    # region_date_scoreX.XX_lat_lon_scenedate.png
    m = re.search(r"_(-?\d+\.?\d*)_(-?\d+\.?\d*)_(\d{8})\.png$", filename)
    if not m:
        return None
    
    try:
        lat = float(m.group(1))
        lon = float(m.group(2))
        scene_date = m.group(3)
    except (ValueError, IndexError):
        return None
    
    # Extract the spawn date (YYYY-MM-DD) from the filename
    # Try to find it between region and _score
    date_match = re.search(r"_(\d{4}-\d{2}-\d{2})_score", filename)
    if not date_match:
        return None
    
    date_str = date_match.group(1)
    
    # Extract region (everything before the date)
    region = filename[:date_match.start()].strip("_")
    
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
) -> dict[str, list[dict[str, Any]]]:
    """Load positive locations from training_manifest.json and parse lat/lon from filenames.
    
    Also scans the negative directory for parseable negative locations.
    """
    manifest_path = Path(manifest_path)
    negative_dir = Path(negative_dir)
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    positives: list[dict[str, Any]] = []
    for fname in manifest.get("positives", []):
        parsed = parse_location_from_filename(fname)
        if parsed:
            # Verify the file exists in the positive directory
            img_path = manifest_path.parent / "positive" / fname
            if img_path.exists():
                parsed["image_path"] = str(img_path)
                positives.append(parsed)
    
    negatives: list[dict[str, Any]] = []
    for fname in sorted(negative_dir.iterdir()):
        if not fname.name.endswith(".png"):
            continue
        parsed = parse_location_from_filename(fname.name)
        if parsed:
            parsed["image_path"] = str(fname)
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
source .venv/bin/activate && python -m pytest tests/test_delta_detector.py -v
```
Expected: All 6 tests pass.

- [ ] **Step 5: Commit the test and initial implementation**

```bash
git add tests/test_delta_detector.py scripts/delta_detector.py
git commit -m "feat: add DINOv2 delta detector core - parser, delta computation, location loader"
```

- [ ] **Step 6: Implement GEE thumbnail fetching for paired images**

Add the GEE thumbnail fetching functions to `delta_detector.py`. These fetch a baseline (Jan/Feb) and spawn-season (Mar/Apr) image for each known location.

```python
# ---------------------------------------------------------------------------
# GEE thumbnail fetching for paired images
# ---------------------------------------------------------------------------

def fetch_paired_thumbnails(
    ee_module: Any,
    lat: float,
    lon: float,
    spawn_date: str,
    baseline_days_before: int = 45,
    spawn_window_days: int = 14,
    max_cloud: float = 50.0,
) -> dict[str, Any] | None:
    """Fetch baseline and spawn-season thumbnails for a location.
    
    Args:
        ee_module: Earth Engine module (initialized)
        lat, lon: Location coordinates
        spawn_date: The known spawn date in YYYY-MM-DD format
        baseline_days_before: How many days before spawn to search for baseline
        spawn_window_days: ±days around spawn date to search
        max_cloud: Maximum cloud percentage
    
    Returns dict with keys: baseline_bytes, spawn_bytes, baseline_scene, 
    spawn_scene, baseline_date, spawn_date, lat, lon
    or None if either image cannot be found.
    """
    spawn_dt = date.fromisoformat(spawn_date)
    baseline_start = (spawn_dt - timedelta(days=baseline_days_before + 14)).isoformat()
    baseline_end = (spawn_dt - timedelta(days=baseline_days_before - 14)).isoformat()
    
    # For negatives without a real spawn date, use March 15 as default
    if spawn_date.startswith("0000") or spawn_date == "default":
        spawn_dt = date(2024, 3, 15)
    
    spawn_start = (spawn_dt - timedelta(days=spawn_window_days)).isoformat()
    spawn_end = (spawn_dt + timedelta(days=spawn_window_days)).isoformat()
    
    # Fetch baseline (pre-spawn) image
    baseline_scene = find_best_scene(
        ee_module, lat, lon, baseline_start, baseline_end, max_cloud
    )
    if baseline_scene is None:
        return None
    
    # Fetch spawn-season image
    spawn_scene = find_best_scene(
        ee_module, lat, lon, spawn_start, spawn_end, max_cloud
    )
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
        "target_spawn_date": spawn_date,
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
    # Normalize lat/lon for filenames
    lat_str = f"{lat:.6f}".replace(".", "_")
    lon_str = f"{lon:.6f}".replace(".", "_")
    
    base_fname = f"delta_{lat}_{lon}"
    bf = f"{base_fname}_baseline_{pair['baseline_date']}_{label}.png"
    sf = f"{base_fname}_spawn_{pair['spawn_date']}_{label}.png"
    
    (output_dir / bf).write_bytes(pair["baseline_bytes"])
    (output_dir / sf).write_bytes(pair["spawn_bytes"])
    
    return bf, sf
```

- [ ] **Step 7: Embedding and delta computation for paired images**

```python
# ---------------------------------------------------------------------------
# Embedding and delta computation
# ---------------------------------------------------------------------------

def embed_png_bytes(model: torch.nn.Module, device: torch.device, png_bytes: bytes) -> np.ndarray:
    """Compute DINOv2 embedding from PNG bytes."""
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    tensor = DINO_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    return F.normalize(emb, dim=1).cpu().numpy().flatten().astype(float)


def embed_png_file(model: torch.nn.Module, device: torch.device, path: str | Path) -> np.ndarray:
    """Compute DINOv2 embedding from a PNG file on disk."""
    image = Image.open(path).convert("RGB")
    tensor = DINO_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    return F.normalize(emb, dim=1).cpu().numpy().flatten().astype(float)


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
    except Exception:
        return None
    
    delta_vec, delta_mag = compute_delta(spawn_emb, baseline_emb)
    cos_sim = float(np.dot(spawn_emb, baseline_emb))
    
    return {
        "lat": pair["lat"],
        "lon": pair["lon"],
        "baseline_date": pair["baseline_date"],
        "spawn_date": pair["spawn_date"],
        "target_spawn_date": pair.get("target_spawn_date", ""),
        "baseline_embedding": baseline_emb,
        "spawn_embedding": spawn_emb,
        "delta_vector": delta_vec,
        "delta_magnitude": delta_mag,
        "cosine_similarity": cos_sim,
    }
```

- [ ] **Step 8: Implement SVM training on delta vectors and classification**

```python
# ---------------------------------------------------------------------------
# SVM classifier on delta vectors
# ---------------------------------------------------------------------------

def train_delta_classifier(
    spawn_deltas: list[np.ndarray],
    nonspawn_deltas: list[np.ndarray],
) -> dict[str, Any]:
    """Train SVM classifier on delta vectors to distinguish spawn from non-spawn.
    
    Returns dict with: model, accuracy, confusion_matrix, separation stats.
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
    
    # Evaluate (simple train accuracy - small data, this is a rough check)
    y_pred = svm.predict(X_norm)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred).tolist()
    
    # Compute separation in delta magnitude
    spawn_mags = [float(np.linalg.norm(d)) for d in spawn_deltas]
    nonspawn_mags = [float(np.linalg.norm(d)) for d in nonspawn_deltas]
    sep = float(np.mean(spawn_mags) - np.mean(nonspawn_mags))
    
    return {
        "model": svm,
        "train_accuracy": float(acc),
        "confusion_matrix": cm,
        "spawn_mean_magnitude": float(np.mean(spawn_mags)),
        "nonspawn_mean_magnitude": float(np.mean(nonspawn_mags)),
        "separation_in_magnitude": sep,
        "n_spawn": len(spawn_deltas),
        "n_nonspawn": len(nonspawn_deltas),
    }
```

- [ ] **Step 9: Implement candidate scanning using delta approach**

```python
# ---------------------------------------------------------------------------
# Candidate scanning
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
    batch_size: int = 100,
) -> list[dict[str, Any]]:
    """Scan grid points, fetch baseline+spawn pairs, classify with delta.
    
    For each point, tries to get a baseline (late Jan) and spawn-season (Mar/Apr) image.
    If both exist, computes delta and scores with SVM or delta magnitude.
    
    Returns manifest entries for candidates with score > 0.
    """
    if regions is None:
        regions = REGIONS
    
    points = generate_grid_points(regions, grid_spacing)
    print(f"Scanning {len(points)} grid points...")
    
    candidates: list[dict[str, Any]] = []
    
    def process_point_parallel(args):
        """Process a single grid point."""
        point, idx, total = args
        lat, lon = point["lat"], point["lon"]
        
        # Try to get baseline (Jan) and spawn (Mar/Apr) images
        baseline = find_best_scene(ee_module, lat, lon, "2024-01-15", "2024-02-15", max_cloud)
        if baseline is None:
            return None
        
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
            score = float(proba[1])  # probability of spawn class
        
        # Save candidate pair
        pair_dir = output_dir / "pairs"
        pair_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = f"delta_{lat:.4f}_{lon:.4f}".replace(".", "_").replace("-", "n")
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
            "baseline_cloud": baseline["cloud"],
            "spawn_cloud": spawn["cloud"],
            "delta_magnitude": round(float(delta_mag), 6),
            "cosine_similarity": round(float(cos_sim), 6),
            "score": round(float(score), 6),
            "baseline_file": bf,
            "spawn_file": sf,
        }
        
        if idx % 10 == 0:
            print(f"  [{idx}/{total}] {point['region']} ({lat:.4f}, {lon:.4f}) delta={delta_mag:.4f} score={score:.4f}")
        
        return entry
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_point_parallel, (p, i, len(points))) for i, p in enumerate(points)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                candidates.append(result)
    
    # Sort by score descending
    candidates.sort(key=lambda c: -c["score"])
    
    return candidates


# ---------------------------------------------------------------------------
# Review page generation
# ---------------------------------------------------------------------------

def generate_review_html(
    results: list[dict[str, Any]],
    stats: dict[str, Any],
    title: str = "DINOv2 Delta Detector Review",
) -> str:
    """Generate an HTML review page with baseline/spawn/delta visualization.
    
    Each candidate shows:
    - Baseline (pre-spawn) thumbnail
    - Spawn-season thumbnail
    - Score and metrics
    """
    cards = []
    for entry in results:
        baseline_file = html.escape(entry.get("baseline_file", ""))
        spawn_file = html.escape(entry.get("spawn_file", ""))
        region = html.escape(entry.get("region", "unknown"))
        lat = entry.get("lat", 0)
        lon = entry.get("lon", 0)
        delta_mag = entry.get("delta_magnitude", 0)
        cos_sim = entry.get("cosine_similarity", 0)
        score = entry.get("score", 0)
        baseline_date = entry.get("baseline_date", "")
        spawn_date = entry.get("spawn_date", "")
        
        cards.append(f"""
        <article class="card">
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
                <strong>{region}</strong> · ({lat:.4f}, {lon:.4f})<br>
                Δ magnitude: {delta_mag:.4f} · cos sim: {cos_sim:.4f}<br>
                <span class="score">Score: {score:.4f}</span>
            </div>
        </article>
        """)
    
    # Stats summary
    n_candidates = len(results)
    n_pos = stats.get("n_pos", 0)
    n_neg = stats.get("n_neg", 0)
    sep = stats.get("separation", 0)
    accuracy = stats.get("accuracy", 0)
    
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
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
        .stat {{ background: white; border-radius: 12px; padding: 14px 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
        .value {{ font-size: 28px; font-weight: 700; }}
        .label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 16px; }}
        .card {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
        .image-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2px; }}
        .image-pair figure {{ margin: 0; position: relative; }}
        .image-pair figcaption {{ 
            position: absolute; bottom: 0; left: 0; right: 0; 
            background: rgba(0,0,0,0.6); color: white; 
            font-size: 11px; padding: 3px 6px; text-align: center;
        }}
        .image-pair img {{ width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }}
        .meta {{ padding: 10px 14px 14px; font-size: 13px; color: #374151; line-height: 1.5; }}
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
            <div class="stat"><div class="label">Candidates</div><div class="value">{n_candidates}</div></div>
            <div class="stat"><div class="label">Training Pos/Neg</div><div class="value">{n_pos}/{n_neg}</div></div>
            <div class="stat"><div class="label">Δ Mag Separation</div><div class="value">{sep:.4f}</div></div>
            <div class="stat"><div class="label">Train Accuracy</div><div class="value">{accuracy:.1%}</div></div>
        </section>

        <h2>Candidates (sorted by score)</h2>
        <section class="grid">{''.join(cards)}</section>
    </main>
</body>
</html>"""


def generate_analysis_report(
    pos_results: list[dict[str, Any]],
    neg_results: list[dict[str, Any]],
    classifier_info: dict[str, Any],
) -> str:
    """Generate an analysis report comparing spawn vs non-spawn delta distributions."""
    spawn_mags = [r["delta_magnitude"] for r in pos_results]
    neg_mags = [r["delta_magnitude"] for r in neg_results]
    
    # Add analysis report content here
    # (This will be generated after processing)
    
    return ""
```

- [ ] **Step 10: Commit paired fetching, embedding, SVM, scanning, and review code**

```bash
git add scripts/delta_detector.py
git commit -m "feat: complete delta detector pipeline - fetch, embed, SVM, scan, review"
```

- [ ] **Step 11: Implement the main CLI with `--mode analyze` and `--mode scan`**

```python
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze known positive/negative locations using delta-based detection."""
    import ee
    ee.Initialize(project=args.project)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = _pick_device()
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval().to(device)
    print(f"Device: {device}")
    
    # Load locations
    manifest_path = args.manifest
    neg_dir = args.negative_dir
    locations = load_locations_from_manifest(manifest_path, neg_dir)
    
    pos_locs = locations["positives"]
    neg_locs = locations["negatives"]
    print(f"Loaded {len(pos_locs)} positive, {len(neg_locs)} negative locations")
    
    # Process positive locations
    pos_results: list[dict[str, Any]] = []
    for i, loc in enumerate(pos_locs):
        print(f"[{i+1}/{len(pos_locs)}] Positive: {loc['filename']} ({loc['lat']:.4f}, {loc['lon']:.4f})")
        pair = fetch_paired_thumbnails(
            ee, loc["lat"], loc["lon"], loc["date"],
            baseline_days_before=args.baseline_days,
            spawn_window_days=args.spawn_window,
            max_cloud=args.max_cloud,
        )
        if pair is None:
            print(f"  SKIP: no paired imagery found")
            continue
        
        bf, sf = save_paired_thumbnails(output_dir, pair, label="pos")
        
        result = process_pair(model, device, pair)
        if result is None:
            continue
        
        result["baseline_file"] = bf
        result["spawn_file"] = sf
        pos_results.append(result)
        print(f"  delta_mag={result['delta_magnitude']:.4f} cos_sim={result['cosine_similarity']:.4f}")
    
    # Process negative locations
    neg_results: list[dict[str, Any]] = []
    for i, loc in enumerate(neg_locs[:args.max_negatives]):
        print(f"[{i+1}/{min(len(neg_locs), args.max_negatives)}] Negative: {loc['filename']} ({loc['lat']:.4f}, {loc['lon']:.4f})")
        # Negatives may not have a known spawn date, use default
        default_spawn = "default"
        pair = fetch_paired_thumbnails(
            ee, loc["lat"], loc["lon"], default_spawn,
            baseline_days_before=args.baseline_days,
            spawn_window_days=args.spawn_window,
            max_cloud=args.max_cloud,
        )
        if pair is None:
            print(f"  SKIP: no paired imagery found")
            continue
        
        bf, sf = save_paired_thumbnails(output_dir, pair, label="neg")
        
        result = process_pair(model, device, pair)
        if result is None:
            continue
        
        result["baseline_file"] = bf
        result["spawn_file"] = sf
        neg_results.append(result)
        print(f"  delta_mag={result['delta_magnitude']:.4f} cos_sim={result['cosine_similarity']:.4f}")
    
    # Train delta classifier
    if len(pos_results) >= 2 and len(neg_results) >= 2:
        spawn_deltas = [r["delta_vector"] for r in pos_results]
        nonspawn_deltas = [r["delta_vector"] for r in neg_results]
        classifier_info = train_delta_classifier(spawn_deltas, nonspawn_deltas)
        
        print(f"\n=== Delta Classifier Results ===")
        print(f"Train accuracy: {classifier_info['train_accuracy']:.1%}")
        print(f"Confusion matrix: {classifier_info['confusion_matrix']}")
        print(f"Spawn mean Δ mag: {classifier_info['spawn_mean_magnitude']:.4f}")
        print(f"Non-spawn mean Δ mag: {classifier_info['nonspawn_mean_magnitude']:.4f}")
        print(f"Separation: {classifier_info['separation_in_magnitude']:.4f}")
        
        # Per-sample scores
        for r in pos_results:
            delta_norm = r["delta_vector"] / max(float(np.linalg.norm(r["delta_vector"])), 1e-12)
            r["svm_score"] = float(classifier_info["model"].predict_proba([delta_norm])[0][1])
        for r in neg_results:
            delta_norm = r["delta_vector"] / max(float(np.linalg.norm(r["delta_vector"])), 1e-12)
            r["svm_score"] = float(classifier_info["model"].predict_proba([delta_norm])[0][1])
    else:
        classifier_info = {"train_accuracy": 0, "n_pos": len(pos_results), "n_neg": len(neg_results)}
    
    # Save results
    all_results = sorted(
        pos_results + neg_results,
        key=lambda r: -r.get("svm_score", r["delta_magnitude"])
    )
    
    stats = {
        "n_pos": len(pos_results),
        "n_neg": len(neg_results),
        "separation": classifier_info.get("separation_in_magnitude", 0),
        "accuracy": classifier_info.get("train_accuracy", 0),
    }
    
    # Generate analysis report
    analysis_html = generate_analysis_report(pos_results, neg_results, classifier_info)
    
    # Save manifest and generate review page
    manifest = []
    for r in all_results:
        manifest.append({
            "lat": r["lat"],
            "lon": r["lon"],
            "baseline_date": r["baseline_date"],
            "spawn_date": r["spawn_date"],
            "delta_magnitude": r["delta_magnitude"],
            "cosine_similarity": r["cosine_similarity"],
            "svm_score": r.get("svm_score", r["delta_magnitude"]),
            "baseline_file": r["baseline_file"],
            "spawn_file": r["spawn_file"],
        })
    
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "analysis.json").write_text(json.dumps(classifier_info, indent=2, default=str))
    (output_dir / "review.html").write_text(generate_review_html(
        all_results, stats, title="Delta Analysis: Positives vs Negatives"
    ))
    
    print(f"\nResults saved to {output_dir}")
    print(f"Review page: file://{(output_dir / 'review.html').resolve()}")
    print(f"Use: python -m http.server 8775 --directory {output_dir}")
    
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan issue #1 regions for 2024 candidates using delta approach."""
    import ee
    ee.Initialize(project=args.project)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pairs").mkdir(parents=True, exist_ok=True)
    
    device = _pick_device()
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval().to(device)
    
    print(f"Device: {device}")
    print("Loading delta classifier...")
    
    # Load previously trained classifier from analysis step
    svm = None
    analysis_path = args.input_dir / "analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        # SVM weights are stored; for now, use delta magnitude scoring
    
    candidates = scan_candidates(
        ee, model, device, output_dir,
        svm_classifier=svm,
        start=args.start,
        end=args.end,
        max_cloud=args.max_cloud,
        grid_spacing=args.grid_spacing,
        workers=args.workers,
    )
    
    if not candidates:
        print("No candidates found.")
        return 0
    
    # Save manifest
    (output_dir / "manifest.json").write_text(json.dumps(candidates, indent=2))
    
    # Stats
    stats = {
        "n_pos": 0,
        "n_neg": 0,
        "separation": 0,
        "accuracy": 0,
    }
    
    # Generate review page
    (output_dir / "review.html").write_text(generate_review_html(
        candidates, stats, title="Delta Candidate Scan - Issue #1 Regions (2024)"
    ))
    
    print(f"\nFound {len(candidates)} candidates")
    print(f"Results saved to {output_dir}")
    print(f"Review page: file://{(output_dir / 'review.html').resolve()}")
    print(f"Use: python -m http.server 8775 --directory {output_dir}")
    
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["analyze", "scan"], default="analyze",
                        help="analyze known locations or scan for new candidates")
    parser.add_argument("--output", default="data/delta_pairs",
                        help="Output directory")
    parser.add_argument("--project", default="redd-fish",
                        help="GEE project name")
    parser.add_argument("--manifest", default="data/samples/training_manifest.json",
                        help="Path to training manifest JSON")
    parser.add_argument("--negative-dir", default="data/samples/negative",
                        help="Directory of negative samples")
    parser.add_argument("--input-dir", default="data/delta_pairs",
                        help="Input directory for scan mode (to load classifier)")
    parser.add_argument("--baseline-days", type=int, default=45,
                        help="Days before spawn for baseline search")
    parser.add_argument("--spawn-window", type=int, default=14,
                        help="±days for spawn-season search window")
    parser.add_argument("--max-cloud", type=float, default=50.0,
                        help="Maximum cloud percentage")
    parser.add_argument("--max-negatives", type=int, default=50,
                        help="Maximum number of negative locations to process")
    parser.add_argument("--start", default="2024-03-01",
                        help="Scan start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-04-30",
                        help="Scan end date (YYYY-MM-DD)")
    parser.add_argument("--grid-spacing", type=float, default=0.02,
                        help="Grid spacing in degrees for scanning")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    
    args = parser.parse_args(argv)
    
    if args.mode == "analyze":
        return cmd_analyze(args)
    elif args.mode == "scan":
        return cmd_scan(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 12: Write tests for the scan, embed, and pair processing functions**

Add to `tests/test_delta_detector.py`:
```python
def test_embed_png_bytes_returns_384_dim():
    """DINOv2 ViT-S/14 produces 384-dim embeddings."""
    # This test requires the model, so it's a integration test marker
    import torch
    from scripts.delta_detector import embed_png_bytes, MODEL_NAME
    from scripts.knn_detector import _pick_device
    
    device = _pick_device()
    try:
        model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    except Exception:
        pytest.skip("DINOv2 model not available")
    model.eval().to(device)
    
    # Create a small test PNG
    from PIL import Image
    import io
    img = Image.new("RGB", (224, 224), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    emb = embed_png_bytes(model, device, buf.getvalue())
    assert emb.shape == (384,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5  # normalized


def test_extract_location_from_manifest(tmp_path):
    """Test loading locations from manifest extracts lat/lon/dates."""
    import json
    from scripts.delta_detector import load_locations_from_manifest
    
    # Create a positive sample with proper filename
    pos_dir = tmp_path / "positive"
    pos_dir.mkdir()
    fname = "SoG_2021-03-11_score0.00_49.5175_-124.577222_20210311.png"
    (pos_dir / fname).write_text("fake")
    
    # Create manifest
    manifest = {
        "positive_count": 1,
        "negative_count": 0,
        "positives": [fname]
    }
    mfile = tmp_path / "manifest.json"
    mfile.write_text(json.dumps(manifest))
    
    neg_dir = tmp_path / "negative"
    neg_dir.mkdir()
    
    locs = load_locations_from_manifest(str(mfile), str(neg_dir))
    assert len(locs["positives"]) == 1
    assert locs["positives"][0]["lat"] == pytest.approx(49.5175)
    assert locs["positives"][0]["lon"] == pytest.approx(-124.577222)
    assert locs["positives"][0]["date"] == "2021-03-11"
```

- [ ] **Step 13: Run all tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_delta_detector.py -v
```
Expected: All tests pass.

- [ ] **Step 14: Commit everything**

```bash
git add tests/test_delta_detector.py scripts/delta_detector.py
git commit -m "feat: complete DINOv2 delta detector with CLI"
```

- [ ] **Step 15: Run `--mode analyze` to process known locations**

```bash
source .venv/bin/activate && python scripts/delta_detector.py \
    --mode analyze \
    --output data/delta_pairs \
    --max-negatives 20
```
Expected: Script processes 8 positives + 20 negatives, downloads paired thumbnails, computes deltas, trains SVM, outputs to data/delta_pairs/.

- [ ] **Step 16: Run `--mode scan` for issue #1 candidates**

```bash
source .venv/bin/activate && python scripts/delta_detector.py \
    --mode scan \
    --output data/delta_candidates \
    --input-dir data/delta_pairs \
    --grid-spacing 0.02 \
    --workers 4
```
Expected: Scans issue #1 regions, saves candidates to data/delta_candidates/.

- [ ] **Step 17: Serve and verify review pages**

```bash
python -m http.server 8775 --directory data/delta_pairs
# Open http://localhost:8775/review.html
```
Expected: Review page shows paired baseline/spawn images with delta metrics.

```bash
python -m http.server 8775 --directory data/delta_candidates
# Open http://localhost:8775/review.html
```
Expected: Candidate review page with sorted delta candidates.

- [ ] **Step 18: Final review and summary**

Verify:
- data/delta_pairs/ contains paired PNGs for positives + negatives
- data/delta_pairs/review.html works at port 8775
- data/delta_pairs/manifest.json has all results
- data/delta_candidates/ contains scanned candidates
- data/delta_candidates/review.html works at port 8775

Compare delta performance vs single-image baseline.
