#!/usr/bin/env python3
"""GeoRSCLIP-based BC coast scan for herring spawn candidates.

Replaces DINOv2 with GeoRSCLIP ViT-B-32 (AUROC 0.969 on unified set).
Uses unified labels (28 pos, 340 neg) as KNN reference. Stores candidates
on external drive to keep main disk lean.

Usage:
    python scripts/scan_bc_georsclip.py \
        --output "/Volumes/Z Slim/data/candidates_georsclip" \
        --start 2024-02-01 --end 2024-05-31 \
        --grid-spacing 0.02 --workers 8 --k 3
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# GeoRSCLIP setup
# ---------------------------------------------------------------------------

import open_clip
from huggingface_hub import hf_hub_download

GEO_MODEL = None
GEO_PREPROCESS = None
MODEL_LOCK = threading.Lock()

def _init_model():
    global GEO_MODEL, GEO_PREPROCESS
    with MODEL_LOCK:
        if GEO_MODEL is None:
            ckpt = hf_hub_download("Zilun/GeoRSCLIP", "ckpt/RS5M_ViT-B-32.pt")
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=ckpt
            )
            model.eval()
            GEO_MODEL = model
            GEO_PREPROCESS = preprocess
            print(f"  GeoRSCLIP ViT-B-32 loaded (512-dim)")

# ---------------------------------------------------------------------------
# Imports from existing pipeline
# ---------------------------------------------------------------------------

from scripts.scan_bc_coast import (
    REGIONS,
    download_thumbnail,
    find_best_scene,
    generate_grid_points,
    print_progress as _print_progress_orig,
    save_candidate,
    update_manifest,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = Path("/Volumes/Z Slim/data/candidates_georsclip")
DEFAULT_START = "2024-02-01"
DEFAULT_END = "2024-05-31"
DEFAULT_MAX_CLOUD = 50.0
DEFAULT_GRID_SPACING = 0.02
DEFAULT_WORKERS = 8
DEFAULT_K = 3

MANIFEST_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()
STATS = {"processed": 0, "candidates": 0, "no_scene": 0, "download_errors": 0, "skipped": 0}

# ---------------------------------------------------------------------------
# Reference embeddings from unified labels
# ---------------------------------------------------------------------------

def load_reference_embeddings(repo_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and embed all labeled images from the unified dataset."""
    manifest_path = repo_root / "data/unified/manifest.json"
    labels_path = repo_root / "data/unified/labels.json"
    thumbs_dir = repo_root / "data/unified/thumbs"

    if not manifest_path.exists():
        print("WARNING: unified manifest not found, using golden set fallback")
        return _load_golden_embeddings(repo_root)

    manifest = json.loads(manifest_path.read_text())
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}

    pos_embs, neg_embs = [], []
    for entry in manifest:
        fname = entry["filename"]
        label = labels.get(fname, entry.get("label"))
        if label not in ("spawn", "no-spawn"):
            continue

        img_path = thumbs_dir / fname
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            t = GEO_PREPROCESS(img).unsqueeze(0)
            with torch.no_grad():
                emb = GEO_MODEL.encode_image(t).squeeze(0).numpy()
            if label == "spawn":
                pos_embs.append(emb)
            else:
                neg_embs.append(emb)
        except Exception:
            continue

    print(f"  Reference: {len(pos_embs)} spawn, {len(neg_embs)} no-spawn embeddings")
    return np.array(pos_embs), np.array(neg_embs)


def _load_golden_embeddings(repo_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: load from golden set."""
    manifest = json.loads((repo_root / "data/samples/training_manifest.json").read_text())
    pos_dir = repo_root / "data/samples/positive"
    neg_dir = repo_root / "data/samples/negative"

    pos_embs, neg_embs = [], []
    for fname in manifest.get("positives", []):
        img = Image.open(pos_dir / fname).convert("RGB")
        t = GEO_PREPROCESS(img).unsqueeze(0)
        with torch.no_grad():
            emb = GEO_MODEL.encode_image(t).squeeze(0).numpy()
        pos_embs.append(emb)

    for png in sorted((neg_dir).glob("*.png")):
        img = Image.open(png).convert("RGB")
        t = GEO_PREPROCESS(img).unsqueeze(0)
        with torch.no_grad():
            emb = GEO_MODEL.encode_image(t).squeeze(0).numpy()
        neg_embs.append(emb)

    print(f"  Reference (golden): {len(pos_embs)} spawn, {len(neg_embs)} no-spawn")
    return np.array(pos_embs), np.array(neg_embs)


# ---------------------------------------------------------------------------
# KNN scoring
# ---------------------------------------------------------------------------

def knn_score(query: np.ndarray, ref_pos: np.ndarray, ref_neg: np.ndarray, k: int) -> dict:
    """Score a query embedding using KNN majority vote."""
    all_ref = np.vstack([ref_pos, ref_neg])
    all_labels = np.array([1] * len(ref_pos) + [0] * len(ref_neg))

    # Cosine distance
    query_norm = query / (np.linalg.norm(query) + 1e-8)
    ref_norm = all_ref / (np.linalg.norm(all_ref, axis=1, keepdims=True) + 1e-8)
    similarities = np.dot(ref_norm, query_norm)

    top_k = np.argsort(similarities)[-k:]
    votes = all_labels[top_k].sum()
    score = votes / k

    # Also compute similarity to nearest neighbors
    nn_sims = similarities[top_k][::-1]
    nn_labels = all_labels[top_k][::-1]

    return {
        "score": float(score),
        "votes": int(votes),
        "k": k,
        "nn_similarities": nn_sims.tolist(),
        "nn_labels": nn_labels.tolist(),
    }


# ---------------------------------------------------------------------------
# Thumbnail embedding
# ---------------------------------------------------------------------------

def embed_thumbnail(thumb_bytes: bytes) -> np.ndarray | None:
    """Embed a thumbnail PNG as GeoRSCLIP vector."""
    try:
        img = Image.open(io.BytesIO(thumb_bytes)).convert("RGB")
        t = GEO_PREPROCESS(img).unsqueeze(0)
        with torch.no_grad():
            return GEO_MODEL.encode_image(t).squeeze(0).numpy()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Point processing
# ---------------------------------------------------------------------------

def process_point(
    point: dict,
    args: argparse.Namespace,
    output_dir: Path,
    ref_pos: np.ndarray,
    ref_neg: np.ndarray,
    idx: int,
    total: int,
    ee_module: Any,
) -> dict:
    """Process a single grid point."""
    result = {"processed": 1, "candidates": 0, "no_scene": 0, "errors": 0}

    scene_info = find_best_scene(
        ee_module, point["lat"], point["lon"], args.start, args.end, args.max_cloud
    )
    if scene_info is None:
        with PRINT_LOCK:
            print(f"  [{idx+1}/{total}] {point['region']} ({point['lat']:.4f},{point['lon']:.4f}) no-scene")
        result["no_scene"] = 1
        return result

    thumb_bytes = download_thumbnail(
        ee_module, point["lat"], point["lon"], scene_info["scene_id"]
    )
    if thumb_bytes is None:
        with PRINT_LOCK:
            print(f"  [{idx+1}/{total}] {point['region']} dl-error")
        result["errors"] = 1
        return result

    emb = embed_thumbnail(thumb_bytes)
    if emb is None:
        with PRINT_LOCK:
            print(f"  [{idx+1}/{total}] {point['region']} embed-error")
        result["errors"] = 1
        return result

    knn_result = knn_score(emb, ref_pos, ref_neg, args.k)
    score = knn_result["score"]

    if score >= 0.5:  # majority spawn vote
        info = {
            "region": point["region"],
            "lat": point["lat"],
            "lon": point["lon"],
            "date": scene_info["date"],
            "scene_id": scene_info["scene_id"],
            "cloud": scene_info["cloud"],
            "score": round(score, 4),
            "votes": knn_result["votes"],
            "k": args.k,
        }
        fname = save_candidate(output_dir, thumb_bytes, info, score)
        with MANIFEST_LOCK:
            update_manifest(output_dir, {**info, "thumbnail_path": fname})

        with PRINT_LOCK:
            print(f"  [{idx+1}/{total}] CANDIDATE {point['region']} score={score:.3f} ({knn_result['votes']}/{args.k} votes)")
        result["candidates"] = 1
    else:
        result["candidates"] = 0

    with PRINT_LOCK:
        STATS["processed"] += 1
        if (STATS["processed"] % 100) == 0:
            elapsed = time.time() - STATS.get("_t0", time.time())
            rate = STATS["processed"] / max(elapsed, 1)
            eta = (total - STATS["processed"]) / max(rate, 0.01) / 60
            print(f"  [{STATS['processed']}/{total}] PROGRESS: {STATS['candidates']} cand, {rate:.0f}/min, ETA {eta:.0f}min")

    return result


# ---------------------------------------------------------------------------
# Review HTML
# ---------------------------------------------------------------------------

def build_review_html(output_dir: Path) -> None:
    """Generate a review page for candidates."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text())
    if not manifest:
        return

    # Sort by score descending
    manifest.sort(key=lambda e: e["score"], reverse=True)

    regions = {}
    for e in manifest:
        regions.setdefault(e["region"], 0)
        regions[e["region"]] += 1

    cards = []
    for e in manifest:
        fname = e["thumbnail_path"]
        cards.append(f"""
        <article class="card">
          <img src="{fname}" alt="candidate" loading="lazy">
          <div class="meta"><strong>{e['region']}</strong> · {e['date']}</div>
          <div class="meta">score {e['score']:.3f} · votes {e.get('votes','?')}/{e.get('k','?')} · cloud {e['cloud']:.1f}%</div>
          <div class="meta">({e['lat']:.4f}, {e['lon']:.4f})</div>
        </article>""")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GeoRSCLIP BC Coast Candidates</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #f5f6fa; color: #1f2937; }}
    header {{ background: linear-gradient(135deg, #1e40af, #1e3a8a); color: white; padding: 24px; }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .stat {{ background: white; border-radius: 12px; padding: 14px 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .value {{ font-size: 28px; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 14px; }}
    .card {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .card img {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; }}
    .meta {{ padding: 0 14px 8px; font-size: 13px; color: #374151; }}
    table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-top: 12px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
    th {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <header>
    <h1>GeoRSCLIP BC Coast Candidates</h1>
    <p>KNN majority-vote candidates from GeoRSCLIP ViT-B-32 scan.</p>
  </header>
  <main>
    <section class="stats">
      <div class="stat"><div class="label">Candidates</div><div class="value">{len(manifest)}</div></div>
      <div class="stat"><div class="label">Regions</div><div class="value">{len(regions)}</div></div>
    </section>
    <h2>By Region</h2>
    <table><thead><tr><th>Region</th><th>Candidates</th></tr></thead><tbody>
      {"".join(f'<tr><td>{r}</td><td>{c}</td></tr>' for r, c in sorted(regions.items(), key=lambda x: -x[1]))}
    </tbody></table>
    <h2>Candidates (sorted by score)</h2>
    <section class="grid">{''.join(cards)}</section>
  </main>
</body>
</html>"""

    (output_dir / "review.html").write_text(html)
    print(f"  Review page: {output_dir / 'review.html'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GeoRSCLIP BC coast scan for herring spawn candidates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory on Z Slim")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="End date YYYY-MM-DD")
    parser.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD, help="Max cloud %")
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING, help="Grid spacing in degrees")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent threads")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="KNN neighbors")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: Load model & reference embeddings ----
    print("=== GeoRSCLIP BC Coast Scan ===\n")
    print("Phase 1: Loading model and reference data...")
    _init_model()
    ref_pos, ref_neg = load_reference_embeddings(repo_root)

    print(f"\n  Output: {output_dir}")
    print(f"  Grid spacing: {args.grid_spacing}°")
    print(f"  Date range: {args.start} to {args.end}")
    print(f"  Max cloud: {args.max_cloud}%")
    print(f"  Workers: {args.workers}")
    print(f"  K: {args.k}")

    # ---- Phase 2: Generate grid points ----
    print("\nPhase 2: Generating grid points...")
    points = generate_grid_points(REGIONS, args.grid_spacing)
    print(f"  Total points: {len(points)}")

    # ---- Phase 3: Import GEE ----
    print("\nPhase 3: Initializing Earth Engine...")
    import ee
    ee.Initialize(project="redd-fish")
    print("  Authenticated")

    # ---- Phase 4: Scan ----
    print(f"\nPhase 4: Scanning {len(points)} points...")
    STATS["_t0"] = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_point, point, args, output_dir, ref_pos, ref_neg, i, len(points), ee
            ): i
            for i, point in enumerate(points)
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"  Worker error: {exc}")

    elapsed = time.time() - STATS["_t0"]

    # ---- Phase 5: Report ----
    candidates = json.loads((output_dir / "manifest.json").read_text()) if (output_dir / "manifest.json").exists() else []
    print(f"\nPhase 5: Complete in {elapsed/60:.1f} min")
    print(f"  Points: {len(points)}")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Rate: {len(points)/max(elapsed,1)*60:.0f} points/min")

    # ---- Phase 6: Review page ----
    print("\nPhase 6: Building review page...")
    build_review_html(output_dir)

    print(f"\nDONE. Candidates in {output_dir}")
    print(f"  View: {output_dir / 'review.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
