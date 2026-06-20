#!/usr/bin/env python3
"""Few-shot SubspaceAD — score candidates with PCA trained on positives+negatives.

Trains PCA on DINOv2 patch tokens from both positive (spawn) and negative
(non-spawn) images (134 total). Then scores each candidate by reconstruction
residual. Low residual = looks like spawn or normal coastal water (in-distribution).
High residual = true anomaly (neither spawn nor normal — worth investigating).

Generates heatmaps + segmentation overlays, a static review HTML page, and
a JSON manifest with per-candidate metrics.

Usage:
    python scripts/run_fewshot_subspace_review.py

Output: data/fewshot_subspace_ad_review/

Serve the review page:
    python -m http.server 8773 --directory data/fewshot_subspace_ad_review
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.label_subspace_app import (  # noqa: E402
    ANOMALOUS_PATCH_FRAC,
    DEFAULT_N_COMPONENTS,
    N_PATCHES,
    PATCH_GRID_SIZE,
    _resolve_device,
    extract_patch_embeddings_single,
    load_dinov2,
    make_heatmap_overlay,
    make_segmentation_overlay,
    load_labels,
    save_label,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NEGATIVE_DIR = PROJECT_ROOT / "data" / "samples" / "negative"
DEFAULT_POSITIVE_DIR = PROJECT_ROOT / "data" / "samples" / "positive"
DEFAULT_POSITIVE_MANIFEST = PROJECT_ROOT / "data" / "samples" / "training_manifest.json"
DEFAULT_CANDIDATE_DIRS = [
    PROJECT_ROOT / "data" / "candidates_knn",
    PROJECT_ROOT / "data" / "candidates_knn_expanded",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "fewshot_subspace_ad_review"
DEFAULT_N_COMPONENTS = 64
DEFAULT_SAMPLE_FRAC = 0.15

# Default PCA prediction threshold for few-shot (will be auto-computed)
DEFAULT_PREDICTION_THRESHOLD = 0.00015


# ---------------------------------------------------------------------------
# PCA Training on Combined Positives + Negatives
# ---------------------------------------------------------------------------

def extract_patch_embeddings_from_list(
    image_paths: list[Path], device: str = "auto",
) -> np.ndarray:
    """Extract DINOv2 patch tokens for a list of image paths.

    Returns:
        np.ndarray of shape (N*256, 384) with all patch tokens.
    """
    resolved = _resolve_device(device)
    model = load_dinov2(resolved)
    all_patches: list[np.ndarray] = []

    for p in tqdm(image_paths, desc="Extracting patches", unit="img", leave=False):
        try:
            img = Image.open(p).convert("RGB")
            tensor = DINO_TRANSFORM(img).unsqueeze(0).to(resolved)
            with torch.no_grad():
                patch_tokens, _ = model.get_intermediate_layers(
                    tensor, n=1, reshape=True, return_class_token=True,
                )[0]
            pt = (patch_tokens.flatten(2).transpose(1, 2).squeeze(0)
                  .cpu().numpy().astype(np.float32))
            all_patches.append(pt)
        except Exception as exc:
            print(f"  WARNING: Failed to embed {p.name}: {exc}")

    if not all_patches:
        return np.array([])

    return np.vstack(all_patches).astype(np.float32)


def train_pca_on_combined(
    negative_dir: Path,
    positive_paths: list[Path],
    n_components: int | None = None,
    device: str = "auto",
    sample_frac: float = DEFAULT_SAMPLE_FRAC,
) -> dict:
    """Train PCA on DINOv2 patch embeddings from both negatives and positives.

    Args:
        negative_dir: Directory of negative (non-spawn) images.
        positive_paths: List of paths to positive (spawn) images.
        n_components: Number of PCA components. None = auto-select.
        device: Device for DINOv2.
        sample_frac: Fraction of patches to sample per image (0.15 = 15%).

    Returns:
        dict with 'pca_model' (fitted PCA) and metadata.
    """
    resolved = _resolve_device(device)
    print("=" * 60)
    print("  Few-shot SubspaceAD — Training PCA on Combined Set")
    print("=" * 60)
    print(f"  Negative dir: {negative_dir}")
    print(f"  Positive images: {len(positive_paths)}")
    print(f"  Device: {resolved}")
    print(f"  Patch sample fraction: {sample_frac}")

    # ---- Collect all image paths ----
    neg_paths = sorted(negative_dir.glob("*.png"))
    if not neg_paths:
        print("  ERROR: No negative PNG images found")
        return {"error": "No negative images"}

    all_image_paths = list(neg_paths) + positive_paths
    n_neg = len(neg_paths)
    n_pos = len(positive_paths)
    print(f"  Training images: {len(all_image_paths)} ({n_neg} neg + {n_pos} pos)")

    # ---- Extract all patch tokens ----
    all_patches = extract_patch_embeddings_from_list(all_image_paths, device=resolved)
    if len(all_patches) == 0:
        return {"error": "No patch embeddings extracted"}

    n_total_patches = len(all_patches)
    n_images = len(all_image_paths)
    print(f"  Extracted {n_total_patches} patch tokens from {n_images} images")

    # ---- Sample patches per image ----
    if sample_frac < 1.0:
        sampled: list[np.ndarray] = []
        rng = np.random.RandomState(42)
        for img_idx in range(n_images):
            start = img_idx * N_PATCHES
            end = start + N_PATCHES
            img_patches = all_patches[start:end]
            n_sample = max(1, int(N_PATCHES * sample_frac))
            indices = rng.choice(N_PATCHES, size=n_sample, replace=False)
            sampled.append(img_patches[indices])
        training_patches = np.vstack(sampled).astype(np.float32)
        print(f"  Sampled {len(training_patches)} patch tokens ({sample_frac:.0%} per image)")
    else:
        training_patches = all_patches

    # ---- Auto-select n_components ----
    n_patches, n_features = training_patches.shape
    if n_components is None:
        max_comp = min(n_patches - 1, n_features)
        n_components = min(DEFAULT_N_COMPONENTS, n_patches // 2, max_comp)
        n_components = max(1, n_components)

    max_components = min(n_patches - 1, n_features)
    if n_components > max_components:
        n_components = max(1, max_components)

    print(f"  Fitting PCA with {n_components} components on {n_patches} patches...")
    pca = PCA(n_components=n_components, whiten=False, random_state=42)
    pca.fit(training_patches)

    var_ratio = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance: {var_ratio:.4f} ({n_components} components)")
    print(f"  Training complete: {n_images} images ({n_neg}+{n_pos}), {n_patches} patches")

    return {
        "pca_model": pca,
        "n_components": n_components,
        "n_images_used": n_images,
        "n_neg_used": n_neg,
        "n_pos_used": n_pos,
        "n_patches_trained": n_patches,
        "explained_variance_ratio": var_ratio,
        "sample_frac": sample_frac,
    }


# ---------------------------------------------------------------------------
# Scoring + Overlay Generation
# ---------------------------------------------------------------------------

def score_and_segment(
    patches: np.ndarray, pca: PCA,
) -> dict:
    """Compute patch residuals and segmentation from DINOv2 patch tokens.

    Args:
        patches: (256, 384) numpy array of DINOv2 patch tokens.
        pca: Fitted PCA model.

    Returns:
        dict with scores, heatmap (224x224), mask (224x224), etc.
    """
    # Project through PCA and reconstruct
    projected = pca.transform(patches)
    reconstructed = pca.inverse_transform(projected)

    # Per-patch MSE residuals
    residuals = np.mean((patches - reconstructed) ** 2, axis=1)

    # Aggregate scores
    sorted_r = np.sort(residuals)[::-1]
    n_anom = max(1, int(N_PATCHES * ANOMALOUS_PATCH_FRAC))
    score_top10p = float(np.mean(sorted_r[:n_anom]))
    score_mean = float(np.mean(residuals))
    score_max = float(np.max(residuals))

    # Reshape to 16x16 heatmap
    patch_grid = residuals.reshape(PATCH_GRID_SIZE, PATCH_GRID_SIZE)

    # Upsample to 224x224
    heatmap_t = torch.from_numpy(patch_grid).float().unsqueeze(0).unsqueeze(0)
    heatmap_up = F.interpolate(
        heatmap_t, size=(224, 224), mode="bilinear", align_corners=False,
    )
    heatmap = heatmap_up.squeeze().cpu().numpy().astype(np.float32)

    # Auto-threshold: mean + 2*std
    auto_threshold = float(np.mean(residuals) + 2.0 * np.std(residuals))

    # Binary mask
    mask = (heatmap > auto_threshold).astype(np.float32)
    spawn_area_frac = float(np.mean(mask))
    n_spawn_patches = int((patch_grid > auto_threshold).sum())

    return {
        "score_top10p": score_top10p,
        "score_mean": score_mean,
        "score_max": score_max,
        "auto_threshold": auto_threshold,
        "spawn_area_frac": spawn_area_frac,
        "n_spawn_patches": n_spawn_patches,
        "n_patches": len(residuals),
        "heatmap": heatmap,              # 224x224 float32
        "mask": mask,                    # 224x224 float32
        "patch_residuals": patch_grid,   # 16x16
    }


def score_single_candidate(
    image_path: Path, pca: PCA, device: str,
    output_dir: Path,
) -> dict | None:
    """Score a single candidate image and save overlays.

    Args:
        image_path: Path to candidate PNG.
        pca: Fitted PCA model.
        device: Device string.
        output_dir: Where to save overlays.

    Returns:
        dict with scores and paths, or None on failure.
    """
    try:
        patches = extract_patch_embeddings_single(str(image_path), device=device)
    except Exception as exc:
        print(f"  WARNING: Failed to extract patches for {image_path.name}: {exc}")
        return None

    result = score_and_segment(patches, pca)

    # Create overlay directory
    overlay_dir = output_dir / "overlays" / image_path.stem
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Save original (resized to 224x224)
    original_img = Image.open(image_path).convert("RGB").resize((224, 224))
    original_img.save(overlay_dir / "original.png")

    # Save heatmap overlay
    heatmap_img = make_heatmap_overlay(str(image_path), result["heatmap"])
    heatmap_img.save(overlay_dir / "heatmap.png")

    # Save segmentation overlay
    seg_img = make_segmentation_overlay(str(image_path), result["mask"])
    seg_img.save(overlay_dir / "segmentation.png")

    return {
        "score_top10p": result["score_top10p"],
        "score_mean": result["score_mean"],
        "score_max": result["score_max"],
        "auto_threshold": result["auto_threshold"],
        "spawn_area_frac": result["spawn_area_frac"],
        "n_spawn_patches": result["n_spawn_patches"],
        "n_patches": result["n_patches"],
        "original_path": f"overlays/{image_path.stem}/original.png",
        "heatmap_path": f"overlays/{image_path.stem}/heatmap.png",
        "segmentation_path": f"overlays/{image_path.stem}/segmentation.png",
    }


# ---------------------------------------------------------------------------
# Manifest and Review Page
# ---------------------------------------------------------------------------

def _get_scores(item: dict) -> dict:
    """Extract scores dict from either flat format or nested 'scores' format."""
    if "score_top10p" in item:
        return item  # flat format
    return item.get("scores", item)


def _compute_score_buckets(scores: list[float], n_buckets: int = 10) -> dict[str, int]:
    """Compute histogram buckets for score distribution."""
    if not scores:
        return {}
    min_s, max_s = min(scores), max(scores)
    if max_s - min_s < 1e-12:
        return {f"{min_s:.3f}": len(scores)}
    bucket_size = (max_s - min_s) / n_buckets
    buckets: dict[str, int] = {}
    for i in range(n_buckets):
        lo = min_s + i * bucket_size
        hi = lo + bucket_size
        label = f"{lo:.3f}-{hi:.3f}"
        count = sum(1 for s in scores if lo <= s < hi)
        if count > 0:
            buckets[label] = count
    # Include max bucket
    last_label = f"{min_s + (n_buckets-1) * bucket_size:.3f}-{max_s:.3f}"
    last_count = sum(1 for s in scores if s >= min_s + (n_buckets-1) * bucket_size)
    buckets[last_label] = last_count
    return dict(sorted(buckets.items()))


def build_review_html(
    candidates: list[dict],
    train_meta: dict,
    elapsed: float,
) -> str:
    """Generate a static review HTML page with all scored candidates."""
    # Sort by score descending
    sorted_candidates = sorted(
        candidates,
        key=lambda c: _get_scores(c).get("score_top10p", 0),
        reverse=True,
    )

    # Normalize training metadata keys (support both formats)
    tm = dict(train_meta)
    key_map = {"n_positive_images": "n_pos_used", "n_negative_images": "n_neg_used",
               "n_total_images": "n_images_used"}
    for old_k, new_k in key_map.items():
        if old_k in tm and new_k not in tm:
            tm[new_k] = tm[old_k]
    train_meta = tm

    # Stats
    n_total = len(sorted_candidates)
    scores_top10p = [_get_scores(c).get("score_top10p", 0) for c in sorted_candidates]
    scores_mean = [_get_scores(c).get("score_mean", 0) for c in sorted_candidates]
    areas = [_get_scores(c).get("spawn_area_frac", 0) for c in sorted_candidates]

    mean_score = float(np.mean(scores_top10p)) if scores_top10p else 0
    max_score = float(np.max(scores_top10p)) if scores_top10p else 0
    min_score = float(np.min(scores_top10p)) if scores_top10p else 0

    # Region counts
    region_counts: dict[str, int] = {}
    for c in sorted_candidates:
        fname = c.get("filename", "")
        region = fname.split("_")[0] if "_" in fname else "unknown"
        region_counts[region] = region_counts.get(region, 0) + 1

    region_rows = "".join(
        f"<tr><td>{_html_escape(region)}</td><td>{count}</td></tr>"
        for region, count in sorted(region_counts.items(), key=lambda x: (-x[1], x[0]))
    )

    # Score distribution
    score_buckets = _compute_score_buckets(scores_top10p)
    bucket_rows = "".join(
        f"<tr><td>{_html_escape(label)}</td><td>{count}</td></tr>"
        for label, count in score_buckets.items()
    )

    # Candidate cards
    cards: list[str] = []
    for c in sorted_candidates:
        fname = c["filename"]
        scores = _get_scores(c)
        overlay_dir = f"overlays/{Path(fname).stem}"

        # Extract metadata from filename
        parts = fname.replace(".png", "").split("_")
        region = parts[0] if len(parts) > 0 else "unknown"
        date_str = parts[1] if len(parts) > 1 else "unknown"

        # Extract lat/lon from filename pattern
        lat_lon = "unknown"
        for part in parts:
            if part.startswith(("49.", "48.", "50.", "51.", "52.", "53.", "54.")):
                lat_lon = part
                break

        cards.append(f"""
            <article class="card" data-region="{_html_escape(region)}" data-score="{scores['score_top10p']:.4f}">
              <div class="card-image" data-fname="{_html_escape(fname)}">
                <img src="{overlay_dir}/original.png" alt="{_html_escape(fname)}" loading="lazy"
                     class="card-img" id="img-{hash(fname) & 0xFFFFFFFF:x}">
                <div class="overlay-buttons">
                  <button class="ov-btn ov-orig active" onclick="showOverlay('{hash(fname) & 0xFFFFFFFF:x}', 'orig')">O</button>
                  <button class="ov-btn ov-heat" onclick="showOverlay('{hash(fname) & 0xFFFFFFFF:x}', 'heat')">H</button>
                  <button class="ov-btn ov-seg" onclick="showOverlay('{hash(fname) & 0xFFFFFFFF:x}', 'seg')">S</button>
                </div>
              </div>
              <div class="card-body">
                <div class="meta"><strong>{_html_escape(region)}</strong> · {_html_escape(date_str)}</div>
                <div class="meta">Score (top10p): <span class="score-val">{scores['score_top10p']:.4f}</span></div>
                <div class="meta">Mean: {scores['score_mean']:.4f} · Max: {scores['score_max']:.4f}</div>
                <div class="meta">Spawn area: {scores['spawn_area_frac']:.4f} ({scores['n_spawn_patches']}/{scores['n_patches']} patches)</div>
                <div class="meta">Lat/Lon: {_html_escape(lat_lon)} · Auto-thr: {scores['auto_threshold']:.4f}</div>
                <div class="meta fname">{_html_escape(fname)}</div>
              </div>
            </article>""")

    # Build the HTML page
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Few-shot SubspaceAD Review</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
           background: #f0f2f5; color: #1f2937; }}
    header {{ background: linear-gradient(135deg, #0f172a, #1e3a5f); color: white; padding: 24px 32px; }}
    header h1 {{ font-size: 24px; margin-bottom: 4px; }}
    header p {{ font-size: 14px; opacity: 0.85; }}
    main {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .stat {{ background: white; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    .stat .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #6b7280; }}
    .stat .value {{ font-size: 22px; font-weight: 700; margin-top: 2px; }}
    .stat .sub {{ font-size: 12px; color: #6b7280; }}
    .filters {{ margin-bottom: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    .filters label {{ font-size: 13px; font-weight: 500; }}
    .filters select, .filters input {{ padding: 6px 10px; border: 1px solid #d1d5db; border-radius: 6px;
                                       font-size: 13px; background: white; }}
    .filters input {{ width: 100px; }}
    .filters .count {{ font-size: 13px; color: #6b7280; margin-left: auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }}
    .card {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08);
             transition: box-shadow .15s; }}
    .card:hover {{ box-shadow: 0 3px 12px rgba(0,0,0,.12); }}
    .card-image {{ position: relative; width: 100%; aspect-ratio: 1; overflow: hidden; background: #e5e7eb; }}
    .card-img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .overlay-buttons {{ position: absolute; top: 6px; right: 6px; display: flex; gap: 3px; }}
    .ov-btn {{ width: 26px; height: 26px; border: none; border-radius: 4px; font-size: 11px; font-weight: 700;
              cursor: pointer; opacity: 0.7; transition: opacity .15s; }}
    .ov-btn:hover {{ opacity: 1; }}
    .ov-orig {{ background: #374151; color: white; }}
    .ov-heat {{ background: #ef4444; color: white; }}
    .ov-seg {{ background: #22c55e; color: white; }}
    .ov-btn.active {{ opacity: 1; box-shadow: 0 0 0 2px white; }}
    .card-body {{ padding: 10px 12px 12px; }}
    .meta {{ font-size: 12px; color: #4b5563; margin-bottom: 2px; line-height: 1.5; }}
    .meta.fname {{ font-size: 10px; color: #9ca3af; word-break: break-all; margin-top: 4px; }}
    .score-val {{ font-weight: 600; color: #059669; }}
    /* lower score = more like spawn/normal = green, higher = anomaly = red */
    .score-val.high {{ color: #dc2626; }}
    .score-val.mid {{ color: #d97706; }}
    h2 {{ font-size: 18px; margin: 24px 0 12px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 20px; }}
    th, td {{ padding: 8px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 12px; }}
    th {{ background: #f9fafb; font-weight: 600; }}
    .training-info {{ background: white; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,.08);
                     margin-bottom: 20px; font-size: 13px; line-height: 1.6; }}
    .training-info strong {{ color: #111827; }}
  </style>
</head>
<body>
  <header>
    <h1>Few-shot SubspaceAD Review</h1>
    <p>PCA trained on <strong>{train_meta['n_pos_used']}</strong> positives + <strong>{train_meta['n_neg_used']}</strong> negatives
       · {train_meta['n_components']} components · {train_meta['sample_frac']:.0%} patch sampling</p>
  </header>
  <main>
    <section class="stats">
      <div class="stat"><div class="label">Candidates</div><div class="value">{n_total}</div></div>
      <div class="stat"><div class="label">Mean Score</div><div class="value">{mean_score:.4f}</div></div>
      <div class="stat"><div class="label">Max Score</div><div class="value">{max_score:.4f}</div></div>
      <div class="stat"><div class="label">Min Score</div><div class="value">{min_score:.4f}</div></div>
      <div class="stat"><div class="label">Runtime</div><div class="value">{elapsed:.1f}s</div></div>
      <div class="stat"><div class="label">PCA Var Ratio</div><div class="value">{train_meta.get('explained_variance_ratio', 0):.4f}</div></div>
    </section>

    <div class="training-info">
      <strong>Training:</strong> {train_meta['n_pos_used']} positives + {train_meta['n_neg_used']} negatives
      = {train_meta['n_images_used']} images
      · {train_meta['n_patches_trained']} patch tokens ({train_meta['sample_frac']:.0%} sampling)
      · {train_meta['n_components']} PCA components
      · Var ratio: {train_meta.get('explained_variance_ratio', 0):.4f}
      · <strong>Low score</strong> = in-distribution (spawn or normal) · <strong>High score</strong> = anomaly
    </div>

    <div class="filters">
      <label for="regionFilter">Region:</label>
      <select id="regionFilter" onchange="applyFilters()">
        <option value="all">All regions</option>
        {"".join(f'<option value="{_html_escape(r)}">{_html_escape(r)} ({c})</option>'
                for r, c in sorted(region_counts.items(), key=lambda x: (-x[1], x[0])))}
      </select>
      <label for="minScore">Min score:</label>
      <input type="number" id="minScore" step="0.1" min="0" placeholder="0.0" oninput="applyFilters()">
      <label for="sortBy">Sort:</label>
      <select id="sortBy" onchange="applyFilters()">
        <option value="score-desc">Score ↓</option>
        <option value="score-asc">Score ↑</option>
        <option value="spawn-desc">Spawn area ↓</option>
        <option value="region">Region</option>
      </select>
      <span class="count" id="visibleCount">Showing {n_total}/{n_total}</span>
    </div>

    <div class="grid" id="candidateGrid">
      {''.join(cards)}
    </div>

    <h2>Region Distribution</h2>
    <table><thead><tr><th>Region</th><th>Candidates</th></tr></thead><tbody>{region_rows}</tbody></table>

    <h2>Score Distribution</h2>
    <table><thead><tr><th>Score Range</th><th>Count</th></tr></thead><tbody>{bucket_rows}</tbody></table>
  </main>

  <script>
    // Overlay toggle
    function showOverlay(id, type) {{
      const img = document.getElementById('img-' + id);
      if (!img) return;
      const fname = img.closest('.card-image').dataset.fname;
      const stem = fname.replace(/\\.png$/, '');
      const base = 'overlays/' + stem + '/';
      const urls = {{ orig: base + 'original.png', heat: base + 'heatmap.png', seg: base + 'segmentation.png' }};
      img.src = urls[type] || urls.orig;
      // Update active button
      const btns = img.closest('.card-image').querySelectorAll('.ov-btn');
      btns.forEach(b => b.classList.remove('active'));
      const idx = {{ orig: 0, heat: 1, seg: 2 }};
      if (btns[idx[type]]) btns[idx[type]].classList.add('active');
    }}

    // Filtering and sorting
    function applyFilters() {{
      const regionFilter = document.getElementById('regionFilter').value;
      const minScore = parseFloat(document.getElementById('minScore').value) || 0;
      const sortBy = document.getElementById('sortBy').value;
      const grid = document.getElementById('candidateGrid');
      const cards = Array.from(grid.querySelectorAll('.card'));

      // Filter
      cards.forEach(card => {{
        const region = card.dataset.region;
        const score = parseFloat(card.dataset.score);
        const regionMatch = regionFilter === 'all' || region === regionFilter;
        const scoreMatch = score >= minScore;
        card.style.display = (regionMatch && scoreMatch) ? '' : 'none';
      }});

      // Sort
      const visible = cards.filter(c => c.style.display !== 'none');
      visible.sort((a, b) => {{
        if (sortBy === 'score-desc') return parseFloat(b.dataset.score) - parseFloat(a.dataset.score);
        if (sortBy === 'score-asc') return parseFloat(a.dataset.score) - parseFloat(b.dataset.score);
        if (sortBy === 'spawn-desc') {{
          // Need to parse spawn area from card body
          const aArea = parseFloat(a.querySelector('.meta:nth-child(4)')?.textContent?.match(/[\\d.]+/)?.[0] || '0');
          const bArea = parseFloat(b.querySelector('.meta:nth-child(4)')?.textContent?.match(/[\\d.]+/)?.[0] || '0');
          return bArea - aArea;
        }}
        if (sortBy === 'region') return a.dataset.region.localeCompare(b.dataset.region);
        return 0;
      }});

      // Re-append in sorted order
      visible.forEach(card => grid.appendChild(card));

      // Update count
      const total = cards.length;
      document.getElementById('visibleCount').textContent = 'Showing ' + visible.length + '/' + total;
    }}
  </script>
</body>
</html>"""

    return html


def _html_escape(value: Any) -> str:
    import html
    return html.escape(str(value))


# ---------------------------------------------------------------------------
# CSV Export for comparison
# ---------------------------------------------------------------------------

def export_comparison_csv(
    candidates: list[dict],
    output_path: Path,
) -> None:
    """Export a CSV with candidate scores for easy analysis."""
    import csv

    rows = []
    for c in sorted(
        candidates,
        key=lambda x: x.get("score_top10p", x.get("scores", {}).get("score_top10p", 0)),
        reverse=True,
    ):
        scores = c if "score_top10p" in c else c.get("scores", c)
        # Parse region from filename
        fname = c["filename"]
        region = fname.split("_")[0] if "_" in fname else "unknown"
        rows.append({
            "filename": fname,
            "region": region,
            "score_top10p": f"{scores['score_top10p']:.6f}",
            "score_mean": f"{scores['score_mean']:.6f}",
            "score_max": f"{scores['score_max']:.6f}",
            "spawn_area_frac": f"{scores['spawn_area_frac']:.6f}",
            "n_spawn_patches": scores["n_spawn_patches"],
            "auto_threshold": f"{scores['auto_threshold']:.6f}",
        })

    fieldnames = ["filename", "region", "score_top10p", "score_mean", "score_max",
                  "spawn_area_frac", "n_spawn_patches", "auto_threshold"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV exported: {output_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# DINOv2 transform (reuse from train_classifier)
# ---------------------------------------------------------------------------

from scripts.train_classifier import DINO_TRANSFORM  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Few-shot SubspaceAD — score candidates with PCA on positives+negatives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--negative-dir", type=Path, default=DEFAULT_NEGATIVE_DIR,
        help="Directory of negative (non-spawn) images for PCA training",
    )
    parser.add_argument(
        "--positive-dir", type=Path, default=DEFAULT_POSITIVE_DIR,
        help="Directory of positive (spawn) images for PCA training",
    )
    parser.add_argument(
        "--positive-manifest", type=Path, default=None,
        help="Path to JSON manifest listing which filenames in --positive-dir to use "
             "(default: use data/labeled_candidates/manifest.json 'accept' entries)",
    )
    parser.add_argument(
        "--candidate-dir", type=Path, action="append", default=None,
        help="Directories containing candidate PNGs (can specify multiple)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for scored results (default: data/fewshot_subspace_ad_review)",
    )
    parser.add_argument(
        "--n-components", type=int, default=DEFAULT_N_COMPONENTS,
        help=f"Number of PCA components (default: {DEFAULT_N_COMPONENTS})",
    )
    parser.add_argument(
        "--sample-frac", type=float, default=DEFAULT_SAMPLE_FRAC,
        help="Fraction of patches sampled per image for PCA (default: 0.15)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for DINOv2 inference (default: auto)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for candidate scoring (default: 1, "
             "increase only if using GPU with batch support)",
    )
    parser.add_argument(
        "--export-csv", action="store_true",
        help="Export a CSV of all scores alongside the manifest",
    )
    args = parser.parse_args(argv)

    t_start = time.time()
    resolved = _resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overlays").mkdir(parents=True, exist_ok=True)

    # ---- Resolve candidate directories ----
    candidate_dirs = args.candidate_dir or DEFAULT_CANDIDATE_DIRS

    # ---- Determine which positive files to use ----
    pos_dir = args.positive_dir
    if not pos_dir.is_dir():
        print(f"ERROR: Positive directory not found: {pos_dir}")
        return 1

    # Try to use manifest to filter positives
    if args.positive_manifest is not None:
        manifest_path = args.positive_manifest
    else:
        manifest_path = PROJECT_ROOT / "data" / "labeled_candidates" / "manifest.json"

    positive_paths: list[Path] = []
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text())
            # Handle both list-of-dicts (labeled manifest) and
            # dict-with-'positives' (training manifest) formats
            if isinstance(manifest_data, list):
                # labeled manifest: list of {filename, label, ...}
                positive_fnames = {
                    e["filename"] for e in manifest_data
                    if e.get("label") == "accept"
                }
            elif isinstance(manifest_data, dict):
                # training manifest: {"positives": ["file1.png", ...]}
                positive_fnames = set(manifest_data.get("positives", []))
            else:
                positive_fnames = set()

            for fname in sorted(positive_fnames):
                fp = pos_dir / fname
                if fp.exists():
                    positive_paths.append(fp)
                else:
                    print(f"  WARNING: Positive file listed but not found: {fp}")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  WARNING: Could not parse manifest {manifest_path}: {exc}")

    # Fallback: use all files in positive dir
    if not positive_paths:
        positive_paths = sorted(pos_dir.glob("*.png"))
        print(f"  Using all {len(positive_paths)} files from {pos_dir} (no manifest filter)")
    else:
        print(f"  Using {len(positive_paths)} positive files from manifest ({manifest_path.name})")

    # ---- Validate directories ----
    neg_dir = args.negative_dir
    if not neg_dir.is_dir():
        print(f"ERROR: Negative directory not found: {neg_dir}")
        return 1

    valid_candidate_dirs = [d for d in candidate_dirs if d.is_dir()]
    if not valid_candidate_dirs:
        print("ERROR: No valid candidate directories found")
        print(f"  Checked: {candidate_dirs}")
        return 1

    # ---- Step 1: Train PCA on combined positives + negatives ----
    print()
    train_result = train_pca_on_combined(
        neg_dir, positive_paths,
        n_components=args.n_components,
        device=resolved,
        sample_frac=args.sample_frac,
    )
    if "error" in train_result:
        print(f"\nERROR: PCA training failed: {train_result['error']}")
        return 1

    pca = train_result["pca_model"]

    # ---- Step 2: Find all candidate PNGs ----
    candidate_pngs: list[Path] = []
    for d in valid_candidate_dirs:
        pngs = sorted(d.glob("*.png"))
        candidate_pngs.extend(pngs)
        print(f"\n  Found {len(pngs)} candidates in {d.name}")

    # Deduplicate by filename (in case a file appears in both dirs)
    seen_fnames: set[str] = set()
    unique_pngs: list[Path] = []
    for p in candidate_pngs:
        if p.name not in seen_fnames:
            seen_fnames.add(p.name)
            unique_pngs.append(p)
    candidate_pngs = unique_pngs
    n_total = len(candidate_pngs)
    print(f"  Total unique candidates: {n_total}")

    # ---- Step 3: Load existing manifest (for checkpointing) ----
    manifest_path = output_dir / "manifest.json"
    existing_results: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing_data = json.loads(manifest_path.read_text())
            existing_candidates = existing_data.get("candidates", [])
            for ec in existing_candidates:
                existing_results[ec["filename"]] = ec
            print(f"  Loaded {len(existing_results)} previously scored results")
        except (json.JSONDecodeError, Exception):
            pass

    # ---- Step 4: Score all candidates ----
    to_process = [p for p in candidate_pngs if p.name not in existing_results]
    already_done = len(candidate_pngs) - len(to_process)
    print(f"\n  Already scored: {already_done}, remaining: {len(to_process)}")

    if to_process:
        print(f"\n  {'=' * 60}")
        print(f"  Scoring {len(to_process)} candidates with few-shot SubspaceAD")
        print(f"  {'=' * 60}")
        print(f"  PCA components: {train_result['n_components']}")
        print(f"  Training set: {train_result['n_images_used']} images "
              f"({train_result['n_pos_used']} pos + {train_result['n_neg_used']} neg)")
        print()

        # Preload DINOv2 model
        load_dinov2(resolved)

        if args.workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                fut_to_path = {
                    executor.submit(
                        score_single_candidate, p, pca, resolved, output_dir,
                    ): p for p in to_process
                }
                for future in tqdm(
                    as_completed(fut_to_path), total=len(to_process),
                    desc="Scoring", unit="img",
                ):
                    p = fut_to_path[future]
                    try:
                        result = future.result()
                        if result is not None:
                            existing_results[p.name] = {
                                "filename": p.name,
                                **result,
                            }
                    except Exception as exc:
                        print(f"  ERROR scoring {p.name}: {exc}")
        else:
            # Sequential processing
            for p in tqdm(to_process, desc="Scoring candidates", unit="img"):
                result = score_single_candidate(p, pca, resolved, output_dir)
                if result is not None:
                    existing_results[p.name] = {
                        "filename": p.name,
                        **result,
                    }

    # ---- Step 5: Build final manifest ----
    all_results = list(existing_results.values())
    all_results.sort(key=lambda r: r.get("score_top10p", 0), reverse=True)

    # Compute threshold statistics
    all_top10p = np.array([r.get("score_top10p", 0) for r in all_results])
    all_mean = np.array([r.get("score_mean", 0) for r in all_results])

    manifest = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "dinov2_vits14 + Few-shot SubspaceAD (combined pos+neg PCA)",
        "training": {
            "n_positive_images": train_result["n_pos_used"],
            "n_negative_images": train_result["n_neg_used"],
            "n_total_images": train_result["n_images_used"],
            "n_components": train_result["n_components"],
            "n_patches_trained": train_result["n_patches_trained"],
            "explained_variance_ratio": train_result["explained_variance_ratio"],
            "sample_frac": train_result["sample_frac"],
        },
        "scoring": {
            "anomalous_patch_frac": ANOMALOUS_PATCH_FRAC,
            "n_candidates_total": n_total,
            "n_candidates_scored": len(all_results),
            "score_top10p_mean": float(np.mean(all_top10p)) if len(all_top10p) > 0 else 0,
            "score_top10p_std": float(np.std(all_top10p)) if len(all_top10p) > 0 else 0,
            "score_top10p_max": float(np.max(all_top10p)) if len(all_top10p) > 0 else 0,
            "score_top10p_min": float(np.min(all_top10p)) if len(all_top10p) > 0 else 0,
        },
        "candidate_sources": [str(d) for d in valid_candidate_dirs],
        "elapsed_seconds": time.time() - t_start,
        "candidates": all_results,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n  Manifest saved: {manifest_path}")

    # ---- Step 6: Export CSV if requested ----
    if args.export_csv:
        csv_path = output_dir / "scores.csv"
        export_comparison_csv(all_results, csv_path)

    # ---- Step 7: Generate review page ----
    elapsed_total = time.time() - t_start
    def _remap_training_meta(meta: dict) -> dict:
        """Normalize training metadata keys for the HTML template."""
        return {
            "n_pos_used": meta.get("n_pos_used", meta.get("n_positive_images", 0)),
            "n_neg_used": meta.get("n_neg_used", meta.get("n_negative_images", 0)),
            "n_images_used": meta.get("n_images_used", meta.get("n_total_images", 0)),
            "n_components": meta["n_components"],
            "n_patches_trained": meta["n_patches_trained"],
            "explained_variance_ratio": meta["explained_variance_ratio"],
            "sample_frac": meta.get("sample_frac", 0.15),
        }

    train_meta_for_html = _remap_training_meta(train_result)
    review_html = build_review_html(all_results, train_meta_for_html, elapsed_total)
    review_path = output_dir / "review.html"
    review_path.write_text(review_html, encoding="utf-8")
    print(f"  Review page: {review_path}")

    # ---- Step 8: Summary ----
    elapsed_str = f"{elapsed_total:.1f}s"
    print(f"\n  {'=' * 60}")
    print(f"  Few-shot SubspaceAD Review Complete")
    print(f"  {'=' * 60}")
    print(f"  Candidates scored: {len(all_results)}/{n_total}")
    print(f"  PCA components: {train_result['n_components']}")
    print(f"  Training: {train_result['n_images_used']} images "
          f"({train_result['n_pos_used']} pos + {train_result['n_neg_used']} neg)")
    print(f"  Top 5 by score_top10p:")
    for r in all_results[:5]:
        print(f"    {r['score_top10p']:.4f}  {r['filename']}")
    print(f"  Time: {elapsed_str}")
    print(f"  Output: {output_dir}")
    print(f"  Serve: python -m http.server 8773 --directory {output_dir}")
    print(f"  URL: http://localhost:8773/review.html")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
