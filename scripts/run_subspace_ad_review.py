#!/usr/bin/env python3
"""Batch DINOv2 SubspaceAD review page generator for herring spawn candidates.

Trains PCA subspace on DINOv2 patch embeddings from negative (non-spawn)
images, scores all candidates, generates heatmap + segmentation overlays,
and creates a standalone web review page at data/subspace_ad_review/.

Usage:
    python scripts/run_subspace_ad_review.py                     # default: candidates_knn
    python scripts/run_subspace_ad_review.py --candidate-dir data/candidates_knn
    python scripts/run_subspace_ad_review.py --port 8770

Output:
    data/subspace_ad_review/
        review.html          — standalone web review page
        manifest.json        — per-candidate metrics
        overlays/{filename}/ — heatmap + segmentation PNGs per candidate
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from datetime import UTC, datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.label_subspace_app import (
    ANOMALOUS_PATCH_FRAC,
    DEFAULT_N_COMPONENTS,
    EMBED_DIM,
    MODEL_NAME,
    N_PATCHES,
    PATCH_GRID_SIZE,
    _resolve_device,
    extract_patch_embeddings_single,
    load_dinov2,
    make_heatmap_overlay,
    make_segmentation_overlay,
    score_and_segment,
    train_pca_on_negatives,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = "data/subspace_ad_review"
DEFAULT_NEGATIVE_DIR = "data/samples/negative"
DEFAULT_CANDIDATE_DIR = "data/candidates_knn"
DEFAULT_PORT = 8770


# ---------------------------------------------------------------------------
# Batch scoring and overlay generation
# ---------------------------------------------------------------------------

def batch_process(
    pca_model: PCA,
    candidate_dir: str,
    output_dir: str,
    device: str = "cpu",
) -> list[dict]:
    """Score all candidates, generate overlays, return manifest entries.

    Args:
        pca_model: Fitted PCA model.
        candidate_dir: Directory of candidate PNG images.
        output_dir: Directory to save results (review.html, manifest.json, overlays/).
        device: Device for DINOv2 inference.

    Returns:
        list of dicts sorted by score descending, each with:
        - filename, score_top10p, score_mean, score_max, auto_threshold,
          spawn_area_frac, n_spawn_patches, n_patches,
          heatmap_path, seg_path, original_path (all relative to output_dir)
    """
    resolved = _resolve_device(device)
    load_dinov2(resolved)

    cand_dir = Path(candidate_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    pngs = sorted(cand_dir.glob("*.png"))
    print(f"  Found {len(pngs)} candidate PNG images in {candidate_dir}")

    if not pngs:
        return []

    manifest_entries: list[dict] = []

    for p in tqdm(pngs, desc="Scoring candidates", unit="img"):
        try:
            # Extract patch embeddings
            patches = extract_patch_embeddings_single(str(p), device=resolved)

            # Score and segment
            result = score_and_segment(patches, pca_model)

            # Generate and save overlays
            stem = p.stem
            img_overlay_dir = overlays_dir / stem
            img_overlay_dir.mkdir(exist_ok=True)

            # Original (resized to 224x224 for consistency)
            orig = Image.open(p).convert("RGB").resize((224, 224))
            orig_path = img_overlay_dir / "original.png"
            orig.save(orig_path)

            # Heatmap overlay
            heat_img = make_heatmap_overlay(str(p), result["heatmap"])
            heat_path = img_overlay_dir / "heatmap.png"
            heat_img.save(heat_path)

            # Segmentation overlay
            seg_img = make_segmentation_overlay(str(p), result["mask"])
            seg_path = img_overlay_dir / "segmentation.png"
            seg_img.save(seg_path)

            # Relative paths for manifest
            rel_orig = f"overlays/{stem}/original.png"
            rel_heat = f"overlays/{stem}/heatmap.png"
            rel_seg = f"overlays/{stem}/segmentation.png"

            manifest_entries.append({
                "filename": p.name,
                "score_top10p": result["score_top10p"],
                "score_mean": result["score_mean"],
                "score_max": result["score_max"],
                "auto_threshold": result["auto_threshold"],
                "spawn_area_frac": result["spawn_area_frac"],
                "n_spawn_patches": result["n_spawn_patches"],
                "n_patches": result["n_patches"],
                "heatmap_path": rel_heat,
                "segmentation_path": rel_seg,
                "original_path": rel_orig,
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

        except Exception as exc:
            print(f"  WARNING: Failed to score {p.name}: {exc}")
            manifest_entries.append({
                "filename": p.name,
                "error": str(exc),
                "score_top10p": 0.0,
                "score_mean": 0.0,
                "score_max": 0.0,
                "auto_threshold": 0.0,
                "spawn_area_frac": 0.0,
                "n_spawn_patches": 0,
                "n_patches": N_PATCHES,
            })

    # Sort by score descending
    manifest_entries.sort(key=lambda e: e.get("score_top10p", 0), reverse=True)

    print(f"  Successfully scored {len(manifest_entries)} candidates")
    return manifest_entries


# ---------------------------------------------------------------------------
# Review page generator
# ---------------------------------------------------------------------------

def generate_review_page(
    manifest: list[dict],
    output_dir: str,
    train_meta: dict | None = None,
) -> str:
    """Generate standalone review HTML page.

    Returns:
        Path to the generated HTML file.
    """
    out_dir = Path(output_dir)
    html_path = out_dir / "review.html"

    # Compute stats
    total = len(manifest)
    scored = [e for e in manifest if e.get("score_top10p", 0) > 0 or e.get("score_max", 0) > 0]
    n_positive_like = sum(1 for e in manifest if e.get("spawn_area_frac", 0) > 0.05)
    high_score = max((e.get("score_top10p", 0) for e in manifest), default=0)

    # Training info
    if train_meta:
        train_info = (
            f"PCA: {train_meta.get('n_components', '?')} components &middot; "
            f"Trained on {train_meta.get('n_images_used', '?')} negative images &middot; "
            f"{train_meta.get('n_patches_trained', '?')} patches &middot; "
            f"Var: {train_meta.get('explained_variance_ratio', 0) * 100:.1f}%"
        )
    else:
        train_info = "Training info unavailable"

    # Build candidates JSON for the frontend
    candidates_json = json.dumps(manifest)

    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SubspaceAD Review — Herring Spawn Candidates</title>
<style>
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #0d0d1a; color: #ddd; }
h1, h2, h3, p { margin: 0; }
a { color: #64B5F6; text-decoration: none; }
a:hover { text-decoration: underline; }

/* Header */
.header { background: #1a1a2e; padding: 12px 24px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #2a2a4e; flex-wrap: wrap; }
.header h1 { font-size: 16px; color: #fff; }
.header .subtitle { font-size: 12px; color: #888; }
.header .stats { margin-left: auto; display: flex; gap: 16px; font-size: 12px; flex-wrap: wrap; }
.header .stats span { color: #888; }
.header .stats strong { color: #fff; }

/* Sort / filter controls */
.controls { background: #12121e; padding: 10px 24px; border-bottom: 1px solid #2a2a4e; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; font-size: 13px; }
.controls label { color: #888; }
.controls select, .controls input { background: #0a0a14; border: 1px solid #2a2a4e; color: #ddd; padding: 4px 8px; border-radius: 4px; font-size: 13px; }
.controls .count { color: #888; margin-left: auto; }

/* Grid */
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; padding: 16px 24px; }
.card { background: #12121e; border: 1px solid #2a2a4e; border-radius: 8px; overflow: hidden; transition: border-color 0.2s; }
.card:hover { border-color: #64B5F6; }
.card.highlight { border-color: #4CAF50; box-shadow: 0 0 12px rgba(76,175,80,0.3); }

.card-header { padding: 8px 12px; background: #0a0a14; border-bottom: 1px solid #1a1a2e; display: flex; justify-content: space-between; align-items: center; }
.card-header .filename { font-size: 10px; color: #666; font-family: monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 60%; }
.card-header .batch { font-size: 10px; color: #555; }

.card-images { display: flex; gap: 0; }
.card-images .img-wrap { flex: 1; position: relative; }
.card-images img { width: 100%; height: auto; display: block; aspect-ratio: 1; object-fit: cover; }
.card-images .img-label { position: absolute; bottom: 0; left: 0; right: 0; text-align: center; font-size: 9px; padding: 2px; background: rgba(0,0,0,0.75); color: #888; text-transform: uppercase; letter-spacing: 0.3px; }

.card-body { padding: 8px 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 11px; }
.card-body .metric { display: flex; justify-content: space-between; padding: 2px 4px; background: #0a0a14; border-radius: 3px; }
.card-body .metric .label { color: #666; }
.card-body .metric .value { color: #ddd; font-weight: 600; font-variant-numeric: tabular-nums; }
.card-body .metric .value.good { color: #4CAF50; }
.card-body .metric .value.warn { color: #FFC107; }
.card-body .metric .value.bad { color: #f44336; }
.card-body .metric .value.high { color: #64B5F6; }

.score-bar { height: 4px; background: #0a0a14; border-radius: 2px; margin: 0 12px 8px; overflow: hidden; }
.score-bar .fill { height: 100%; border-radius: 2px; transition: width 0.3s; }

/* Detail modal */
.modal-overlay { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.8); z-index: 1000; justify-content: center; align-items: center; }
.modal-overlay.active { display: flex; }
.modal { background: #1a1a2e; border: 1px solid #2a2a4e; border-radius: 12px; max-width: 800px; width: 90%; max-height: 90vh; overflow-y: auto; padding: 24px; }
.modal h2 { font-size: 14px; color: #fff; margin-bottom: 12px; word-break: break-all; }
.modal-images { display: flex; gap: 8px; margin-bottom: 16px; }
.modal-images img { width: 33%; aspect-ratio: 1; object-fit: cover; border-radius: 6px; }
.modal-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 16px; }
.modal-metrics .metric { background: #0a0a14; border-radius: 6px; padding: 10px; text-align: center; }
.modal-metrics .metric .label { font-size: 10px; color: #666; text-transform: uppercase; }
.modal-metrics .metric .value { font-size: 20px; font-weight: 700; color: #fff; margin-top: 4px; }
.modal-close { float: right; background: #2a2a4e; border: none; color: #ddd; padding: 6px 16px; border-radius: 4px; cursor: pointer; font-size: 13px; }
.modal-close:hover { background: #3a3a5e; }

/* Empty state */
.empty { text-align: center; padding: 60px 24px; color: #555; }
.empty h2 { font-size: 20px; color: #888; margin-bottom: 8px; }

/* Toast */
.toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); background: #1a1a2e; border: 1px solid #2a2a4e; border-radius: 8px; padding: 10px 20px; font-size: 13px; color: #ddd; box-shadow: 0 4px 20px rgba(0,0,0,0.5); opacity: 0; transition: opacity 0.3s; pointer-events: none; z-index: 100; }
.toast.show { opacity: 1; }

/* Train info */
.train-info { font-size: 11px; color: #555; background: #0a0a14; padding: 6px 12px; border-radius: 4px; margin: 0 24px; }

/* Responsive */
@media (max-width: 700px) {
    .grid { grid-template-columns: 1fr; padding: 12px; }
    .header { padding: 10px 12px; }
    .header .stats { margin-left: 0; width: 100%; }
    .controls { padding: 8px 12px; }
    .modal-images { flex-direction: column; }
    .modal-images img { width: 100%; }
    .modal-metrics { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>

<div class="header">
    <h1>SubspaceAD Review</h1>
    <span class="subtitle">DINOv2 patch-level anomaly detection on """ + str(len(manifest)) + """ candidates</span>
    <div class="stats">
        <span>Candidates: <strong id="statTotal">""" + str(total) + """</strong></span>
        <span>High area (<span style="color:#4CAF50;">&gt;5%</span>): <strong id="statHigh" style="color:#4CAF50">""" + str(n_positive_like) + """</strong></span>
        <span>Max score: <strong id="statMaxScore">""" + f"{high_score:.6f}" + """</strong></span>
    </div>
</div>

<div class="controls">
    <label>Sort:</label>
    <select id="sortSelect" onchange="applyFilters()">
        <option value="score">Score (top-10%) descending</option>
        <option value="area">Spawn area descending</option>
        <option value="mean">Mean residual descending</option>
        <option value="filename">Filename A-Z</option>
    </select>
    <label>Min area:</label>
    <input type="range" id="minArea" min="0" max="100" value="0" oninput="applyFilters()" style="width:100px;">
    <span id="minAreaLabel" style="color:#888;font-size:12px;">0%</span>
    <label>Search:</label>
    <input type="text" id="searchInput" placeholder="Filter by filename..." oninput="applyFilters()" style="width:180px;">
    <span class="count" id="filterCount">Showing """ + str(total) + """/""" + str(total) + """</span>
</div>

<div class="train-info" id="trainInfo">""" + train_info + """</div>

<div class="grid" id="candidateGrid"></div>
<div class="empty" id="emptyState" style="display:none;">
    <h2>No candidates match filter</h2>
    <p>Try adjusting the search or minimum area threshold.</p>
</div>

<!-- Modal -->
<div class="modal-overlay" id="modalOverlay" onclick="closeModal(event)">
    <div class="modal" id="modalContent">
        <button class="modal-close" onclick="closeModal()">Close</button>
        <h2 id="modalFilename"></h2>
        <div class="modal-images">
            <img id="modalOriginal" alt="Original">
            <img id="modalHeatmap" alt="Heatmap">
            <img id="modalSeg" alt="Segmentation">
        </div>
        <div class="modal-metrics" id="modalMetrics"></div>
    </div>
</div>

<div class="toast" id="toast"></div>

<script>
// ── Data ─────────────────────────────────────────────────────────────

const allCandidates = """ + candidates_json + r""";

// ── Rendering ──────────────────────────────────────────────────────────

function getScoreColor(score, maxScore) {
    const ratio = maxScore > 0 ? score / maxScore : 0;
    if (ratio > 0.8) return '#4CAF50';
    if (ratio > 0.4) return '#FFC107';
    return '#f44336';
}

function getAreaColor(frac) {
    if (frac > 0.05) return '#4CAF50';
    if (frac > 0.01) return '#FFC107';
    return '#888';
}

function render() {
    const sortBy = document.getElementById('sortSelect').value;
    const minArea = parseFloat(document.getElementById('minArea').value) / 100;
    const search = document.getElementById('searchInput').value.toLowerCase();

    let filtered = allCandidates.filter(c => {
        if (c.spawn_area_frac < minArea) return false;
        if (search && !c.filename.toLowerCase().includes(search)) return false;
        return true;
    });

    // Sort
    filtered.sort((a, b) => {
        switch (sortBy) {
            case 'score': return (b.score_top10p || 0) - (a.score_top10p || 0);
            case 'area': return (b.spawn_area_frac || 0) - (a.spawn_area_frac || 0);
            case 'mean': return (b.score_mean || 0) - (a.score_mean || 0);
            case 'filename': return a.filename.localeCompare(b.filename);
            default: return 0;
        }
    });

    const grid = document.getElementById('candidateGrid');
    const empty = document.getElementById('emptyState');

    document.getElementById('filterCount').textContent = `Showing ${filtered.length}/${allCandidates.length}`;

    if (filtered.length === 0) {
        grid.innerHTML = '';
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';

    const maxScore = filtered.length > 0 ? Math.max(...filtered.map(c => c.score_top10p || 0)) : 1;
    const globalMax = Math.max(...allCandidates.map(c => c.score_top10p || 0));

    let html = '';
    for (const c of filtered) {
        const scorePct = globalMax > 0 ? (c.score_top10p || 0) / globalMax * 100 : 0;
        const isHighlight = c.spawn_area_frac > 0.05;

        html += `
        <div class="card ${isHighlight ? 'highlight' : ''}" onclick="openModal('${c.filename}')">
            <div class="card-header">
                <span class="filename" title="${escapeHtml(c.filename)}">${escapeHtml(c.filename)}</span>
                <span class="batch"></span>
            </div>
            <div class="card-images">
                <div class="img-wrap"><img src="${c.original_path}" alt="orig"><div class="img-label">RGB</div></div>
                <div class="img-wrap"><img src="${c.heatmap_path}" alt="heatmap"><div class="img-label">Heatmap</div></div>
                <div class="img-wrap"><img src="${c.segmentation_path}" alt="seg"><div class="img-label">Segmentation</div></div>
            </div>
            <div class="score-bar"><div class="fill" style="width:${scorePct.toFixed(1)}%;background:${getScoreColor(c.score_top10p || 0, globalMax)}"></div></div>
            <div class="card-body">
                <div class="metric"><span class="label">Score</span><span class="value" style="color:${getScoreColor(c.score_top10p || 0, globalMax)}">${(c.score_top10p || 0).toFixed(6)}</span></div>
                <div class="metric"><span class="label">Area</span><span class="value" style="color:${getAreaColor(c.spawn_area_frac)}">${((c.spawn_area_frac || 0) * 100).toFixed(1)}%</span></div>
                <div class="metric"><span class="label">Mean</span><span class="value">${(c.score_mean || 0).toFixed(6)}</span></div>
                <div class="metric"><span class="label">Patches</span><span class="value">${c.n_spawn_patches || 0}/${c.n_patches || 256}</span></div>
            </div>
        </div>`;
    }
    grid.innerHTML = html;
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function applyFilters() {
    const val = document.getElementById('minArea').value;
    document.getElementById('minAreaLabel').textContent = val + '%';
    render();
}

// ── Modal ──────────────────────────────────────────────────────────────

function openModal(filename) {
    const c = allCandidates.find(x => x.filename === filename);
    if (!c) return;

    document.getElementById('modalFilename').textContent = c.filename;
    document.getElementById('modalOriginal').src = c.original_path;
    document.getElementById('modalHeatmap').src = c.heatmap_path;
    document.getElementById('modalSeg').src = c.segmentation_path;

    const globalMax = Math.max(...allCandidates.map(x => x.score_top10p || 0));

    document.getElementById('modalMetrics').innerHTML = `
        <div class="metric"><div class="label">Score (top-10%)</div><div class="value" style="color:${getScoreColor(c.score_top10p || 0, globalMax)}">${(c.score_top10p || 0).toFixed(6)}</div></div>
        <div class="metric"><div class="label">Mean Residual</div><div class="value">${(c.score_mean || 0).toFixed(6)}</div></div>
        <div class="metric"><div class="label">Max Residual</div><div class="value">${(c.score_max || 0).toFixed(6)}</div></div>
        <div class="metric"><div class="label">Auto Threshold</div><div class="value">${(c.auto_threshold || 0).toFixed(6)}</div></div>
        <div class="metric"><div class="label">Spawn Area</div><div class="value" style="color:${getAreaColor(c.spawn_area_frac)}">${((c.spawn_area_frac || 0) * 100).toFixed(2)}%</div></div>
        <div class="metric"><div class="label">Anomalous Patches</div><div class="value">${c.n_spawn_patches || 0} / ${c.n_patches || 256}</div></div>
    `;

    document.getElementById('modalOverlay').classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeModal(e) {
    if (e && e.target !== e.currentTarget) return;
    document.getElementById('modalOverlay').classList.remove('active');
    document.body.style.overflow = '';
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

// ── Init ───────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', render);
</script>

</body>
</html>"""

    html_path.write_text(html)
    print(f"  Review page generated: {html_path}")
    return str(html_path)


# ---------------------------------------------------------------------------
# Manifest saver
# ---------------------------------------------------------------------------

def save_manifest(manifest: list[dict], output_dir: str, train_meta: dict | None = None) -> str:
    """Save manifest.json with full metadata.

    Returns:
        Path to the saved manifest file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": f"{MODEL_NAME} + SubspaceAD (patch PCA)",
        "n_components": train_meta.get("n_components", 0) if train_meta else 0,
        "n_negative_images": train_meta.get("n_images_used", 0) if train_meta else 0,
        "n_patches_trained": train_meta.get("n_patches_trained", 0) if train_meta else 0,
        "explained_variance_ratio": train_meta.get("explained_variance_ratio", 0) if train_meta else 0,
        "anomalous_patch_frac": ANOMALOUS_PATCH_FRAC,
        "n_candidates_total": len(manifest),
        "n_candidates_scored": sum(1 for e in manifest if "error" not in e),
        "candidates": manifest,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2))
    print(f"  Manifest saved: {manifest_path} ({len(manifest)} candidates)")
    return str(manifest_path)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler that doesn't log every request."""
    def log_message(self, format_, *args):
        pass


def start_server(directory: str, port: int) -> HTTPServer:
    """Start a simple HTTP server serving the given directory."""
    os.chdir(directory)
    server = HTTPServer(("0.0.0.0", port), QuietHTTPRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch DINOv2 SubspaceAD review page generator for herring spawn candidates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--negative-dir", type=str, default=None,
        help=f"Directory of negative (non-spawn) PNG images (default: {DEFAULT_NEGATIVE_DIR})",
    )
    parser.add_argument(
        "--candidate-dir", type=str, default=None,
        help=f"Directory of candidate PNG images (default: {DEFAULT_CANDIDATE_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory for review artifacts (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"HTTP server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for DINOv2 inference (default: auto)",
    )
    parser.add_argument(
        "--n-components", type=int, default=DEFAULT_N_COMPONENTS,
        help=f"Number of PCA components (default: {DEFAULT_N_COMPONENTS})",
    )
    parser.add_argument(
        "--sample-frac", type=float, default=0.15,
        help="Fraction of patches sampled per image for PCA training (default: 0.15)",
    )
    parser.add_argument(
        "--serve-only", action="store_true",
        help="Skip processing, just start the HTTP server for an existing review",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open the browser",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print debug info",
    )
    args = parser.parse_args()

    # Resolve paths
    repo_root = Path(__file__).resolve().parent.parent

    def resolve_path(given: str | None, default: str) -> str:
        if given is None:
            return str(repo_root / default)
        p = Path(given)
        return str(p if p.is_absolute() else (repo_root / given))

    negative_dir = resolve_path(args.negative_dir, DEFAULT_NEGATIVE_DIR)
    candidate_dir = resolve_path(args.candidate_dir, DEFAULT_CANDIDATE_DIR)
    output_dir = resolve_path(args.output_dir, OUTPUT_DIR)

    # ==================================================================
    # Serve-only mode
    # ==================================================================
    if args.serve_only:
        out_path = Path(output_dir)
        if not (out_path / "review.html").exists():
            print(f"ERROR: No review page found in {output_dir}")
            print("  Run without --serve-only to generate the review first.")
            return 1

        print(f"\n  Serving existing review from {output_dir}")
        print(f"  URL: http://localhost:{args.port}")
        print(f"  Press Ctrl+C to stop.\n")

        server = start_server(output_dir, args.port)

        if not args.no_browser:
            webbrowser.open(f"http://localhost:{args.port}/review.html")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")
        return 0

    # ==================================================================
    # Validate directories
    # ==================================================================
    if not os.path.isdir(negative_dir):
        print(f"ERROR: Negative directory not found: {negative_dir}")
        return 1

    if not os.path.isdir(candidate_dir):
        print(f"ERROR: Candidate directory not found: {candidate_dir}")
        return 1

    print("=" * 60)
    print("  DINOv2 SubspaceAD — Batch Review Generator")
    print("=" * 60)
    print(f"  Negative directory:  {negative_dir}")
    print(f"  Candidate directory: {candidate_dir}")
    print(f"  Output directory:    {output_dir}")
    print(f"  Device:              {args.device}")
    print(f"  PCA components:      {args.n_components}")
    print(f"  Sample fraction:     {args.sample_frac}")
    print()

    # ==================================================================
    # Train PCA on negatives
    # ==================================================================
    device = _resolve_device(args.device)

    print("--- Step 1: Training PCA Subspace on Negatives ---")
    train_result = train_pca_on_negatives(
        negative_dir,
        n_components=args.n_components,
        device=device,
        sample_frac=args.sample_frac,
    )

    if "error" in train_result:
        print(f"ERROR: PCA training failed: {train_result['error']}")
        return 1

    pca_model = train_result["pca_model"]
    print(f"  PCA trained: {train_result['n_components']} components, "
          f"{train_result['n_images_used']} images, "
          f"{train_result['n_patches_trained']} patches")
    print()

    # ==================================================================
    # Batch score candidates
    # ==================================================================
    print("--- Step 2: Scoring Candidates & Generating Overlays ---")
    manifest = batch_process(
        pca_model,
        candidate_dir,
        output_dir,
        device=device,
    )

    if not manifest:
        print("ERROR: No candidates were successfully scored.")
        return 1

    print()

    # ==================================================================
    # Generate review page
    # ==================================================================
    print("--- Step 3: Generating Review Page ---")
    generate_review_page(manifest, output_dir, train_meta=train_result)
    print()

    # ==================================================================
    # Save manifest
    # ==================================================================
    print("--- Step 4: Saving Manifest ---")
    save_manifest(manifest, output_dir, train_meta=train_result)
    print()

    # ==================================================================
    # Print summary
    # ==================================================================
    top = manifest[:5]
    print("=" * 60)
    print("  Top 5 Candidates by SubspaceAD Score")
    print("=" * 60)
    for i, c in enumerate(top, 1):
        print(f"  {i}. score={c['score_top10p']:.6f}  area={c['spawn_area_frac']*100:.1f}%  "
              f"patches={c['n_spawn_patches']}/{c['n_patches']}  {c['filename']}")
    print()

    # ==================================================================
    # Start HTTP server
    # ==================================================================
    print("--- Starting Web Server ---")
    print(f"  Output directory: {output_dir}")
    print(f"  Review page URL:  http://localhost:{args.port}/review.html")
    print(f"  Press Ctrl+C to stop.")
    print()

    server = start_server(output_dir, args.port)

    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}/review.html")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
