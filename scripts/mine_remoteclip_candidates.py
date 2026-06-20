#!/usr/bin/env python3
"""Mine unlabeled candidate satellite thumbnails through RemoteCLIP scoring,
deduplicate against existing training data, and build an HTML review page for
human labeling.

Usage:
    python scripts/mine_remoteclip_candidates.py \\
        --candidate-dirs data/candidates_knn data/candidates_final data/sog_candidates/thumbnails \\
        --output-dir data/remoteclip_mined \\
        --top 200 \\
        --device cpu

Output files (in --output-dir):
    scores.json   — Full list of all scored candidates
    review.html   — Self-contained HTML review page (top-N, keyboard shortcuts,
                    localStorage labels, export, lightbox, sortable)
    manifest.json — Summary statistics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Import RemoteCLIP utilities from sibling module
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from remoteclip_zero_shot import (  # noqa: E402
    NEGATIVE_PROMPTS,
    POSITIVE_PROMPTS,
    _resolve_device,
    get_image_embedding,
    get_text_embedding,
    load_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_DIM = 768
CANDIDATE_CACHE_PATH = Path("data/embeddings/remoteclip_candidates.npz")

ROOT_DIR = _SCRIPT_DIR.parent

# Directories to check for known training targets (filename dedup)
TRAINING_DIRS = [
    ROOT_DIR / "data" / "samples" / "positive",
    ROOT_DIR / "data" / "samples" / "negative",
]

# Suffixes accepted as images
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_training_filenames() -> set[str]:
    """Return the set of bare filenames already present in training dirs."""
    names: set[str] = set()
    for d in TRAINING_DIRS:
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix.lower() in IMAGE_SUFFIXES:
                    names.add(f.name)
    return names


def _extract_numeric_score(filename: str) -> float:
    """Parse numeric score from a filename like ``*_score0.01_*``.

    Returns 0.0 if no score pattern is found.
    """
    m = re.search(r"_score([\d.]+(?:e[+-]?\d+)?)_", filename)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    return 0.0


def _candidate_dirs_from_args(dirs: list[str]) -> list[Path]:
    """Resolve candidate directory paths relative to the repo root if needed."""
    resolved: list[Path] = []
    for d in dirs:
        p = Path(d)
        if not p.is_absolute():
            p = ROOT_DIR / d
        resolved.append(p)
    return resolved


# ---------------------------------------------------------------------------
# Candidate discovery + dedup
# ---------------------------------------------------------------------------

def discover_candidates(
    candidate_dir_paths: list[Path],
    training_fnames: set[str],
) -> list[dict]:
    """Walk candidate directories for PNGs, deduplicate by filename.

    Returns a list of dicts with keys:
        filename, image_path, source_dir, score
    sorted by score descending.
    Images whose base filename appears in *training_fnames* are excluded.
    If the same filename appears in multiple directories, only the instance
    with the highest numeric score is kept.
    """
    # Collect all candidates keyed by filename
    by_name: dict[str, dict] = {}

    for dir_path in candidate_dir_paths:
        if not dir_path.is_dir():
            print(f"  WARNING: candidate directory not found, skipping: {dir_path}")
            continue

        # Look for images both at top level and in a "thumbnails/" subdir
        search_globs = ["*", "thumbnails/*"]
        for pattern in search_globs:
            for fpath in sorted(dir_path.glob(pattern)):
                if fpath.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                fname = fpath.name

                # Skip training images
                if fname in training_fnames:
                    continue

                score = _extract_numeric_score(fname)
                existing = by_name.get(fname)

                if existing is None:
                    by_name[fname] = {
                        "filename": fname,
                        "image_path": str(fpath.resolve()),
                        "source_dir": str(dir_path),
                        "score": score,
                    }
                elif score > existing["score"]:
                    # Keep the higher-scored version
                    existing["image_path"] = str(fpath.resolve())
                    existing["source_dir"] = str(dir_path)
                    existing["score"] = score

    candidates = list(by_name.values())
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _abs_paths_key(paths: list[str]) -> str:
    """Deterministic hash key for a set of absolute paths."""
    combined = "\x00".join(sorted(paths))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def _load_embedding_cache() -> dict[str, np.ndarray]:
    """Load the embedding cache from ``CANDIDATE_CACHE_PATH``.

    Returns a dict mapping absolute image path -> embedding (1-D float32).
    Returns empty dict if cache does not exist.
    """
    cache_path = ROOT_DIR / CANDIDATE_CACHE_PATH
    if not cache_path.exists():
        return {}

    try:
        loaded = np.load(cache_path, allow_pickle=True)
        embeddings = loaded["embeddings"]
        img_paths = loaded["img_paths"]
        result: dict[str, np.ndarray] = {}
        for i in range(len(img_paths)):
            result[str(img_paths[i])] = embeddings[i]
        return result
    except Exception as exc:
        print(f"  WARNING: failed to load embedding cache: {exc}")
        return {}


def _save_embedding_cache(cache: dict[str, np.ndarray]) -> None:
    """Persist the embedding cache."""
    if not cache:
        return
    cache_path = ROOT_DIR / CANDIDATE_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    paths = list(cache.keys())
    embs = np.stack([cache[p] for p in paths]).astype(np.float32)
    np.savez_compressed(
        cache_path,
        embeddings=embs,
        img_paths=np.array(paths, dtype=object),
    )
    print(f"  Saved {len(paths)} embeddings to cache: {cache_path}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_candidates(
    candidates: list[dict],
    model,
    preprocess,
    tokenize,
    device: str,
    cache: dict[str, np.ndarray],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Score each candidate with RemoteCLIP zero-shot.

    Returns (scored_list, updated_cache).
    """
    # Pre-compute text embeddings once
    all_texts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    text_emb = get_text_embedding(model, tokenize, all_texts, device)  # (N, D)
    n_pos = len(POSITIVE_PROMPTS)

    scored: list[dict] = []
    uncached_paths: list[str] = []

    # First pass: identify which images need embedding
    for cand in candidates:
        img_path = cand["image_path"]
        if img_path not in cache:
            uncached_paths.append(img_path)

    # Compute missing embeddings
    if uncached_paths:
        print(f"  Computing {len(uncached_paths)} new image embeddings...")
        for img_path in tqdm(uncached_paths, desc="Embedding", unit="img"):
            emb = get_image_embedding(model, preprocess, img_path, device)
            if emb is not None:
                cache[img_path] = emb.astype(np.float32)

    # Score all candidates from cache
    for cand in tqdm(candidates, desc="Scoring", unit="img"):
        img_path = cand["image_path"]
        emb = cache.get(img_path)
        if emb is None:
            continue

        img_tensor = torch.from_numpy(emb).to(device)
        sims = text_emb @ img_tensor  # (N,) cosine similarities

        pos_sims = sims[:n_pos].cpu().tolist()
        neg_sims = sims[n_pos:].cpu().tolist()

        pos_mean = float(np.mean(pos_sims))
        neg_mean = float(np.mean(neg_sims))
        score_val = pos_mean - neg_mean
        prediction = 1 if score_val > 0 else 0

        scored.append({
            "filename": cand["filename"],
            "image_path": img_path,
            "source_dir": cand["source_dir"],
            "score": score_val,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "prediction": prediction,
            "pos_scores": pos_sims,
            "neg_scores": neg_sims,
        })

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored, cache


# ---------------------------------------------------------------------------
# HTML review page builder
# ---------------------------------------------------------------------------

def build_review_html(
    scored: list[dict],
    top_n: int,
    output_dir: Path,
) -> str:
    """Generate a self-contained HTML review page.

    The candidate data is embedded as JSON in a ``<script>`` tag.  The page
    renders cards dynamically, supports keyboard shortcuts, localStorage
    labels, lightbox, sorting, and label export.
    """
    # Build JSON data for the top-N scored candidates (include all if top_n <= 0)
    display = scored[:top_n] if top_n > 0 else scored[:]

    # Compute relative paths from the output directory to each image
    rows = []
    for r in display:
        img_abs = Path(r["image_path"])
        try:
            rel = os.path.relpath(img_abs, output_dir)
        except ValueError:
            rel = img_abs  # fallback: absolute path if rel fails (cross-volume)
        rows.append({
            "filename": r["filename"],
            "image_path": rel,
            "score": round(r["score"], 6),
            "pos_mean": round(r["pos_mean"], 6),
            "neg_mean": round(r["neg_mean"], 6),
            "prediction": r["prediction"],
        })

    total_count = len(scored)
    shown_count = len(rows)
    data_json = json.dumps({"candidates": rows, "total": total_count})

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RemoteCLIP Mined Candidates</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #f0f2f5; color: #1f2937; }}
    header {{ background: linear-gradient(135deg, #111827, #0f172a); color: #fff; padding: 20px 24px; }}
    header h1 {{ margin: 0 0 4px; font-size: 22px; }}
    header p {{ margin: 0; font-size: 14px; opacity: .85; }}
    .toolbar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding: 12px 24px; background: #fff; border-bottom: 1px solid #e2e8f0; }}
    .toolbar .progress {{ flex: 1; min-width: 140px; }}
    .toolbar .progress-bar {{ height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }}
    .toolbar .progress-fill {{ height: 100%; background: #3b82f6; border-radius: 4px; transition: width .2s; width: 0%; }}
    .toolbar .progress-label {{ font-size: 12px; color: #64748b; margin-top: 2px; }}
    .toolbar button {{ padding: 6px 14px; border: 1px solid #cbd5e1; border-radius: 6px; background: #fff; cursor: pointer; font-size: 13px; }}
    .toolbar button:hover {{ background: #f1f5f9; }}
    .toolbar button.primary {{ background: #3b82f6; color: #fff; border-color: #3b82f6; }}
    .toolbar button.primary:hover {{ background: #2563eb; }}
    .toolbar select {{ padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; background: #fff; }}
    .toolbar .stats {{ font-size: 13px; color: #475569; }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 20px 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }}
    .card {{ background: #fff; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); transition: box-shadow .15s; position: relative; }}
    .card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,.12); }}
    .card img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; cursor: pointer; }}
    .card .meta {{ padding: 6px 12px 4px; font-size: 12px; color: #475569; line-height: 1.5; }}
    .card .meta .fn {{ font-family: ui-monospace, monospace; font-size: 10px; word-break: break-all; color: #64748b; }}
    .card .score-good {{ color: #059669; font-weight: 600; }}
    .card .score-bad {{ color: #dc2626; font-weight: 600; }}
    .card .pred-badge {{ display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 10px; font-weight: 600; }}
    .card .pred-1 {{ background: #d1fae5; color: #065f46; }}
    .card .pred-0 {{ background: #fee2e2; color: #991b1b; }}
    .card .label-buttons {{ display: flex; gap: 4px; padding: 6px 12px 10px; }}
    .card .label-buttons button {{ flex: 1; padding: 4px 0; border: 1px solid #d1d5db; border-radius: 5px; font-size: 11px; font-weight: 600; cursor: pointer; background: #f9fafb; }}
    .card .label-buttons button:hover {{ background: #e5e7eb; }}
    .card .label-buttons button.active-spawn {{ background: #10b981; color: #fff; border-color: #059669; }}
    .card .label-buttons button.active-nospawn {{ background: #ef4444; color: #fff; border-color: #dc2626; }}
    .card .label-buttons button.active-unsure {{ background: #f59e0b; color: #fff; border-color: #d97706; }}
    .card .label-status {{ padding: 0 12px 8px; font-size: 11px; color: #6b7280; }}
    .card.highlight {{ box-shadow: 0 0 0 3px #3b82f6; }}
    .lightbox {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,.85); z-index: 1000; justify-content: center; align-items: center; cursor: pointer; }}
    .lightbox.open {{ display: flex; }}
    .lightbox img {{ max-width: 92vw; max-height: 92vh; object-fit: contain; border-radius: 4px; }}
    .shortcut-hint {{ font-size: 11px; color: #94a3b8; margin-left: auto; }}
    .empty {{ text-align: center; padding: 60px 20px; color: #94a3b8; }}
  </style>
</head>
<body>
  <header>
    <h1>RemoteCLIP Mined Candidates</h1>
    <p>Zero-shot spawn scoring · <span id="shown-count">{shown_count}</span> of <span id="total-count">{total_count}</span> candidates shown · sorted by score</p>
  </header>
  <div class="toolbar">
    <div class="stats" id="stats-bar">Labeled: <strong id="labeled-count">0</strong> / {total_count}</div>
    <div class="progress">
      <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
      <div class="progress-label" id="progress-label">0% labeled</div>
    </div>
    <label>
      Sort:
      <select id="sort-select">
        <option value="score-desc">Score ↓</option>
        <option value="score-asc">Score ↑</option>
        <option value="filename">Filename A–Z</option>
      </select>
    </label>
    <button id="export-btn" class="primary">⬇ Export Labels</button>
    <button id="clear-btn">Clear All Labels</button>
    <span class="shortcut-hint">Keys: <kbd>y</kbd> spawn · <kbd>n</kbd> no · <kbd>u</kbd> unsure · <kbd>←</kbd><kbd>→</kbd> navigate</span>
  </div>
  <main>
    <div class="grid" id="candidate-grid"></div>
  </main>
  <div class="lightbox" id="lightbox">
    <img id="lightbox-img" src="" alt="full-size view">
  </div>
  <script>
    const DATA = {data_json};
    const STORAGE_KEY = 'remoteclip_mined_labels';
    let currentIndex = 0;

    // ----- localStorage labels -----
    function loadLabels() {{
      try {{
        const raw = localStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : {{}};
      }} catch {{ return {{}}; }}
    }}
    function saveLabels(labels) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
    }}
    const labels = loadLabels();

    function setLabel(filename, value) {{
      labels[filename] = value;
      saveLabels(labels);
      renderGrid();
    }}

    // ----- export -----
    function exportLabels() {{
      const classifications = DATA.candidates
        .filter(c => labels[c.filename] !== undefined)
        .map(c => ({{
          filename: c.filename,
          spawn: labels[c.filename] === 'spawn' ? true : (labels[c.filename] === 'nospawn' ? false : null),
          confidence: 'medium',
          notes: ''
        }}));
      const payload = JSON.stringify({{ classifications }}, null, 2);
      const blob = new Blob([payload], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'remoteclip_mined_labels.json';
      a.click();
      URL.revokeObjectURL(url);
    }}

    // ----- clear labels -----
    function clearLabels() {{
      if (!confirm('Clear all labels from localStorage?')) return;
      Object.keys(labels).forEach(k => delete labels[k]);
      saveLabels(labels);
      renderGrid();
    }}

    // ----- lightbox -----
    function openLightbox(src) {{
      document.getElementById('lightbox-img').src = src;
      document.getElementById('lightbox').classList.add('open');
    }}
    function closeLightbox() {{
      document.getElementById('lightbox').classList.remove('open');
    }}
    document.getElementById('lightbox').addEventListener('click', closeLightbox);

    // ----- rendering -----
    function getSortedCandidates() {{
      const sortVal = document.getElementById('sort-select').value;
      const arr = [...DATA.candidates];
      if (sortVal === 'score-desc') arr.sort((a, b) => b.score - a.score);
      else if (sortVal === 'score-asc') arr.sort((a, b) => a.score - b.score);
      else if (sortVal === 'filename') arr.sort((a, b) => a.filename.localeCompare(b.filename));
      return arr;
    }}

    function renderGrid() {{
      const grid = document.getElementById('candidate-grid');
      const sorted = getSortedCandidates();
      let html = '';
      let labeledCount = 0;

      sorted.forEach((c, idx) => {{
        const lbl = labels[c.filename];
        if (lbl) labeledCount++;
        const scoreClass = c.score >= 0 ? 'score-good' : 'score-bad';
        const predClass = c.prediction === 1 ? 'pred-1' : 'pred-0';
        const predLabel = c.prediction === 1 ? 'spawn' : 'no spawn';

        let btnSpawn = 'Spawn', btnNo = 'No', btnUnsure = 'Unsure';
        let clsSpawn = '', clsNo = '', clsUnsure = '';
        if (lbl === 'spawn') clsSpawn = 'active-spawn';
        else if (lbl === 'nospawn') clsNo = 'active-nospawn';
        else if (lbl === 'unsure') clsUnsure = 'active-unsure';

        let statusText = '';
        if (lbl === 'spawn') statusText = '✅ Labeled: spawn';
        else if (lbl === 'nospawn') statusText = '❌ Labeled: no spawn';
        else if (lbl === 'unsure') statusText = '❓ Labeled: unsure';
        else statusText = '⏺ Not labeled';

        html += `
          <article class="card" data-index="${{idx}}">
            <img src="${{c.image_path}}" alt="${{c.filename}}" loading="lazy" onclick="openLightbox(this.src)">
            <div class="meta">
              <div class="fn">${{c.filename}}</div>
              <div><span class="${{scoreClass}}">score ${{c.score.toFixed(4)}}</span></div>
              <div><span class="pred-badge ${{predClass}}">${{predLabel}}</span> · pos ${{c.pos_mean.toFixed(4)}} · neg ${{c.neg_mean.toFixed(4)}}</div>
            </div>
            <div class="label-buttons">
              <button class="${{clsSpawn}}" data-filename="${{c.filename}}" data-value="spawn">${{btnSpawn}}</button>
              <button class="${{clsNo}}" data-filename="${{c.filename}}" data-value="nospawn">${{btnNo}}</button>
              <button class="${{clsUnsure}}" data-filename="${{c.filename}}" data-value="unsure">${{btnUnsure}}</button>
            </div>
            <div class="label-status">${{statusText}}</div>
          </article>
        `;
      }});

      grid.innerHTML = html;

      // Attach label button events
      grid.querySelectorAll('.label-buttons button').forEach(btn => {{
        btn.addEventListener('click', () => {{
          setLabel(btn.dataset.filename, btn.dataset.value);
        }});
      }});

      // Update progress
      document.getElementById('labeled-count').textContent = labeledCount;
      const pct = DATA.total > 0 ? Math.round((labeledCount / DATA.total) * 100) : 0;
      document.getElementById('progress-fill').style.width = pct + '%';
      document.getElementById('progress-label').textContent = pct + '% labeled';
    }}

    // ----- keyboard shortcuts -----
    document.addEventListener('keydown', (e) => {{
      const grid = document.getElementById('candidate-grid');
      const cards = grid.querySelectorAll('.card');
      if (!cards.length) return;

      let focused = grid.querySelector('.card.highlight');
      let curIdx = focused ? parseInt(focused.dataset.index) : 0;

      if (e.key === 'ArrowRight') {{
        e.preventDefault();
        curIdx = Math.min(curIdx + 1, cards.length - 1);
      }} else if (e.key === 'ArrowLeft') {{
        e.preventDefault();
        curIdx = Math.max(curIdx - 1, 0);
      }}

      // Find the card at the new index based on sorted order
      const sorted = getSortedCandidates();
      if (curIdx < 0 || curIdx >= sorted.length) return;
      const target = sorted[curIdx];

      cards.forEach(c => c.classList.remove('highlight'));
      const targetCard = grid.querySelector(`.card[data-index="${{curIdx}}"]`);
      if (targetCard) {{
        targetCard.classList.add('highlight');
        targetCard.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
      }}

      if (e.key === 'y' || e.key === 'Y') {{
        if (target) setLabel(target.filename, 'spawn');
      }} else if (e.key === 'n' || e.key === 'N') {{
        if (target) setLabel(target.filename, 'nospawn');
      }} else if (e.key === 'u' || e.key === 'U') {{
        if (target) setLabel(target.filename, 'unsure');
      }}
    }});

    // ----- event wiring -----
    document.getElementById('sort-select').addEventListener('change', renderGrid);
    document.getElementById('export-btn').addEventListener('click', exportLabels);
    document.getElementById('clear-btn').addEventListener('click', clearLabels);

    // ----- initial render -----
    renderGrid();
  </script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine unlabeled candidate satellite thumbnails through RemoteCLIP scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--candidate-dirs", type=str, nargs="+", required=True,
        help="One or more directories containing candidate PNG thumbnails",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/remoteclip_mined",
        help="Output directory for scores.json, review.html, manifest.json",
    )
    parser.add_argument(
        "--top", type=int, default=200,
        help="Number of top-scoring candidates to show in review.html (0 = all)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for RemoteCLIP inference",
    )
    args = parser.parse_args(argv)

    resolved_device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  RemoteCLIP Candidate Mining")
    print("=" * 60)
    print(f"  Output dir: {output_dir}")
    print(f"  Device:     {resolved_device}")
    print()

    # ---- Step 1: Discover candidates ----
    print("[1/5] Discovering candidates...")
    training_fnames = _get_training_filenames()
    print(f"  Training files to exclude: {len(training_fnames)}")
    candidate_paths = _candidate_dirs_from_args(args.candidate_dirs)
    print(f"  Candidate directories: {[str(p) for p in candidate_paths]}")

    candidates = discover_candidates(candidate_paths, training_fnames)
    if not candidates:
        print("  No new candidates found after dedup.")
        # Still write empty output files
        _write_outputs([], output_dir, args.top)
        return 0
    print(f"  Unique candidates found: {len(candidates)}")
    print(f"  Score range: {candidates[-1]['score']:.4f} – {candidates[0]['score']:.4f}")

    # ---- Step 2: Load model ----
    print("\n[2/5] Loading RemoteCLIP model...")
    model, preprocess, tokenize = load_model(device=resolved_device)

    # ---- Step 3: Load embedding cache ----
    print("\n[3/5] Loading embedding cache...")
    cache = _load_embedding_cache()
    print(f"  Cached embeddings: {len(cache)}")

    # ---- Step 4: Score candidates ----
    print(f"\n[4/5] Scoring {len(candidates)} candidates with RemoteCLIP...")
    scored, updated_cache = score_candidates(candidates, model, preprocess, tokenize, resolved_device, cache)

    if not scored:
        print("  No images scored successfully.")
        _write_outputs([], output_dir, args.top)
        return 0

    print(f"  Successfully scored: {len(scored)}")
    print(f"  Top score: {scored[0]['score']:.4f} (filename: {scored[0]['filename']})")
    n_pos_pred = sum(1 for s in scored if s["prediction"] == 1)
    print(f"  Predicted spawn: {n_pos_pred}  /  no spawn: {len(scored) - n_pos_pred}")

    # Save updated cache
    _save_embedding_cache(updated_cache)

    # ---- Step 5: Write outputs ----
    print("\n[5/5] Writing output files...")
    _write_outputs(scored, output_dir, args.top)

    print()
    print(f"  Done. Review page: {output_dir / 'review.html'}")
    print(f"  To serve: python -m http.server 8766 --directory {output_dir.parent}")
    print(f"  Then open http://localhost:8766/{output_dir.name}/review.html")
    return 0


def _write_outputs(scored: list[dict], output_dir: Path, top_n: int) -> None:
    """Write scores.json, review.html, and manifest.json to output_dir."""

    # Prepare clean output records (no large arrays for storage)
    clean_records = []
    for s in scored:
        clean_records.append({
            "filename": s["filename"],
            "image_path": s["image_path"],
            "source_dir": s["source_dir"],
            "score": round(s["score"], 6),
            "pos_mean": round(s["pos_mean"], 6),
            "neg_mean": round(s["neg_mean"], 6),
            "prediction": s["prediction"],
        })

    # scores.json
    scores_path = output_dir / "scores.json"
    scores_path.write_text(json.dumps(clean_records, indent=2))
    print(f"  scores.json: {len(clean_records)} entries -> {scores_path}")

    # manifest.json
    n_spawn_pred = sum(1 for s in scored if s["prediction"] == 1)
    n_nospawn_pred = len(scored) - n_spawn_pred
    top_scores = [s["score"] for s in scored[:5]] if scored else []
    manifest = {
        "total_candidates": len(scored),
        "predicted_spawn": n_spawn_pred,
        "predicted_no_spawn": n_nospawn_pred,
        "top_n_shown": min(top_n, len(scored)) if top_n > 0 else len(scored),
        "top_5_scores": top_scores,
        "device": "cpu",
        "model": "RemoteCLIP ViT-L-14",
        "positive_prompts": len(POSITIVE_PROMPTS),
        "negative_prompts": len(NEGATIVE_PROMPTS),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest.json written -> {manifest_path}")

    # review.html
    html = build_review_html(scored, top_n, output_dir)
    review_path = output_dir / "review.html"
    review_path.write_text(html, encoding="utf-8")
    print(f"  review.html: {len(scored[:top_n]) if top_n > 0 else len(scored)} cards -> {review_path}")


if __name__ == "__main__":
    sys.exit(main())
