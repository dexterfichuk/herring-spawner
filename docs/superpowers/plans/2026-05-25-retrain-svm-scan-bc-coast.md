# Retrain DINOv2 SVM + BC Coast Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain DINOv2 SVM classifier on 8 verified positives + 126 negatives, scan BC coast for spawn candidates, and generate a review page.

**Architecture:** Training uses `scripts/train_classifier.py`-style pipeline (load DINOv2 embeddings → fit RBF SVM with 5-fold CV). Scanning uses `scripts/scan_bc_coast.py` with the retrained model. Review page is generated from the candidate manifest JSON.

**Tech Stack:** DINOv2 ViT-S/14, scikit-learn SVM (RBF), Google Earth Engine Sentinel-2, Python 3.11+

**Files:**
- Create: `scripts/train_from_manifest.py` — Trains SVM from training_manifest.json
- Create: `scripts/generate_review_page.py` — Generates HTML review page from manifest.json
- Run: `scripts/scan_bc_coast.py` — BC coast scan (existing, no changes needed)

**Outputs:**
- `data/models/svm_8pos_126neg.pkl` — Trained model
- `data/candidates_svm_8pos/` — Candidate thumbnails + manifest.json
- `data/candidates_svm_8pos/review.html` — Sortable review page

---

### Task 0: Verify training data matches manifest

**Files:** `data/samples/training_manifest.json`, `data/samples/positive/`, `data/samples/negative/`

- [ ] **Check that manifest positive files exist in positive/ directory**

Run: `python3 -c "
import json
from pathlib import Path
m = json.loads(Path('data/samples/training_manifest.json').read_text())
pos_dir = Path('data/samples/positive')
neg_dir = Path('data/samples/negative')
missing = [f for f in m['positives'] if not (pos_dir / f).exists()]
print(f'Manifest: {m[\"positive_count\"]} positives, {m[\"negative_count\"]} negatives')
print(f'Positive dir: {len(list(pos_dir.glob(\"*.png\")))} PNGs')
print(f'Negative dir: {len(list(neg_dir.glob(\"*.png\")))} PNGs')
if missing: print(f'MISSING: {missing}')
else: print('All manifest files present OK')
"`

Expected: All 8 positives present, 126 negatives, no missing files.

---

### Task 1: Write `scripts/train_from_manifest.py`

**Files:**
- Create: `scripts/train_from_manifest.py`

This script reads `training_manifest.json`, loads only the specific positive files listed + all negative PNGs, computes DINOv2 embeddings, trains RBF SVM with 5-fold stratified CV, saves model to `data/models/svm_8pos_126neg.pkl`.

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""Train SVM classifier from training_manifest.json — specific 8 positives + all negatives.

Reads the manifest to get exact positive filenames (to exclude rose-verified
spawns that are also in the directory), loads all negatives, trains DINOv2+SVM.

Usage:
    python scripts/train_from_manifest.py \
        --manifest data/samples/training_manifest.json \
        --positive-dir data/samples/positive \
        --negative-dir data/samples/negative \
        --output-model data/models/svm_8pos_126neg.pkl
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import transforms

MODEL_NAME = "dinov2_vits14"
EMBED_DIM = 384

DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_embeddings_from_manifest(
    manifest_path: Path,
    pos_dir: Path,
    neg_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load embeddings for files specified in manifest + all negatives.

    Returns (embeddings, labels, filenames, errors).
    labels[i] = 1 for positive (spawn), 0 for negative (no spawn).
    """
    manifest = json.loads(manifest_path.read_text())
    pos_fnames: list[str] = manifest["positives"]

    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    filenames: list[str] = []
    errors: list[str] = []

    # Load only manifest-specified positives
    for fname in pos_fnames:
        fpath = pos_dir / fname
        if not fpath.exists():
            errors.append(f"{fname}: file not found in {pos_dir}")
            continue
        try:
            img = Image.open(fpath).convert("RGB")
            tensor = DINO_TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor)
            emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
            embeddings.append(emb)
            labels.append(1)
            filenames.append(fname)
        except Exception as exc:
            errors.append(f"{fname}: {exc}")

    # Load all negatives
    for fpath in sorted(neg_dir.glob("*.png")):
        try:
            img = Image.open(fpath).convert("RGB")
            tensor = DINO_TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor)
            emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
            embeddings.append(emb)
            labels.append(0)
            filenames.append(fpath.name)
        except Exception as exc:
            errors.append(f"{fpath.name}: {exc}")

    return np.array(embeddings), np.array(labels), filenames, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train SVM from training manifest (specific positives + all negatives)",
    )
    parser.add_argument("--manifest", default="data/samples/training_manifest.json")
    parser.add_argument("--positive-dir", default="data/samples/positive")
    parser.add_argument("--negative-dir", default="data/samples/negative")
    parser.add_argument("--output-model", default="data/models/svm_8pos_126neg.pkl")
    parser.add_argument("--kernel", default="rbf", choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = repo_root / args.manifest
    pos_dir = repo_root / args.positive_dir
    neg_dir = repo_root / args.negative_dir
    model_path = repo_root / args.output_model

    # 1. Load DINOv2
    print("=" * 60)
    print("  Loading DINOv2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval()
    model = model.to(device)
    print(f"  Model: {MODEL_NAME} ({EMBED_DIM}-dim)")

    # 2. Load embeddings from manifest
    print(f"\n{'=' * 60}")
    print("  Loading labeled samples from manifest...")
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text())
    print(f"  Manifest: {manifest['positive_count']} positives, {manifest['negative_count']} negatives")

    X, y, fnames, errors = load_embeddings_from_manifest(
        manifest_path, pos_dir, neg_dir, model, device
    )
    n_pos = int(y.sum())
    n_neg = int(len(y) - y.sum())
    print(f"  Loaded {len(X)} samples: {n_pos} positive, {n_neg} negative")
    print(f"  Embedding dim: {X.shape[1]}")
    if errors:
        print(f"  WARNING: {len(errors)} errors:")
        for e in errors[:5]:
            print(f"    - {e}")

    if len(X) < 10:
        print("ERROR: Too few samples (<10)")
        return 1
    if n_pos < 2 or n_neg < 2:
        print("ERROR: Need >=2 per class")
        return 1

    # 3. Train SVM
    print(f"\n{'=' * 60}")
    print(f"  Training SVM (kernel={args.kernel}, class_weight='balanced')...")
    svm = SVC(
        kernel=args.kernel,
        class_weight="balanced",
        probability=True,
        random_state=args.random_state,
        gamma="scale",
    )
    svm.fit(X, y)

    # Full-dataset metrics
    y_pred = svm.predict(X)
    full_acc = accuracy_score(y, y_pred)
    y_decision = svm.decision_function(X)
    pos_scores = y_decision[y == 1]
    neg_scores = y_decision[y == 0]
    separation = float(np.mean(pos_scores) - np.mean(neg_scores))

    print(f"\n  FULL DATASET RESULTS")
    print(f"  Accuracy: {full_acc:.4f}")
    print(classification_report(y, y_pred, target_names=["negative", "positive"]))
    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                Neg   Pos")
    print(f"  Actual Neg    {cm[0][0]:<5} {cm[0][1]:<5}")
    print(f"         Pos    {cm[1][0]:<5} {cm[1][1]:<5}")
    print(f"  Separation: {separation:.4f}")

    # 4. Cross-validation
    n_folds = min(args.cv_folds, min(n_pos, n_neg))
    if n_folds >= 3:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.random_state)
        cv_svm = SVC(
            kernel=args.kernel,
            class_weight="balanced",
            random_state=args.random_state,
            gamma="scale",
        )
        cv_scores = cross_val_score(cv_svm, X, y, cv=cv, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        print(f"\n  CV ({n_folds}-fold): accuracy = {cv_mean:.4f} +/- {cv_std:.4f}")
    else:
        cv_mean = cv_std = 0.0
        print("\n  CV skipped (too few per class)")

    # 5. Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "svm": svm,
        "embed_dim": EMBED_DIM,
        "model_name": MODEL_NAME,
        "kernel": args.kernel,
        "class_weight": "balanced",
        "n_train": len(X),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "full_accuracy": float(full_acc),
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "separation": separation,
        "manifest_source": str(manifest_path),
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    summary_path = model_path.with_suffix(".summary.json")
    summary_data = {k: v for k, v in model_data.items() if k != "svm"}
    summary_path.write_text(json.dumps(summary_data, indent=2))

    print(f"\n  Model saved:   {model_path}")
    print(f"  Summary saved: {summary_path}")
    print(f"\n{'=' * 60}")
    print(f"  Summary: SVM {args.kernel.upper()} | CV {cv_mean:.1%} +/- {cv_std:.1%} | "
          f"Full {full_acc:.1%} | Sep {separation:.4f}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify no syntax errors**

Run: `python3 -c "import py_compile; py_compile.compile('scripts/train_from_manifest.py', doraise=True)"`

Expected: no error

---

### Task 2: Run retraining

**Files:** `scripts/train_from_manifest.py`

- [ ] **Execute training**

Run:
```bash
source .venv/bin/activate && python scripts/train_from_manifest.py \
    --manifest data/samples/training_manifest.json \
    --positive-dir data/samples/positive \
    --negative-dir data/samples/negative \
    --output-model data/models/svm_8pos_126neg.pkl \
    --kernel rbf \
    --cv-folds 5
```

Expected: Training completes in 1-3 minutes, prints CV accuracy and separation metrics.

- [ ] **Verify model file was created**

Run: `python3 -c "
import pickle
d = pickle.load(open('data/models/svm_8pos_126neg.pkl', 'rb'))
print(f'Training samples: {d[\"n_train\"]} ({d[\"n_pos\"]} pos + {d[\"n_neg\"]} neg)')
print(f'CV accuracy: {d[\"cv_accuracy_mean\"]:.4f} +/- {d[\"cv_accuracy_std\"]:.4f}')
print(f'Separation: {d[\"separation\"]:.4f}')
print(f'Full accuracy: {d[\"full_accuracy\"]:.4f}')
"`

Expected: Model has 134 training samples, metrics reported.

---

### Task 3: Run BC coast scan with retrained SVM

**Files:** `scripts/scan_bc_coast.py`, `data/models/svm_8pos_126neg.pkl`

- [ ] **Create output directory**

Run: `mkdir -p data/candidates_svm_8pos`

Expected: Directory created.

- [ ] **Run the BC coast scan**

Run:
```bash
source .venv/bin/activate && python scripts/scan_bc_coast.py \
    --output data/candidates_svm_8pos \
    --threshold 0.0 \
    --start 2024-02-01 \
    --end 2024-05-31 \
    --max-cloud 50 \
    --grid-spacing 0.02 \
    --workers 6 \
    --classifier svm \
    --svm-model data/models/svm_8pos_126neg.pkl
```

Expected: Scan processes all grid points, saves candidates above threshold 0.0.

---

### Task 4: Write review page generator

**Files:**
- Create: `scripts/generate_review_page.py`

This script reads the manifest.json from the candidates directory, sorts entries by score descending, and generates a review.html with thumbnail previews.

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""Generate a sortable review HTML page from a candidates manifest.json.

Usage:
    python scripts/generate_review_page.py \
        --candidates-dir data/candidates_svm_8pos \
        --port 8774
"""
import argparse
import json
import sys
from pathlib import Path


def generate_review_page(candidates_dir: Path) -> str:
    """Generate HTML review page from candidate manifest and return file path."""
    manifest_path = candidates_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: No manifest.json found in {candidates_dir}")
        return ""

    entries = json.loads(manifest_path.read_text())
    if not entries:
        print("ERROR: Manifest is empty")
        return ""

    # Sort by score descending
    entries_sorted = sorted(entries, key=lambda e: e.get("score", 0), reverse=True)

    # Build cards
    cards = []
    for e in entries_sorted:
        thumb = e.get("thumbnail_path", "")
        score = e.get("score", 0)
        region = e.get("region", "?")
        lat = e.get("lat", 0)
        lon = e.get("lon", 0)
        date = e.get("date", "?")
        scene_id = e.get("scene_id", "?")
        cloud = e.get("cloud", "?")
        # Color class based on score
        score_class = "high" if score > 0.5 else ("mid" if score > 0.2 else "low")

        cards.append(f"""
    <div class="card {score_class}">
        <img src="{thumb}" alt="" loading="lazy" onclick="toggleZoom(this)">
        <div class="body">
            <div class="region">{region}</div>
            <div class="coords">{lat:.4f}, {lon:.4f}</div>
            <div class="meta">{date} &middot; {scene_id[:30]} &middot; cloud {cloud}%</div>
            <div class="score">Score: {score:.4f}</div>
        </div>
    </div>""")

    cards_html = "\n".join(cards)

    # Count top scores
    high_count = sum(1 for e in entries_sorted if e.get("score", 0) > 0.5)
    mid_count = sum(1 for e in entries_sorted if 0.2 < e.get("score", 0) <= 0.5)
    low_count = sum(1 for e in entries_sorted if e.get("score", 0) <= 0.2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Herring Spawn Candidates — SVM 8pos+126neg Review</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #0d0d1a; color: #eee; }}
.bar {{ background: #1a1a2e; padding: 14px 20px; position: sticky; top: 0; z-index: 99; border-bottom: 1px solid #333; }}
.bar h1 {{ margin: 0; font-size: 20px; }}
.bar .sub {{ font-size: 13px; color: #888; margin-top: 4px; }}
.g {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 12px; padding: 12px; }}
.card {{ background: #1a1a2e; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.5); cursor: pointer; }}
.high {{ border-left: 5px solid #00E676; }}
.mid {{ border-left: 5px solid #FFD740; }}
.low {{ border-left: 5px solid #FF5252; }}
.card img {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; transition: transform 0.2s; }}
.card img.zoomed {{ transform: scale(2); transform-origin: top left; position: relative; z-index: 10; }}
.body {{ padding: 10px 14px; }}
.region {{ font-weight: 700; font-size: 15px; color: #fff; }}
.coords {{ font-size: 12px; color: #888; }}
.meta {{ font-size: 11px; color: #666; margin-top: 2px; }}
.score {{ margin-top: 6px; font-size: 14px; font-weight: 600; }}
.high .score {{ color: #00E676; }}
.mid .score {{ color: #FFD740; }}
.low .score {{ color: #FF5252; }}
.summary {{ background: #1a1a2e; margin: 12px; padding: 16px 20px; border-radius: 8px; font-size: 13px; }}
.summary-stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; }}
.stat {{ text-align: center; padding: 8px 16px; background: #0a0a18; border-radius: 8px; min-width: 100px; }}
.stat .num {{ font-size: 24px; font-weight: 700; color: #fff; }}
.stat .lbl {{ font-size: 11px; color: #888; text-transform: uppercase; }}
.stat.high .num {{ color: #00E676; }}
.stat.mid .num {{ color: #FFD740; }}
.stat.low .num {{ color: #FF5252; }}
.sort-bar {{ display: flex; gap: 8px; padding: 8px 12px; background: #151528; align-items: center; }}
.sort-bar label {{ font-size: 12px; color: #888; }}
.sort-bar select {{ background: #1a1a2e; color: #eee; border: 1px solid #333; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
#search {{ flex: 1; max-width: 300px; background: #1a1a2e; color: #eee; border: 1px solid #333; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>
<div class="bar">
    <h1>🦈 Herring Spawn Candidates — SVM (8 pos + 126 neg)</h1>
    <div class="sub">{len(entries_sorted)} candidates from {len(set(e.get("region","?") for e in entries_sorted))} regions &middot; DINOv2+SVM RBF &middot; Sorted by score</div>
</div>
<div class="summary">
    <strong>Score distribution:</strong>
    <div class="summary-stats">
        <div class="stat high"><div class="num">{high_count}</div><div class="lbl">High (&gt;0.5)</div></div>
        <div class="stat mid"><div class="num">{mid_count}</div><div class="lbl">Mid (0.2-0.5)</div></div>
        <div class="stat low"><div class="num">{low_count}</div><div class="lbl">Low (&le;0.2)</div></div>
    </div>
</div>
<div class="sort-bar">
    <label for="sortSelect">Sort:</label>
    <select id="sortSelect" onchange="sortCards()">
        <option value="score-desc">Score ↓</option>
        <option value="score-asc">Score ↑</option>
        <option value="region">Region A-Z</option>
        <option value="date">Date</option>
    </select>
    <label for="filterSelect">Filter:</label>
    <select id="filterSelect" onchange="filterCards()">
        <option value="all">All</option>
        <option value="high">High (&gt;0.5)</option>
        <option value="mid">Mid (0.2-0.5)</option>
        <option value="low">Low (≤0.2)</option>
    </select>
    <input type="text" id="search" placeholder="Search region/coords..." oninput="filterCards()">
    <span style="font-size:12px;color:#888;" id="count">{len(entries_sorted)} shown</span>
</div>
<div class="g" id="grid">
{cards_html}
</div>
<script>
function toggleZoom(img) {{
    img.classList.toggle('zoomed');
}}
function sortCards() {{
    const grid = document.getElementById('grid');
    const cards = Array.from(grid.children);
    const sortVal = document.getElementById('sortSelect').value;
    cards.sort((a, b) => {{
        const sA = parseFloat(a.querySelector('.score').textContent.replace('Score: ', ''));
        const sB = parseFloat(b.querySelector('.score').textContent.replace('Score: ', ''));
        const rA = a.querySelector('.region').textContent;
        const rB = b.querySelector('.region').textContent;
        const dA = a.querySelector('.meta').textContent.split('·')[0].trim();
        const dB = b.querySelector('.meta').textContent.split('·')[0].trim();
        if (sortVal === 'score-desc') return sB - sA;
        if (sortVal === 'score-asc') return sA - sB;
        if (sortVal === 'region') return rA.localeCompare(rB);
        if (sortVal === 'date') return dA.localeCompare(dB);
    }});
    cards.forEach(c => grid.appendChild(c));
}}
function filterCards() {{
    const filter = document.getElementById('filterSelect').value;
    const search = document.getElementById('search').value.toLowerCase();
    const cards = document.querySelectorAll('.card');
    let shown = 0;
    cards.forEach(c => {{
        const score = parseFloat(c.querySelector('.score').textContent.replace('Score: ', ''));
        const text = c.textContent.toLowerCase();
        const cls = score > 0.5 ? 'high' : (score > 0.2 ? 'mid' : 'low');
        const matchFilter = filter === 'all' || cls === filter;
        const matchSearch = !search || text.includes(search);
        c.style.display = (matchFilter && matchSearch) ? '' : 'none';
        if (matchFilter && matchSearch) shown++;
    }});
    document.getElementById('count').textContent = shown + ' shown';
}}
</script>
</body>
</html>"""

    review_path = candidates_dir / "review.html"
    review_path.write_text(html, encoding="utf-8")
    print(f"Review page generated: {review_path}")
    print(f"  {len(entries_sorted)} candidates")
    print(f"  High (>0.5): {high_count}")
    print(f"  Mid (0.2-0.5): {mid_count}")
    print(f"  Low (≤0.2): {low_count}")
    return str(review_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate review HTML from candidate manifest")
    parser.add_argument("--candidates-dir", default="data/candidates_svm_8pos")
    parser.add_argument("--port", type=int, default=8774, help="Port for serving (informational)")
    args = parser.parse_args(argv)

    candidates_dir = Path(args.candidates_dir).resolve()
    if not candidates_dir.exists():
        print(f"ERROR: Candidates directory not found: {candidates_dir}")
        return 1

    result = generate_review_page(candidates_dir)
    if not result:
        return 1

    print(f"\nTo serve:")
    print(f"  python -m http.server {args.port} --directory {candidates_dir.parent}")
    print(f"  http://localhost:{args.port}/candidates_svm_8pos/review.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import py_compile; py_compile.compile('scripts/generate_review_page.py', doraise=True)"`

Expected: no error

---

### Task 5: Generate review page and serve

**Files:** `scripts/generate_review_page.py`, `data/candidates_svm_8pos/`

- [ ] **Run review page generator**

Run:
```bash
source .venv/bin/activate && python scripts/generate_review_page.py \
    --candidates-dir data/candidates_svm_8pos \
    --port 8774
```

Expected: review.html generated with all candidates sorted by score.

- [ ] **Start HTTP server**

Run: `python -m http.server 8774 --directory data`

Expected: server accessible at http://localhost:8774/candidates_svm_8pos/review.html

- [ ] **Run final summary**

Run: `python3 -c "
import json
m = json.loads(open('data/candidates_svm_8pos/manifest.json').read())
scores = [e['score'] for e in m]
print(f'Total candidates: {len(m)}')
print(f'Score range: {min(scores):.4f} to {max(scores):.4f}')
print(f'Mean score: {sum(scores)/len(scores):.4f}')
print(f'Top 5:')
for e in sorted(m, key=lambda x: x['score'], reverse=True)[:5]:
    print(f'  {e[\"region\"]} ({e[\"lat\"]:.4f}, {e[\"lon\"]:.4f}) | {e[\"date\"]} | score={e[\"score\"]:.4f}')
# Count by region
from collections import Counter
regions = Counter(e['region'] for e in m)
print(f'\\nBy region:')
for r, c in regions.most_common():
    print(f'  {r}: {c}')
"`

Expected: Full summary printed.
