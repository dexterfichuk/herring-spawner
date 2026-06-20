#!/usr/bin/env python3
"""Zero-shot herring spawn detection using RemoteCLIP (ViT-L-14).

Downloads the RemoteCLIP checkpoint from HuggingFace (chendelong/RemoteCLIP)
and uses OpenAI CLIP ViT-L-14 as the base architecture. Scores satellite
thumbnails against hardcoded positive and negative text prompts.

Usage:
    # Score all PNGs in a directory
    python scripts/remoteclip_zero_shot.py \\
        --image-dir data/samples/positive \\
        --output-json data/remoteclip_results.json

    # Validate against human labels
    python scripts/remoteclip_zero_shot.py \\
        --image-dir data/samples/ \\
        --labels-json data/samples/labels.json \\
        --output-json data/remoteclip_results.json

    # Validate only (skip scoring individual images)
    python scripts/remoteclip_zero_shot.py \\
        --validate-only \\
        --image-dir data/samples/ \\
        --labels-json data/samples/labels.json

    # Few-shot classification on labeled images
    python scripts/remoteclip_zero_shot.py --mode fewshot \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --output-json data/remoteclip_fewshot_results.json

Dependencies:
    pip install open-clip-torch huggingface-hub torch torchvision Pillow scikit-learn tqdm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ARCH = "ViT-L-14"
REMOTECLIP_REPO = "chendelong/RemoteCLIP"
REMOTECLIP_CKPT = "RemoteCLIP-ViT-L-14.pt"
EMBED_DIM = 768
CACHE_DIR = "data/models/remoteclip"
EMBEDDINGS_CACHE_PATH = "data/embeddings/remoteclip_embeddings.npz"

POSITIVE_PROMPTS = [
    "a satellite image of turquoise milky water from herring spawning in coastal waters",
    "shallow coastal bay with bright turquoise discoloration indicating fish spawn",
    "milky blue-green plume hugging a shoreline in a sheltered inlet",
    "satellite view of bright cyan water discoloration along a coast during spawning season",
    "pale turquoise surface water in a coastal fjord from dense fish eggs",
    "white-blue milky patches in nearshore water contrasting with darker ocean",
    "aerial view of coastal bay with opaque turquoise streaks from herring milt",
    "bright aquamarine water along a Pacific coastline, biological bloom or fish spawn",
    "light blue turbid water concentrated in a sheltered cove, distinct from surrounding dark ocean",
    "satellite imagery showing extensive turquoise surface slick in a spawning ground",
    "translucent milky white-blue water in a narrow coastal embayment",
    "coastal water with bright pastel blue discoloration typical of mass fish spawning events",
]

NEGATIVE_PROMPTS = [
    "a satellite image of clear blue ocean water along a rocky coastline",
    "dark deep ocean water with no visible biological activity",
    "coastal shoreline with typical green or brown kelp beds and dark water",
    "sediment plume showing brown and grey turbid water from river outflow",
    "open ocean water with uniform dark navy surface and no discoloration",
    "typical coastal water with moderate wave patterns and no surface anomalies",
    "satellite view of a sandy beach with normal blue water and breaking surf",
    "estuary outflow with tan and brown sediment mixing into blue ocean water",
    "greenish algae bloom in a lake or coastal area, distinct from milky spawn water",
    "ocean water with whitecaps and wind streaks, no unusual coloration near shore",
    "cloud shadows and sunglint on dark ocean surface obscuring any water features",
    "coastal water showing typical chlorophyll pattern with diffuse green tint, not bright turquoise",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu', pass through others."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _download_remoteclip_ckpt(cache_dir: str | Path) -> Path:
    """Download the RemoteCLIP-ViT-L-14 checkpoint via huggingface_hub."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=REMOTECLIP_REPO,
        filename=REMOTECLIP_CKPT,
        cache_dir=str(cache_path),
    )
    return Path(local_path)


def load_model(device: str = "auto") -> tuple:
    """Load RemoteCLIP model, preprocess transform, and tokenizer.

    Args:
        device: 'auto', 'cuda', or 'cpu'.

    Returns:
        (model, preprocess, tokenize) — the model in eval mode, the
        OpenCLIP image transform pipeline, and the tokenizer callable.
    """
    resolved = _resolve_device(device)
    print(f"  RemoteCLIP device: {resolved}")

    # 1. Build base architecture from open_clip
    print(f"  Creating {MODEL_ARCH} base model (pretrained='openai')...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_ARCH, pretrained="openai"
    )
    tokenize = open_clip.get_tokenizer(MODEL_ARCH)

    # 2. Download and load RemoteCLIP weights
    print(f"  Downloading RemoteCLIP checkpoint ({REMOTECLIP_CKPT})...")
    ckpt_path = _download_remoteclip_ckpt(CACHE_DIR)
    print(f"  Loading weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=resolved, weights_only=True)

    # Handle DataParallel-wrapped checkpoints (keys prefixed with 'module.')
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (typically unused heads)")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys in checkpoint")

    model.eval()
    model = model.to(resolved)
    print(f"  Model loaded: {MODEL_ARCH} ({EMBED_DIM}-dim embeddings)")

    return model, preprocess, tokenize


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def get_text_embedding(
    model, tokenize, texts: list[str], device: str
) -> torch.Tensor:
    """Tokenize and encode a list of text prompts.

    Returns a normalized (N, embed_dim) tensor.
    """
    tokens = tokenize(texts).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb


def get_image_embedding(
    model, preprocess, image_path: str, device: str
) -> np.ndarray | None:
    """Load and preprocess an image, encode with model.

    Returns a normalized 1-D numpy array, or None on failure.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    try:
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensor)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_image(
    model,
    preprocess,
    tokenize,
    image_path: str,
    positive_texts: list[str],
    negative_texts: list[str],
    device: str,
) -> dict | None:
    """Score a single image against positive and negative text prompts.

    Returns a dict with keys:
        pos_scores, neg_scores, score, pos_mean, neg_mean, prediction, image_path
    or None if the image could not be loaded.
    """
    # Get image embedding
    img_emb = get_image_embedding(model, preprocess, image_path, device)
    if img_emb is None:
        return None

    img_tensor = torch.from_numpy(img_emb).to(device)

    # Get text embeddings (cached per batch — compute all at once)
    all_texts = positive_texts + negative_texts
    text_emb = get_text_embedding(model, tokenize, all_texts, device)  # (N, D)

    # Cosine similarities (all at once)
    sims = text_emb @ img_tensor  # (N,) — dot product of unit vectors = cosine

    n_pos = len(positive_texts)
    pos_sims = sims[:n_pos].cpu().tolist()
    neg_sims = sims[n_pos:].cpu().tolist()

    pos_mean = float(np.mean(pos_sims))
    neg_mean = float(np.mean(neg_sims))
    score_val = pos_mean - neg_mean
    prediction = 1 if score_val > 0 else 0

    return {
        "pos_scores": pos_sims,
        "neg_scores": neg_sims,
        "score": score_val,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "prediction": prediction,
        "image_path": image_path,
    }


def score_directory(
    image_dir: str, device: str = "auto", batch_size: int = 16
) -> list[dict]:
    """Score all PNG images in a directory against default prompts.

    Returns list of score dicts sorted by score descending.
    """
    resolved = _resolve_device(device)
    print(f"  Scoring directory: {image_dir}")
    model, preprocess, tokenize = load_model(device=resolved)

    img_dir = Path(image_dir)
    pngs = sorted(img_dir.glob("*.png"))
    print(f"  Found {len(pngs)} PNG images")

    if not pngs:
        print("  No images to score.")
        return []

    results: list[dict] = []
    for p in tqdm(pngs, desc="Scoring", unit="img"):
        result = score_image(
            model, preprocess, tokenize,
            str(p), POSITIVE_PROMPTS, NEGATIVE_PROMPTS,
            device=resolved,
        )
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"  Scored {len(results)}/{len(pngs)} images successfully")
    return results


# ---------------------------------------------------------------------------
# Validation against human labels
# ---------------------------------------------------------------------------

def validate(
    labels_json_path: str, image_dir: str, device: str = "auto"
) -> dict:
    """Validate against human labels.

    Labels JSON format:
        {"labels": [{"filename": "image.png", "label": 1}, ...]}
    where label=1 means positive (spawn), label=0 means negative (no spawn).

    Returns dict with accuracy, best_accuracy, best_threshold, auc_roc,
    avg_precision, confusion_matrix, per_sample list, and counts.
    """
    resolved = _resolve_device(device)
    print(f"  Validate device: {resolved}")

    # Load labels
    labels_path = Path(labels_json_path)
    if not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        return {"error": f"Labels file not found: {labels_path}"}

    labels_data = json.loads(labels_path.read_text())
    label_entries = labels_data.get("labels", [])
    print(f"  Loaded {len(label_entries)} label entries")

    if not label_entries:
        return {
            "accuracy": 0.0,
            "best_accuracy": 0.0,
            "best_threshold": 0.0,
            "auc_roc": 0.0,
            "avg_precision": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "per_sample": [],
            "n_total": 0,
            "n_pos": 0,
            "n_neg": 0,
        }

    # Load model once
    model, preprocess, tokenize = load_model(device=resolved)

    img_dir = Path(image_dir)
    per_sample: list[dict] = []

    for entry in tqdm(label_entries, desc="Validating", unit="img"):
        fname = entry["filename"]
        true_label = entry["label"]
        img_path = img_dir / fname

        if not img_path.exists():
            print(f"  WARNING: Image not found: {img_path}")
            continue

        result = score_image(
            model, preprocess, tokenize,
            str(img_path), POSITIVE_PROMPTS, NEGATIVE_PROMPTS,
            device=resolved,
        )
        if result is None:
            continue

        per_sample.append({
            "filename": fname,
            "true_label": true_label,
            "prediction": result["prediction"],
            "score": result["score"],
            "pos_mean": result["pos_mean"],
            "neg_mean": result["neg_mean"],
            "pos_scores": result["pos_scores"],
            "neg_scores": result["neg_scores"],
        })

    if not per_sample:
        print("  No samples successfully scored.")
        return {
            "accuracy": 0.0,
            "best_accuracy": 0.0,
            "best_threshold": 0.0,
            "auc_roc": 0.0,
            "avg_precision": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "per_sample": [],
            "n_total": 0,
            "n_pos": 0,
            "n_neg": 0,
        }

    # Aggregate metrics
    y_true = np.array([s["true_label"] for s in per_sample])
    y_pred = np.array([s["prediction"] for s in per_sample])
    y_score = np.array([s["score"] for s in per_sample])

    n_total = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_total - n_pos

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Best accuracy via threshold sweep
    thresholds = np.linspace(y_score.min() - 0.1, y_score.max() + 0.1, 201)
    best_acc = 0.0
    best_thr = 0.0
    for thr in thresholds:
        thr_pred = (y_score > thr).astype(int)
        thr_acc = accuracy_score(y_true, thr_pred)
        if thr_acc > best_acc:
            best_acc = thr_acc
            best_thr = float(thr)

    # AUROC (requires at least one sample from each class)
    auroc = 0.0
    if n_pos > 0 and n_neg > 0:
        try:
            auroc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auroc = 0.0

    # Average precision
    ap = 0.0
    if n_pos > 0:
        try:
            ap = float(average_precision_score(y_true, y_score))
        except Exception:
            ap = 0.0

    print("\n  Validation results:")
    print(f"    Total: {n_total}  Pos: {n_pos}  Neg: {n_neg}")
    print(f"    Accuracy (thr=0):  {acc:.4f}")
    print(f"    Best accuracy:     {best_acc:.4f} @ thr={best_thr:.4f}")
    print(f"    AUROC:             {auroc:.4f}")
    print(f"    Avg Precision:     {ap:.4f}")
    print(f"    Confusion Matrix:  {cm}")

    return {
        "accuracy": acc,
        "best_accuracy": best_acc,
        "best_threshold": best_thr,
        "auc_roc": auroc,
        "avg_precision": ap,
        "confusion_matrix": cm,
        "per_sample": per_sample,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


# ---------------------------------------------------------------------------
# Few-shot helpers
# ---------------------------------------------------------------------------


def extract_embeddings(
    image_dir: str, device: str = "auto"
) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract RemoteCLIP image embeddings for all PNGs in a directory.

    Checks for a cached .npz file at ``EMBEDDINGS_CACHE_PATH`` first; if it
    exists the cached embeddings are returned without loading the model.

    Args:
        image_dir: Directory containing PNG images.
        device: 'auto', 'cuda', or 'cpu'.

    Returns:
        (embeddings, img_paths, errors) where:
        - embeddings: np.ndarray of shape (N, 768), dtype float32.
        - img_paths: list of str paths for successfully embedded images.
        - errors: list of str error messages for images that failed.
    """
    resolved = _resolve_device(device)

    # ---- Check cache ----
    cache_path = Path(EMBEDDINGS_CACHE_PATH)
    if cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        embeddings = loaded["embeddings"]
        img_paths = loaded["img_paths"].tolist() if loaded["img_paths"].ndim > 0 else []
        print(f"  Loaded {len(img_paths)} cached embeddings (shape {embeddings.shape})")
        return embeddings, img_paths, []

    # ---- Load model ----
    print(f"  Extracting RemoteCLIP embeddings from: {image_dir}")
    model, preprocess, tokenize = load_model(device=resolved)

    img_dir = Path(image_dir)
    pngs = sorted(img_dir.glob("*.png"))
    if not pngs:
        print("  No PNG images found.")
        return np.array([]), [], []

    print(f"  Found {len(pngs)} PNG images")

    embeddings: list[np.ndarray] = []
    img_paths: list[str] = []
    errors: list[str] = []

    for p in tqdm(pngs, desc="Extracting embeddings", unit="img"):
        emb = get_image_embedding(model, preprocess, str(p), device=resolved)
        if emb is not None:
            embeddings.append(emb)
            img_paths.append(str(p))
        else:
            errors.append(f"Failed to embed: {p.name}")

    if not embeddings:
        print("  No embeddings extracted successfully.")
        return np.array([]), [], errors

    emb_array = np.stack(embeddings).astype(np.float32)
    print(f"  Extracted {len(img_paths)} embeddings (shape {emb_array.shape})")

    # ---- Save cache ----
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=emb_array,
        img_paths=np.array(img_paths, dtype=object),
    )
    print(f"  Saved embeddings cache to {cache_path}")

    return emb_array, img_paths, errors


def train_few_shot(
    image_dir: str,
    labels_json_path: str,
    device: str = "auto",
    cv_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Train a logistic regression classifier on RemoteCLIP image embeddings
    using stratified cross-validation.

    Also trains an SVM (RBF) on the full dataset for comparison.

    Args:
        image_dir: Directory containing PNG images referenced in labels.
        labels_json_path: Path to JSON file with ``{"labels": [...]}`` structure.
            Each entry: ``{"filename": "image.png", "label": 1}``
            where label=1 means positive (spawn) and label=0 means negative.
        device: 'auto', 'cuda', or 'cpu'.
        cv_folds: Number of stratified CV folds.
        seed: Random seed for reproducibility.

    Returns:
        dict with CV metrics, full-train accuracy, and per-fold breakdown.
        See test expectations for the full key set.
    """
    resolved = _resolve_device(device)

    # ---- Load labels ----
    labels_path = Path(labels_json_path)
    if not labels_path.exists():
        msg = f"Labels file not found: {labels_path}"
        print(f"ERROR: {msg}")
        return {"error": msg}

    labels_data = json.loads(labels_path.read_text())
    label_entries = labels_data.get("labels", [])
    print(f"\n  Loaded {len(label_entries)} label entries")

    if not label_entries:
        msg = "No labels found in labels file"
        print(f"  WARNING: {msg}")
        return {"error": msg}

    # ---- Extract embeddings ----
    embeddings, img_paths, errors = extract_embeddings(image_dir, device=resolved)

    if errors:
        print(f"  WARNING: {len(errors)} images failed to embed")
        for e in errors[:5]:
            print(f"    {e}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    if len(img_paths) == 0:
        msg = "No embeddings extracted successfully"
        print(f"  ERROR: {msg}")
        return {"error": msg}

    # ---- Match labels to embeddings ----
    path_to_label: dict[str, int] = {}
    for entry in label_entries:
        # Search by basename
        fname = entry["filename"]
        matched = False
        for full_path in img_paths:
            if Path(full_path).name == fname:
                path_to_label[full_path] = entry["label"]
                matched = True
                break
        if not matched:
            print(f"  WARNING: Label references image not found: {fname}")

    if not path_to_label:
        msg = "No matching images found for labels"
        print(f"  ERROR: {msg}")
        return {"error": msg}

    # Build aligned X, y
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    used_paths: list[str] = []

    for p in img_paths:
        if p in path_to_label:
            idx = img_paths.index(p)
            X_list.append(embeddings[idx])
            y_list.append(path_to_label[p])
            used_paths.append(p)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)

    n_total = len(y)
    n_pos = int(y.sum())
    n_neg = n_total - n_pos
    print(f"\n  Aligned {n_total} samples ({n_pos} positive, {n_neg} negative)")

    if n_pos == 0 or n_neg == 0:
        msg = "Need at least one positive and one negative sample for CV"
        print(f"  ERROR: {msg}")
        return {"error": msg}

    if cv_folds > min(n_pos, n_neg):
        cv_folds = min(n_pos, n_neg)
        print(f"  Adjusted CV folds to {cv_folds} (limited by class size)")

    # ---- Stratified K-Fold CV with Logistic Regression ----
    print(f"\n  Running {cv_folds}-fold stratified CV with Logistic Regression...")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    fold_results: list[dict] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)

        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())

        fold_results.append({
            "fold": fold_idx,
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        })

    # Aggregate CV metrics
    cv_acc_mean = float(np.mean([f["accuracy"] for f in fold_results]))
    cv_acc_std = float(np.std([f["accuracy"] for f in fold_results]))
    cv_prec_mean = float(np.mean([f["precision"] for f in fold_results]))
    cv_recall_mean = float(np.mean([f["recall"] for f in fold_results]))
    cv_f1_mean = float(np.mean([f["f1"] for f in fold_results]))

    # ---- Train final model on ALL data ----
    final_clf = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=seed
    )
    final_clf.fit(X, y)
    y_full_pred = final_clf.predict(X)
    full_train_accuracy = float(accuracy_score(y, y_full_pred))

    print(f"\n  CV results ({cv_folds} folds):")
    print(f"    Accuracy:  {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
    print(f"    Precision: {cv_prec_mean:.4f}")
    print(f"    Recall:    {cv_recall_mean:.4f}")
    print(f"    F1:        {cv_f1_mean:.4f}")
    print(f"  Full-train accuracy: {full_train_accuracy:.4f}")

    # ---- Train SVM (RBF) on full data for comparison ----
    print("\n  Training SVM (RBF) on full dataset for comparison...")
    svm_clf = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
    svm_clf.fit(X, y)
    y_svm_pred = svm_clf.predict(X)
    svm_full_train_accuracy = float(accuracy_score(y, y_svm_pred))
    print(f"  SVM full-train accuracy: {svm_full_train_accuracy:.4f}")

    print()

    return {
        "cv_accuracy_mean": cv_acc_mean,
        "cv_accuracy_std": cv_acc_std,
        "cv_precision_mean": cv_prec_mean,
        "cv_recall_mean": cv_recall_mean,
        "cv_f1_mean": cv_f1_mean,
        "cv_folds": cv_folds,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "full_train_accuracy": full_train_accuracy,
        "svm_full_train_accuracy": svm_full_train_accuracy,
        "per_fold": fold_results,
        "classifier": "logistic",
        "embed_dim": EMBED_DIM,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-shot / few-shot herring spawn detection with RemoteCLIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory of PNG images to score, validate, or train on",
    )
    parser.add_argument(
        "--mode", type=str, default="zeroshot",
        choices=["zeroshot", "fewshot"],
        help="Classification mode: zeroshot (prompt-based) or fewshot (embedding classifier)",
    )
    parser.add_argument(
        "--labels-json", type=str, default=None,
        help=(
            "Path to validation / training labels JSON file. "
            "Required for --validate-only and --mode=fewshot."
        ),
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save output JSON results",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on (default: auto)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for image processing (default: 16)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip scoring all images, just run validation against labels",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of CV folds for few-shot training (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args(argv)

    resolved = _resolve_device(args.device)

    # ----- Required: --image-dir -----
    if not args.image_dir:
        parser.print_help()
        print("\nERROR: --image-dir is required")
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    given = Path(args.image_dir)
    img_dir = given if given.is_absolute() else repo_root / args.image_dir

    if not img_dir.is_dir():
        print(f"ERROR: Image directory not found: {img_dir}")
        return 1

    # ----- Handle --labels-json path -----
    labels_path = None
    if args.labels_json:
        lp = Path(args.labels_json)
        labels_path = repo_root / args.labels_json if not lp.is_absolute() else lp
        if not labels_path.exists():
            print(f"ERROR: Labels file not found: {labels_path}")
            return 1

    # ----- Few-shot mode -----
    if args.mode == "fewshot":
        if labels_path is None:
            print("ERROR: --mode=fewshot requires --labels-json")
            return 1

        print("=" * 60)
        print("  RemoteCLIP Few-Shot Training")
        print("=" * 60)
        result = train_few_shot(
            str(img_dir),
            str(labels_path),
            device=resolved,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )

        # If error, print and return non-zero
        if "error" in result:
            print(f"\nERROR: {result['error']}")
            if args.output_json:
                out_path = Path(args.output_json)
                if not out_path.is_absolute():
                    out_path = repo_root / args.output_json
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result, indent=2, default=str))
            return 1

    elif args.validate_only:
        # ----- Validate-only mode (zero-shot validation) -----
        if labels_path is None:
            print("ERROR: --validate-only requires --labels-json")
            return 1
        result = validate(str(labels_path), str(img_dir), device=resolved)
    else:
        # ----- Zero-shot scoring mode -----
        print("=" * 60)
        print("  RemoteCLIP Zero-Shot Herring Spawn Scoring")
        print("=" * 60)
        result = score_directory(str(img_dir), device=resolved, batch_size=args.batch_size)
        result = {
            "device": resolved,
            "model": f"{MODEL_ARCH} + RemoteCLIP",
            "n_images_scored": len(result),
            "positive_prompts": POSITIVE_PROMPTS,
            "negative_prompts": NEGATIVE_PROMPTS,
            "results": result,
        }

        if labels_path is not None:
            print("\n" + "=" * 60)
            print("  Running validation against labels...")
            val_result = validate(str(labels_path), str(img_dir), device=resolved)
            result["validation"] = val_result

    # ----- Save output -----
    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = repo_root / args.output_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Results saved to: {out_path}")
    else:
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
