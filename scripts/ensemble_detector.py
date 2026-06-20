#!/usr/bin/env python3
"""Zero-shot ensemble detector combining the best individual detectors
via score normalization and averaging.

Each detector produces a score on a different scale. This script normalizes
all scores to [0,1] using min-max scaling across the validation set, then
ensembles via weighted average. The ensemble smooths out individual weaknesses.

Ensembled methods:
  1. SubspaceAD (PCA on DINOv2 global embeddings) — AUROC 0.997
  2. SHSI 95th percentile — AUROC 0.745
  3. Spectral threshold (G-R)/(G+R) — AUROC 0.761
  4. RemoteCLIP zero-shot — AUROC 0.736

Usage:
    # Full ensemble validation with comparison table
    python scripts/ensemble_detector.py \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --output-json data/ensemble_results.json

    # Custom weights (subspace_ad, shsi_95p, spectral, remoteclip)
    python scripts/ensemble_detector.py \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --weights 0.4,0.2,0.2,0.2

    # Validate only (skip full per-image JSON output)
    python scripts/ensemble_detector.py --validate-only \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json

    # Custom SubspaceAD training dir and component count
    python scripts/ensemble_detector.py \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --subspace-train-dir data/samples/negative \\
        --subspace-n-components 64

Dependencies: torch, torchvision, Pillow, scikit-learn, numpy, open-clip-torch,
              huggingface-hub, tqdm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Individual method imports
# ---------------------------------------------------------------------------

# SubspaceAD — PCA reconstruction residual on DINOv2 embeddings
from scripts.subspace_ad import (
    train_on_negatives,
    score_image as score_subspace,
    _load_dinov2_model,
)

# SHSI — spectral herring spawning index (Green - 2*Red)
from scripts.shsi_detector import score_image as score_shsi

# Spectral threshold — (G - R) / (G + R) index
from scripts.spectral_threshold import score_image as score_spectral

# RemoteCLIP — zero-shot vision-language scoring
from scripts.remoteclip_zero_shot import (
    score_image as score_remoteclip,
    load_model as load_remoteclip,
    POSITIVE_PROMPTS,
    NEGATIVE_PROMPTS,
)

# DINOv2 constants
from scripts.train_classifier import DINO_TRANSFORM, MODEL_NAME, EMBED_DIM

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHODS = [
    "subspace_ad",
    "shsi_95p",
    "spectral",
    "remoteclip",
]

METHOD_LABELS: dict[str, str] = {
    "subspace_ad": "SubspaceAD (PCA residual)",
    "shsi_95p": "SHSI 95th percentile",
    "spectral": "Spectral (G-R)/(G+R)",
    "remoteclip": "RemoteCLIP zero-shot",
}

# Which key in each method's score dict carries the primary score
SCORE_KEYS: dict[str, str] = {
    "subspace_ad": "score",       # PCA reconstruction residual (higher = more anomalous)
    "shsi_95p": "score_95p",      # 95th percentile SHSI (higher = more spawn-like)
    "spectral": "score",          # Mean (G-R)/(G+R) (higher = more spawn-like)
    "remoteclip": "score",        # pos_mean - neg_mean (higher = more spawn-like)
}

KNOWN_AUROCS: dict[str, float] = {
    "subspace_ad": 0.997,
    "shsi_95p": 0.745,
    "spectral": 0.761,
    "remoteclip": 0.736,
}

DEFAULT_NEGATIVE_DIR = "data/samples/negative"

# Cache for raw per-method scores (avoids re-running all methods)
SCORE_CACHE_PATH = "data/embeddings/ensemble_raw_scores.npz"

EPS = 1e-8

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu', pass through others."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _compute_labels_hash(labels_json_path: str, image_dir: str) -> str:
    """Compute a content hash of labels + image filenames for cache invalidation.

    Combines the SHA-256 of the labels file content with the sorted list of
    PNG filenames in image_dir.
    """
    labels_path = Path(labels_json_path)
    labels_content = labels_path.read_bytes() if labels_path.exists() else b""

    img_dir = Path(image_dir)
    png_names = sorted(p.name.encode("utf-8") for p in img_dir.glob("*.png"))

    h = hashlib.sha256(labels_content)
    for name in png_names:
        h.update(name)
    return h.hexdigest()


def _load_cached_scores(data_hash: str) -> dict | None:
    """Load raw per-method scores from cache if the hash matches."""
    cache_path = Path(SCORE_CACHE_PATH)
    if not cache_path.exists():
        return None
    try:
        loaded = np.load(cache_path, allow_pickle=True)
        cached_hash = str(loaded.get("data_hash", b""))
        if cached_hash != data_hash:
            return None
        # Reconstruct per_method_scores dict from arrays
        methods = loaded.get("methods", [])
        if len(methods) == 0:
            return None
        per_method_scores: dict = {}
        for method in methods:
            method = str(method)
            arr = loaded[method]
            if arr.ndim == 0 or len(arr) == 0:
                per_method_scores[method] = []
            else:
                per_method_scores[method] = arr.tolist()
        return per_method_scores
    except Exception:
        return None


def _save_cached_scores(data_hash: str, per_method_scores: dict) -> None:
    """Save raw per-method scores to cache with data_hash."""
    cache_path = Path(SCORE_CACHE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict = {
        "data_hash": data_hash,
        "methods": np.array(list(per_method_scores.keys()), dtype=object),
    }
    for method, samples in per_method_scores.items():
        save_dict[method] = np.array(samples, dtype=object)

    np.savez_compressed(str(cache_path), **save_dict)
    print(f"  Cached raw scores to {cache_path}")


# ---------------------------------------------------------------------------
# DINOv2 embedding helper (single image)
# ---------------------------------------------------------------------------


def _dinov2_embedding(
    model: torch.nn.Module, image_path: str, device: str,
) -> np.ndarray | None:
    """Extract a single normalized DINOv2 embedding from an image file."""
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = DINO_TRANSFORM(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(tensor)
        emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
        return emb
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SubspaceAD helper for single-image scoring
# ---------------------------------------------------------------------------


def _score_subspace_image(
    pca_model: object,
    dinov2_model: torch.nn.Module,
    image_path: str,
    device: str,
) -> float:
    """Score a single image with SubspaceAD.

    Extracts DINOv2 embedding, then computes PCA reconstruction residual.
    Returns 0.0 on failure.
    """
    emb = _dinov2_embedding(dinov2_model, image_path, device)
    if emb is None:
        return 0.0
    try:
        result = score_subspace(pca_model, emb)
        return float(result["score"])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Run all methods
# ---------------------------------------------------------------------------


def run_all_methods(
    image_dir: str,
    labels_json: str,
    device: str = "auto",
    subspace_train_dir: str | None = None,
    subspace_n_components: int | None = None,
    no_cache: bool = False,
) -> dict:
    """Run all individual methods on the labeled validation set.

    For each labeled image, produces a score from each method.
    Results are cached to avoid re-running when labels haven't changed.

    Args:
        image_dir: Directory containing PNG images referenced in labels.
        labels_json: Path to labels JSON file (``{"labels": [...]}``).
        device: 'auto', 'cuda', or 'cpu'.
        subspace_train_dir: Directory of negative images for PCA training.
            Defaults to ``data/samples/negative``.
        subspace_n_components: PCA component count (None = auto-select).
        no_cache: If True, bypass cached scores.

    Returns:
        dict mapping method_name -> list of per-sample score dicts.
        Each per-sample score dict contains at minimum:
            'filename', 'true_label', and the method's primary score key
            (e.g. 'score', 'score_95p').
    """
    resolved = _resolve_device(device)

    # ---- Load labels ----
    labels_path = Path(labels_json)
    if not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        return {}

    labels_data = json.loads(labels_path.read_text())
    label_entries = labels_data.get("labels", [])
    print(f"  Loaded {len(label_entries)} label entries from {labels_json}")

    if not label_entries:
        print("  No label entries found.")
        return {}

    # ---- Check cache ----
    data_hash = _compute_labels_hash(labels_json, image_dir)
    if not no_cache:
        cached = _load_cached_scores(data_hash)
        if cached is not None:
            print(f"  Loaded cached raw scores (hash={data_hash[:12]}...)")
            # Verify all methods present
            if all(m in cached for m in METHODS):
                return cached
            print("  Cache incomplete — re-running all methods")

    print("  Running all detection methods on validation set...")

    # ---- Resolve paths ----
    img_dir = Path(image_dir)
    repo_root = Path(__file__).resolve().parent.parent

    def _resolve(given: str | None, default: str | None = None) -> Path | None:
        p = Path(given) if given else (Path(default) if default else None)
        if p is None:
            return None
        return p if p.is_absolute() else (repo_root / str(p))

    resolved_img_dir = _resolve(image_dir)
    if resolved_img_dir is None or not resolved_img_dir.is_dir():
        print(f"ERROR: Image directory not found: {image_dir}")
        return {}

    # ---- Load models once ----
    print(f"\n  {'─' * 50}")
    print(f"  Loading models...")
    print(f"  Device: {resolved}")

    # DINOv2 for SubspaceAD
    print(f"  Loading DINOv2 ({MODEL_NAME})...")
    dinov2_model = _load_dinov2_model(resolved)

    # RemoteCLIP
    print(f"  Loading RemoteCLIP...")
    remoteclip_model, remoteclip_preprocess, remoteclip_tokenize = load_remoteclip(
        device=resolved,
    )

    # ---- Train SubspaceAD PCA on negatives ----
    neg_dir = _resolve(subspace_train_dir, DEFAULT_NEGATIVE_DIR)
    if neg_dir is None or not neg_dir.is_dir():
        print(f"WARNING: Negative training directory not found: {neg_dir}")
        print("  SubspaceAD will use a zero-score fallback for all images.")
        pca_model = None
    else:
        print(f"\n  {'─' * 50}")
        print(f"  Training SubspaceAD PCA on negatives from: {neg_dir}")
        train_result = train_on_negatives(
            str(neg_dir),
            n_components=subspace_n_components,
            device=resolved,
        )
        if "error" in train_result:
            print(f"  WARNING: SubspaceAD training failed: {train_result['error']}")
            pca_model = None
        else:
            pca_model = train_result["pca_model"]
            n_comp = train_result["n_components"]
            n_neg = train_result["n_negative_images"]
            print(f"  Trained PCA with {n_comp} components on {n_neg} negatives")

    # ---- Process each labeled image ----
    print(f"\n  {'─' * 50}")
    print(f"  Scoring {len(label_entries)} labeled images with all 4 methods...")

    # Initialize per-method score lists
    per_method_scores: dict[str, list[dict]] = {m: [] for m in METHODS}

    for entry in label_entries:
        fname = entry["filename"]
        true_label = entry["label"]
        img_path = resolved_img_dir / fname

        if not img_path.exists():
            print(f"  WARNING: Image not found: {img_path}")
            # Still add entries with zero scores to keep alignment
            for m in METHODS:
                per_method_scores[m].append({
                    "filename": fname,
                    "true_label": true_label,
                    SCORE_KEYS[m]: 0.0,
                    "prediction": 0,
                })
            continue

        # --- SubspaceAD ---
        if pca_model is not None:
            subspace_result = _score_subspace_image(
                pca_model, dinov2_model, str(img_path), resolved,
            )
        else:
            subspace_result = 0.0
        per_method_scores["subspace_ad"].append({
            "filename": fname,
            "true_label": true_label,
            "score": subspace_result,
            "prediction": 1 if subspace_result > 0.00015 else 0,
        })

        # --- SHSI 95p ---
        try:
            shsi_result = score_shsi(str(img_path))
            shsi_score = shsi_result["score_95p"]
            shsi_pred = shsi_result["prediction"]
        except Exception as exc:
            print(f"  WARNING: SHSI failed for {fname}: {exc}")
            shsi_score = 0.0
            shsi_pred = 0
        per_method_scores["shsi_95p"].append({
            "filename": fname,
            "true_label": true_label,
            "score_95p": shsi_score,
            "score": shsi_result.get("score", 0.0) if isinstance(shsi_result, dict) else 0.0,
            "prediction": shsi_pred,
        })

        # --- Spectral ---
        try:
            spectral_result = score_spectral(str(img_path))
            spectral_score = spectral_result["score"]
            spectral_pred = spectral_result["prediction"]
        except Exception as exc:
            print(f"  WARNING: Spectral failed for {fname}: {exc}")
            spectral_score = 0.0
            spectral_pred = 0
        per_method_scores["spectral"].append({
            "filename": fname,
            "true_label": true_label,
            "score": spectral_score,
            "prediction": spectral_pred,
        })

        # --- RemoteCLIP ---
        try:
            clip_result = score_remoteclip(
                remoteclip_model, remoteclip_preprocess, remoteclip_tokenize,
                str(img_path), POSITIVE_PROMPTS, NEGATIVE_PROMPTS,
                device=resolved,
            )
            if clip_result is not None:
                clip_score = clip_result["score"]
                clip_pred = clip_result["prediction"]
            else:
                clip_score = 0.0
                clip_pred = 0
        except Exception as exc:
            print(f"  WARNING: RemoteCLIP failed for {fname}: {exc}")
            clip_score = 0.0
            clip_pred = 0
        per_method_scores["remoteclip"].append({
            "filename": fname,
            "true_label": true_label,
            "score": clip_score,
            "prediction": clip_pred,
        })

    # Verify all methods scored the same number of samples
    n_samples = len(label_entries)
    for m in METHODS:
        actual = len(per_method_scores[m])
        if actual != n_samples:
            print(f"  WARNING: {m} scored {actual}/{n_samples} samples")

    counts = {m: len(per_method_scores[m]) for m in METHODS}
    print(f"\n  Scored samples per method: {counts}")

    # ---- Cache ----
    if not no_cache:
        _save_cached_scores(data_hash, per_method_scores)

    return per_method_scores


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------


def normalize_scores(per_method_scores: dict, method: str) -> np.ndarray:
    """Min-max normalize scores to [0, 1] for a given method.

    Args:
        per_method_scores: Dict from ``run_all_methods()``.
        method: One of ``METHODS``.

    Returns:
        ndarray of normalized scores in [0, 1] (same order as samples).
    """
    score_key = SCORE_KEYS.get(method, "score")
    samples = per_method_scores.get(method, [])
    scores = np.array([s.get(score_key, 0.0) for s in samples], dtype=np.float64)

    s_min = scores.min()
    s_max = scores.max()

    if s_max - s_min < EPS:
        return np.zeros_like(scores)

    normalized = (scores - s_min) / (s_max - s_min + EPS)
    return normalized


# ---------------------------------------------------------------------------
# Ensemble combination
# ---------------------------------------------------------------------------


def ensemble_score(
    normalized_scores: np.ndarray,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Weighted average of normalized scores.

    Args:
        normalized_scores: Array of shape (n_methods, n_samples) with
            scores normalized to [0, 1].
        weights: Optional list of weights (same order as METHODS).
            Defaults to equal weights.

    Returns:
        ndarray of shape (n_samples,) — ensemble scores in [0, 1].
    """
    n_methods = normalized_scores.shape[0]

    if weights is None:
        weights = [1.0 / n_methods] * n_methods
    else:
        weights = np.array(weights, dtype=np.float64)
        weights = weights / weights.sum()  # renormalize to sum=1

    ensemble = np.average(normalized_scores, axis=0, weights=weights)
    return ensemble


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute accuracy, best-accuracy, AUROC, avg precision, confusion matrix.

    Args:
        y_true: Ground truth binary labels.
        y_score: Continuous scores (higher = more spawn-like).

    Returns:
        dict with keys: accuracy, best_accuracy, best_threshold, auc_roc,
        avg_precision, confusion_matrix.
    """
    n_total = len(y_true)
    n_pos = int(y_true.sum()) if n_total > 0 else 0
    n_neg = n_total - n_pos

    if n_total == 0:
        return {
            "accuracy": 0.0,
            "best_accuracy": 0.0,
            "best_threshold": 0.0,
            "auc_roc": 0.0,
            "avg_precision": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "n_total": 0,
            "n_pos": 0,
            "n_neg": 0,
        }

    # Default prediction (threshold = 0 for methods where 0 is baseline,
    # or 0.5 for ensemble where scores are normalized to [0,1])
    threshold = 0.5
    y_pred = (y_score > threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Best accuracy via threshold sweep
    if y_score.min() == y_score.max():
        best_acc = acc
        best_thr = float(y_score[0]) if len(y_score) > 0 else 0.0
    else:
        thresholds = np.linspace(
            y_score.min() - 0.1 * max(1.0, abs(y_score.min())),
            y_score.max() + 0.1 * max(1.0, abs(y_score.max())),
            201,
        )
        best_acc = 0.0
        best_thr = float(threshold)
        for thr in thresholds:
            thr_pred = (y_score > thr).astype(int)
            thr_acc = accuracy_score(y_true, thr_pred)
            if thr_acc > best_acc:
                best_acc = thr_acc
                best_thr = float(thr)

    # AUROC
    auroc = 0.0
    if n_pos > 0 and n_neg > 0 and not np.all(y_score == y_score[0]):
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

    return {
        "accuracy": acc,
        "threshold": threshold,
        "best_accuracy": best_acc,
        "best_threshold": best_thr,
        "auc_roc": auroc,
        "avg_precision": ap,
        "confusion_matrix": cm,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------


def _print_comparison_table(all_metrics: dict, weights: list[float]) -> None:
    """Print a formatted comparison table of all methods + ensemble.

    Args:
        all_metrics: dict mapping method name -> metrics dict from _compute_metrics.
            Includes an 'ensemble' key.
        weights: Weights used for ensemble.
    """
    sep = "─" * 78
    header = (
        f"  {'Method':<36s} {'Accuracy':>9s} {'AUROC':>8s} "
        f"{'Avg Prec':>9s} {'Best Acc':>9s}"
    )
    print(f"\n  {sep}")
    print(f"  Ensemble Detector — Method Comparison")
    print(f"  Ensemble weights: {[f'{w:.3f}' for w in weights]}")
    print(f"  {sep}")
    print(header)
    print(f"  {sep}")

    # Individual methods
    for method in METHODS:
        label = METHOD_LABELS.get(method, method)
        m = all_metrics.get(method, {})
        acc = m.get("accuracy", 0.0)
        auroc = m.get("auc_roc", 0.0)
        ap = m.get("avg_precision", 0.0)
        best = m.get("best_accuracy", 0.0)
        print(
            f"  {label:<36s} {acc:>9.4f} {auroc:>8.4f} "
            f"{ap:>9.4f} {best:>9.4f}"
        )

    # Divider
    print(f"  {sep}")

    # Ensemble
    em = all_metrics.get("ensemble", {})
    e_acc = em.get("accuracy", 0.0)
    e_auroc = em.get("auc_roc", 0.0)
    e_ap = em.get("avg_precision", 0.0)
    e_best = em.get("best_accuracy", 0.0)
    weights_str = "equal" if weights is None else f"w={[f'{w:.2f}' for w in weights]}"
    ensemble_label = f"Ensemble ({weights_str})"
    print(
        f"  {ensemble_label:<36s} {e_acc:>9.4f} {e_auroc:>8.4f} "
        f"{e_ap:>9.4f} {e_best:>9.4f}"
    )
    print(f"  {sep}")

    # Counts
    if em:
        print(f"  n_total={em.get('n_total', 0)}  "
              f"n_pos={em.get('n_pos', 0)}  "
              f"n_neg={em.get('n_neg', 0)}")
    print()


# ---------------------------------------------------------------------------
# Full ensemble validation
# ---------------------------------------------------------------------------


def validate_ensemble(
    labels_json_path: str,
    image_dir: str,
    device: str = "auto",
    weights: list[float] | None = None,
    subspace_train_dir: str | None = None,
    subspace_n_components: int | None = None,
    no_cache: bool = False,
) -> dict:
    """Run all methods, normalize scores, ensemble, and validate.

    Computes and returns metrics for each individual method plus the ensemble
    for comparison.

    Args:
        labels_json_path: Path to labels JSON.
        image_dir: Directory containing PNG images.
        device: 'auto', 'cuda', or 'cpu'.
        weights: Optional weight list in METHODS order (default: equal).
        subspace_train_dir: Directory of negatives for PCA training.
        subspace_n_components: PCA component count (None = auto).
        no_cache: If True, bypass score cache.

    Returns:
        dict with keys:
            'individual_methods': dict of method -> metrics
            'ensemble': ensemble metrics
            'weights': weights used
            'per_sample': list of per-sample dicts with all scores
            'n_total', 'n_pos', 'n_neg'
    """
    resolved = _resolve_device(device)

    print("=" * 78)
    print("  Ensemble Detector — Validate against human labels")
    print("=" * 78)
    print(f"  Labels:    {labels_json_path}")
    print(f"  Images:    {image_dir}")
    print(f"  Device:    {resolved}")

    # ---- Step 1: Run all methods ----
    print()
    per_method_scores = run_all_methods(
        image_dir=str(image_dir),
        labels_json=str(labels_json_path),
        device=resolved,
        subspace_train_dir=subspace_train_dir,
        subspace_n_components=subspace_n_components,
        no_cache=no_cache,
    )

    if not per_method_scores or not per_method_scores.get(METHODS[0]):
        print("ERROR: No scores computed.")
        return {"error": "No scores computed"}

    n_samples = len(per_method_scores[METHODS[0]])
    print(f"\n  Total samples: {n_samples}")

    # ---- Step 2: Normalize each method's scores ----
    print(f"\n  {'─' * 50}")
    print(f"  Normalizing scores per method...")
    normalized: dict[str, np.ndarray] = {}
    for method in METHODS:
        norm = normalize_scores(per_method_scores, method)
        normalized[method] = norm
        score_key = SCORE_KEYS[method]
        raw = np.array([s.get(score_key, 0.0) for s in per_method_scores[method]])
        print(f"    {METHOD_LABELS[method]:<36s} "
              f"raw range [{raw.min():.6f}, {raw.max():.6f}] → "
              f"norm range [{norm.min():.4f}, {norm.max():.4f}]")

    # ---- Step 3: Combine into ensemble ----
    print(f"\n  {'─' * 50}")
    print(f"  Computing ensemble...")

    if weights is not None:
        weights_arr = np.array(weights, dtype=np.float64)
        weights_arr = weights_arr / weights_arr.sum()
    else:
        weights_arr = np.array([1.0 / len(METHODS)] * len(METHODS))

    norm_stack = np.stack([normalized[m] for m in METHODS])  # (n_methods, n_samples)
    ensemble_scores = ensemble_score(norm_stack, weights=list(weights_arr))
    print(f"    Ensemble weights: {[f'{w:.4f}' for w in weights_arr]}")
    print(f"    Ensemble score range: [{ensemble_scores.min():.4f}, {ensemble_scores.max():.4f}]")

    # ---- Step 4: Compute metrics for each method + ensemble ----
    print(f"\n  {'─' * 50}")
    print(f"  Computing metrics...")

    all_metrics: dict = {}

    # Individual methods
    y_true = np.array([s["true_label"] for s in per_method_scores[METHODS[0]]])

    for method in METHODS:
        score_key = SCORE_KEYS[method]
        y_score = np.array([s.get(score_key, 0.0) for s in per_method_scores[method]])
        metrics = _compute_metrics(y_true, y_score)
        all_metrics[method] = metrics

    # Ensemble
    ensemble_metrics = _compute_metrics(y_true, ensemble_scores)
    all_metrics["ensemble"] = ensemble_metrics

    # ---- Step 5: Build per_sample list ----
    per_sample: list[dict] = []
    for i in range(n_samples):
        sample: dict = {
            "filename": per_method_scores[METHODS[0]][i]["filename"],
            "true_label": int(y_true[i]),
        }
        for method in METHODS:
            score_key = SCORE_KEYS[method]
            sample[f"{method}_score"] = float(
                per_method_scores[method][i].get(score_key, 0.0)
            )
            sample[f"{method}_norm"] = float(normalized[method][i])
        sample["ensemble_score"] = float(ensemble_scores[i])
        sample["ensemble_prediction"] = int(ensemble_scores[i] > 0.5)
        per_sample.append(sample)

    # ---- Step 6: Print comparison table ----
    _print_comparison_table(all_metrics, list(weights_arr))

    n_total = ensemble_metrics["n_total"]
    n_pos = ensemble_metrics["n_pos"]
    n_neg = ensemble_metrics["n_neg"]

    return {
        "individual_methods": {
            m: {k: v for k, v in all_metrics[m].items() if k != "per_sample"}
            for m in METHODS
        },
        "ensemble": {k: v for k, v in ensemble_metrics.items() if k != "per_sample"},
        "weights": [float(w) for w in weights_arr],
        "per_sample": per_sample,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-shot ensemble detector for herring spawn — combines "
                    "SubspaceAD, SHSI, Spectral, and RemoteCLIP via score "
                    "normalization and weighted averaging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image-dir", type=str, required=True,
        help="Directory of PNG images to score / validate",
    )
    parser.add_argument(
        "--labels-json", type=str, required=True,
        help="Path to labels JSON file (format: {labels: [{filename, label}, ...]})",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save output JSON results",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip per-sample JSON dump (still prints comparison table)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on (default: auto)",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Comma-separated weights for subspace_ad,shsi_95p,spectral,remoteclip "
             "(default: equal)",
    )
    parser.add_argument(
        "--subspace-train-dir", type=str, default=None,
        help="Directory of negative images for SubspaceAD PCA training "
             "(default: data/samples/negative)",
    )
    parser.add_argument(
        "--subspace-n-components", type=int, default=None,
        help="Number of PCA components for SubspaceAD (default: auto-select)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Bypass cached raw scores",
    )
    args = parser.parse_args(argv)

    # ---- Resolve paths ----
    repo_root = Path(__file__).resolve().parent.parent

    def _resolve(given: str) -> Path:
        p = Path(given)
        return p if p.is_absolute() else (repo_root / given)

    img_dir = _resolve(args.image_dir)
    labels_path = _resolve(args.labels_json)

    if not img_dir.is_dir():
        print(f"ERROR: Image directory not found: {img_dir}")
        return 1

    if not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        return 1

    # ---- Parse weights ----
    weights = None
    if args.weights:
        parts = [w.strip() for w in args.weights.split(",")]
        if len(parts) != len(METHODS):
            print(
                f"ERROR: --weights requires exactly {len(METHODS)} comma-separated "
                f"values (subspace_ad, shsi_95p, spectral, remoteclip)"
            )
            return 1
        try:
            weights = [float(w) for w in parts]
        except ValueError:
            print("ERROR: --weights must be comma-separated numeric values")
            return 1

    # ---- Resolve subspace train dir ----
    subspace_train_dir = None
    if args.subspace_train_dir:
        st_dir = _resolve(args.subspace_train_dir)
        if not st_dir.is_dir():
            print(f"ERROR: Subspace training directory not found: {st_dir}")
            return 1
        subspace_train_dir = str(st_dir)

    # ---- Run ensemble validation ----
    result = validate_ensemble(
        labels_json_path=str(labels_path),
        image_dir=str(img_dir),
        device=args.device,
        weights=weights,
        subspace_train_dir=subspace_train_dir,
        subspace_n_components=args.subspace_n_components,
        no_cache=args.no_cache,
    )

    if "error" in result:
        print(f"\nERROR: {result['error']}")
        return 1

    # ---- Save or print output ----
    if args.output_json:
        out_path = _resolve(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  Results saved to: {out_path}")
    elif not args.validate_only:
        # Full JSON dump to stdout
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
