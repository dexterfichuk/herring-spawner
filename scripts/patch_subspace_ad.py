#!/usr/bin/env python3
"""DINOv2 patch-level SubspaceAD for herring spawn detection.

Instead of using the CLS token (global 384-dim embedding), this extracts all
256 patch tokens (16x16 grid, each 384-dim) from DINOv2 ViT-S/14. These
capture local texture patterns. PCA is fit on pooled patch features from
negative (non-spawn) images. Candidates are scored by mean patch
reconstruction residual, focusing on the top-10% most anomalous patches.

The key insight: spawn events produce local texture anomalies that affect
only a few patches (e.g., milky turquoise water in a bay), while the rest
of the image remains "normal" coastal scenery. The top-10% residual mean
amplifies this signal.

Training:      PCA on pooled patch tokens from negative images.
Inference:     Score each candidate image by patch-level PCA residuals.

Usage:
    # Train on negatives, validate against human labels
    python scripts/patch_subspace_ad.py --validate-only \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --device cpu

    # Score a directory of candidates
    python scripts/patch_subspace_ad.py \\
        --train-dir data/samples/negative \\
        --image-dir data/candidates_knn \\
        --output-json data/patch_subspace_results.json

    # Train with explicit component count and sampling
    python scripts/patch_subspace_ad.py \\
        --train-dir data/samples/negative \\
        --image-dir data/candidates_knn \\
        --n-components 128 \\
        --sample-frac 0.2 \\
        --output-json data/patch_subspace_results.json

Dependencies: torch, torchvision, Pillow, scikit-learn, numpy, tqdm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

# Reuse DINOv2 transform, model name, and embed dim from train_classifier
from scripts.train_classifier import DINO_TRANSFORM, MODEL_NAME, EMBED_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DINOv2 ViT-S/14 produces a 16x16 grid of patch tokens
N_PATCHES = 256  # 16 * 16
PATCH_GRID_SIZE = 16  # 16x16

# PCA variance ratio target when auto-selecting n_components
AUTO_VARIANCE_TARGET = 0.90

# Default number of PCA components (used when auto-selection is disabled
# or when n_components > n_samples)
DEFAULT_N_COMPONENTS = 64

# Cache path for extracted patch embeddings
PATCH_EMBEDDINGS_CACHE_PATH = "data/embeddings/patch_subspace_embeddings.npz"

# Default prediction threshold (mean top-10% patch residual above this = spawn).
# Patch residuals tend to be larger than CLS residuals since patches are
# noisier. Default is set conservatively; the threshold sweep in validate()
# finds the optimal value automatically.
DEFAULT_PREDICTION_THRESHOLD = 0.0005

# Fraction of anomalous patches to consider when computing score
ANOMALOUS_PATCH_FRAC = 0.10  # top 10%


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu', pass through others."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_dinov2_model(device: str) -> torch.nn.Module:
    """Load DINOv2 ViT-S/14 from torchhub and put in eval mode."""
    print(f"  Loading DINOv2 model ({MODEL_NAME})...")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval()
    model = model.to(device)
    print(f"  Model loaded: {MODEL_NAME} ({EMBED_DIM}-dim embeddings, "
          f"{N_PATCHES} patch tokens)")
    return model


# ---------------------------------------------------------------------------
# Patch embedding extraction
# ---------------------------------------------------------------------------

def extract_patch_embeddings(
    image_dir: str, device: str = "auto", cache: bool = True,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract DINOv2 patch tokens (not CLS) for all PNGs in a directory.

    DINOv2 ViT-S/14 outputs 256 patch tokens (16x16 grid), each 384-dim.
    Uses ``model.get_intermediate_layers()`` to access per-patch features.

    Checks for cached embeddings at ``PATCH_EMBEDDINGS_CACHE_PATH`` first.
    If caching is enabled and the cache exists, loads from cache without
    loading the model.

    Args:
        image_dir: Directory containing PNG images.
        device: 'auto', 'cuda', or 'cpu'.
        cache: Whether to check / save the embeddings cache (default True).

    Returns:
        (all_patches, filenames_repeated, filenames_unique) where:
        - all_patches: np.ndarray of shape (N*256, 384), dtype float32.
          Each image contributes 256 patch tokens stacked vertically.
        - filenames_repeated: list of str, each filename repeated 256 times,
          matching the rows of all_patches.
        - filenames_unique: list of str, one per successfully embedded image.
    """
    resolved = _resolve_device(device)
    img_dir = Path(image_dir)

    if not img_dir.is_dir():
        print(f"  ERROR: Not a directory: {img_dir}")
        return np.array([]), [], []

    pngs = sorted(img_dir.glob("*.png"))
    if not pngs:
        print(f"  No PNG images found in {image_dir}")
        return np.array([]), [], []

    # ---- Check cache ----
    cache_path = Path(PATCH_EMBEDDINGS_CACHE_PATH)
    if cache and cache_path.exists():
        print(f"  Loading cached patch embeddings from {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        all_patches = loaded["all_patches"]
        cached_fnames_repeated = loaded["filenames_repeated"].tolist()
        cached_fnames_unique = loaded["filenames_unique"].tolist()
        # Verify cache matches the requested image_dir's basenames
        requested_basenames = {p.name for p in pngs}
        cached_basenames = set(cached_fnames_unique)
        if requested_basenames == cached_basenames:
            print(f"  Loaded {len(cached_fnames_unique)} cached images "
                  f"({len(all_patches)} patch tokens, shape {all_patches.shape})")
            return all_patches, cached_fnames_repeated, cached_fnames_unique
        else:
            print(f"  Cache mismatch — re-extracting patch embeddings")
            print(f"    Requested: {len(requested_basenames)} files, "
                  f"Cached: {len(cached_basenames)} files")

    # ---- Load DINOv2 ----
    print(f"  Extracting DINOv2 patch embeddings from: {image_dir}")
    model = _load_dinov2_model(resolved)

    all_patches_list: list[np.ndarray] = []
    filenames_repeated: list[str] = []
    filenames_unique: list[str] = []

    for p in tqdm(pngs, desc="Patch embedding", unit="img"):
        try:
            img = Image.open(p).convert("RGB")
            tensor = DINO_TRANSFORM(img).unsqueeze(0).to(resolved)

            with torch.no_grad():
                # get_intermediate_layers with reshape=True and
                # return_class_token=True returns patch tokens in spatial
                # format [1, D, H, W] and cls token [1, D].
                # The output is a list of tuples, one per requested layer.
                patch_tokens, cls_tokens = model.get_intermediate_layers(
                    tensor, n=1, reshape=True, return_class_token=True,
                )[0]

            # patch_tokens: [1, 384, 16, 16] -> flatten spatial dims
            # -> [1, 384, 256] -> transpose -> [1, 256, 384]
            pt = (patch_tokens
                  .flatten(2)           # [1, 384, 256]
                  .transpose(1, 2)      # [1, 256, 384]
                  .squeeze(0)           # [256, 384]
                  .cpu()
                  .numpy()
                  .astype(np.float32))

            all_patches_list.append(pt)
            filenames_repeated.extend([p.name] * N_PATCHES)
            filenames_unique.append(p.name)

        except Exception as exc:
            print(f"  WARNING: Failed to embed {p.name}: {exc}")

    if not all_patches_list:
        print("  No patch embeddings extracted successfully.")
        return np.array([]), [], []

    all_patches_arr = np.vstack(all_patches_list).astype(np.float32)
    print(f"  Extracted {len(filenames_unique)} images, "
          f"{len(all_patches_arr)} patch tokens (shape {all_patches_arr.shape})")

    # ---- Save cache ----
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            all_patches=all_patches_arr,
            filenames_repeated=np.array(filenames_repeated, dtype=object),
            filenames_unique=np.array(filenames_unique, dtype=object),
        )
        print(f"  Saved patch embeddings cache to {cache_path}")

    return all_patches_arr, filenames_repeated, filenames_unique


# ---------------------------------------------------------------------------
# Subspace training (PCA on patch tokens)
# ---------------------------------------------------------------------------

def _auto_select_n_components(
    n_samples: int, n_features: int, variance_target: float = AUTO_VARIANCE_TARGET,
) -> int:
    """Automatically select n_components for PCA on patch tokens.

    For patch-level anomaly detection we want a low-dimensional subspace so
    that anomalous patches produce meaningful reconstruction residuals.
    The heuristic:

      1. Start with DEFAULT_N_COMPONENTS (64).
      2. Cap at min(n_samples - 1, n_features) to avoid singular covariance.
      3. Cap at n_samples // 2 to prevent memorizing the training set.

    The user can always override with --n-components.

    Args:
        n_samples: Number of training patch tokens (pooled across images).
        n_features: Number of feature dimensions (384 for DINOv2 ViT-S/14).
        variance_target: Ignored — kept for API compatibility.

    Returns:
        int: Number of PCA components to use.
    """
    max_possible = min(n_samples - 1, n_features)
    if max_possible < 1:
        return 1
    # Pick the smallest of: default, half the samples, or max possible
    suggested = min(DEFAULT_N_COMPONENTS, n_samples // 2, max_possible)
    return max(1, suggested)


def train_patch_subspace(
    negative_dir: str, n_components: int | None = None,
    device: str = "auto", sample_frac: float = 0.1,
) -> dict:
    """Train PCA on patch embeddings from negative images.

    Extracts all patch tokens from negative images, optionally samples a
    fraction of patches per image to keep memory manageable, then fits PCA.

    Args:
        negative_dir: Directory of negative (non-spawn) PNG images.
        n_components: Number of PCA components. None = auto-select.
        device: 'auto', 'cuda', or 'cpu'.
        sample_frac: Fraction of patches to randomly sample per image
            (0.0 to 1.0). Set to 1.0 to use all patches. Default 0.1
            yields ~26 patches per image, keeping memory manageable.

    Returns:
        dict with keys:
        - 'pca_model': fitted PCA object (sklearn.decomposition.PCA)
        - 'mean': ndarray of mean patch embedding
        - 'components': ndarray of PCA components
        - 'explained_variance': ndarray of explained variance per component
        - 'explained_variance_ratio': ndarray of explained variance ratio
        - 'n_components': int
        - 'n_patches_trained': int, number of patch tokens used for training
        - 'n_images_used': int, number of images used
        - 'sample_frac': float, fraction of patches sampled per image
    """
    resolved = _resolve_device(device)

    print("=" * 60)
    print("  Patch SubspaceAD — Train on Negatives")
    print("=" * 60)
    print(f"  Negative directory: {negative_dir}")
    print(f"  Device: {resolved}")
    print(f"  Sample fraction: {sample_frac}")

    # Extract all patch embeddings (with caching)
    all_patches, filenames_repeated, filenames_unique = extract_patch_embeddings(
        negative_dir, device=resolved, cache=True,
    )

    if len(all_patches) == 0:
        print("ERROR: No patch embeddings extracted.")
        return {"error": "No patch embeddings extracted"}

    n_images = len(filenames_unique)
    n_total_patches = len(all_patches)
    print(f"  Total patch tokens available: {n_total_patches} "
          f"from {n_images} images")

    # ---- Sample patches per image ----
    if sample_frac < 1.0:
        # Group patches by image (each image contributes 256 contiguous rows)
        sampled_patches: list[np.ndarray] = []
        rng = np.random.RandomState(42)
        for img_idx in range(n_images):
            start = img_idx * N_PATCHES
            end = start + N_PATCHES
            img_patches = all_patches[start:end]  # (256, 384)
            n_sample = max(1, int(N_PATCHES * sample_frac))
            indices = rng.choice(N_PATCHES, size=n_sample, replace=False)
            sampled_patches.append(img_patches[indices])

        training_patches = np.vstack(sampled_patches).astype(np.float32)
        print(f"  Sampled {len(training_patches)} patch tokens "
              f"({sample_frac:.1%} per image)")
    else:
        training_patches = all_patches
        print(f"  Using all {n_total_patches} patch tokens for training")

    # ---- Fit PCA ----
    n_patches, n_features = training_patches.shape

    if n_components is None:
        n_components = _auto_select_n_components(n_patches, n_features)
        print(f"  Auto-selected n_components={n_components} "
              f"(from {n_patches} patches x {n_features} features)")

    # Clamp: PCA requires n_components <= min(n_patches, n_features)
    max_components = min(n_patches - 1, n_features)
    if n_components > max_components:
        n_components = max(1, max_components)
        print(f"  Clamped n_components to {n_components} "
              f"(limited by data dimensions)")

    print(f"  Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components, whiten=False, random_state=42)
    pca.fit(training_patches)

    total_var_ratio = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance ratio: {total_var_ratio:.4f} "
          f"(with {n_components} components)")

    print(f"\n  Training complete:")
    print(f"    PCA components:             {pca.n_components_}")
    print(f"    PCA explained variance:     {total_var_ratio:.4f}")
    print(f"    Negative training images:   {n_images}")
    print(f"    Training patch tokens:      {n_patches}")
    print(f"    Embedding dimension:        {EMBED_DIM}")

    return {
        "pca_model": pca,
        "mean": pca.mean_,
        "components": pca.components_,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "n_components": int(n_components),
        "n_patches_trained": n_patches,
        "n_images_used": n_images,
        "sample_frac": sample_frac,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_image_patches(
    model: dict, image_path: str, device: str = "auto",
) -> dict:
    """Score a single image by patch-level PCA reconstruction residuals.

    Steps:
      1. Extract all 256 patch embeddings from DINOv2.
      2. Project each through PCA and compute reconstruction residual.
      3. Take mean of top 10% highest residuals (anomalous patches).
      4. Also compute: mean residual (all patches), max residual,
         frac_anomalous (residual > 3 * median residual).

    Args:
        model: dict from train_patch_subspace() containing 'pca_model'.
        image_path: Path to a single PNG image.
        device: 'auto', 'cuda', or 'cpu'.

    Returns:
        dict with keys:
        - 'score_top10p': float, mean residual of top 10% patches
            (higher = more anomalous = more likely spawn).
        - 'score_mean': float, mean residual across all patches.
        - 'score_max': float, maximum patch residual.
        - 'frac_anomalous': float, fraction of patches with residual
            > 3 * median residual.
        - 'n_patches': int, number of patches processed (should be 256).
        - 'prediction': 1 if score_top10p > DEFAULT_PREDICTION_THRESHOLD
            else 0.
        - 'threshold': float, the threshold used.
    """
    resolved = _resolve_device(device)
    # Handle both dict-wrapped PCA and raw PCA object
    if isinstance(model, dict):
        pca_model = model["pca_model"]
        threshold = model.get("threshold", DEFAULT_PREDICTION_THRESHOLD)
    else:
        pca_model = model
        threshold = DEFAULT_PREDICTION_THRESHOLD

    # ---- Load and embed image ----
    img = Image.open(image_path).convert("RGB")
    tensor = DINO_TRANSFORM(img).unsqueeze(0).to(resolved)

    dinov2 = _load_dinov2_model(resolved) if not hasattr(
        score_image_patches, "_model"
    ) or score_image_patches._device != resolved else score_image_patches._model

    # Cache the model to avoid reloading per image
    if not hasattr(score_image_patches, "_model") or \
            score_image_patches._device != resolved:
        score_image_patches._model = dinov2
        score_image_patches._device = resolved
    else:
        dinov2 = score_image_patches._model

    with torch.no_grad():
        patch_tokens, _cls_tokens = dinov2.get_intermediate_layers(
            tensor, n=1, reshape=True, return_class_token=True,
        )[0]

    # patch_tokens: [1, 384, 16, 16] -> [256, 384]
    pt = (patch_tokens
          .flatten(2)
          .transpose(1, 2)
          .squeeze(0)
          .cpu()
          .numpy()
          .astype(np.float32))  # (256, 384)

    # ---- Compute per-patch reconstruction residuals ----
    projected = pca_model.transform(pt)                        # (256, n_components)
    reconstructed = pca_model.inverse_transform(projected)     # (256, 384)

    # Per-patch MSE residual
    residuals = np.mean((pt - reconstructed) ** 2, axis=1)     # (256,)

    # ---- Aggregate residuals ----
    # Sort descending to find top anomalies
    sorted_residuals = np.sort(residuals)[::-1]

    n_anom = max(1, int(N_PATCHES * ANOMALOUS_PATCH_FRAC))
    score_top10p = float(np.mean(sorted_residuals[:n_anom]))
    score_mean = float(np.mean(residuals))
    score_max = float(np.max(residuals))

    # Fraction of patches with residual > 3 * median
    median_residual = float(np.median(residuals))
    if median_residual > 0:
        frac_anomalous = float(np.mean(residuals > 3.0 * median_residual))
    else:
        frac_anomalous = 0.0

    prediction = 1 if score_top10p > threshold else 0

    return {
        "score_top10p": score_top10p,
        "score_mean": score_mean,
        "score_max": score_max,
        "frac_anomalous": frac_anomalous,
        "n_patches": len(residuals),
        "prediction": prediction,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Directory scoring
# ---------------------------------------------------------------------------

def score_directory_patches(
    model: dict, image_dir: str, device: str = "auto",
) -> list[dict]:
    """Score all images in a directory using trained patch PCA subspace.

    For each image:
      1. Extract all 256 DINOv2 patch embeddings.
      2. Compute per-patch PCA reconstruction residuals.
      3. Aggregate: top-10% mean, overall mean, max, frac anomalous.

    Args:
        model: dict from train_patch_subspace().
        image_dir: Directory containing PNG images to score.
        device: 'auto', 'cuda', or 'cpu'.

    Returns:
        list of dicts sorted by score_top10p descending, each with keys:
        - 'filename': str
        - 'score_top10p': float
        - 'score_mean': float
        - 'score_max': float
        - 'frac_anomalous': float
        - 'n_patches': int
        - 'prediction': 0 or 1
    """
    resolved = _resolve_device(device)
    img_dir = Path(image_dir)

    if not img_dir.is_dir():
        print(f"  ERROR: Not a directory: {image_dir}")
        return []

    pngs = sorted(img_dir.glob("*.png"))
    print(f"  Found {len(pngs)} PNG images in {image_dir}")

    if not pngs:
        print("  No images to score.")
        return []

    # Load model once, share across all images
    _ = _load_dinov2_model(resolved)
    # Reuse the caching trick from score_image_patches
    score_image_patches._model = _
    score_image_patches._device = resolved

    results: list[dict] = []

    for p in tqdm(pngs, desc="Patch scoring", unit="img"):
        try:
            score_result = score_image_patches(model, str(p), device=resolved)
            score_result["filename"] = p.name
            results.append(score_result)
        except Exception as exc:
            print(f"  WARNING: Failed to score {p.name}: {exc}")

    # Clean up cached model
    if hasattr(score_image_patches, "_model"):
        del score_image_patches._model
        del score_image_patches._device

    results.sort(key=lambda r: r["score_top10p"], reverse=True)
    print(f"  Scored {len(results)}/{len(pngs)} images successfully")

    if results:
        print(f"  Top 3 scores (top-10% patch residual):")
        for r in results[:3]:
            print(f"    {r['score_top10p']:.6f}  {r['filename']}  "
                  f"[{'spawn' if r['prediction'] else 'normal'}]")

    return results


# ---------------------------------------------------------------------------
# Validation against human labels
# ---------------------------------------------------------------------------

def validate_patches(
    model: dict, labels_json_path: str, image_dir: str, device: str = "auto",
) -> dict:
    """Validate patch-level subspace anomaly detection against human labels.

    Labels JSON format:
        {"labels": [{"filename": "image.png", "label": 1}, ...]}
    where label=1 means positive (spawn), label=0 means negative (no spawn).

    Returns dict with the same schema as subspace_ad.validate():
        accuracy, best_accuracy, best_threshold, auc_roc, avg_precision,
        confusion_matrix, per_sample, n_total, n_pos, n_neg.
    """
    resolved = _resolve_device(device)
    print(f"  Validate device: {resolved}")

    # ---- Load labels ----
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

    # ---- Load DINOv2 (cached in score_image_patches) ----
    dinov2 = _load_dinov2_model(resolved)
    score_image_patches._model = dinov2
    score_image_patches._device = resolved

    img_dir = Path(image_dir)
    per_sample: list[dict] = []

    for entry in tqdm(label_entries, desc="Validating patches", unit="img"):
        fname = entry["filename"]
        true_label = entry["label"]
        img_path = img_dir / fname

        if not img_path.exists():
            print(f"  WARNING: Image not found: {img_path}")
            continue

        try:
            score_result = score_image_patches(model, str(img_path), device=resolved)

            per_sample.append({
                "filename": fname,
                "true_label": true_label,
                "prediction": score_result["prediction"],
                "score_top10p": score_result["score_top10p"],
                "score_mean": score_result["score_mean"],
                "score_max": score_result["score_max"],
                "frac_anomalous": score_result["frac_anomalous"],
            })
        except Exception as exc:
            print(f"  WARNING: Failed to process {fname}: {exc}")

    # Clean up cached model
    if hasattr(score_image_patches, "_model"):
        del score_image_patches._model
        del score_image_patches._device

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

    # ---- Aggregate metrics ----
    y_true = np.array([s["true_label"] for s in per_sample])
    y_pred = np.array([s["prediction"] for s in per_sample])
    y_score = np.array([s["score_top10p"] for s in per_sample])

    n_total = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_total - n_pos

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Best accuracy via threshold sweep on score_top10p
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
        best_thr = 0.0
        for thr in thresholds:
            thr_pred = (y_score > thr).astype(int)
            thr_acc = accuracy_score(y_true, thr_pred)
            if thr_acc > best_acc:
                best_acc = thr_acc
                best_thr = float(thr)

    # AUROC
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

    print("\n  Validation results (patch-level SubspaceAD):")
    print(f"    Total: {n_total}  Pos: {n_pos}  Neg: {n_neg}")
    print(f"    Accuracy (thr={DEFAULT_PREDICTION_THRESHOLD}):  {acc:.4f}")
    print(f"    Best accuracy:         {best_acc:.4f} @ thr={best_thr:.6f}")
    print(f"    AUROC:                 {auroc:.4f}")
    print(f"    Avg Precision:         {ap:.4f}")
    print(f"    Confusion Matrix:      {cm}")

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
# Full pipeline: train + validate
# ---------------------------------------------------------------------------

def train_and_validate(
    negative_dir: str, labels_json_path: str, image_dir: str,
    n_components: int | None = None, device: str = "auto",
    sample_frac: float = 0.1,
) -> dict:
    """Full pipeline: train PCA on negatives, validate against labels.

    Args:
        negative_dir: Directory of negative (non-spawn) PNG images.
        labels_json_path: Path to validation labels JSON.
        image_dir: Directory containing labeled PNG images.
        n_components: Number of PCA components. None = auto-select.
        device: 'auto', 'cuda', or 'cpu'.
        sample_frac: Fraction of patches to sample per image for training.

    Returns:
        dict with keys:
        - 'training': dict from train_patch_subspace()
        - 'validation': dict from validate_patches()
        - 'cv_metrics': dict with accuracy, best_accuracy, best_threshold,
          auc_roc, avg_precision matching standard format.
    """
    resolved = _resolve_device(device)

    print("=" * 60)
    print("  Patch SubspaceAD — Full Pipeline")
    print("=" * 60)

    # ---- Train ----
    train_result = train_patch_subspace(
        negative_dir, n_components=n_components,
        device=resolved, sample_frac=sample_frac,
    )

    if "error" in train_result:
        print(f"\nERROR: Training failed: {train_result['error']}")
        return {"error": train_result["error"]}

    pca_model = train_result["pca_model"]

    # ---- Validate ----
    print("\n" + "=" * 60)
    print("  Validation")
    print("=" * 60)
    val_result = validate_patches(
        pca_model, labels_json_path, image_dir, device=resolved,
    )

    # Build standard cv_metrics format
    cv_metrics = {
        "accuracy": val_result.get("accuracy", 0.0),
        "best_accuracy": val_result.get("best_accuracy", 0.0),
        "best_threshold": val_result.get("best_threshold", 0.0),
        "auc_roc": val_result.get("auc_roc", 0.0),
        "avg_precision": val_result.get("avg_precision", 0.0),
        "confusion_matrix": val_result.get("confusion_matrix", [[0, 0], [0, 0]]),
        "n_total": val_result.get("n_total", 0),
        "n_pos": val_result.get("n_pos", 0),
        "n_neg": val_result.get("n_neg", 0),
        "n_patches_trained": train_result.get("n_patches_trained", 0),
        "n_images_trained": train_result.get("n_images_used", 0),
        "n_components": train_result.get("n_components", 0),
        "explained_variance_ratio": float(
            train_result.get("explained_variance_ratio", np.array([0.0])).sum()
        ),
    }

    return {
        "training": train_result,
        "validation": val_result,
        "cv_metrics": cv_metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DINOv2 patch-level SubspaceAD for herring spawn detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train-dir", type=str, default=None,
        help="Directory of negative (non-spawn) PNG images for training PCA "
             "on patch tokens",
    )
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory of PNG images to score or validate",
    )
    parser.add_argument(
        "--labels-json", type=str, default=None,
        help="Path to validation labels JSON file "
             "(format: {labels: [{filename, label}, ...]})",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save output JSON results",
    )
    parser.add_argument(
        "--n-components", type=int, default=None,
        help="Number of PCA components (default: auto-select)",
    )
    parser.add_argument(
        "--sample-frac", type=float, default=0.1,
        help="Fraction of patches to sample per image for PCA training "
             "(default: 0.1). Use 1.0 for all patches.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip per-image scoring output, just run validation against labels",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on (default: auto)",
    )
    args = parser.parse_args(argv)

    # Validate sample_frac
    if not 0 < args.sample_frac <= 1.0:
        print(f"ERROR: --sample-frac must be in (0, 1], got {args.sample_frac}")
        return 1

    resolved = _resolve_device(args.device)

    # ----- Determine paths -----
    repo_root = Path(__file__).resolve().parent.parent

    def _resolve_path(given: str | None) -> Path | None:
        if given is None:
            return None
        p = Path(given)
        return p if p.is_absolute() else (repo_root / given)

    train_dir = _resolve_path(args.train_dir)
    img_dir = _resolve_path(args.image_dir)
    labels_path = _resolve_path(args.labels_json)

    # Check paths
    if img_dir is not None and not img_dir.is_dir():
        print(f"ERROR: Image directory not found: {img_dir}")
        return 1

    if labels_path is not None and not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        return 1

    if train_dir is not None and not train_dir.is_dir():
        print(f"ERROR: Train directory not found: {train_dir}")
        return 1

    # ==================================================================
    # Mode 1: Validate-only
    # ==================================================================
    if args.validate_only:
        if args.labels_json is None:
            print("ERROR: --validate-only requires --labels-json")
            return 1
        if img_dir is None:
            print("ERROR: --validate-only requires --image-dir")
            return 1

        # Default to negatives directory for training if --train-dir not given
        if train_dir is None:
            default_train = repo_root / "data/samples/negative"
            if default_train.is_dir():
                train_dir = default_train
                print(f"  Using default train dir: {train_dir}")
            else:
                print("ERROR: --validate-only requires --train-dir "
                      "or data/samples/negative")
                return 1

        # Run full pipeline
        pipeline_result = train_and_validate(
            negative_dir=str(train_dir),
            labels_json_path=str(labels_path),
            image_dir=str(img_dir),
            n_components=args.n_components,
            device=resolved,
            sample_frac=args.sample_frac,
        )

        if "error" in pipeline_result:
            print(f"\nERROR: Pipeline failed: {pipeline_result['error']}")
            return 1

        result = pipeline_result

    # ==================================================================
    # Mode 2: Score directory (with optional training)
    # ==================================================================
    else:
        if img_dir is None:
            # If no image-dir, try default scoring on candidates_knn
            if train_dir is not None:
                default_img = repo_root / "data/candidates_knn"
                if default_img.is_dir():
                    img_dir = default_img
                    print(f"  Using default image dir: {img_dir}")
                else:
                    print("ERROR: --image-dir is required")
                    return 1
            else:
                print("ERROR: --image-dir or --train-dir is required")
                return 1

        # Train if train-dir provided
        if train_dir is not None and train_dir.is_dir():
            train_result = train_patch_subspace(
                str(train_dir),
                n_components=args.n_components,
                device=resolved,
                sample_frac=args.sample_frac,
            )
            if "error" in train_result:
                print(f"\nERROR: Training failed: {train_result['error']}")
                return 1
            pca_model = train_result
        elif args.n_components is not None:
            # Try to load cached patch embeddings and train from those
            cache_path = Path(PATCH_EMBEDDINGS_CACHE_PATH)
            if cache_path.exists():
                print(f"  Loading cached patch embeddings for PCA training "
                      f"from {cache_path}")
                loaded = np.load(cache_path, allow_pickle=True)
                all_patches = loaded["all_patches"]

                # Sample if needed
                if args.sample_frac < 1.0:
                    n_images = len(loaded["filenames_unique"])
                    rng = np.random.RandomState(42)
                    sampled: list[np.ndarray] = []
                    for img_idx in range(n_images):
                        start = img_idx * N_PATCHES
                        end = start + N_PATCHES
                        img_patches = all_patches[start:end]
                        n_sample = max(1, int(N_PATCHES * args.sample_frac))
                        indices = rng.choice(N_PATCHES, size=n_sample,
                                             replace=False)
                        sampled.append(img_patches[indices])
                    training_patches = np.vstack(sampled)
                else:
                    training_patches = all_patches

                pca_result = _train_pca_on_patches(
                    training_patches, n_components=args.n_components
                )
                pca_model = pca_result
                train_result = {
                    "n_images_used": len(loaded["filenames_unique"]),
                    "n_patches_trained": len(training_patches),
                    "n_components": pca_result["n_components"],
                    "sample_frac": args.sample_frac,
                }
            else:
                print("ERROR: --train-dir required (no cached embeddings found)")
                return 1
        else:
            print("ERROR: --train-dir required for training the PCA subspace")
            return 1

        # Score directory
        print("\n" + "=" * 60)
        print("  Scoring (patch-level)")
        print("=" * 60)
        scoring_results = score_directory_patches(
            pca_model, str(img_dir), device=resolved,
        )

        result = {
            "model": f"{MODEL_NAME} + Patch SubspaceAD (PCA)",
            "device": resolved,
            "n_components": train_result.get("n_components", 0),
            "n_patches_trained": train_result.get("n_patches_trained", 0),
            "n_negative_training_images": train_result.get(
                "n_images_used", 0),
            "sample_frac": args.sample_frac,
            "n_images_scored": len(scoring_results),
            "results": scoring_results,
        }

        # Optionally validate against labels
        if labels_path is not None and labels_path.exists():
            print("\n" + "=" * 60)
            print("  Running validation against labels...")
            val_result = validate_patches(
                pca_model, str(labels_path), str(img_dir), device=resolved,
            )
            result["validation"] = val_result

    # ==================================================================
    # Save or print output
    # ==================================================================
    if args.output_json:
        out_path = _resolve_path(args.output_json)
        if out_path is None:
            out_path = repo_root / "data/patch_subspace_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Results saved to: {out_path}")
    elif not args.validate_only:
        # In scoring mode without --output-json, print summary
        print(json.dumps(result, indent=2, default=str))
    else:
        # In validate-only mode, the validation result was already printed
        pass

    return 0


# ---------------------------------------------------------------------------
# Internal helper: train PCA on already-loaded patches
# ---------------------------------------------------------------------------

def _train_pca_on_patches(
    patches: np.ndarray, n_components: int | None = None,
) -> dict:
    """Fit PCA on a pre-loaded array of patch tokens.

    This is used when loading cached embeddings rather than re-extracting.

    Args:
        patches: (N, 384) numpy array of patch embeddings.
        n_components: Number of PCA components. None = auto-select.

    Returns:
        Same dict format as partial result from train_patch_subspace().
    """
    n_patches, n_features = patches.shape

    if n_components is None:
        n_components = _auto_select_n_components(n_patches, n_features)

    max_components = min(n_patches - 1, n_features)
    if n_components > max_components:
        n_components = max(1, max_components)

    print(f"  Fitting PCA on {n_patches} patches with "
          f"{n_components} components...")
    pca = PCA(n_components=n_components, whiten=False, random_state=42)
    pca.fit(patches)

    total_var_ratio = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance ratio: {total_var_ratio:.4f}")

    return {
        "pca_model": pca,
        "mean": pca.mean_,
        "components": pca.components_,
        "explained_variance": pca.explained_variance,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "n_components": int(n_components),
        "n_patches_trained": n_patches,
    }


if __name__ == "__main__":
    sys.exit(main())
