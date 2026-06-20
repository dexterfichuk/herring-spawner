#!/usr/bin/env python3
"""Zero-shot herring spawn detection using SubspaceAD (PCA reconstruction residual).

Under this paradigm, patch-level features are extracted from a small set of
normal, non-spawning coastal images using a frozen DINOv2 backbone. A Principal
Component Analysis (PCA) model is fit to these features to estimate the
low-dimensional subspace of normal coastal variation. At inference, anomalies
(spawning events) are detected via high reconstruction residuals.

Training: PCA on DINOv2 embeddings from negative (no-spawn) images.
Inference: Score candidate images by PCA reconstruction error.
  Higher error = more anomalous = more likely spawn.

Usage:
    # Train subspace on negatives, validate against human labels
    python scripts/subspace_ad.py --validate-only \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --device cpu

    # Score a directory of candidates
    python scripts/subspace_ad.py \\
        --train-dir data/samples/negative \\
        --image-dir data/candidates_knn \\
        --output-json data/subspace_results.json

    # Train with explicit component count
    python scripts/subspace_ad.py \\
        --train-dir data/samples/negative \\
        --image-dir data/candidates_knn \\
        --n-components 64 \\
        --output-json data/subspace_results.json

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
from sklearn.ensemble import IsolationForest
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

# PCA variance ratio target when auto-selecting n_components
AUTO_VARIANCE_TARGET = 0.90

# Default number of PCA components (used when auto-selection is disabled
# or when n_components > n_samples)
DEFAULT_N_COMPONENTS = 32

# Cache path for extracted negative embeddings
EMBEDDINGS_CACHE_PATH = "data/embeddings/subspace_embeddings.npz"

# Default prediction threshold (mean reconstruction residual above this = spawn).
# DINOv2 embeddings are L2-normalized unit vectors, so residuals are ~0.00003-0.00065.
# The default 0.00015 is ~2 std above the negative training mean.
DEFAULT_PREDICTION_THRESHOLD = 0.00015


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
    print(f"  Model loaded: {MODEL_NAME} ({EMBED_DIM}-dim embeddings)")
    return model


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_dinov2_embeddings(
    image_dir: str, device: str = "auto", cache: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Extract DINOv2 ViT-S/14 embeddings for all PNGs in a directory.

    Uses the existing DINO_TRANSFORM from train_classifier.py.
    Checks for cached embeddings at ``EMBEDDINGS_CACHE_PATH`` first.
    If caching is enabled and the cache exists, loads from cache without
    loading the model.

    Args:
        image_dir: Directory containing PNG images.
        device: 'auto', 'cuda', or 'cpu'.
        cache: Whether to check / save the embeddings cache (default True).

    Returns:
        (embeddings, filenames) where:
        - embeddings: np.ndarray of shape (N, EMBED_DIM), dtype float32.
        - filenames: list of str filenames for successfully embedded images.
    """
    resolved = _resolve_device(device)
    img_dir = Path(image_dir)

    if not img_dir.is_dir():
        print(f"  ERROR: Not a directory: {img_dir}")
        return np.array([]), []

    pngs = sorted(img_dir.glob("*.png"))
    if not pngs:
        print(f"  No PNG images found in {image_dir}")
        return np.array([]), []

    # ---- Check cache ----
    cache_path = Path(EMBEDDINGS_CACHE_PATH)
    if cache and cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        embeddings = loaded["embeddings"]
        cached_fnames = loaded["filenames"].tolist()
        # Verify cache matches the requested image_dir's basenames
        requested_basenames = {p.name for p in pngs}
        cached_basenames = set(cached_fnames)
        if requested_basenames == cached_basenames:
            print(f"  Loaded {len(cached_fnames)} cached embeddings (shape {embeddings.shape})")
            return embeddings, cached_fnames
        else:
            print(f"  Cache mismatch — re-extracting embeddings")
            print(f"    Requested: {len(requested_basenames)} files, Cached: {len(cached_basenames)} files")

    # ---- Load DINOv2 ----
    print(f"  Extracting DINOv2 embeddings from: {image_dir}")
    model = _load_dinov2_model(resolved)

    embeddings: list[np.ndarray] = []
    filenames: list[str] = []

    for p in tqdm(pngs, desc="Embedding", unit="img"):
        try:
            img = Image.open(p).convert("RGB")
            tensor = DINO_TRANSFORM(img).unsqueeze(0).to(resolved)
            with torch.no_grad():
                emb = model(tensor)
            emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
            embeddings.append(emb)
            filenames.append(p.name)
        except Exception as exc:
            print(f"  WARNING: Failed to embed {p.name}: {exc}")

    if not embeddings:
        print("  No embeddings extracted successfully.")
        return np.array([]), []

    emb_array = np.stack(embeddings).astype(np.float32)
    print(f"  Extracted {len(filenames)} embeddings (shape {emb_array.shape})")

    # ---- Save cache ----
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=emb_array,
            filenames=np.array(filenames, dtype=object),
        )
        print(f"  Saved embeddings cache to {cache_path}")

    return emb_array, filenames


# ---------------------------------------------------------------------------
# Subspace training (PCA)
# ---------------------------------------------------------------------------

def _auto_select_n_components(
    n_samples: int, n_features: int, variance_target: float = AUTO_VARIANCE_TARGET,
) -> int:
    """Automatically select n_components for PCA.

    For anomaly detection we want a *low-dimensional* subspace so that
    anomalies produce meaningful reconstruction residuals. The heuristic:

      1. Start with DEFAULT_N_COMPONENTS (32).
      2. Cap at min(n_samples - 1, n_features) to avoid singular covariance.
      3. Cap at n_samples // 2 to prevent memorizing the training set.

    The user can always override with --n-components.

    Args:
        n_samples: Number of training samples.
        n_features: Number of feature dimensions.
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


def train_subspace(
    embeddings: np.ndarray, n_components: int | None = None,
) -> dict:
    """Fit PCA on normal (negative) embeddings to learn the normal coastal subspace.

    If n_components is None, auto-select to explain ~90% variance.

    Args:
        embeddings: (N, D) numpy array of DINOv2 embeddings from negative images.
        n_components: Number of PCA components. None = auto-select.

    Returns:
        dict with keys:
        - 'pca_model': fitted PCA object (sklearn.decomposition.PCA)
        - 'mean': ndarray of mean embedding
        - 'components': ndarray of PCA components
        - 'explained_variance': ndarray of explained variance per component
        - 'explained_variance_ratio': ndarray of explained variance ratio
        - 'n_components': int, number of components used
        - 'n_train': int, number of training samples
    """
    n_samples, n_features = embeddings.shape

    if n_components is None:
        n_components = _auto_select_n_components(n_samples, n_features)
        print(f"  Auto-selected n_components={n_components} "
              f"(from {n_samples} samples x {n_features} features)")

    # Clamp: PCA requires n_components <= min(n_samples, n_features)
    max_components = min(n_samples - 1, n_features)
    if n_components > max_components:
        n_components = max(1, max_components)
        print(f"  Clamped n_components to {n_components} (limited by data dimensions)")

    print(f"  Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components, whiten=False, random_state=42)
    pca.fit(embeddings)

    total_var_ratio = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance ratio: {total_var_ratio:.4f} "
          f"(with {n_components} components)")

    return {
        "pca_model": pca,
        "mean": pca.mean_,
        "components": pca.components_,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "n_components": int(n_components),
        "n_train": n_samples,
    }


def train_isolation_forest(
    embeddings: np.ndarray, random_state: int = 42,
) -> IsolationForest:
    """Train an IsolationForest on the same embeddings for comparison.

    IsolationForest provides an alternative anomaly detection baseline
    that works well in high dimensions without dimensionality reduction.

    Args:
        embeddings: (N, D) numpy array of DINOv2 embeddings.
        random_state: Random seed.

    Returns:
        Fitted IsolationForest model.
    """
    print(f"  Fitting IsolationForest (n_estimators=100, contamination='auto')...")
    if_model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    if_model.fit(embeddings)
    print(f"  IsolationForest fitted on {len(embeddings)} samples")
    return if_model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_image(
    pca_model: PCA, embedding: np.ndarray,
) -> dict:
    """Score a single embedding by its PCA reconstruction residual.

    reconstruction = pca.inverse_transform(pca.transform(embedding.reshape(1, -1)))
    residual = mean((embedding - reconstruction)^2)

    This is the *mean squared error* across all feature dimensions.
    Higher residual = more anomalous relative to the normal subspace.

    Args:
        pca_model: Fitted PCA object from train_subspace.
        embedding: 1-D numpy array of shape (D,) — a single DINOv2 embedding.

    Returns:
        dict with keys:
        - 'score': float, reconstruction residual (higher = more anomalous)
        - 'n_components': int, number of PCA components used
        - 'prediction': 1 if score > DEFAULT_PREDICTION_THRESHOLD else 0
        - 'reconstruction_error_l2': float, L2 reconstruction error
    """
    emb_2d = embedding.reshape(1, -1)
    projected = pca_model.transform(emb_2d)          # (1, n_components)
    reconstructed = pca_model.inverse_transform(projected)  # (1, D)

    # Mean squared error (MSE) — the standard PCA reconstruction residual
    residual = np.mean((emb_2d - reconstructed) ** 2)

    # L2 reconstruction error (Euclidean distance)
    l2_error = float(np.linalg.norm(emb_2d - reconstructed))

    prediction = 1 if residual > DEFAULT_PREDICTION_THRESHOLD else 0

    return {
        "score": float(residual),
        "n_components": pca_model.n_components_,
        "prediction": prediction,
        "reconstruction_error_l2": l2_error,
    }


def score_directory(
    pca_model: PCA, image_dir: str, device: str = "auto",
) -> list[dict]:
    """Score all images in a directory using trained PCA subspace.

    For each image:
      1. Extract DINOv2 embedding.
      2. Compute PCA reconstruction residual.
      3. Higher residual = more anomalous = more likely spawn.

    Args:
        pca_model: Fitted PCA object from train_subspace.
        image_dir: Directory containing PNG images to score.
        device: 'auto', 'cuda', or 'cpu'.

    Returns:
        list of dicts sorted by score descending, each with keys:
        - 'filename': str
        - 'score': float
        - 'n_components': int
        - 'prediction': 0 or 1
        - 'reconstruction_error_l2': float
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

    # ---- Load DINOv2 ----
    model = _load_dinov2_model(resolved)

    results: list[dict] = []

    for p in tqdm(pngs, desc="Scoring", unit="img"):
        try:
            img = Image.open(p).convert("RGB")
            tensor = DINO_TRANSFORM(img).unsqueeze(0).to(resolved)
            with torch.no_grad():
                emb = model(tensor)
            emb = F.normalize(emb, dim=1).cpu().numpy().flatten()

            score_result = score_image(pca_model, emb)
            score_result["filename"] = p.name
            results.append(score_result)
        except Exception as exc:
            print(f"  WARNING: Failed to score {p.name}: {exc}")

    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"  Scored {len(results)}/{len(pngs)} images successfully")

    if results:
        print(f"  Top 3 scores:")
        for r in results[:3]:
            print(f"    {r['score']:.6f}  {r['filename']}  "
                  f"[{'spawn' if r['prediction'] else 'normal'}]")

    return results


# ---------------------------------------------------------------------------
# Validation against human labels
# ---------------------------------------------------------------------------

def validate(
    pca_model: PCA, labels_json_path: str, image_dir: str, device: str = "auto",
) -> dict:
    """Validate subspace anomaly detection against human labels.

    Labels JSON format (matching remoteclip_zero_shot.py convention):
        {"labels": [{"filename": "image.png", "label": 1}, ...]}
    where label=1 means positive (spawn), label=0 means negative (no spawn).

    Returns dict with the same schema as
    ``remoteclip_zero_shot.validate()``:
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

    # ---- Load DINOv2 ----
    model = _load_dinov2_model(resolved)

    img_dir = Path(image_dir)
    per_sample: list[dict] = []

    for entry in tqdm(label_entries, desc="Validating", unit="img"):
        fname = entry["filename"]
        true_label = entry["label"]
        img_path = img_dir / fname

        if not img_path.exists():
            print(f"  WARNING: Image not found: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            tensor = DINO_TRANSFORM(img).unsqueeze(0).to(resolved)
            with torch.no_grad():
                emb = model(tensor)
            emb = F.normalize(emb, dim=1).cpu().numpy().flatten()

            score_result = score_image(pca_model, emb)

            per_sample.append({
                "filename": fname,
                "true_label": true_label,
                "prediction": score_result["prediction"],
                "score": score_result["score"],
                "n_components": score_result["n_components"],
                "reconstruction_error_l2": score_result["reconstruction_error_l2"],
            })
        except Exception as exc:
            print(f"  WARNING: Failed to process {fname}: {exc}")

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
    y_score = np.array([s["score"] for s in per_sample])

    n_total = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_total - n_pos

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

    print("\n  Validation results:")
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
# Training convenience
# ---------------------------------------------------------------------------

def train_on_negatives(
    negative_dir: str, n_components: int | None = None, device: str = "auto",
) -> dict:
    """Train PCA subspace on all negatives in a directory.

    Extracts DINOv2 embeddings from all PNGs in negative_dir, fits PCA.
    Also trains and returns an IsolationForest on the same embeddings for
    comparison.

    Args:
        negative_dir: Directory of negative (non-spawn) PNG images.
        n_components: Number of PCA components. None = auto-select.
        device: 'auto', 'cuda', or 'cpu'.

    Returns:
        dict with keys:
        - 'pca_model': fitted PCA object (sklearn.decomposition.PCA)
        - 'pca': the full PCA result dict from train_subspace()
        - 'if_model': fitted IsolationForest model
        - 'n_negative_images': int
        - 'explained_variance_ratio': float (total)
        - 'n_components': int
        - 'embedding_dim': int
    """
    resolved = _resolve_device(device)

    print("=" * 60)
    print("  SubspaceAD — Train on Negatives")
    print("=" * 60)
    print(f"  Negative directory: {negative_dir}")
    print(f"  Device: {resolved}")

    # Extract embeddings from negatives
    embeddings, filenames = extract_dinov2_embeddings(
        negative_dir, device=resolved, cache=True,
    )

    if len(embeddings) == 0:
        print("ERROR: No negative embeddings extracted.")
        return {
            "error": "No negative embeddings extracted",
        }

    print(f"  Training on {len(embeddings)} negative images")

    # Train PCA subspace
    pca_result = train_subspace(embeddings, n_components=n_components)
    pca_model = pca_result["pca_model"]

    # Train IsolationForest for comparison
    if_model = train_isolation_forest(embeddings)

    total_var_ratio = float(pca_model.explained_variance_ratio_.sum())

    print(f"\n  Training complete:")
    print(f"    PCA components:             {pca_model.n_components_}")
    print(f"    PCA explained variance:     {total_var_ratio:.4f}")
    print(f"    IsolationForest estimators: {if_model.n_estimators}")
    print(f"    Negative training images:   {len(embeddings)}")
    print(f"    Embedding dimension:        {EMBED_DIM}")

    return {
        "pca_model": pca_model,
        "if_model": if_model,
        "n_negative_images": len(embeddings),
        "explained_variance_ratio": pca_result["explained_variance_ratio"],
        "n_components": pca_result["n_components"],
        "embedding_dim": EMBED_DIM,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-shot herring spawn detection via SubspaceAD (PCA reconstruction residual)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train-dir", type=str, default=None,
        help="Directory of negative (non-spawn) PNG images for training PCA subspace",
    )
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory of PNG images to score or validate",
    )
    parser.add_argument(
        "--labels-json", type=str, default=None,
        help="Path to validation labels JSON file (format: {labels: [{filename, label}, ...]})",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save output JSON results",
    )
    parser.add_argument(
        "--n-components", type=int, default=None,
        help="Number of PCA components (default: auto-select to explain ~90%% variance)",
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

    # Check image dir
    if img_dir is not None and not img_dir.is_dir():
        print(f"ERROR: Image directory not found: {img_dir}")
        return 1

    # Check labels file
    if labels_path is not None and not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        return 1

    # Check train dir
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
                print("ERROR: --validate-only requires --train-dir or data/samples/negative")
                return 1

        # Train subspace on negatives
        train_result = train_on_negatives(
            str(train_dir),
            n_components=args.n_components,
            device=resolved,
        )

        if "error" in train_result:
            print(f"\nERROR: Training failed: {train_result['error']}")
            return 1

        pca_model = train_result["pca_model"]

        # Validate against labels
        print("\n" + "=" * 60)
        print("  Validation")
        print("=" * 60)
        result = validate(
            pca_model, str(labels_path), str(img_dir), device=resolved,
        )

        # Attach training metadata
        result["training"] = {
            "train_dir": str(train_dir),
            "n_negative_images": train_result["n_negative_images"],
            "n_components": train_result["n_components"],
            "explained_variance_ratio": train_result["explained_variance_ratio"],
            "embedding_dim": EMBED_DIM,
            "model_name": MODEL_NAME,
        }

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

        # Train if train-dir provided, or try to load cached
        if train_dir is not None and train_dir.is_dir():
            train_result = train_on_negatives(
                str(train_dir),
                n_components=args.n_components,
                device=resolved,
            )
            if "error" in train_result:
                print(f"\nERROR: Training failed: {train_result['error']}")
                return 1
            pca_model = train_result["pca_model"]
        elif args.n_components is not None:
            # Try to load cached embeddings and train from those
            cache_path = Path(EMBEDDINGS_CACHE_PATH)
            if cache_path.exists():
                print(f"  Loading cached embeddings for PCA training from {cache_path}")
                loaded = np.load(cache_path, allow_pickle=True)
                embeddings = loaded["embeddings"]
                pca_result = train_subspace(embeddings, n_components=args.n_components)
                pca_model = pca_result["pca_model"]
                train_result = {
                    "n_negative_images": len(embeddings),
                    "n_components": pca_result["n_components"],
                    "explained_variance_ratio": float(pca_result["explained_variance_ratio"].sum()),
                }
            else:
                print("ERROR: --train-dir required (no cached embeddings found)")
                return 1
        else:
            print("ERROR: --train-dir required for training the PCA subspace")
            return 1

        # Score directory
        print("\n" + "=" * 60)
        print("  Scoring")
        print("=" * 60)
        scoring_results = score_directory(
            pca_model, str(img_dir), device=resolved,
        )

        result = {
            "model": f"{MODEL_NAME} + SubspaceAD (PCA)",
            "device": resolved,
            "n_components": train_result["n_components"],
            "explained_variance_ratio": train_result["explained_variance_ratio"],
            "n_negative_training_images": train_result["n_negative_images"],
            "n_images_scored": len(scoring_results),
            "results": scoring_results,
        }

        # Optionally validate against labels
        if labels_path is not None and labels_path.exists():
            print("\n" + "=" * 60)
            print("  Running validation against labels...")
            val_result = validate(
                pca_model, str(labels_path), str(img_dir), device=resolved,
            )
            result["validation"] = val_result

    # ==================================================================
    # Save or print output
    # ==================================================================
    if args.output_json:
        out_path = _resolve_path(args.output_json)
        if out_path is None:
            out_path = repo_root / "data/subspace_results.json"
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


if __name__ == "__main__":
    sys.exit(main())
