#!/usr/bin/env python3
"""Few-shot herring spawn detection using SubspaceAD + positive similarity.

Extends the zero-shot SubspaceAD (PCA reconstruction residual) by calibrating
the anomaly score using the few labeled positives.

Approach:
  1. Train PCA on negatives (same as zero-shot) — learns the "normal coastal"
     subspace.
  2. For each image, compute:
     - residual: PCA reconstruction residual (anomaly score, higher = more
       anomalous).
     - pos_sim: cosine similarity to mean positive embedding in original
       DINOv2 space.
  3. Calibrated score = residual * pos_sim (both min-max normalised to [0,1]).
     This combines "how anomalous vs normal water" with "how similar to known
     spawn".
  4. Train a LogisticRegression on [residual_normalized, pos_sim_normalized]
     using the few positives + all negatives.

Few-shot evaluation:
  - Hold out --n-positives (default 4) positives for training.
  - Evaluate on the remaining positives + all negatives.
  - Run multiple random splits (default 10) for robust metrics.

Usage:
    # Few-shot evaluation
    python scripts/fewshot_subspace_ad.py --validate-only \\
        --train-dir data/samples/negative \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --n-positives 4 --n-trials 10 --device cpu

    # Score a directory of candidates with few-shot calibration
    python scripts/fewshot_subspace_ad.py \\
        --train-dir data/samples/negative \\
        --pos-dir data/samples/positive \\
        --image-dir data/candidates_knn \\
        --output-json data/fewshot_results.json

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

from scripts.train_classifier import DINO_TRANSFORM, MODEL_NAME, EMBED_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PCA variance ratio target when auto-selecting n_components
AUTO_VARIANCE_TARGET = 0.90

# Default number of PCA components
DEFAULT_N_COMPONENTS = 32

# Cache path for extracted negative embeddings
EMBEDDINGS_CACHE_PATH = "data/embeddings/subspace_embeddings.npz"

# Default prediction threshold (mean reconstruction residual above this = spawn)
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
    image_dir: str, device: str = "auto",
) -> tuple[np.ndarray, list[str]]:
    """Extract DINOv2 ViT-S/14 embeddings for all PNGs in a directory.

    Args:
        image_dir: Directory containing PNG images.
        device: 'auto', 'cuda', or 'cpu'.

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

    print(f"  Loading DINOv2 model...")
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
    return emb_array, filenames


# ---------------------------------------------------------------------------
# PCA training on negatives
# ---------------------------------------------------------------------------

def _auto_select_n_components(
    n_samples: int, n_features: int, variance_target: float = AUTO_VARIANCE_TARGET,
) -> int:
    """Automatically select n_components for PCA.

    Heuristic:
      1. Start with DEFAULT_N_COMPONENTS (32).
      2. Cap at min(n_samples - 1, n_features) to avoid singular covariance.
      3. Cap at n_samples // 2 to prevent memorizing the training set.

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
    suggested = min(DEFAULT_N_COMPONENTS, n_samples // 2, max_possible)
    return max(1, suggested)


def train_pca_on_negatives(
    negative_dir: str, device: str = "auto", n_components: int | None = None,
) -> dict:
    """Train PCA on negative (non-spawn) embeddings.

    Extracts DINOv2 embeddings from all PNGs in negative_dir, fits PCA.

    Args:
        negative_dir: Directory of negative (non-spawn) PNG images.
        device: 'auto', 'cuda', or 'cpu'.
        n_components: Number of PCA components. None = auto-select.

    Returns:
        dict with keys:
        - 'pca_model': fitted PCA object (sklearn.decomposition.PCA)
        - 'mean': ndarray of mean embedding
        - 'components': ndarray of PCA components
        - 'explained_variance': ndarray of explained variance per component
        - 'explained_variance_ratio': ndarray of explained variance ratio
        - 'n_components': int
        - 'n_train': int, number of training samples
        - 'negative_embeddings': np.ndarray of all negative embeddings
        - 'negative_filenames': list of str
    """
    resolved = _resolve_device(device)

    print("=" * 60)
    print("  Few-shot SubspaceAD — Train PCA on Negatives")
    print("=" * 60)
    print(f"  Negative directory: {negative_dir}")
    print(f"  Device: {resolved}")

    embeddings, filenames = extract_dinov2_embeddings(
        negative_dir, device=resolved,
    )

    if len(embeddings) == 0:
        print("ERROR: No negative embeddings extracted.")
        return {"error": "No negative embeddings extracted"}

    n_samples, n_features = embeddings.shape
    print(f"  Training on {n_samples} negative images")

    if n_components is None:
        n_components = _auto_select_n_components(n_samples, n_features)
        print(f"  Auto-selected n_components={n_components} "
              f"(from {n_samples} samples x {n_features} features)")

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
        "negative_embeddings": embeddings,
        "negative_filenames": filenames,
    }


# ---------------------------------------------------------------------------
# Feature computation (residual + pos_sim)
# ---------------------------------------------------------------------------

def compute_features(
    pca_model: PCA, image_dir: str, device: str = "auto",
    positives_dir: str | None = None,
) -> dict:
    """For all images compute: embedding, PCA residual, pos_sim.

    Args:
        pca_model: Fitted PCA object from train_pca_on_negatives.
        image_dir: Directory containing PNG images to score.
        device: 'auto', 'cuda', or 'cpu'.
        positives_dir: Optional directory of positive images used to compute
            the mean positive embedding for pos_sim.

    Returns:
        dict with keys:
        - 'embeddings': np.ndarray (N, EMBED_DIM)
        - 'residuals': np.ndarray (N,) PCA reconstruction residuals
        - 'pos_sims': np.ndarray (N,) cosine similarity to mean positive
          (or zeros if positives_dir not given)
        - 'filenames': list of str
        - 'mean_positive_embedding': np.ndarray or None
    """
    resolved = _resolve_device(device)
    img_dir = Path(image_dir)

    if not img_dir.is_dir():
        print(f"  ERROR: Not a directory: {img_dir}")
        return {"error": f"Not a directory: {image_dir}"}

    pngs = sorted(img_dir.glob("*.png"))
    if not pngs:
        print(f"  No PNG images found in {image_dir}")
        return {"embeddings": np.array([]), "residuals": np.array([]),
                "pos_sims": np.array([]), "filenames": [],
                "mean_positive_embedding": None}

    # ---- Compute mean positive embedding if positives_dir provided ----
    mean_pos_emb: np.ndarray | None = None
    if positives_dir is not None:
        pos_path = Path(positives_dir)
        if pos_path.is_dir():
            pos_embs, pos_fnames = extract_dinov2_embeddings(
                positives_dir, device=resolved,
            )
            if len(pos_embs) > 0:
                mean_pos_emb = np.mean(pos_embs, axis=0)
                print(f"  Computed mean positive embedding from "
                      f"{len(pos_embs)} images")
            else:
                print("  WARNING: No positive embeddings found")
        else:
            print(f"  WARNING: Positives directory not found: {pos_path}")

    # ---- Load DINOv2 ----
    print(f"  Computing features from: {image_dir}")
    model = _load_dinov2_model(resolved)

    embeddings_list: list[np.ndarray] = []
    residuals_list: list[float] = []
    pos_sims_list: list[float] = []
    fnames_list: list[str] = []

    for p in tqdm(pngs, desc="Computing features", unit="img"):
        try:
            img = Image.open(p).convert("RGB")
            tensor = DINO_TRANSFORM(img).unsqueeze(0).to(resolved)
            with torch.no_grad():
                emb = model(tensor)
            emb_np = F.normalize(emb, dim=1).cpu().numpy().flatten()

            # PCA reconstruction residual
            emb_2d = emb_np.reshape(1, -1)
            projected = pca_model.transform(emb_2d)
            reconstructed = pca_model.inverse_transform(projected)
            residual = float(np.mean((emb_2d - reconstructed) ** 2))

            # Cosine similarity to mean positive embedding
            pos_sim = 0.0
            if mean_pos_emb is not None:
                # Both are L2-normalized, so dot product = cosine similarity
                pos_sim = float(np.dot(emb_np, mean_pos_emb))
                # Clamp to [0, 1] since we use L2-normalized unit vectors
                pos_sim = max(0.0, min(1.0, pos_sim))

            embeddings_list.append(emb_np)
            residuals_list.append(residual)
            pos_sims_list.append(pos_sim)
            fnames_list.append(p.name)

        except Exception as exc:
            print(f"  WARNING: Failed to process {p.name}: {exc}")

    if not embeddings_list:
        print("  No images processed successfully.")
        return {"embeddings": np.array([]), "residuals": np.array([]),
                "pos_sims": np.array([]), "filenames": [],
                "mean_positive_embedding": mean_pos_emb}

    return {
        "embeddings": np.stack(embeddings_list).astype(np.float32),
        "residuals": np.array(residuals_list, dtype=np.float32),
        "pos_sims": np.array(pos_sims_list, dtype=np.float32),
        "filenames": fnames_list,
        "mean_positive_embedding": mean_pos_emb,
    }


# ---------------------------------------------------------------------------
# Calibrated scoring
# ---------------------------------------------------------------------------

def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    v_min = values.min()
    v_max = values.max()
    if v_max - v_min < 1e-12:
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


def train_few_shot(
    pca_model: PCA, labels_json_path: str, image_dir: str,
    device: str = "auto", n_train_pos: int = 4,
    n_trials: int = 10, seed: int = 42,
) -> dict:
    """Few-shot evaluation using LogisticRegression on [residual, pos_sim].

    For each trial:
      1. Randomly select n_train_pos positives as training set.
      2. Train LogisticRegression on [residual, pos_sim] features.
      3. Evaluate on remaining positives + all negatives.

    Also compares against zero-shot (residual-only) baseline on same splits.

    Args:
        pca_model: Fitted PCA object from train_pca_on_negatives.
        labels_json_path: Path to labels JSON.
        image_dir: Directory containing labeled PNG images.
        device: 'auto', 'cuda', or 'cpu'.
        n_train_pos: Number of positives to hold out for training.
        n_trials: Number of random split trials.
        seed: Random seed for reproducibility.

    Returns:
        dict with mean/stdev metrics across trials.
    """
    resolved = _resolve_device(device)
    print("=" * 60)
    print("  Few-shot SubspaceAD — Evaluation")
    print("=" * 60)
    print(f"  Device: {resolved}")
    print(f"  n_train_positives: {n_train_pos}")
    print(f"  n_trials: {n_trials}")

    # ---- Load labels ----
    labels_path = Path(labels_json_path)
    if not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        return {"error": f"Labels file not found: {labels_path}"}

    labels_data = json.loads(labels_path.read_text())
    label_entries = labels_data.get("labels", [])
    print(f"  Loaded {len(label_entries)} label entries")

    if not label_entries:
        return {"error": "No labels found"}

    # ---- Compute features for all labeled images ----
    print(f"\n  Computing features for all labeled images...")
    features = compute_features(
        pca_model, image_dir, device=resolved, positives_dir=None,
    )
    if "error" in features:
        return features

    emb_fnames = set(features["filenames"])
    residuals_dict = dict(zip(features["filenames"], features["residuals"]))

    # ---- Match labels to features ----
    matched: list[dict] = []
    for entry in label_entries:
        fname = entry["filename"]
        if fname not in emb_fnames:
            print(f"  WARNING: Labeled image not found in features: {fname}")
            continue
        matched.append({
            "filename": fname,
            "label": entry["label"],
            "residual": residuals_dict[fname],
        })

    if not matched:
        return {"error": "No labeled images matched features"}

    # ---- Build arrays ----
    all_fnames = np.array([m["filename"] for m in matched])
    all_labels = np.array([m["label"] for m in matched], dtype=int)
    all_residuals = np.array([m["residual"] for m in matched], dtype=np.float32)

    # Normalize residuals to [0, 1]
    residuals_norm = min_max_normalize(all_residuals)

    # Identify positives and negatives
    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    n_pos_total = len(pos_indices)
    n_neg_total = len(neg_indices)
    print(f"\n  Total labeled images matched: {len(matched)}")
    print(f"    Positives: {n_pos_total}")
    print(f"    Negatives: {n_neg_total}")

    if n_pos_total < n_train_pos + 1:
        print(f"  WARNING: Only {n_pos_total} positives available, "
              f"need at least {n_train_pos + 1} for train/test split.")
        n_train_pos = max(1, n_pos_total - 1)
        print(f"  Adjusted n_train_positives to {n_train_pos}")

    if n_pos_total < 2 or n_neg_total < 2:
        return {"error": "Need at least 2 positives and 2 negatives for evaluation"}

    # ---- Compute pos_sim features ----
    # We need to compute pos_sim for each trial using the trial's training positives.
    # For zero-shot baseline, we use residuals only.

    rng = np.random.RandomState(seed)

    fewshot_accs: list[float] = []
    fewshot_aurocs: list[float] = []
    fewshot_aps: list[float] = []
    zeroshot_accs: list[float] = []
    zeroshot_aurocs: list[float] = []
    zeroshot_aps: list[float] = []

    # Pre-load DINOv2 for embedding extraction
    print(f"\n  Loading DINOv2 for embedding extraction...")
    model = _load_dinov2_model(resolved)

    # Pre-extract embeddings for all labeled images
    img_dir = Path(image_dir)
    all_embeddings: dict[str, np.ndarray] = {}
    for fname in tqdm(all_fnames, desc="Loading embeddings", unit="img"):
        img_path = img_dir / fname
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = DINO_TRANSFORM(img).unsqueeze(0).to(resolved)
            with torch.no_grad():
                emb = model(tensor)
            all_embeddings[fname] = F.normalize(emb, dim=1).cpu().numpy().flatten()
        except Exception as exc:
            print(f"  WARNING: Failed to load {fname}: {exc}")

    # ---- Run trials ----
    print(f"\n  Running {n_trials} trials...")
    for trial in range(n_trials):
        # Randomly select n_train_pos positives for training
        rng.shuffle(pos_indices)
        train_pos_idx = pos_indices[:n_train_pos]
        test_pos_idx = pos_indices[n_train_pos:]

        # Training set: selected positives + all negatives
        train_idx = np.concatenate([train_pos_idx, neg_indices])
        test_idx = np.concatenate([test_pos_idx, neg_indices])

        # Training labels
        y_train = all_labels[train_idx]

        # Compute mean positive embedding from training positives
        train_pos_fnames = all_fnames[train_pos_idx]
        pos_embs = np.array([all_embeddings[f] for f in train_pos_fnames
                             if f in all_embeddings])
        if len(pos_embs) == 0:
            continue
        mean_pos_emb = np.mean(pos_embs, axis=0)

        # Compute pos_sim for all images
        pos_sims = np.array([
            max(0.0, min(1.0, float(np.dot(all_embeddings.get(f, np.zeros(EMBED_DIM)),
                                           mean_pos_emb))))
            for f in all_fnames
        ], dtype=np.float32)

        # Normalize pos_sim to [0, 1]
        pos_sims_norm = min_max_normalize(pos_sims)

        # Few-shot feature: [residual_norm, pos_sim_norm]
        X_fewshot = np.column_stack([residuals_norm, pos_sims_norm])

        # Zero-shot feature: [residual_norm] only
        X_zeroshot = residuals_norm.reshape(-1, 1)

        # ---- Train logistic regression (few-shot) ----
        lr = LogisticRegression(
            class_weight="balanced", random_state=trial, max_iter=1000,
        )
        lr.fit(X_fewshot[train_idx], y_train)

        # ---- Evaluate few-shot ----
        if len(test_idx) > 0:
            y_test = all_labels[test_idx]
            y_pred_few = lr.predict(X_fewshot[test_idx])
            y_prob_few = lr.predict_proba(X_fewshot[test_idx])[:, 1]

            fewshot_accs.append(float(accuracy_score(y_test, y_pred_few)))

            if len(np.unique(y_test)) > 1:
                fewshot_aurocs.append(float(roc_auc_score(y_test, y_prob_few)))
                fewshot_aps.append(float(average_precision_score(y_test, y_prob_few)))
            else:
                fewshot_aurocs.append(0.0)
                fewshot_aps.append(0.0)

            # ---- Zero-shot baseline on same test set ----
            # Use residual as decision score (higher = more likely spawn)
            y_score_zero = X_zeroshot[test_idx].flatten()
            # Optimal threshold via sweep on training set
            best_thr = 0.0
            best_train_acc = 0.0
            train_scores = X_zeroshot[train_idx].flatten()
            thr_space = np.linspace(
                train_scores.min() - 0.01,
                train_scores.max() + 0.01,
                101,
            )
            for thr in thr_space:
                thr_pred = (train_scores > thr).astype(int)
                thr_acc = accuracy_score(y_train, thr_pred)
                if thr_acc > best_train_acc:
                    best_train_acc = thr_acc
                    best_thr = thr

            y_pred_zero = (y_score_zero > best_thr).astype(int)
            zeroshot_accs.append(float(accuracy_score(y_test, y_pred_zero)))

            if len(np.unique(y_test)) > 1:
                zeroshot_aurocs.append(float(roc_auc_score(y_test, y_score_zero)))
                zeroshot_aps.append(float(average_precision_score(y_test, y_score_zero)))
            else:
                zeroshot_aurocs.append(0.0)
                zeroshot_aps.append(0.0)

    # ---- Aggregate results ----
    if not fewshot_accs:
        return {"error": "No valid trials completed"}

    result = {
        "mean_accuracy": float(np.mean(fewshot_accs)),
        "mean_auc_roc": float(np.mean(fewshot_aurocs)),
        "mean_avg_precision": float(np.mean(fewshot_aps)),
        "std_accuracy": float(np.std(fewshot_accs)),
        "std_auc_roc": float(np.std(fewshot_aurocs)),
        "std_avg_precision": float(np.std(fewshot_aps)),
        "n_train_positives": n_train_pos,
        "n_trials": len(fewshot_accs),
        "n_total_positives": n_pos_total,
        "n_total_negatives": n_neg_total,
        "n_total_images": len(matched),
        "zeroshot_baseline": {
            "mean_accuracy": float(np.mean(zeroshot_accs)),
            "mean_auc_roc": float(np.mean(zeroshot_aurocs)),
            "mean_avg_precision": float(np.mean(zeroshot_aps)),
            "std_accuracy": float(np.std(zeroshot_accs)),
            "std_auc_roc": float(np.std(zeroshot_aurocs)),
            "std_avg_precision": float(np.std(zeroshot_aps)),
        },
        "per_trial_fewshot": {
            "accuracies": fewshot_accs,
            "auc_rocs": fewshot_aurocs,
            "avg_precisions": fewshot_aps,
        },
        "per_trial_zeroshot": {
            "accuracies": zeroshot_accs,
            "auc_rocs": zeroshot_aurocs,
            "avg_precisions": zeroshot_aps,
        },
    }

    # Print summary
    print(f"\n  {'=' * 60}")
    print(f"  Few-shot SubspaceAD Results ({n_train_pos} training positives)")
    print(f"  {'=' * 60}")
    print(f"  Few-shot (residual + pos_sim):")
    print(f"    Accuracy:        {result['mean_accuracy']:.4f} +/- "
          f"{result['std_accuracy']:.4f}")
    print(f"    AUROC:           {result['mean_auc_roc']:.4f} +/- "
          f"{result['std_auc_roc']:.4f}")
    print(f"    Avg Precision:   {result['mean_avg_precision']:.4f} +/- "
          f"{result['std_avg_precision']:.4f}")
    print(f"  Zero-shot (residual only):")
    zb = result["zeroshot_baseline"]
    print(f"    Accuracy:        {zb['mean_accuracy']:.4f} +/- "
          f"{zb['std_accuracy']:.4f}")
    print(f"    AUROC:           {zb['mean_auc_roc']:.4f} +/- "
          f"{zb['std_auc_roc']:.4f}")
    print(f"    Avg Precision:   {zb['mean_avg_precision']:.4f} +/- "
          f"{zb['std_avg_precision']:.4f}")
    print(f"  {'=' * 60}")

    return result


# ---------------------------------------------------------------------------
# Single-image segmentation
# ---------------------------------------------------------------------------

def segment_image(
    pca_model: PCA, image_path: str, device: str = "auto",
) -> dict:
    """Score + segment a single image using PCA reconstruction residual
    and per-patch anomalies.

    For each 16x16 patch, compute the PCA residual, then upsample to 224x224.
    The segmentation mask highlights patches with residual above
    mean + 2*std of the patch residuals.

    Args:
        pca_model: Fitted PCA object from train_pca_on_negatives.
        image_path: Path to a single PNG image.
        device: 'auto', 'cuda', or 'cpu'.

    Returns:
        dict with keys:
        - 'score': float, overall image reconstruction residual.
        - 'patch_residuals': np.ndarray (16, 16) of per-patch residuals.
        - 'heatmap': np.ndarray (224, 224) of per-pixel anomaly scores.
        - 'mask': np.ndarray (224, 224) of binary spawn mask.
        - 'spawn_area_frac': float, fraction of image classified as spawn.
        - 'prediction': int, 1 if score > DEFAULT_PREDICTION_THRESHOLD else 0.
    """
    resolved = _resolve_device(device)

    # ---- Load and embed image ----
    img = Image.open(image_path).convert("RGB")
    tensor = DINO_TRANSFORM(img).unsqueeze(0).to(resolved)

    dinov2 = _load_dinov2_model(resolved)

    with torch.no_grad():
        # Get intermediate layers with reshape for patch tokens
        patch_tokens, cls_tokens = dinov2.get_intermediate_layers(
            tensor, n=1, reshape=True, return_class_token=True,
        )[0]

    # patch_tokens: [1, 384, 16, 16] -> [256, 384]
    pt = (patch_tokens
          .flatten(2)           # [1, 384, 256]
          .transpose(1, 2)      # [1, 256, 384]
          .squeeze(0)           # [256, 384]
          .cpu()
          .numpy()
          .astype(np.float32))  # (256, 384)

    # ---- Compute per-patch PCA reconstruction residuals ----
    projected = pca_model.transform(pt)                        # (256, n_components)
    reconstructed = pca_model.inverse_transform(projected)     # (256, 384)

    # Per-patch MSE residual
    patch_residuals = np.mean((pt - reconstructed) ** 2, axis=1)  # (256,)

    # Reshape to 16x16 grid
    patch_grid = patch_residuals.reshape(16, 16)  # (16, 16)

    # Overall score (mean residual of CLS token)
    cls_emb = cls_tokens.cpu().numpy().flatten()
    cls_2d = cls_emb.reshape(1, -1)
    cls_proj = pca_model.transform(cls_2d)
    cls_recon = pca_model.inverse_transform(cls_proj)
    score = float(np.mean((cls_2d - cls_recon) ** 2))

    # ---- Upsample heatmap to 224x224 ----
    heatmap_tensor = torch.from_numpy(patch_grid).float().unsqueeze(0).unsqueeze(0)
    # (1, 1, 16, 16) -> (1, 1, 224, 224) bilinear
    heatmap_up = F.interpolate(
        heatmap_tensor, size=(224, 224), mode="bilinear", align_corners=False,
    )
    heatmap = heatmap_up.squeeze().cpu().numpy()  # (224, 224)

    # ---- Auto-threshold for binary mask ----
    # Use mean + 2*std of patch residuals as threshold
    auto_threshold = float(np.mean(patch_residuals) + 2.0 * np.std(patch_residuals))
    mask = (heatmap > auto_threshold).astype(np.float32)

    spawn_area_frac = float(np.mean(mask))
    prediction = 1 if score > DEFAULT_PREDICTION_THRESHOLD else 0

    return {
        "score": score,
        "patch_residuals": patch_grid,  # (16, 16)
        "heatmap": heatmap,             # (224, 224)
        "mask": mask,                   # (224, 224)
        "auto_threshold": auto_threshold,
        "spawn_area_frac": spawn_area_frac,
        "n_spawn_patches": int((patch_grid > auto_threshold).sum()),
        "prediction": prediction,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Few-shot herring spawn detection via SubspaceAD + positive similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train-dir", type=str, default=None,
        help="Directory of negative (non-spawn) PNG images for training PCA subspace",
    )
    parser.add_argument(
        "--pos-dir", type=str, default=None,
        help="Directory of positive (spawn) PNG images for computing mean positive embedding",
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
        "--n-positives", type=int, default=4,
        help="Number of positives to use for few-shot training (default: 4)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=10,
        help="Number of random split trials for robust evaluation (default: 10)",
    )
    parser.add_argument(
        "--n-components", type=int, default=None,
        help="Number of PCA components (default: auto-select)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Run few-shot evaluation against labels instead of per-image scoring",
    )
    parser.add_argument(
        "--segment", action="store_true",
        help="Also output per-patch segmentation for each scored image",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on (default: auto)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
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
    pos_dir = _resolve_path(args.pos_dir)

    # ==================================================================
    # Mode 1: Validate-only (few-shot evaluation)
    # ==================================================================
    if args.validate_only:
        if args.labels_json is None:
            print("ERROR: --validate-only requires --labels-json")
            return 1
        if img_dir is None:
            print("ERROR: --validate-only requires --image-dir")
            return 1

        # Default to negatives directory for training
        if train_dir is None:
            default_train = repo_root / "data/samples/negative"
            if default_train.is_dir():
                train_dir = default_train
                print(f"  Using default train dir: {train_dir}")
            else:
                print("ERROR: --validate-only requires --train-dir or data/samples/negative")
                return 1

        # Train PCA on negatives
        train_result = train_pca_on_negatives(
            str(train_dir),
            n_components=args.n_components,
            device=resolved,
        )
        if "error" in train_result:
            print(f"\nERROR: Training failed: {train_result['error']}")
            return 1

        pca_model = train_result["pca_model"]

        # Few-shot evaluation
        print("\n" + "=" * 60)
        print("  Few-shot Evaluation")
        print("=" * 60)
        fewshot_result = train_few_shot(
            pca_model,
            labels_json_path=str(labels_path),
            image_dir=str(img_dir),
            device=resolved,
            n_train_pos=args.n_positives,
            n_trials=args.n_trials,
            seed=args.seed,
        )

        if "error" in fewshot_result:
            print(f"\nERROR: Few-shot evaluation failed: {fewshot_result['error']}")
            return 1

        result = fewshot_result

    # ==================================================================
    # Mode 2: Score directory with few-shot calibration
    # ==================================================================
    else:
        if img_dir is None:
            print("ERROR: --image-dir is required")
            return 1
        if train_dir is None:
            default_train = repo_root / "data/samples/negative"
            if default_train.is_dir():
                train_dir = default_train
                print(f"  Using default train dir: {train_dir}")
            else:
                print("ERROR: --train-dir is required")
                return 1

        # Default positives directory
        if pos_dir is None:
            default_pos = repo_root / "data/samples/positive"
            if default_pos.is_dir():
                pos_dir = default_pos
                print(f"  Using default pos dir: {pos_dir}")

        # Train PCA on negatives
        train_result = train_pca_on_negatives(
            str(train_dir),
            n_components=args.n_components,
            device=resolved,
        )
        if "error" in train_result:
            print(f"\nERROR: Training failed: {train_result['error']}")
            return 1

        pca_model = train_result["pca_model"]

        # Compute features
        print("\n" + "=" * 60)
        print("  Scoring with Few-shot Calibration")
        print("=" * 60)
        features = compute_features(
            pca_model, str(img_dir), device=resolved,
            positives_dir=str(pos_dir) if pos_dir else None,
        )
        if "error" in features:
            print(f"\nERROR: Feature computation failed: {features['error']}")
            return 1

        if len(features["filenames"]) == 0:
            print("  No images to score.")
            return 0

        # Normalize residuals and pos_sims
        residuals_norm = min_max_normalize(features["residuals"])
        pos_sims_norm = min_max_normalize(features["pos_sims"])

        # Calibrated score = residual * pos_sim
        calibrated_scores = residuals_norm * pos_sims_norm

        # Build results
        scoring_results: list[dict] = []
        for i, fname in enumerate(features["filenames"]):
            scoring_results.append({
                "filename": fname,
                "residual": float(features["residuals"][i]),
                "residual_normalized": float(residuals_norm[i]),
                "pos_sim": float(features["pos_sims"][i]),
                "pos_sim_normalized": float(pos_sims_norm[i]),
                "calibrated_score": float(calibrated_scores[i]),
            })

        scoring_results.sort(key=lambda r: r["calibrated_score"], reverse=True)

        # Optionally segment each image
        if args.segment:
            print(f"\n  Computing per-image segmentations...")
            for r in tqdm(scoring_results, desc="Segmenting", unit="img"):
                img_path = Path(str(img_dir)) / r["filename"]
                try:
                    seg_result = segment_image(
                        pca_model, str(img_path), device=resolved,
                    )
                    r["segmentation"] = {
                        "score": seg_result["score"],
                        "spawn_area_frac": seg_result["spawn_area_frac"],
                        "n_spawn_patches": seg_result["n_spawn_patches"],
                        "auto_threshold": seg_result["auto_threshold"],
                        "prediction": seg_result["prediction"],
                    }
                except Exception as exc:
                    print(f"  WARNING: Segmentation failed for {r['filename']}: {exc}")

        result = {
            "model": f"{MODEL_NAME} + Few-shot SubspaceAD",
            "device": resolved,
            "n_components": train_result["n_components"],
            "n_negative_training_images": train_result["n_train"],
            "n_images_scored": len(scoring_results),
            "mean_positive_embedding_computed": features["mean_positive_embedding"] is not None,
            "results": scoring_results,
        }

        if features["mean_positive_embedding"] is not None:
            result["mean_positive_embedding_norm"] = float(
                np.linalg.norm(features["mean_positive_embedding"])
            )

        # Print top 5
        print(f"\n  Top 5 calibrated scores:")
        for r in scoring_results[:5]:
            print(f"    {r['calibrated_score']:.6f}  "
                  f"(res={r['residual']:.6f}, sim={r['pos_sim']:.4f})  "
                  f"{r['filename']}")

    # ==================================================================
    # Save or print output
    # ==================================================================
    if args.output_json:
        out_path = _resolve_path(args.output_json)
        if out_path is None:
            out_path = repo_root / "data/fewshot_subspace_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Results saved to: {out_path}")
    elif not args.validate_only:
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
