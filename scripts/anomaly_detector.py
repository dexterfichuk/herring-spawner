#!/usr/bin/env python3
"""Zero-shot herring spawn detection using one-class anomaly detection.

Trains an IsolationForest or OneClassSVM on pixel-level RGB features extracted
from known negative (no-spawn) images. Then scores candidate images by how
anomalous their pixel distributions are relative to the normal (no-spawn) model.
Herring spawn images are expected to have anomalous bright turquoise pixels that
don't match normal coastal water patterns.

Usage:
    # Train on negatives, score candidates
    python scripts/anomaly_detector.py \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --output-json data/anomaly_results.json

    # Validate only
    python scripts/anomaly_detector.py \\
        --validate-only \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json

    # Score a single directory with a specific method
    python scripts/anomaly_detector.py \\
        --train-dir data/samples/negative \\
        --image-dir data/samples/positive \\
        --output-json data/anomaly_positive_scores.json \\
        --method isolation_forest

Dependencies:
    pip install numpy scikit-learn Pillow tqdm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.svm import OneClassSVM
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPSILON = 1e-8
DEFAULT_N_SAMPLES = 500
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Pixel feature extraction
# ---------------------------------------------------------------------------


def extract_pixel_features(
    image_array: np.ndarray, n_samples: int = DEFAULT_N_SAMPLES
) -> np.ndarray:
    """Extract pixel-level RGB features from an image.

    Randomly samples *n_samples* pixels from the image (or all pixels if the
    image contains fewer than *n_samples*). Each pixel yields an 8-dimensional
    feature vector:

      [R, G, B, G-R, G / (R+G+B+eps), H, S, V]

    where H, S, V are the Hue, Saturation, and Value components from the
    HSV colour space.

    Args:
        image_array: (H, W, 3) uint8 RGB array.
        n_samples: Maximum number of pixels to sample.

    Returns:
        (N, 8) float32 array of pixel features, where N <= n_samples.
    """
    h, w = image_array.shape[:2]
    pixels = image_array.reshape(-1, 3).astype(np.float32)  # (H*W, 3)

    n_total = pixels.shape[0]
    if n_total > n_samples:
        rng = np.random.default_rng(RANDOM_SEED)
        indices = rng.choice(n_total, size=n_samples, replace=False)
        pixels = pixels[indices]

    # --- Raw RGB ---
    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]

    # --- Derived features ---
    g_minus_r = g - r
    g_div_sum = g / (r + g + b + EPSILON)

    # --- HSV ---
    # Manual HSV conversion (faster than per-pixel rgb_to_hsv loop)
    max_rgb = pixels.max(axis=1)
    min_rgb = pixels.min(axis=1)
    diff = max_rgb - min_rgb + EPSILON

    # Hue
    hue = np.zeros(n_total if n_total <= n_samples else n_samples, dtype=np.float32)
    max_mask = max_rgb > 0
    rc = (max_rgb - r) / diff
    gc = (max_rgb - g) / diff
    bc = (max_rgb - b) / diff

    r_max = (r == max_rgb) & (r > 0)
    g_max = (g == max_rgb) & (r != max_rgb)
    b_max = ~(r_max | g_max)

    hue_vals = np.zeros_like(rc)
    hue_vals[r_max] = (bc[r_max] - gc[r_max]) % 6.0
    hue_vals[g_max] = 2.0 + rc[g_max] - bc[g_max]
    hue_vals[b_max] = 4.0 + gc[b_max] - rc[b_max]
    hue = hue_vals * 60.0 * max_mask.astype(np.float32)

    # Saturation (use np.divide with where= to avoid div-by-zero warnings)
    saturation = np.divide(diff, max_rgb, out=np.zeros_like(diff), where=max_rgb > 0).astype(np.float32)

    # Value
    value = (max_rgb / 255.0).astype(np.float32)

    # --- Stack ---
    features = np.column_stack(
        [r, g, b, g_minus_r, g_div_sum, hue, saturation, value]
    ).astype(np.float32)

    return features


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_normal_model(
    negative_dir: str, method: str = "isolation_forest"
) -> tuple:
    """Train an anomaly detection model on pixels extracted from negative images.

    Loads every PNG in *negative_dir*, samples up to ``DEFAULT_N_SAMPLES``
    pixels per image, and trains a one-class model on the pooled pixel
    features.

    Args:
        negative_dir: Directory containing PNG images of known negative
            (no-spawn) scenes.
        method: ``"isolation_forest"`` (default) or ``"oneclass_svm"``.

    Returns:
        (model, feature_dim) where *model* is the trained sklearn estimator
        and *feature_dim* is the number of pixel features (always 8).

    Raises:
        FileNotFoundError: If *negative_dir* contains no PNGs.
    """
    neg_path = Path(negative_dir)
    pngs = sorted(neg_path.glob("*.png"))

    if not pngs:
        raise FileNotFoundError(
            f"No PNG images found in {negative_dir}"
        )

    print(f"  Loading {len(pngs)} negative images from {negative_dir}")

    all_features: list[np.ndarray] = []
    skipped = 0

    for p in tqdm(pngs, desc="Extracting pixel features", unit="img"):
        try:
            img = Image.open(p).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            feats = extract_pixel_features(arr, n_samples=DEFAULT_N_SAMPLES)
            all_features.append(feats)
        except Exception as exc:
            print(f"    WARNING: Skipping {p.name}: {exc}")
            skipped += 1

    if not all_features:
        raise RuntimeError(
            "No pixel features could be extracted from any image in "
            f"{negative_dir}"
        )

    X = np.concatenate(all_features, axis=0)
    n_pixels = X.shape[0]
    print(f"  Pooled {n_pixels} pixels from {len(pngs) - skipped} images")

    print(f"  Training {method} on {n_pixels} pixels...")
    if method == "isolation_forest":
        model = IsolationForest(
            contamination=0.01,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    elif method == "oneclass_svm":
        model = OneClassSVM(
            nu=0.01,
            kernel="rbf",
            gamma="scale",
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'isolation_forest' or "
            f"'oneclass_svm'."
        )

    model.fit(X)
    print(f"  Model trained: {method} on {n_pixels} pixels x {X.shape[1]} features")

    return model, X.shape[1]


# ---------------------------------------------------------------------------
# Image scoring
# ---------------------------------------------------------------------------


def score_image(model, image_path: str) -> dict:
    """Score a single image using a trained anomaly model.

    Extracts pixel features, predicts the anomaly score for each sampled
    pixel, and aggregates into per-image statistics.

    *score* is normalised so that **positive values indicate anomalous
    pixels** (herring-spawn-like):

    - ``IsolationForest``: ``-decision_function(x)``
    - ``OneClassSVM``: ``-decision_function(x)``

    Args:
        model: Trained ``IsolationForest`` or ``OneClassSVM``.
        image_path: Path to a PNG image.

    Returns:
        dict with keys:
        - ``score``: Mean anomaly score across all sampled pixels.
        - ``frac_anomalous``: Fraction of pixels classified as anomalous
          (prediction == -1).
        - ``mean_bg_score``: Mean anomaly score of the least anomalous 80 %
          of pixels (lower tail).
        - ``top_5p_score``: Mean anomaly score of the most anomalous 5 % of
          pixels (upper tail).
        - ``prediction``: 1 if ``score > 0`` else 0.
        - ``image_path``: Input path.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        return {
            "score": 0.0,
            "frac_anomalous": 0.0,
            "mean_bg_score": 0.0,
            "top_5p_score": 0.0,
            "prediction": 0,
            "image_path": image_path,
            "error": str(exc),
        }

    features = extract_pixel_features(arr, n_samples=DEFAULT_N_SAMPLES)

    # decision_function: more negative = more anomalous for both IF and OCSVM
    raw_scores = model.decision_function(features)  # (N,) — neg = anomalous
    anomaly_scores = -raw_scores  # positive = anomalous

    mean_score = float(np.mean(anomaly_scores))

    # Predict: model returns 1 for inliers, -1 for outliers
    preds = model.predict(features)
    frac_anom = float(np.mean(preds == -1))

    # Background: bottom 80 % least anomalous
    sorted_scores = np.sort(anomaly_scores)
    n_bg = max(1, int(0.8 * len(sorted_scores)))
    mean_bg = float(np.mean(sorted_scores[:n_bg]))

    # Top 5 % most anomalous
    n_top5 = max(1, int(0.05 * len(sorted_scores)))
    top5 = float(np.mean(sorted_scores[-n_top5:]))

    # Overall prediction: score > 0 means more anomalous than typical
    prediction = 1 if mean_score > 0 else 0

    return {
        "score": mean_score,
        "frac_anomalous": frac_anom,
        "mean_bg_score": mean_bg,
        "top_5p_score": top5,
        "prediction": prediction,
        "image_path": image_path,
    }


def score_directory(model, image_dir: str) -> list[dict]:
    """Score all PNG images in a directory using a trained anomaly model.

    Args:
        model: Trained ``IsolationForest`` or ``OneClassSVM``.
        image_dir: Directory containing PNG images.

    Returns:
        List of score dicts (see ``score_image``), sorted by *score*
        descending (most anomalous first).
    """
    img_dir = Path(image_dir)
    pngs = sorted(img_dir.glob("*.png"))
    print(f"  Scoring {len(pngs)} images in {image_dir}")

    if not pngs:
        print("  No images to score.")
        return []

    results: list[dict] = []
    for p in tqdm(pngs, desc="Scoring", unit="img"):
        result = score_image(model, str(p))
        results.append(result)

    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"  Scored {len(results)}/{len(pngs)} images")
    return results


# ---------------------------------------------------------------------------
# Validation against human labels
# ---------------------------------------------------------------------------


def validate(
    model, labels_json_path: str, image_dir: str
) -> dict:
    """Validate a trained anomaly model against human-annotated labels.

    Labels JSON format (same as ``remoteclip_zero_shot.py``):

    .. code-block:: json

        {"labels": [{"filename": "image.png", "label": 1}, ...]}

    where label=1 means positive (spawn) and label=0 means negative
    (no spawn).

    Args:
        model: Trained ``IsolationForest`` or ``OneClassSVM``.
        labels_json_path: Path to the labels JSON file.
        image_dir: Directory containing the PNG images referenced in the
            labels file.

    Returns:
        dict with keys: ``accuracy``, ``best_accuracy``, ``best_threshold``,
        ``auc_roc``, ``avg_precision``, ``confusion_matrix``,
        ``per_sample``, ``n_total``, ``n_pos``, ``n_neg``.
    """
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

    img_dir = Path(image_dir)
    per_sample: list[dict] = []

    for entry in tqdm(label_entries, desc="Validating", unit="img"):
        fname = entry["filename"]
        true_label = entry["label"]
        img_path = img_dir / fname

        if not img_path.exists():
            print(f"  WARNING: Image not found: {img_path}")
            continue

        result = score_image(model, str(img_path))
        if result.get("error"):
            print(f"  WARNING: Could not score {fname}: {result['error']}")
            continue

        per_sample.append({
            "filename": fname,
            "true_label": true_label,
            "prediction": result["prediction"],
            "score": result["score"],
            "frac_anomalous": result["frac_anomalous"],
            "mean_bg_score": result["mean_bg_score"],
            "top_5p_score": result["top_5p_score"],
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
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot herring spawn detection using one-class anomaly "
            "detection trained on negative (no-spawn) images"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train-dir", type=str, default=None,
        help=(
            "Directory of known negative (no-spawn) images for training "
            "(default: data/samples/negative)"
        ),
    )
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory of PNG images to score or validate",
    )
    parser.add_argument(
        "--labels-json", type=str, default=None,
        help="Path to validation labels JSON file",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save output JSON results",
    )
    parser.add_argument(
        "--method", type=str, default="isolation_forest",
        choices=["isolation_forest", "oneclass_svm"],
        help=(
            "Anomaly detection method: isolation_forest (default) or "
            "oneclass_svm"
        ),
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip training/new scoring, just run validation against labels",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent

    # ----- Resolve paths -----
    train_dir: str | None = None
    if args.train_dir:
        td = Path(args.train_dir)
        train_dir = str(td if td.is_absolute() else repo_root / args.train_dir)
    else:
        train_dir = str(repo_root / "data/samples/negative")

    img_dir: str | None = None
    if args.image_dir:
        id_ = Path(args.image_dir)
        img_dir = str(id_ if id_.is_absolute() else repo_root / args.image_dir)

    labels_path: str | None = None
    if args.labels_json:
        lp = Path(args.labels_json)
        labels_path = str(lp if lp.is_absolute() else repo_root / args.labels_json)
        if not Path(labels_path).exists():
            print(f"ERROR: Labels file not found: {labels_path}")
            return 1

    # ----- Validate-only path -----
    if args.validate_only:
        if img_dir is None:
            print("ERROR: --validate-only requires --image-dir")
            return 1
        if labels_path is None:
            print("ERROR: --validate-only requires --labels-json")
            return 1

        print("=" * 60)
        print("  Anomaly Detector — Validate Only")
        print("=" * 60)

        # Train model first (needed for scoring)
        print("\n  Training anomaly model on negatives...")
        try:
            model, feat_dim = train_normal_model(train_dir, method=args.method)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"ERROR: {exc}")
            return 1

        print(f"\n  Validating against {labels_path}...")
        result = validate(model, labels_path, img_dir)

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

    # ----- Train + score path -----
    print("=" * 60)
    print("  Anomaly Detector — Training & Scoring")
    print("=" * 60)

    print(f"\n  Training directory: {train_dir}")
    print(f"  Method:             {args.method}")

    try:
        model, feat_dim = train_normal_model(train_dir, method=args.method)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    result: dict = {
        "method": args.method,
        "train_dir": train_dir,
        "feature_dim": feat_dim,
        "training_samples_per_image": DEFAULT_N_SAMPLES,
    }

    # Score images if --image-dir provided
    if img_dir is not None:
        if not Path(img_dir).is_dir():
            print(f"ERROR: Image directory not found: {img_dir}")
            return 1

        print(f"\n  Scoring images in {img_dir}...")
        scores = score_directory(model, img_dir)
        result["n_images_scored"] = len(scores)
        result["results"] = scores

        # Also validate if --labels-json provided
        if labels_path is not None:
            print(f"\n  Validating against {labels_path}...")
            val_result = validate(model, labels_path, img_dir)
            result["validation"] = val_result
    else:
        print("  No --image-dir provided; skipping scoring.")
        result["n_images_scored"] = 0
        result["results"] = []

    # ----- Save output -----
    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = repo_root / args.output_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Results saved to: {out_path}")
    else:
        # Only print summary if no file output
        scores = result.get("results", [])
        if scores:
            print(f"\n  Top 5 most anomalous images:")
            for s in scores[:5]:
                print(
                    f"    {Path(s['image_path']).name}: "
                    f"score={s['score']:.4f}  "
                    f"frac_anom={s['frac_anomalous']:.3f}  "
                    f"pred={s['prediction']}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
