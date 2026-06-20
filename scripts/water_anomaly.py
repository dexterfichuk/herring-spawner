#!/usr/bin/env python3
"""Zero-shot herring spawn detection using local water anomaly scoring.

For each scene, compute the mean color of "background water" pixels (dark
pixels, likely open water). Then score each pixel by how anomalously
bright/turquoise it is relative to that background. Herring spawn = bright
turquoise patches that deviate strongly from surrounding dark water.

No model weights needed — purely pixel-based anomaly detection.

Usage:
    # Score all PNGs in a directory
    python scripts/water_anomaly.py \\
        --image-dir data/samples/positive \\
        --output-json data/water_anomaly_results.json

    # Validate against human labels
    python scripts/water_anomaly.py \\
        --image-dir data/samples/ \\
        --labels-json data/samples/remoteclip_labels.json \\
        --output-json data/water_anomaly_validation.json

    # Validate only (skip per-image output)
    python scripts/water_anomaly.py \\
        --validate-only \\
        --image-dir data/samples/ \\
        --labels-json data/samples/remoteclip_labels.json

    # Use simple brightness z-score method instead of Mahalanobis
    python scripts/water_anomaly.py \\
        --image-dir data/samples/ \\
        --method simple \\
        --output-json results.json

Dependencies: numpy, Pillow (standard scientific Python stack)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pixels with max(R,G,B) below this threshold are considered "background water"
BG_WATER_THRESHOLD = 80

# Mahalanobis distance threshold for "strong outlier"
ANOMALY_OUTLIER_THRESHOLD = 3.0

# Regularization added to covariance diagonal for numerical stability
COV_REG = 1e-6

# Default threshold for prediction (score above this = spawn)
DEFAULT_PREDICTION_THRESHOLD = 3.0

# Fraction of top anomalous pixels to average for the final score
TOP_FRACTION = 0.10


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def _load_image(path: str) -> np.ndarray | None:
    """Load a PNG image as an (H, W, 3) uint8 numpy array.

    Returns None on failure (corrupt file, wrong format, etc.).
    """
    try:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        return arr
    except Exception:
        return None


def _identify_bg_water(
    image: np.ndarray, bg_threshold: int = BG_WATER_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify background water pixels and return their RGB values.

    Background water = pixels where max(R, G, B) < bg_threshold.
    These are the dark ocean pixels that serve as the baseline distribution.

    Args:
        image: (H, W, 3) uint8 array.
        bg_threshold: Max pixel brightness to classify as background water.

    Returns:
        (bg_pixels, mask) where:
        - bg_pixels: (N, 3) float64 array of background water RGB values.
        - mask: (H, W) bool array, True where pixel is background water.
    """
    rgb = image.astype(np.float64)
    mask = image.max(axis=2) < bg_threshold
    bg_pixels = rgb[mask]
    return bg_pixels, mask


def _mahalanobis_distances(
    pixels: np.ndarray, mean: np.ndarray, cov: np.ndarray,
) -> np.ndarray:
    """Compute Mahalanobis distance for each pixel relative to a distribution.

    d = sqrt((x - mu)^T @ inv(cov) @ (x - mu))

    Uses np.linalg.solve for numerical stability instead of explicitly
    computing the inverse.

    Args:
        pixels: (N, D) array of pixel RGB values.
        mean: (D,) array of distribution mean.
        cov: (D, D) array of covariance matrix (regularized).

    Returns:
        (N,) array of Mahalanobis distances.
    """
    centered = pixels - mean  # (N, D)
    # Solve: cov @ x = centered  →  x = inv(cov) @ centered
    # Then distance^2 = sum(centered * x, axis=1)
    try:
        solved = np.linalg.solve(cov, centered.T)  # (D, N)
        dist_sq = np.sum(centered * solved.T, axis=1)
    except np.linalg.LinAlgError:
        # Fallback: pseudo-inverse if singular despite regularization
        cov_inv = np.linalg.pinv(cov)
        dist_sq = np.sum(centered @ cov_inv * centered, axis=1)

    # Guard against tiny negatives from numerical noise
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq)


def score_image(
    image_path: str,
    anomaly_method: str = "mahalanobis",
    bg_threshold: int = BG_WATER_THRESHOLD,
) -> dict:
    """Score a single PNG using local water anomaly.

    Steps:
    1. Load image as numpy array.
    2. Identify background water pixels: max(R,G,B) < bg_threshold (dark
       ocean).
    3. Compute mean and covariance of background water pixels.
    4. For each non-background pixel, compute anomaly distance.
    5. Score = mean of top 10% anomaly distances (higher = more anomalous).

    Args:
        image_path: Path to PNG image.
        anomaly_method: "mahalanobis" (RGB Mahalanobis distance) or
            "simple" (brightness z-score).
        bg_threshold: Max pixel brightness to classify as background water.

    Returns:
        dict with keys:
        - 'score': anomaly score (higher = more spawn-like)
        - 'n_bg_pixels': number of background water pixels identified
        - 'n_anomalous_pixels': count of pixels with Mahalanobis > 3
        - 'frac_anomalous': fraction of non-bg pixels that are anomalous
        - 'mean_bg_brightness': mean brightness of background water
        - 'prediction': 1 if score > threshold else 0
        - 'image_path': input path
        - 'error': error message if processing failed (only on failure)
    """
    # Load image
    image = _load_image(image_path)
    if image is None:
        return {
            "score": 0.0,
            "n_bg_pixels": 0,
            "n_anomalous_pixels": 0,
            "frac_anomalous": 0.0,
            "mean_bg_brightness": 0.0,
            "prediction": 0,
            "image_path": image_path,
            "error": "Failed to load image",
        }

    h, w, _ = image.shape
    total_pixels = h * w

    # Identify background water pixels
    bg_pixels, bg_mask = _identify_bg_water(image, bg_threshold)
    n_bg = bg_pixels.shape[0]

    if n_bg == 0:
        # No dark water found — image is all bright (clouds, land, ice, etc.)
        return {
            "score": 0.0,
            "n_bg_pixels": 0,
            "n_anomalous_pixels": 0,
            "frac_anomalous": 0.0,
            "mean_bg_brightness": float(image.mean()),
            "prediction": 0,
            "image_path": image_path,
            "error": "No background water pixels found (image too bright)",
        }

    if n_bg < 100:
        # Very few dark pixels — unreliable background estimate
        mean_bg_brightness = float(bg_pixels.mean())

        return {
            "score": 0.0,
            "n_bg_pixels": n_bg,
            "n_anomalous_pixels": 0,
            "frac_anomalous": 0.0,
            "mean_bg_brightness": mean_bg_brightness,
            "prediction": 0,
            "image_path": image_path,
            "error": f"Too few background water pixels ({n_bg}), estimate unreliable",
        }

    # Compute background statistics
    bg_mean = bg_pixels.mean(axis=0)  # (3,)
    mean_bg_brightness = float(bg_mean.mean())

    # Non-background pixels (potentially anomalous)
    non_bg_mask = ~bg_mask
    non_bg_pixels = image.astype(np.float64)[non_bg_mask]

    if anomaly_method == "simple":
        # Simple brightness z-score: how many std above background mean?
        bg_std = bg_pixels.std(axis=0).mean()  # scalar: mean std across channels
        if bg_std < 1e-10:
            bg_std = 1.0
        pixel_brightness = non_bg_pixels.mean(axis=1)  # mean R,G,B per pixel
        distances = (pixel_brightness - bg_mean.mean()) / bg_std
        # Clip negative distances (darker than bg) to 0 — we only care about
        # anomalously bright pixels
        distances = np.maximum(distances, 0.0)
    else:
        # Mahalanobis distance in RGB space
        bg_cov = np.cov(bg_pixels, rowvar=False)  # (3, 3)
        # Regularize for numerical stability
        bg_cov += np.eye(3) * COV_REG * np.trace(bg_cov) / 3.0

        distances = _mahalanobis_distances(non_bg_pixels, bg_mean, bg_cov)

    if len(distances) == 0:
        # All pixels were classified as background water
        return {
            "score": 0.0,
            "n_bg_pixels": n_bg,
            "n_anomalous_pixels": 0,
            "frac_anomalous": 0.0,
            "mean_bg_brightness": mean_bg_brightness,
            "prediction": 0,
            "image_path": image_path,
        }

    # Sort distances descending, take top fraction
    distances_sorted = np.sort(distances)[::-1]
    n_top = max(1, int(len(distances_sorted) * TOP_FRACTION))
    top_distances = distances_sorted[:n_top]
    score_val = float(top_distances.mean())

    # Count strong outliers
    n_anomalous = int((distances > ANOMALY_OUTLIER_THRESHOLD).sum())
    frac_anomalous = float(n_anomalous / len(distances))

    # Prediction: 1 if score exceeds default threshold
    prediction = 1 if score_val > DEFAULT_PREDICTION_THRESHOLD else 0

    return {
        "score": score_val,
        "n_bg_pixels": n_bg,
        "n_anomalous_pixels": n_anomalous,
        "frac_anomalous": frac_anomalous,
        "mean_bg_brightness": mean_bg_brightness,
        "prediction": prediction,
        "image_path": image_path,
    }


# ---------------------------------------------------------------------------
# Directory scoring
# ---------------------------------------------------------------------------

def score_directory(
    image_dir: str, method: str = "mahalanobis",
    bg_threshold: int = BG_WATER_THRESHOLD,
) -> list[dict]:
    """Score all PNGs in a directory using local water anomaly.

    Args:
        image_dir: Directory containing PNG images.
        method: "mahalanobis" or "simple".
        bg_threshold: Max pixel brightness to classify as background water.

    Returns:
        list of score dicts sorted by score descending.
    """
    img_dir = Path(image_dir)
    if not img_dir.is_dir():
        print(f"  WARNING: Not a directory: {image_dir}")
        return []

    pngs = sorted(img_dir.glob("*.png"))
    print(f"  Found {len(pngs)} PNG images in {image_dir}")

    if not pngs:
        print("  No images to score.")
        return []

    results: list[dict] = []
    for p in pngs:
        result = score_image(str(p), anomaly_method=method, bg_threshold=bg_threshold)
        results.append(result)

    results.sort(key=lambda r: r["score"], reverse=True)

    n_ok = sum(1 for r in results if "error" not in r)
    print(f"  Scored {n_ok}/{len(results)} images successfully")
    print(f"  Top 3 scores:")
    for r in results[:3]:
        status = "OK" if "error" not in r else f"ERROR: {r['error']}"
        print(f"    {r['score']:.2f}  {Path(r['image_path']).name}  [{status}]")

    return results


# ---------------------------------------------------------------------------
# Validation against human labels
# ---------------------------------------------------------------------------

def validate(
    labels_json_path: str, image_dir: str,
    bg_threshold: int = BG_WATER_THRESHOLD,
) -> dict:
    """Validate anomaly scoring against human labels.

    Labels JSON format (matching remoteclip_zero_shot.py convention):
        {"labels": [{"filename": "image.png", "label": 1}, ...]}
    where label=1 means positive (spawn), label=0 means negative (no spawn).

    Returns dict with the same schema as
    ``remoteclip_zero_shot.validate()``:
        accuracy, best_accuracy, best_threshold, auc_roc, avg_precision,
        confusion_matrix, per_sample, n_total, n_pos, n_neg.
    """
    # Load labels
    labels_path = Path(labels_json_path)
    if not labels_path.exists():
        msg = f"Labels file not found: {labels_path}"
        print(f"ERROR: {msg}")
        return {"error": msg}

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

    for entry in label_entries:
        fname = entry["filename"]
        true_label = entry["label"]
        img_path = img_dir / fname

        if not img_path.exists():
            print(f"  WARNING: Image not found: {img_path}")
            continue

        result = score_image(str(img_path), bg_threshold=bg_threshold)
        if "error" in result:
            continue

        per_sample.append({
            "filename": fname,
            "true_label": true_label,
            "prediction": result["prediction"],
            "score": result["score"],
            "n_bg_pixels": result["n_bg_pixels"],
            "n_anomalous_pixels": result["n_anomalous_pixels"],
            "frac_anomalous": result["frac_anomalous"],
            "mean_bg_brightness": result["mean_bg_brightness"],
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

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        roc_auc_score,
    )

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Best accuracy via threshold sweep
    if y_score.min() == y_score.max():
        # All scores identical — no meaningful sweep
        best_acc = acc
        best_thr = float(y_score[0]) if len(y_score) > 0 else 0.0
    else:
        thresholds = np.linspace(
            y_score.min() - 0.1, y_score.max() + 0.1, 201
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
    print(f"    Best accuracy:         {best_acc:.4f} @ thr={best_thr:.4f}")
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
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-shot herring spawn detection via local water anomaly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        "--method", type=str, default="mahalanobis",
        choices=["mahalanobis", "simple"],
        help=(
            "Anomaly detection method. 'mahalanobis' uses full RGB covariance "
            "(default), 'simple' uses brightness z-score."
        ),
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip per-image scoring output, just run validation against labels",
    )
    parser.add_argument(
        "--bg-threshold", type=int, default=BG_WATER_THRESHOLD,
        help=f"Max pixel brightness to classify as background water (default: {BG_WATER_THRESHOLD})",
    )
    args = parser.parse_args(argv)

    bg_threshold = args.bg_threshold

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

    method = args.method

    if args.validate_only:
        # ----- Validate-only mode -----
        if labels_path is None:
            print("ERROR: --validate-only requires --labels-json")
            return 1

        print("=" * 60)
        print("  Water Anomaly Validation")
        print("=" * 60)
        print(f"  Method: {method}")
        print(f"  Background threshold: max(R,G,B) < {bg_threshold}")
        result = validate(str(labels_path), str(img_dir), bg_threshold=bg_threshold)
    else:
        # ----- Scoring mode -----
        print("=" * 60)
        print("  Water Anomaly Herring Spawn Scoring")
        print("=" * 60)
        print(f"  Method: {method}")
        print(f"  Background threshold: max(R,G,B) < {bg_threshold}")
        print()

        results = score_directory(str(img_dir), method=method, bg_threshold=bg_threshold)

        result = {
            "method": method,
            "bg_water_threshold": bg_threshold,
            "n_images_scored": len(results),
            "results": results,
        }

        if labels_path is not None:
            print("\n" + "=" * 60)
            print("  Running validation against labels...")
            val_result = validate(str(labels_path), str(img_dir), bg_threshold=bg_threshold)
            result["validation"] = val_result

    # ----- Save or print output -----
    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = repo_root / args.output_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Results saved to: {out_path}")
    elif not args.validate_only:
        # In scoring mode without --output-json, print summary
        print(json.dumps(result, indent=2, default=str))
    else:
        # In validate-only mode, print the validation result already shown
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
