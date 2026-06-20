#!/usr/bin/env python3
"""Zero-shot herring spawn detection using the SHSI (green-minus-red) spectral index.

Herring spawn creates turquoise/milky water. In RGB terms, spawn pixels have
high Green values and low Red values compared to normal ocean water (dark, all
channels low) or sediment (brown, high Red).

The index is::

    score = (G - R) / (G + R + eps)

normalised per pixel.  Only pixels with brightness (G+R+B)/3 > 10 are scored;
dark/near-black pixels are skipped.

Usage:
    # Score all PNGs in a directory
    python scripts/spectral_threshold.py \\
        --image-dir data/samples/unified \\
        --output-json data/spectral_results.json

    # Validate against human labels
    python scripts/spectral_threshold.py \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --output-json data/spectral_results.json

    # Validate only (skip per-image output)
    python scripts/spectral_threshold.py --validate-only \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json

Dependencies:
    pip install numpy Pillow scikit-learn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS = 1e-6
"""Small constant to avoid division by zero in the spectral index."""

BRIGHTNESS_THRESHOLD = 10.0
"""Minimum mean brightness (0-255) for a pixel to be scored (skip dark/near-black)."""

DEFAULT_FRAC_THRESHOLD = 0.1
"""G-R index threshold used for the ``frac_above_threshold`` metric."""

DEFAULT_PREDICTION_THRESHOLD = 0.0
"""Default threshold for converting mean score to binary prediction.
Positive values mean G > R on average, which indicates spawn-like pixels."""


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def _compute_gr_index(rgb: np.ndarray) -> np.ndarray:
    """Compute the green-minus-red index per pixel.

    Args:
        rgb: uint8 array of shape (H, W, 3).

    Returns:
        Float array of shape (H, W) with values in approximately [-1, 1].
    """
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    return (g - r) / (g + r + EPS)


def _brightness_mask(rgb: np.ndarray) -> np.ndarray:
    """Return boolean mask of pixels with mean brightness > threshold."""
    mean_brightness = rgb.astype(np.float32).mean(axis=-1)  # (H, W)
    return mean_brightness > BRIGHTNESS_THRESHOLD


def score_image(image_path: str, threshold: float = DEFAULT_PREDICTION_THRESHOLD) -> dict:
    """Score a single PNG image using the green-minus-red index.

    Args:
        image_path: Path to a PNG image.
        threshold: Decision threshold for binary prediction
            (score > threshold => prediction = 1).

    Returns:
        Dict with keys:
        - ``score``: mean G-R index across all scored pixels
          (higher = more spawn-like).
        - ``percentile_95``: 95th percentile of G-R index (focus on brightest
          patches).
        - ``frac_above_threshold``: fraction of scored pixels with
          G-R index > 0.1.
        - ``top_patch_score``: mean G-R index of the top 10% brightest scored
          pixels.
        - ``prediction``: 1 if score > threshold else 0.
        - ``n_scored_pixels``: number of non-dark pixels scored.
        - ``total_pixels``: total pixels in the image.
        - ``image_path``: input path.
        - ``threshold``: threshold used.
    """
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)  # (H, W, 3)

    total_pixels = rgb.shape[0] * rgb.shape[1]

    # Identify non-dark pixels
    mask = _brightness_mask(rgb)  # (H, W), bool
    n_scored = int(mask.sum())

    if n_scored == 0:
        return {
            "score": 0.0,
            "percentile_95": 0.0,
            "frac_above_threshold": 0.0,
            "top_patch_score": 0.0,
            "prediction": 0,
            "n_scored_pixels": 0,
            "total_pixels": total_pixels,
            "image_path": image_path,
            "threshold": threshold,
        }

    # Compute index only for scored pixels
    gr = _compute_gr_index(rgb)  # (H, W), float32
    gr_scored = gr[mask]

    mean_score = float(np.mean(gr_scored))
    p95 = float(np.percentile(gr_scored, 95))
    frac_above = float(np.mean(gr_scored > DEFAULT_FRAC_THRESHOLD))

    # Top 10% by brightness among scored pixels
    brightness = rgb.astype(np.float32).mean(axis=-1)  # (H, W)
    bright_values = brightness[mask]
    # Threshold for top 10%
    bright_thr = np.percentile(bright_values, 90)
    top_mask = bright_values >= bright_thr
    top_scores = gr_scored[top_mask]
    top_patch_score = float(np.mean(top_scores)) if top_scores.size > 0 else mean_score

    return {
        "score": mean_score,
        "percentile_95": p95,
        "frac_above_threshold": frac_above,
        "top_patch_score": top_patch_score,
        "prediction": 1 if mean_score > threshold else 0,
        "n_scored_pixels": n_scored,
        "total_pixels": total_pixels,
        "image_path": image_path,
        "threshold": threshold,
    }


def score_directory(
    image_dir: str,
    threshold: float = DEFAULT_PREDICTION_THRESHOLD,
) -> list[dict]:
    """Score all PNG images in a directory using the green-minus-red index.

    Args:
        image_dir: Directory containing PNG images.
        threshold: Decision threshold for binary prediction.

    Returns:
        List of score dicts (see :func:`score_image`) sorted by ``score``
        descending.
    """
    img_dir = Path(image_dir)
    pngs = sorted(img_dir.glob("*.png"))
    print(f"  Found {len(pngs)} PNG images in {image_dir}")

    if not pngs:
        return []

    results: list[dict] = []
    errors: list[str] = []

    for p in pngs:
        try:
            result = score_image(str(p), threshold=threshold)
            results.append(result)
        except Exception as exc:
            errors.append(f"{p.name}: {exc}")

    results.sort(key=lambda r: r["score"], reverse=True)

    if errors:
        print(f"  WARNING: {len(errors)} images failed to score")
        for err in errors[:5]:
            print(f"    {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    print(f"  Scored {len(results)}/{len(pngs)} images successfully")
    return results


# ---------------------------------------------------------------------------
# Validation against human labels
# ---------------------------------------------------------------------------


def validate(
    labels_json_path: str,
    image_dir: str,
    threshold: float = DEFAULT_PREDICTION_THRESHOLD,
    quiet: bool = False,
) -> dict:
    """Validate the spectral index against human labels.

    Labels JSON format::

        {"labels": [{"filename": "image.png", "label": 1}, ...]}

    where ``label`` = 1 means positive (spawn) and ``label`` = 0 means
    negative (no spawn).

    Args:
        labels_json_path: Path to labels JSON file.
        image_dir: Directory containing the PNG images referenced in labels.
        threshold: Decision threshold for binary prediction.
        quiet: If True, suppress print statements.

    Returns:
        Dict with accuracy, best_accuracy, best_threshold, auc_roc,
        avg_precision, confusion_matrix, per_sample list, and counts.
        Matching the return format of :func:`remoteclip_zero_shot.validate`.
    """
    if not quiet:
        print(f"  Loading labels from: {labels_json_path}")

    labels_path = Path(labels_json_path)
    if not labels_path.exists():
        msg = f"Labels file not found: {labels_path}"
        print(f"ERROR: {msg}")
        return {"error": msg}

    labels_data = json.loads(labels_path.read_text())
    label_entries = labels_data.get("labels", [])
    if not quiet:
        print(f"  Loaded {len(label_entries)} label entries")

    if not label_entries:
        empty = {
            "accuracy": 0.0,
            "best_accuracy": 0.0,
            "best_threshold": threshold,
            "auc_roc": 0.0,
            "avg_precision": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "per_sample": [],
            "n_total": 0,
            "n_pos": 0,
            "n_neg": 0,
        }
        return empty

    img_dir = Path(image_dir)
    per_sample: list[dict] = []

    for entry in label_entries:
        fname = entry["filename"]
        true_label = entry["label"]
        img_path = img_dir / fname

        if not img_path.exists():
            if not quiet:
                print(f"  WARNING: Image not found: {img_path}")
            continue

        try:
            result = score_image(str(img_path), threshold=threshold)
        except Exception as exc:
            if not quiet:
                print(f"  WARNING: Failed to score {fname}: {exc}")
            continue

        per_sample.append({
            "filename": fname,
            "true_label": true_label,
            "prediction": result["prediction"],
            "score": result["score"],
            "percentile_95": result["percentile_95"],
            "frac_above_threshold": result["frac_above_threshold"],
            "top_patch_score": result["top_patch_score"],
        })

    if not per_sample:
        if not quiet:
            print("  No samples successfully scored.")
        return {
            "accuracy": 0.0,
            "best_accuracy": 0.0,
            "best_threshold": threshold,
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

    if not quiet:
        print(f"\n  Validation results (threshold={threshold}):")
        print(f"    Total: {n_total}  Pos: {n_pos}  Neg: {n_neg}")
        print(f"    Accuracy (thr={threshold}):  {acc:.4f}")
        print(f"    Best accuracy:              {best_acc:.4f} @ thr={best_thr:.4f}")
        print(f"    AUROC:                      {auroc:.4f}")
        print(f"    Avg Precision:              {ap:.4f}")
        print(f"    Confusion Matrix:           {cm}")

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
        "prediction_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-shot herring spawn detection using the SHSI (green-minus-red) spectral index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory of PNG images to score or validate",
    )
    parser.add_argument(
        "--labels-json", type=str, default=None,
        help="Path to labels JSON file for validation",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save output JSON results",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip scoring all images, just run validation against labels",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_PREDICTION_THRESHOLD,
        help=f"Decision threshold for binary prediction (default: {DEFAULT_PREDICTION_THRESHOLD})",
    )
    args = parser.parse_args(argv)

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

    # ----- Resolve --labels-json path -----
    labels_path = None
    if args.labels_json:
        lp = Path(args.labels_json)
        labels_path = repo_root / args.labels_json if not lp.is_absolute() else lp
        if not labels_path.exists():
            print(f"ERROR: Labels file not found: {labels_path}")
            return 1

    # ----- Validate-only mode -----
    if args.validate_only:
        if labels_path is None:
            print("ERROR: --validate-only requires --labels-json")
            return 1

        print("=" * 60)
        print("  SHSI Spectral Index Validation")
        print("=" * 60)
        result = validate(
            str(labels_path),
            str(img_dir),
            threshold=args.threshold,
        )

    else:
        print("=" * 60)
        print("  SHSI Spectral Index Scoring")
        print("=" * 60)

        # Score directory
        results = score_directory(str(img_dir), threshold=args.threshold)
        result = {
            "method": "shsi_green_minus_red",
            "description": "Green-minus-red spectral index for herring spawn detection",
            "threshold": args.threshold,
            "n_images_scored": len(results),
            "results": results,
        }

        # Also validate if labels are available
        if labels_path is not None:
            print("\n" + "-" * 60)
            print("  Running validation against labels...")
            val_result = validate(
                str(labels_path),
                str(img_dir),
                threshold=args.threshold,
            )
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
