#!/usr/bin/env python3
"""Zero-shot herring spawn detection using Spectral Angle Mapping (SAM).

Spectral Angle Mapping computes the angle between a pixel's RGB vector and a
reference "spawn prototype" spectrum. A low angle means the pixel's spectral
shape is similar to the prototype — independently of brightness.

The key insight from Qi et al. (2021): herring-spawn waters have a distinct
reflectance shape in the visible range (490, 510, 560 nm). The spectral angle
captures the shape of the spectrum independently of brightness — spawn has a
characteristic "curve" even if it varies in intensity.

Usage:
    # Build prototype from positive samples and score all PNGs in a directory
    python scripts/spectral_angle.py \\
        --image-dir data/samples/unified \\
        --output-json data/sam_results.json

    # Validate against human labels
    python scripts/spectral_angle.py \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --output-json data/sam_validation.json

    # Validate only (skip per-image scoring output)
    python scripts/spectral_angle.py --validate-only \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json

    # Use a custom positive directory for building the prototype
    python scripts/spectral_angle.py \\
        --positive-dir data/samples/positive \\
        --image-dir data/samples/unified \\
        --output-json data/sam_results.json

Dependencies: numpy, Pillow, scikit-learn
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

EPS = 1e-8
"""Small constant to avoid division by zero in spectral angle computation."""

DEFAULT_BRIGHTNESS_THRESHOLD = 30.0
"""Minimum mean brightness (0-255) for a pixel to be scored (dark pixels have
unreliable spectral angles)."""

SPAWN_PIXEL_BRIGHTNESS_THRESHOLD = 30.0
"""Minimum mean brightness for a pixel to be considered spawn-like when
building the prototype."""

LOW_ANGLE_THRESHOLD = 0.5
"""Spectral angle threshold (radians) for ``frac_low_angle`` metric."""

DEFAULT_PREDICTION_THRESHOLD = 0.0
"""Default threshold for converting the mean score to binary prediction.
The score is -mean_angle, so score > 0 means mean_angle < 0, which is always
true since angles are non-negative. The actual threshold sweep in validate()
finds the optimal threshold automatically."""


# ---------------------------------------------------------------------------
# Prototype computation
# ---------------------------------------------------------------------------


def compute_spawn_prototype(positive_dir: str) -> tuple[np.ndarray, int]:
    """Compute the mean RGB spectrum of spawn pixels from positive images.

    Steps:
    1. For each positive PNG, load and extract pixels that look spawn-like
       (high G, low R, turquoise appearance).
    2. Take the mean RGB vector across all such pixels.
    3. Normalize to unit length.

    Args:
        positive_dir: Directory containing positive (spawn) PNG images.

    Returns:
        (prototype, n_pixels_used) where:
        - prototype: np.ndarray shape (3,), unit-normalized RGB spectrum.
        - n_pixels_used: total number of spawn-like pixels used.
    """
    pos_dir = Path(positive_dir)
    if not pos_dir.is_dir():
        print(f"ERROR: Positive directory not found: {pos_dir}")
        return np.zeros(3), 0

    pngs = sorted(pos_dir.glob("*.png"))
    if not pngs:
        print(f"ERROR: No PNG files found in {pos_dir}")
        return np.zeros(3), 0

    print(f"  Computing spawn prototype from {len(pngs)} positive images...")

    all_spawn_pixels: list[np.ndarray] = []

    for p in pngs:
        try:
            img = Image.open(str(p)).convert("RGB")
            rgb = np.array(img, dtype=np.float32)  # (H, W, 3)
        except Exception as exc:
            print(f"  WARNING: Could not load {p.name}: {exc}")
            continue

        # Identify spawn-like pixels: Green > Red (turquoise) and bright enough
        brightness = rgb.mean(axis=-1)  # (H, W)
        g_gt_r = rgb[..., 1] > rgb[..., 0]  # Green channel > Red channel
        bright_enough = brightness > SPAWN_PIXEL_BRIGHTNESS_THRESHOLD
        spawn_mask = g_gt_r & bright_enough

        n_candidate = int(spawn_mask.sum())
        if n_candidate == 0:
            print(f"  WARNING: No spawn-like pixels found in {p.name}")
            continue

        spawn_pixels = rgb[spawn_mask]  # (N, 3)
        all_spawn_pixels.append(spawn_pixels)

    if not all_spawn_pixels:
        print("  ERROR: No spawn-like pixels found in any positive image.")
        return np.zeros(3), 0

    all_pixels = np.concatenate(all_spawn_pixels, axis=0)  # (N_total, 3)
    n_pixels_used = all_pixels.shape[0]

    # Mean RGB vector
    prototype = all_pixels.mean(axis=0)  # (3,)

    # Normalize to unit length
    norm = np.linalg.norm(prototype)
    if norm < EPS:
        print("  WARNING: Prototype norm is near zero, using unnormalized mean.")
        return prototype, n_pixels_used

    prototype = prototype / norm

    print(f"  Prototype: R={prototype[0]:.4f}  G={prototype[1]:.4f}  B={prototype[2]:.4f}")
    print(f"  Pixels used: {n_pixels_used}")
    print(f"  Norm: {np.linalg.norm(prototype):.4f} (should be 1.0)")

    return prototype, n_pixels_used


# ---------------------------------------------------------------------------
# Spectral angle computation
# ---------------------------------------------------------------------------


def spectral_angle(pixel_rgb: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    """Compute spectral angle between pixel(s) and prototype.

    The spectral angle is defined as::

        angle = arccos(dot(pixel, prototype) / (norm(pixel) * norm(prototype)))

    Both inputs should be normalized to unit length first.

    Args:
        pixel_rgb: (..., 3) float array of pixel RGB values (already
            unit-normalized). Shape can be (3,) for a single pixel or
            (N, 3) for multiple pixels.
        prototype: (3,) float array of the unit-normalized prototype spectrum.

    Returns:
        Spectral angle(s) in radians. Scalar if input is 1-D, or array
        matching leading dimensions of input.
    """
    # Compute dot product along the last axis
    dot = np.dot(pixel_rgb, prototype)  # (...) or ()

    # Ensure prototype is unit length (should already be)
    prot_norm = np.linalg.norm(prototype)
    if prot_norm < EPS:
        return np.full(pixel_rgb.shape[:-1], np.pi / 2, dtype=np.float64)

    # Compute pixel norms
    pix_norm = np.linalg.norm(pixel_rgb, axis=-1)  # (...) or scalar

    # Clip to valid domain for arccos
    cos_angle = dot / (pix_norm * prot_norm + EPS)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.arccos(cos_angle)
    return angle


# ---------------------------------------------------------------------------
# Image scoring
# ---------------------------------------------------------------------------


def _load_and_normalize(image_path: str) -> np.ndarray | None:
    """Load a PNG image and normalize each pixel to unit L2 norm.

    Args:
        image_path: Path to a PNG image.

    Returns:
        (H, W, 3) float32 array where each pixel vector has unit L2 norm,
        or None on failure.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        rgb = np.array(img, dtype=np.float32)  # (H, W, 3)
    except Exception:
        return None

    # Normalize each pixel to unit length
    norms = np.linalg.norm(rgb, axis=-1, keepdims=True)  # (H, W, 1)
    norms = np.maximum(norms, EPS)
    return rgb / norms


def _brightness_mask(rgb: np.ndarray, threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD) -> np.ndarray:
    """Return boolean mask of pixels with mean brightness > threshold.

    Args:
        rgb: (H, W, 3) uint8 or float32 array.
        threshold: Minimum mean brightness to include a pixel.

    Returns:
        (H, W) bool array.
    """
    mean_brightness = rgb.astype(np.float32).mean(axis=-1)  # (H, W)
    return mean_brightness > threshold


def score_image(
    image_path: str,
    prototype: np.ndarray,
    brightness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD,
    low_angle_threshold: float = LOW_ANGLE_THRESHOLD,
) -> dict:
    """Score a single image using Spectral Angle Mapping.

    Steps:
    1. Load image, normalize to unit vectors per pixel.
    2. Compute spectral angle for each pixel against prototype.
    3. Low angle = similar to spawn spectrum.

    Args:
        image_path: Path to a PNG image.
        prototype: (3,) unit-normalized prototype spectrum.
        brightness_threshold: Only score pixels brighter than this.
        low_angle_threshold: Angle threshold for ``frac_low_angle`` metric.

    Returns:
        Dict with keys:
        - ``mean_angle``: mean spectral angle (radians) across all bright pixels.
        - ``median_angle``: median spectral angle.
        - ``min_angle``: minimum angle (best match).
        - ``frac_low_angle``: fraction of bright pixels with angle < threshold.
        - ``bright_pixel_frac``: fraction of pixels that are bright enough.
        - ``score``: -mean_angle (negated so higher = more spawn-like).
        - ``score_min_angle``: -min_angle (alternative score focusing on best
          match).
        - ``prediction``: 1 if score > 0 else 0 (positive score means
          mean_angle < 0, which is always true; threshold sweep in
          validation finds the optimal threshold).
        - ``image_path``: input path.
        - ``n_scored_pixels``: number of bright pixels scored.
        - ``total_pixels``: total pixels in the image.
    """
    # Load the original uint8 image for brightness masking
    try:
        img = Image.open(image_path).convert("RGB")
        rgb_uint8 = np.array(img, dtype=np.uint8)  # (H, W, 3)
    except Exception as exc:
        return {
            "mean_angle": 0.0,
            "median_angle": 0.0,
            "min_angle": 0.0,
            "frac_low_angle": 0.0,
            "bright_pixel_frac": 0.0,
            "score": float("-inf"),
            "score_min_angle": 0.0,
            "prediction": 0,
            "image_path": image_path,
            "n_scored_pixels": 0,
            "total_pixels": 0,
            "error": f"Failed to load image: {exc}",
        }

    total_pixels = rgb_uint8.shape[0] * rgb_uint8.shape[1]

    # Brightness mask
    mask = _brightness_mask(rgb_uint8, threshold=brightness_threshold)
    n_bright = int(mask.sum())
    bright_pixel_frac = float(n_bright / total_pixels) if total_pixels > 0 else 0.0

    if n_bright == 0:
        return {
            "mean_angle": 0.0,
            "median_angle": 0.0,
            "min_angle": 0.0,
            "frac_low_angle": 0.0,
            "bright_pixel_frac": bright_pixel_frac,
            "score": 0.0,
            "score_min_angle": 0.0,
            "prediction": 0,
            "image_path": image_path,
            "n_scored_pixels": 0,
            "total_pixels": total_pixels,
        }

    # Load and normalize pixels (re-load is simplest to avoid coupling
    # with the uint8 version above)
    unit_rgb = _load_and_normalize(image_path)
    if unit_rgb is None:
        return {
            "mean_angle": 0.0,
            "median_angle": 0.0,
            "min_angle": 0.0,
            "frac_low_angle": 0.0,
            "bright_pixel_frac": bright_pixel_frac,
            "score": 0.0,
            "score_min_angle": 0.0,
            "prediction": 0,
            "image_path": image_path,
            "n_scored_pixels": 0,
            "total_pixels": total_pixels,
            "error": "Failed to normalize image",
        }

    # Compute spectral angles for bright pixels only
    angles = spectral_angle(unit_rgb, prototype)  # (H, W)
    angles_scored = angles[mask]

    mean_angle = float(np.mean(angles_scored))
    median_angle = float(np.median(angles_scored))
    min_angle = float(np.min(angles_scored))
    frac_low = float(np.mean(angles_scored < low_angle_threshold))

    return {
        "mean_angle": mean_angle,
        "median_angle": median_angle,
        "min_angle": min_angle,
        "frac_low_angle": frac_low,
        "bright_pixel_frac": bright_pixel_frac,
        "score": -mean_angle,
        "score_min_angle": -min_angle,
        "prediction": 1 if (-mean_angle) > DEFAULT_PREDICTION_THRESHOLD else 0,
        "image_path": image_path,
        "n_scored_pixels": n_bright,
        "total_pixels": total_pixels,
    }


def score_directory(
    prototype: np.ndarray,
    image_dir: str,
    brightness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD,
) -> list[dict]:
    """Score all PNGs in a directory using Spectral Angle Mapping.

    Args:
        prototype: (3,) unit-normalized prototype spectrum.
        image_dir: Directory containing PNG images.
        brightness_threshold: Minimum brightness to score a pixel.

    Returns:
        List of score dicts sorted by ``score`` descending.
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
    errors: list[str] = []

    for p in pngs:
        try:
            result = score_image(str(p), prototype, brightness_threshold=brightness_threshold)
            if "error" in result:
                errors.append(f"{p.name}: {result['error']}")
            results.append(result)
        except Exception as exc:
            errors.append(f"{p.name}: {exc}")

    results.sort(key=lambda r: r["score"], reverse=True)

    if errors:
        print(f"  WARNING: {len(errors)} images had errors")
        for err in errors[:5]:
            print(f"    {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    n_ok = sum(1 for r in results if "error" not in r)
    print(f"  Scored {n_ok}/{len(results)} images successfully")
    if results:
        print(f"  Top 3 scores:")
        for r in results[:3]:
            name = Path(r["image_path"]).name
            print(f"    {r['score']:.4f}  {name}  (mean_angle={r['mean_angle']:.4f})")

    return results


# ---------------------------------------------------------------------------
# Validation against human labels
# ---------------------------------------------------------------------------


def validate(
    prototype: np.ndarray,
    labels_json_path: str,
    image_dir: str,
    brightness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD,
) -> dict:
    """Validate Spectral Angle Mapping against human labels.

    Evaluates two scoring methods:
    - ``score`` = -mean_angle (overall spectral similarity).
    - ``score_min_angle`` = -min_angle (best-match spectral similarity).

    Labels JSON format::

        {"labels": [{"filename": "image.png", "label": 1}, ...]}

    where ``label`` = 1 means positive (spawn) and ``label`` = 0 means
    negative (no spawn).  Matches the format of
    ``remoteclip_zero_shot.validate()``.

    Args:
        prototype: (3,) unit-normalized prototype spectrum.
        labels_json_path: Path to labels JSON file.
        image_dir: Directory containing the PNG images referenced in labels.
        brightness_threshold: Minimum brightness to score a pixel.

    Returns:
        Dict with accuracy, best_accuracy, best_threshold, auc_roc,
        avg_precision, confusion_matrix, per_sample, n_total, n_pos, n_neg.
    """
    print(f"  Loading labels from: {labels_json_path}")

    labels_path = Path(labels_json_path)
    if not labels_path.exists():
        msg = f"Labels file not found: {labels_path}"
        print(f"ERROR: {msg}")
        return {"error": msg}

    labels_data = json.loads(labels_path.read_text())
    label_entries = labels_data.get("labels", [])
    print(f"  Loaded {len(label_entries)} label entries")

    if not label_entries:
        empty = {
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
        return empty

    img_dir = Path(image_dir)
    per_sample: list[dict] = []

    for entry in label_entries:
        fname = entry["filename"]
        true_label = entry["label"]
        img_path = img_dir / fname

        if not img_path.exists():
            print(f"  WARNING: Image not found: {img_path}")
            continue

        try:
            result = score_image(str(img_path), prototype, brightness_threshold=brightness_threshold)
        except Exception as exc:
            print(f"  WARNING: Failed to score {fname}: {exc}")
            continue

        if "error" in result:
            continue

        per_sample.append({
            "filename": fname,
            "true_label": true_label,
            "prediction": result["prediction"],
            "score": result["score"],
            "score_min_angle": result["score_min_angle"],
            "mean_angle": result["mean_angle"],
            "median_angle": result["median_angle"],
            "min_angle": result["min_angle"],
            "frac_low_angle": result["frac_low_angle"],
            "bright_pixel_frac": result["bright_pixel_frac"],
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

    # Aggregate metrics for "score" (negated mean angle)
    y_true = np.array([s["true_label"] for s in per_sample])
    y_score = np.array([s["score"] for s in per_sample])
    y_pred = np.array([s["prediction"] for s in per_sample])

    n_total = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_total - n_pos

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Best accuracy via threshold sweep (for "score")
    thresholds = np.linspace(y_score.min() - 0.1, y_score.max() + 0.1, 201)
    best_acc = 0.0
    best_thr = 0.0
    for thr in thresholds:
        thr_pred = (y_score > thr).astype(int)
        thr_acc = accuracy_score(y_true, thr_pred)
        if thr_acc > best_acc:
            best_acc = thr_acc
            best_thr = float(thr)

    # AUROC (for "score")
    auroc = 0.0
    if n_pos > 0 and n_neg > 0 and not np.all(y_score == y_score[0]):
        try:
            auroc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auroc = 0.0

    # Average precision (for "score")
    ap = 0.0
    if n_pos > 0:
        try:
            ap = float(average_precision_score(y_true, y_score))
        except Exception:
            ap = 0.0

    # ----- Also evaluate "score_min_angle" (negated min angle) -----
    y_score_min = np.array([s["score_min_angle"] for s in per_sample])

    # Best threshold for score_min_angle
    thr_min = np.linspace(y_score_min.min() - 0.1, y_score_min.max() + 0.1, 201)
    best_acc_min = 0.0
    best_thr_min = 0.0
    for thr in thr_min:
        thr_pred = (y_score_min > thr).astype(int)
        thr_acc = accuracy_score(y_true, thr_pred)
        if thr_acc > best_acc_min:
            best_acc_min = thr_acc
            best_thr_min = float(thr)

    # AUROC for score_min_angle
    auroc_min = 0.0
    if n_pos > 0 and n_neg > 0 and not np.all(y_score_min == y_score_min[0]):
        try:
            auroc_min = float(roc_auc_score(y_true, y_score_min))
        except Exception:
            auroc_min = 0.0

    # Average precision for score_min_angle
    ap_min = 0.0
    if n_pos > 0:
        try:
            ap_min = float(average_precision_score(y_true, y_score_min))
        except Exception:
            ap_min = 0.0

    print(f"\n  Validation results (score = -mean_angle):")
    print(f"    Total: {n_total}  Pos: {n_pos}  Neg: {n_neg}")
    print(f"    Accuracy (thr=0):  {acc:.4f}")
    print(f"    Best accuracy:     {best_acc:.4f} @ thr={best_thr:.4f}")
    print(f"    AUROC:             {auroc:.4f}")
    print(f"    Avg Precision:     {ap:.4f}")
    print(f"    Confusion Matrix:  {cm}")
    print(f"\n  Validation results (score_min_angle = -min_angle):")
    print(f"    Best accuracy:     {best_acc_min:.4f} @ thr={best_thr_min:.4f}")
    print(f"    AUROC:             {auroc_min:.4f}")
    print(f"    Avg Precision:     {ap_min:.4f}")

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
        "min_angle_metrics": {
            "best_accuracy": best_acc_min,
            "best_threshold": best_thr_min,
            "auc_roc": auroc_min,
            "avg_precision": ap_min,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-shot herring spawn detection via Spectral Angle Mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--positive-dir", type=str, default="data/samples/positive",
        help="Directory of positive images to build the spawn prototype "
             "(default: data/samples/positive)",
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
        "--brightness-threshold", type=float, default=DEFAULT_BRIGHTNESS_THRESHOLD,
        help=f"Minimum brightness to score a pixel (default: {DEFAULT_BRIGHTNESS_THRESHOLD})",
    )
    args = parser.parse_args(argv)

    # ----- Resolve paths -----
    repo_root = Path(__file__).resolve().parent.parent

    # --positive-dir
    given_pos = Path(args.positive_dir)
    pos_dir = given_pos if given_pos.is_absolute() else repo_root / args.positive_dir
    if not pos_dir.is_dir():
        print(f"ERROR: Positive directory not found: {pos_dir}")
        return 1

    # --image-dir (required unless we can infer from --validate-only)
    if not args.image_dir:
        parser.print_help()
        print("\nERROR: --image-dir is required")
        return 1

    given = Path(args.image_dir)
    img_dir = given if given.is_absolute() else repo_root / args.image_dir
    if not img_dir.is_dir():
        print(f"ERROR: Image directory not found: {img_dir}")
        return 1

    # --labels-json path
    labels_path = None
    if args.labels_json:
        lp = Path(args.labels_json)
        labels_path = repo_root / args.labels_json if not lp.is_absolute() else lp
        if not labels_path.exists():
            print(f"ERROR: Labels file not found: {labels_path}")
            return 1

    # ----- Build the spawn prototype -----
    print("=" * 60)
    print("  Spectral Angle Mapping — Spawn Prototype")
    print("=" * 60)
    print(f"  Positive images: {pos_dir}")
    prototype, n_pixels = compute_spawn_prototype(str(pos_dir))

    if n_pixels == 0:
        print("ERROR: Could not build prototype (no spawn-like pixels found).")
        return 1

    # ----- Validate-only mode -----
    if args.validate_only:
        if labels_path is None:
            print("ERROR: --validate-only requires --labels-json")
            return 1

        print("\n" + "=" * 60)
        print("  SAM Validation")
        print("=" * 60)
        print(f"  Image directory: {img_dir}")
        print(f"  Labels:          {labels_path}")
        print(f"  Brightness thr:  {args.brightness_threshold}")
        result = validate(
            prototype,
            str(labels_path),
            str(img_dir),
            brightness_threshold=args.brightness_threshold,
        )
    else:
        # ----- Scoring mode -----
        print("\n" + "=" * 60)
        print("  SAM Herring Spawn Scoring")
        print("=" * 60)
        print(f"  Image directory: {img_dir}")
        print(f"  Brightness thr:  {args.brightness_threshold}")
        print()

        results = score_directory(prototype, str(img_dir), brightness_threshold=args.brightness_threshold)

        result = {
            "method": "spectral_angle_mapping",
            "prototype": {
                "r": float(prototype[0]),
                "g": float(prototype[1]),
                "b": float(prototype[2]),
            },
            "prototype_n_pixels": n_pixels,
            "brightness_threshold": args.brightness_threshold,
            "n_images_scored": len(results),
            "results": results,
        }

        # Also validate if labels are available
        if labels_path is not None:
            print("\n" + "-" * 60)
            print("  Running validation against labels...")
            val_result = validate(
                prototype,
                str(labels_path),
                str(img_dir),
                brightness_threshold=args.brightness_threshold,
            )
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
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
