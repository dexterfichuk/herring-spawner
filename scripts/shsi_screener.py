#!/usr/bin/env python3
"""High-recall SHSI candidate screener using the correct UVic formula.

The Spectral Herring Spawning Index (SHSI) from Loïc Dallaire's UVic thesis::

    SHSI = Green² / Red

This is a *screener* — designed for high recall as a cheap pre-filter before
expensive DINOv2 models.  False positives are acceptable; they will be filtered
later.

Pre-filtering pipeline (approximating Earth Engine's multispectral logic on RGB
thumbnails):
    1. **NIR water mask approximation**: ``max(R,G,B) < brightness_max``
       (removes clouds/land/surf).
    2. **Green background suppression**: ``G > green_min`` (0–255)
       (removes dark deep water).
    3. **Red minimum**: ``R > red_min`` (0–255) — avoids division-by-zero
       noise from tiny red values.
    4. *Skip* the Blue>Coastal filter from EE — no coastal band in RGB.

Usage:
    # Benchmark against the golden set
    python scripts/shsi_screener.py --benchmark

    # Score all images in a directory
    python scripts/shsi_screener.py --image-dir data/samples/unified

    # Score a single image
    python scripts/shsi_screener.py --single-image data/samples/positive/example.png

    # Save results
    python scripts/shsi_screener.py --image-dir data/samples/unified \\
        --output-json data/shsi_screener_results.json

Dependencies:
    pip install numpy Pillow scikit-learn
"""

from __future__ import annotations

import argparse
import itertools
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
# Defaults
# ---------------------------------------------------------------------------

SHSI_THRESHOLD = 0.10
"""SHSI threshold — pixels above this are considered spawn-like."""

BRIGHTNESS_MAX = 220
"""Maximum channel value for water mask (NIR approximation)."""

GREEN_MIN = 5.0
"""Minimum Green value (0–255) for green suppression."""

RED_MIN = 1.275
"""Minimum Red value (0–255) — corresponds to ~0.005 in 0–1 reflectance.
Avoids division-by-zero noise.
"""

PREDICTION_THRESHOLD = 0.0
"""Default threshold for converting mean SHSI to binary prediction."""

EPSILON = 1e-8
"""Small constant to avoid division by zero."""


# ---------------------------------------------------------------------------
# Pre-filters
# ---------------------------------------------------------------------------


def apply_prefilters(
    image: np.ndarray,
    brightness_max: float = BRIGHTNESS_MAX,
    green_min: float = GREEN_MIN,
    red_min: float = RED_MIN,
) -> np.ndarray:
    """Apply pre-filters for RGB thumbnails.

    Args:
        image: uint8 array of shape (H, W, 3).
        brightness_max: NIR water mask approx — ``max(R,G,B) <`` this.
        green_min: Green background suppression — ``G >`` this (0–255).
        red_min: Minimum red to avoid division noise — ``R >`` this (0–255).

    Returns:
        Boolean mask — ``True`` where pixel is eligible for SHSI scoring.
    """
    max_channel = image.max(axis=-1).astype(np.float32)  # (H, W)
    r = image[..., 0].astype(np.float32)
    g = image[..., 1].astype(np.float32)

    water_mask = max_channel < brightness_max
    green_mask = g > green_min
    red_mask = r > red_min

    return water_mask & green_mask & red_mask


# ---------------------------------------------------------------------------
# Core SHSI: Green² / Red
# ---------------------------------------------------------------------------


def compute_shsi(
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute SHSI = Green² / Red for each pixel.

    Values computed on the 0–1 normalised range (uint8 input divided by 255).

    Args:
        image: uint8 array of shape (H, W, 3).
        mask: Optional boolean mask.  Non-masked pixels set to 0.

    Returns:
        Float array (H, W) of per-pixel SHSI values.
    """
    normalised = image.astype(np.float32) / 255.0
    r = normalised[..., 0]
    g = normalised[..., 1]

    # SHSI = G² / R, guard against division by zero
    shsi = np.divide(g * g, r, out=np.zeros_like(g), where=(r > EPSILON))

    if mask is not None:
        shsi = shsi * mask.astype(np.float32)

    return shsi


# ---------------------------------------------------------------------------
# Single-image scoring
# ---------------------------------------------------------------------------


def score_image(
    image_path: str,
    shsi_threshold: float = SHSI_THRESHOLD,
    brightness_max: float = BRIGHTNESS_MAX,
    green_min: float = GREEN_MIN,
    red_min: float = RED_MIN,
    prediction_threshold: float = PREDICTION_THRESHOLD,
    aggregation: str = "mean",
) -> dict:
    """Score a single image using the correct SHSI formula.

    Args:
        image_path: Path to a PNG image.
        shsi_threshold: Per-pixel SHSI threshold for ``frac_positive``.
        brightness_max: NIR water mask parameter.
        green_min: Green background suppression parameter.
        red_min: Minimum red value parameter.
        prediction_threshold: Threshold on the primary score for binary
            prediction.
        aggregation: Primary score aggregation — ``mean``, ``95p``, ``max``,
            or ``frac_above``.

    Returns:
        Dict with per-pixel and aggregated scores.
    """
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)  # (H, W, 3)

    total_pixels = rgb.shape[0] * rgb.shape[1]

    # Pre-filters
    mask = apply_prefilters(rgb, brightness_max, green_min, red_min)
    n_eligible = int(mask.sum())
    frac_eligible = n_eligible / total_pixels if total_pixels > 0 else 0.0

    if n_eligible == 0:
        return {
            "shsi_mean": 0.0,
            "shsi_95p": 0.0,
            "shsi_max": 0.0,
            "frac_eligible": 0.0,
            "frac_positive": 0.0,
            "frac_above": 0.0,
            "score": 0.0,
            "n_eligible_pixels": 0,
            "total_pixels": total_pixels,
            "prediction": 0,
            "image_path": image_path,
            "shsi_threshold": shsi_threshold,
            "brightness_max": brightness_max,
            "green_min": green_min,
            "red_min": red_min,
            "prediction_threshold": prediction_threshold,
            "aggregation": aggregation,
        }

    # Compute SHSI
    shsi = compute_shsi(rgb, mask=mask)
    shsi_eligible = shsi[mask]

    shsi_mean = float(np.mean(shsi_eligible))
    shsi_95p = float(np.percentile(shsi_eligible, 95))
    shsi_max_val = float(np.max(shsi_eligible))
    frac_positive = float(np.mean(shsi_eligible > shsi_threshold))

    # Primary score
    if aggregation == "mean":
        score = shsi_mean
    elif aggregation == "95p":
        score = shsi_95p
    elif aggregation == "max":
        score = shsi_max_val
    elif aggregation == "frac_above":
        score = frac_positive
    else:
        score = shsi_mean

    return {
        "shsi_mean": shsi_mean,
        "shsi_95p": shsi_95p,
        "shsi_max": shsi_max_val,
        "frac_eligible": frac_eligible,
        "frac_positive": frac_positive,
        "frac_above": frac_positive,
        "score": score,
        "n_eligible_pixels": n_eligible,
        "total_pixels": total_pixels,
        "prediction": 1 if score > prediction_threshold else 0,
        "image_path": image_path,
        "shsi_threshold": shsi_threshold,
        "brightness_max": brightness_max,
        "green_min": green_min,
        "red_min": red_min,
        "prediction_threshold": prediction_threshold,
        "aggregation": aggregation,
    }


def score_directory(
    image_dir: str,
    **kwargs,
) -> list[dict]:
    """Score all PNGs in a directory.

    Args:
        image_dir: Directory of PNG images.
        **kwargs: Passed to :func:`score_image`.

    Returns:
        List of score dicts sorted by ``score`` descending.
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
            result = score_image(str(p), **kwargs)
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
# Benchmark against golden set
# ---------------------------------------------------------------------------


def load_golden_set(
    manifest_path: str | Path,
    pos_dir: str | Path,
    neg_dir: str | Path,
) -> tuple[list[str], list[int]]:
    """Load images and labels from the golden training set.

    The manifest specifies positive and negative filenames.  Images are
    resolved from ``pos_dir`` and ``neg_dir`` respectively.  If a file
    appears in both directories, the manifest label determines its class.

    Args:
        manifest_path: Path to ``training_manifest.json``.
        pos_dir: Directory of positive images.
        neg_dir: Directory of negative images.

    Returns:
        (image_paths, labels) where each entry is (str, int).
    """
    manifest = json.loads(Path(manifest_path).read_text())

    pos_names: list[str] = manifest.get("positives", [])
    neg_names: list[str] = manifest.get("rejected", manifest.get("negatives", []))

    pos_dir_p = Path(pos_dir)
    neg_dir_p = Path(neg_dir)

    image_paths: list[str] = []
    labels: list[int] = []

    seen: set[str] = set()

    for fname in pos_names:
        candidate = pos_dir_p / fname
        if candidate.exists():
            key = fname.lower()
            if key not in seen:
                seen.add(key)
                image_paths.append(str(candidate.resolve()))
                labels.append(1)
        else:
            # Fallback to neg_dir (some positives might be stored there)
            candidate = neg_dir_p / fname
            if candidate.exists():
                key = fname.lower()
                if key not in seen:
                    seen.add(key)
                    image_paths.append(str(candidate.resolve()))
                    labels.append(1)

    for fname in neg_names:
        candidate = neg_dir_p / fname
        key = fname.lower()
        if candidate.exists() and key not in seen:
            seen.add(key)
            image_paths.append(str(candidate.resolve()))
            labels.append(0)
        elif key not in seen:
            # Try pos_dir fallback (some negatives might be there)
            candidate = pos_dir_p / fname
            if candidate.exists() and key not in seen:
                seen.add(key)
                image_paths.append(str(candidate.resolve()))
                labels.append(0)

    print(f"  Loaded {len(image_paths)} golden samples ({sum(labels)} pos, {len(labels) - sum(labels)} neg)")
    return image_paths, labels


def run_benchmark(
    image_paths: list[str],
    labels: list[int],
    shsi_thresholds: list[float] | None = None,
    brightness_max_values: list[float] | None = None,
    green_min_values: list[float] | None = None,
    red_min: float = RED_MIN,
    aggregations: list[str] | None = None,
) -> list[dict]:
    """Run a grid sweep over SHSI parameters to find the best high-recall
    configuration.

    Args:
        image_paths: List of image file paths.
        labels: Ground-truth labels (0/1).
        shsi_thresholds: SHSI per-pixel thresholds to sweep.
        brightness_max_values: Brightness max values to sweep.
        green_min_values: Green min values to sweep.
        red_min: Fixed red minimum.
        aggregations: Score aggregation methods to sweep.

    Returns:
        List of result dicts, each containing the configuration and metrics.
    """
    if shsi_thresholds is None:
        shsi_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    if brightness_max_values is None:
        brightness_max_values = [180.0, 200.0, 220.0, 240.0, 255.0]
    if green_min_values is None:
        green_min_values = [0.0, 3.0, 5.0, 6.375, 10.0, 15.0]
    if aggregations is None:
        aggregations = ["mean", "95p", "max", "frac_above"]

    y_true = np.array(labels)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    results: list[dict] = []

    total_combos = (
        len(shsi_thresholds)
        * len(brightness_max_values)
        * len(green_min_values)
        * len(aggregations)
    )
    print(f"\n  Sweeping {total_combos} parameter combinations...")

    combo_idx = 0

    for shsi_thr, b_max, g_min, agg in itertools.product(
        shsi_thresholds, brightness_max_values, green_min_values, aggregations,
    ):
        combo_idx += 1
        if combo_idx % 50 == 0:
            print(f"    ... {combo_idx}/{total_combos}")

        scores = []
        for img_path in image_paths:
            result = score_image(
                img_path,
                shsi_threshold=shsi_thr,
                brightness_max=b_max,
                green_min=g_min,
                red_min=red_min,
                prediction_threshold=0.0,
                aggregation=agg,
            )
            scores.append(result["score"])

        y_score = np.array(scores)

        # ---- Metrics ----
        # Determine valid scores (non-constant)
        has_variance = not np.all(y_score == y_score[0]) and n_pos > 0 and n_neg > 0

        auroc = float(roc_auc_score(y_true, y_score)) if has_variance else 0.0
        ap = float(average_precision_score(y_true, y_score)) if has_variance else 0.0

        # Best F1 via threshold sweep
        thr_grid = np.linspace(
            y_score.min() - 0.05,
            y_score.max() + 0.05,
            301,
        )
        best_f1 = 0.0
        best_f1_thr = 0.0
        best_f1_recall = 0.0
        best_f1_precision = 0.0
        best_f1_tn = 0
        best_f1_fp = 0
        best_f1_fn = 0
        best_f1_tp = 0

        for thr in thr_grid:
            pred = (y_score > thr).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_f1_thr = float(thr)
                best_f1_recall = recall
                best_f1_precision = precision
                best_f1_tn = int(((pred == 0) & (y_true == 0)).sum())
                best_f1_fp = fp
                best_f1_fn = fn
                best_f1_tp = tp

        # Recall at the default prediction threshold (0.0)
        pred_default = (y_score > 0.0).astype(int)
        tp_def = int(((pred_default == 1) & (y_true == 1)).sum())
        fp_def = int(((pred_default == 1) & (y_true == 0)).sum())
        fn_def = int(((pred_default == 0) & (y_true == 1)).sum())
        tn_def = int(((pred_default == 0) & (y_true == 0)).sum())
        recall_default = tp_def / (tp_def + fn_def) if (tp_def + fn_def) > 0 else 0.0
        precision_default = tp_def / (tp_def + fp_def) if (tp_def + fp_def) > 0 else 0.0
        fpr_default = fp_def / (fp_def + tn_def) if (fp_def + tn_def) > 0 else 0.0

        results.append({
            "config": {
                "shsi_threshold": float(shsi_thr),
                "brightness_max": float(b_max),
                "green_min": float(g_min),
                "aggregation": agg,
            },
            "auroc": round(auroc, 4),
            "avg_precision": round(ap, 4),
            "best_f1": round(best_f1, 4),
            "best_f1_threshold": round(best_f1_thr, 4),
            "best_f1_recall": round(best_f1_recall, 4),
            "best_f1_precision": round(best_f1_precision, 4),
            "best_f1_cm": [[best_f1_tn, best_f1_fp], [best_f1_fn, best_f1_tp]],
            "recall_at_zero": round(recall_default, 4),
            "precision_at_zero": round(precision_default, 4),
            "fpr_at_zero": round(fpr_default, 4),
            "n_pos_caught": tp_def,
            "n_pos_total": n_pos,
            "n_false_positives": fp_def,
        })

    results.sort(key=lambda r: r["recall_at_zero"], reverse=True)

    return results


def print_benchmark_table(results: list[dict], top_n: int = 15) -> None:
    """Print a formatted benchmark results table."""
    print()
    print("=" * 140)
    print("  SHSI Screener Benchmark — Top Configurations by Recall")
    print("=" * 140)
    print(
        f"  {'Config':<55} {'AUROC':>7} {'AP':>7} {'F1':>7} "
        f"{'Recall@0':>9} {'Prec@0':>8} {'FPR@0':>7} {'PosCght':>7} {'FP':>5}"
    )
    print("  " + "-" * 130)

    for r in results[:top_n]:
        c = r["config"]
        config_str = (
            f"SHSI={c['shsi_threshold']:.2f} "
            f"Bmax={c['brightness_max']:.0f} "
            f"Gmin={c['green_min']:.1f} "
            f"{c['aggregation']}"
        )
        print(
            f"  {config_str:<55} "
            f"{r['auroc']:>7.4f} "
            f"{r['avg_precision']:>7.4f} "
            f"{r['best_f1']:>7.4f} "
            f"{r['recall_at_zero']:>9.4f} "
            f"{r['precision_at_zero']:>8.4f} "
            f"{r['fpr_at_zero']:>7.4f} "
            f"{r['n_pos_caught']:>3}/{r['n_pos_total']:<2} "
            f"{r['n_false_positives']:>5}"
        )

    print("=" * 140)

    # Find configurations with perfect recall
    print()
    print("  Configurations with 100% recall (all positives caught):")
    print("  " + "-" * 80)
    perfect_recall = [r for r in results if r["recall_at_zero"] >= 1.0]
    if perfect_recall:
        # Sort by FPR ascending (most efficient high-recall)
        perfect_recall.sort(key=lambda r: r["fpr_at_zero"])
        for r in perfect_recall[:10]:
            c = r["config"]
            config_str = (
                f"SHSI={c['shsi_threshold']:.2f} "
                f"Bmax={c['brightness_max']:.0f} "
                f"Gmin={c['green_min']:.1f} "
                f"{c['aggregation']}"
            )
            print(
                f"    {config_str:<50} "
                f"AUROC={r['auroc']:.4f} AP={r['avg_precision']:.4f} "
                f"FPR={r['fpr_at_zero']:.4f} FP={r['n_false_positives']:>3}/{len(perfect_recall[0]['n_pos_total'] if 'n_pos_total' in r else 0)}"
            )
    else:
        print("    (none — no configuration caught all positives)")


def benchmark_main(
    manifest_path: str | Path = "data/samples/training_manifest.json",
    pos_dir: str | Path = "data/samples/positive",
    neg_dir: str | Path = "data/samples/negative",
) -> dict:
    """Full benchmark workflow: load golden set, run sweep, print results.

    Returns:
        Dict with ``results`` list and ``best_config``.
    """
    repo_root = Path(__file__).resolve().parent.parent

    manifest_p = Path(manifest_path)
    if not manifest_p.is_absolute():
        manifest_p = repo_root / manifest_path

    pos_p = Path(pos_dir)
    if not pos_p.is_absolute():
        pos_p = repo_root / pos_dir

    neg_p = Path(neg_dir)
    if not neg_p.is_absolute():
        neg_p = repo_root / neg_dir

    print("=" * 60)
    print("  SHSI Screener Benchmark")
    print("  Formula: Green² / Red (correct UVic SHSI)")
    print("=" * 60)

    print("\n  Loading golden set...")
    image_paths, labels = load_golden_set(manifest_p, pos_p, neg_p)

    print("\n  Running parameter grid sweep...")
    results = run_benchmark(image_paths, labels)

    print_benchmark_table(results)

    # Determine best configuration (high recall first, then low FPR)
    high_recall = [r for r in results if r["recall_at_zero"] >= 1.0]
    if high_recall:
        best = min(high_recall, key=lambda r: r["fpr_at_zero"])
    else:
        # Best F1 among all
        best = max(results, key=lambda r: r["best_f1"])

    print()
    print("  Recommended configuration:")
    print(f"    SHSI threshold:    {best['config']['shsi_threshold']:.2f}")
    print(f"    Brightness max:    {best['config']['brightness_max']:.0f}")
    print(f"    Green min:         {best['config']['green_min']:.1f}")
    print(f"    Aggregation:       {best['config']['aggregation']}")
    print(f"    AUROC:             {best['auroc']:.4f}")
    print(f"    Avg Precision:     {best['avg_precision']:.4f}")
    print(f"    Best F1:           {best['best_f1']:.4f}")
    if best["recall_at_zero"] >= 1.0:
        print(f"    Recall (thr=0):    1.0000 (ALL positives caught)")
    else:
        print(f"    Recall (thr=0):    {best['recall_at_zero']:.4f}")
    print(f"    FPR (thr=0):       {best['fpr_at_zero']:.4f}")
    print(f"    False positives:   {best['n_false_positives']}")

    return {
        "results": results,
        "best_config": best,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="High-recall SHSI candidate screener (Green²/Red)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory of PNG images to score",
    )
    parser.add_argument(
        "--single-image", type=str, default=None,
        help="Score a single image",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark sweep against golden set",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save output JSON",
    )

    # Scoring parameters (tunable)
    parser.add_argument(
        "--shsi-threshold", type=float, default=SHSI_THRESHOLD,
        help=f"Per-pixel SHSI threshold (default: {SHSI_THRESHOLD})",
    )
    parser.add_argument(
        "--brightness-max", type=float, default=BRIGHTNESS_MAX,
        help=f"Brightness max for water mask (default: {BRIGHTNESS_MAX})",
    )
    parser.add_argument(
        "--green-min", type=float, default=GREEN_MIN,
        help=f"Green minimum (0-255) (default: {GREEN_MIN})",
    )
    parser.add_argument(
        "--red-min", type=float, default=RED_MIN,
        help=f"Red minimum (0-255) (default: {RED_MIN})",
    )
    parser.add_argument(
        "--prediction-threshold", type=float, default=PREDICTION_THRESHOLD,
        help=f"Score threshold for binary prediction (default: {PREDICTION_THRESHOLD})",
    )
    parser.add_argument(
        "--aggregation", type=str, default="mean",
        choices=["mean", "95p", "max", "frac_above"],
        help="Score aggregation method (default: mean)",
    )

    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent

    # ---- Benchmark mode ----
    if args.benchmark:
        result = benchmark_main()
        if args.output_json:
            out_path = Path(args.output_json)
            if not out_path.is_absolute():
                out_path = repo_root / args.output_json
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, default=str))
            print(f"\n  Benchmark results saved to: {out_path}")
        return 0

    # ---- Single-image mode ----
    if args.single_image:
        result = score_image(
            args.single_image,
            shsi_threshold=args.shsi_threshold,
            brightness_max=args.brightness_max,
            green_min=args.green_min,
            red_min=args.red_min,
            prediction_threshold=args.prediction_threshold,
            aggregation=args.aggregation,
        )
        print(json.dumps(result, indent=2, default=str))
        if args.output_json:
            out_path = Path(args.output_json)
            if not out_path.is_absolute():
                out_path = repo_root / args.output_json
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, default=str))
            print(f"  Saved to: {out_path}")
        return 0

    # ---- Directory mode ----
    if not args.image_dir:
        parser.print_help()
        print("\nERROR: Provide --image-dir, --single-image, or --benchmark")
        return 1

    given = Path(args.image_dir)
    img_dir = given if given.is_absolute() else repo_root / given

    if not img_dir.is_dir():
        print(f"ERROR: Image directory not found: {img_dir}")
        return 1

    print("=" * 60)
    print("  SHSI (Green²/Red) Screener")
    print("=" * 60)

    kwargs = {
        "shsi_threshold": args.shsi_threshold,
        "brightness_max": args.brightness_max,
        "green_min": args.green_min,
        "red_min": args.red_min,
        "prediction_threshold": args.prediction_threshold,
        "aggregation": args.aggregation,
    }
    results = score_directory(str(img_dir), **kwargs)

    output = {
        "method": "shsi_green_squared_over_red",
        "description": "High-recall SHSI screener: Green²/Red with RGB pre-filters",
        "config": {k: v for k, v in kwargs.items()},
        "n_images_scored": len(results),
        "results": results,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = repo_root / args.output_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"\n  Results saved to: {out_path}")
    else:
        print(json.dumps(output, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
