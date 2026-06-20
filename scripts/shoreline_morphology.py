#!/usr/bin/env python3
"""Zero-shot herring spawn detector using shoreline morphology and spatial
pattern analysis on RGB thumbnails.

Theory
------
Herring spawn plumes have characteristic spatial patterns:
- Narrow, elongated patches (not round blobs)
- Attached to or very close to shoreline
- Located in sheltered bays/inlets
- Form a "halo" around the shoreline
- Contrast with surrounding dark water

This is a purely geometric/rule-based filter — no training needed.

Usage
-----
    # Score all PNGs in a directory
    python scripts/shoreline_morphology.py \\
        --image-dir data/samples/unified \\
        --output-json data/morphology_results.json

    # Validate against human labels
    python scripts/shoreline_morphology.py \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --output-json data/morphology_results.json

    # Validate only (skip scoring individual images to stdout)
    python scripts/shoreline_morphology.py \\
        --validate-only \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json

Dependencies: numpy, scipy, Pillow
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# HSV thresholds for turquoise (spawn-like) pixels
# Hue range for turquoise/cyan in OpenCV/PIL HSV (0-255 scale):
#   H in [80, 120] maps to 180° range -> roughly 125-188 in 0-255
# We use PIL HSV (H in [0, 255]), so convert: H_deg/360 * 255
HUE_MIN = 55   # ~78°  — lower bound of turquoise
HUE_MAX = 150  # ~212° — upper bound of turquoise
SAT_MIN = 80   # ~0.31 — minimum saturation (0-255)
VAL_MIN = 80   # ~0.31 — minimum value/brightness (0-255)

# Score formula weights
W_ELONGATION = 0.4    # weight for (1 - mean_circularity)
W_EDGE_PROX = 0.3     # weight for (1 - min(edge_dist/100, 1))
W_FRAGMENTATION = 0.3 # weight for min(n_components/10, 1)

# Edge distance normalisation (pixels). Components within this many pixels
# of the image border are considered shoreline-attached.
EDGE_DIST_CAP = 100

# Maximum components for fragmentation normalisation
MAX_COMPONENTS_NORM = 10

# Default score threshold for binary prediction
DEFAULT_THRESHOLD = 0.3

# Minimum component area in pixels to consider (noise filter)
MIN_COMPONENT_AREA = 20


# ---------------------------------------------------------------------------
# Tunable parameter container
# ---------------------------------------------------------------------------

class HSVThresholds:
    """HSV threshold parameters for turquoise pixel detection."""

    def __init__(
        self,
        hue_min: int = HUE_MIN,
        hue_max: int = HUE_MAX,
        sat_min: int = SAT_MIN,
        val_min: int = VAL_MIN,
    ):
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.val_min = val_min

    def __repr__(self) -> str:
        return (
            f"HSVThresholds(hue=[{self.hue_min}, {self.hue_max}], "
            f"sat>={self.sat_min}, val>={self.val_min})"
        )


# ---------------------------------------------------------------------------
# Core morphology analysis
# ---------------------------------------------------------------------------

def score_image(
    image_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    hsv_params: HSVThresholds | None = None,
) -> dict:
    """Analyse spatial patterns in a single image.

    Steps:
    1. Load image, convert to HSV.
    2. Threshold on turquoise-like pixels.
    3. Find connected components of thresholded mask.
    4. For each component, calculate area, perimeter, circularity etc.
    5. Compute morphology score from circularity, edge-distance, and
       fragmentation features.

    Args:
        image_path: Path to a PNG image.
        threshold: Score threshold for binary prediction (default 0.3).
        hsv_params: HSV threshold parameters. Uses module-level defaults if None.

    Returns:
        dict with keys:
        - score: morphology score (higher = more spawn-like)
        - n_components: number of turquoise components found
        - mean_circularity: mean circularity of components (lower = more elongated)
        - mean_edge_distance: mean distance from components to nearest image edge
        - total_turquoise_area: total pixels classified as turquoise
        - frac_turquoise: fraction of image with turquoise pixels
        - prediction: 1 if score > threshold else 0
        - image_path: input path
    """
    if hsv_params is None:
        hsv_params = HSVThresholds()
    # Load image
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        return {
            "score": 0.0,
            "n_components": 0,
            "mean_circularity": 0.0,
            "mean_edge_distance": float(EDGE_DIST_CAP),
            "total_turquoise_area": 0,
            "frac_turquoise": 0.0,
            "prediction": 0,
            "image_path": image_path,
            "error": str(exc),
        }

    img = np.array(pil_img, dtype=np.uint8)
    h, w, _ = img.shape
    total_pixels = h * w

    # Convert RGB to HSV using PIL (H in [0, 255], S in [0, 255], V in [0, 255])
    pil_hsv = pil_img.convert("HSV")
    hsv_arr = np.array(pil_hsv, dtype=np.uint8)

    # Threshold on turquoise-like pixels
    hue_mask = (hsv_arr[:, :, 0] >= hsv_params.hue_min) & (hsv_arr[:, :, 0] <= hsv_params.hue_max)
    sat_mask = hsv_arr[:, :, 1] >= hsv_params.sat_min
    val_mask = hsv_arr[:, :, 2] >= hsv_params.val_min
    turquoise_mask = hue_mask & sat_mask & val_mask

    total_turquoise = int(turquoise_mask.sum())
    frac_turquoise = total_turquoise / total_pixels if total_pixels > 0 else 0.0

    # If no turquoise pixels, return zero score
    if total_turquoise == 0:
        return {
            "score": 0.0,
            "n_components": 0,
            "mean_circularity": 0.0,
            "mean_edge_distance": float(EDGE_DIST_CAP),
            "total_turquoise_area": 0,
            "frac_turquoise": 0.0,
            "prediction": 0,
            "image_path": image_path,
        }

    # Connected components
    labeled, n_features = ndi.label(turquoise_mask)

    # Extract component properties
    component_stats = []
    for comp_id in range(1, n_features + 1):
        comp_mask = labeled == comp_id
        area = int(comp_mask.sum())

        # Skip tiny noise components
        if area < MIN_COMPONENT_AREA:
            continue

        # Perimeter: count boundary pixels (including holes)
        # Erode and subtract to find the boundary
        struct = np.ones((3, 3), dtype=bool)
        eroded = ndi.binary_erosion(comp_mask, structure=struct)
        perimeter_mask = comp_mask & (~eroded)
        perimeter = int(perimeter_mask.sum())

        # Circularity: 4*pi*area / perimeter^2
        # 1.0 = perfect circle, lower = more elongated
        if perimeter > 0:
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        else:
            circularity = 0.0

        # Component bounding box
        rows, cols = np.where(comp_mask)
        min_row, max_row = int(rows.min()), int(rows.max())
        min_col, max_col = int(cols.min()), int(cols.max())
        bb_height = max_row - min_row + 1
        bb_width = max_col - min_col + 1
        aspect_ratio = bb_width / bb_height if bb_height > 0 else 1.0

        # Distance to nearest image edge
        edge_dist = min(min_row, h - 1 - max_row, min_col, w - 1 - max_col)

        component_stats.append({
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "edge_distance": edge_dist,
        })

    if not component_stats:
        return {
            "score": 0.0,
            "n_components": 0,
            "mean_circularity": 0.0,
            "mean_edge_distance": float(EDGE_DIST_CAP),
            "total_turquoise_area": total_turquoise,
            "frac_turquoise": frac_turquoise,
            "prediction": 0,
            "image_path": image_path,
        }

    n_valid = len(component_stats)
    circularities = np.array([c["circularity"] for c in component_stats])
    edge_distances = np.array([c["edge_distance"] for c in component_stats])

    mean_circularity = float(np.mean(circularities))
    mean_edge_dist = float(np.mean(edge_distances))
    # Also compute area-weighted circularity (large components matter more)
    areas_arr = np.array([c["area"] for c in component_stats], dtype=float)
    weighted_circularity = float(
        np.average(circularities, weights=areas_arr) if areas_arr.sum() > 0
        else mean_circularity
    )

    # --- Score computation ---
    # Elongation component: spawn plumes have low circularity
    # Use area-weighted circularity so large blobs dominate
    elongation_score = 1.0 - weighted_circularity
    elongation_term = elongation_score * W_ELONGATION

    # Edge proximity: components near image edges are more likely
    # shoreline-attached. Normalise by EDGE_DIST_CAP.
    normalised_edge = min(mean_edge_dist / EDGE_DIST_CAP, 1.0)
    edge_term = (1.0 - normalised_edge) * W_EDGE_PROX

    # Fragmentation: multiple separate components suggest fragmented
    # spawn patches along a shoreline. Cap at MAX_COMPONENTS_NORM.
    frag_score = min(n_valid / MAX_COMPONENTS_NORM, 1.0)
    frag_term = frag_score * W_FRAGMENTATION

    score_val = elongation_term + edge_term + frag_term

    return {
        "score": round(score_val, 6),
        "n_components": n_valid,
        "mean_circularity": round(mean_circularity, 6),
        "mean_edge_distance": round(mean_edge_dist, 4),
        "total_turquoise_area": total_turquoise,
        "frac_turquoise": round(frac_turquoise, 6),
        "area_weighted_circularity": round(weighted_circularity, 6),
        "prediction": 1 if score_val > threshold else 0,
        "image_path": image_path,
    }


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def score_directory(
    image_dir: str,
    threshold: float = DEFAULT_THRESHOLD,
    hsv_params: HSVThresholds | None = None,
) -> list[dict]:
    """Score all PNGs in a directory.

    Args:
        image_dir: Path to directory containing PNG images.
        threshold: Score threshold for binary prediction.
        hsv_params: HSV threshold parameters. Uses module defaults if None.

    Returns:
        List of score dicts sorted by score descending.
    """
    if hsv_params is None:
        hsv_params = HSVThresholds()

    img_dir = Path(image_dir)
    if not img_dir.is_dir():
        print(f"ERROR: Image directory not found: {img_dir}")
        return []

    pngs = sorted(img_dir.glob("*.png"))
    print(f"  Found {len(pngs)} PNG images in {img_dir}")

    if not pngs:
        return []

    results: list[dict] = []
    for p in pngs:
        result = score_image(str(p), threshold=threshold, hsv_params=hsv_params)
        results.append(result)

    results.sort(key=lambda r: r["score"], reverse=True)

    n_ok = sum(1 for r in results if "error" not in r)
    print(f"  Scored {n_ok}/{len(pngs)} images successfully")
    return results


# ---------------------------------------------------------------------------
# Validation against human labels
# ---------------------------------------------------------------------------

def validate(
    labels_json_path: str,
    image_dir: str,
    threshold: float = DEFAULT_THRESHOLD,
    hsv_params: HSVThresholds | None = None,
) -> dict:
    """Validate against human labels.

    Labels JSON format (matching remoteclip_zero_shot):
        {"labels": [{"filename": "image.png", "label": 1}, ...]}
    where label=1 means positive (spawn), label=0 means negative (no spawn).

    Returns dict with accuracy, best_accuracy, best_threshold, auc_roc,
    avg_precision, confusion_matrix, per_sample list, and counts.
    """
    if hsv_params is None:
        hsv_params = HSVThresholds()

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

    for entry in label_entries:
        fname = entry["filename"]
        true_label = entry["label"]
        img_path = img_dir / fname

        if not img_path.exists():
            print(f"  WARNING: Image not found: {img_path}")
            continue

        result = score_image(str(img_path), threshold=threshold, hsv_params=hsv_params)
        if "error" in result:
            print(f"  WARNING: Could not score {fname}: {result['error']}")
            continue

        per_sample.append({
            "filename": fname,
            "true_label": true_label,
            "prediction": result["prediction"],
            "score": result["score"],
            "n_components": result["n_components"],
            "mean_circularity": result["mean_circularity"],
            "mean_edge_distance": result["mean_edge_distance"],
            "frac_turquoise": result["frac_turquoise"],
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
    thresholds = np.linspace(
        max(y_score.min() - 0.1, 0.0),
        min(y_score.max() + 0.1, 1.0),
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

    print(f"\n  Validation results (threshold={threshold}):")
    print(f"    Total: {n_total}  Pos: {n_pos}  Neg: {n_neg}")
    print(f"    Accuracy (thr={threshold}):  {acc:.4f}")
    print(f"    Best accuracy:                {best_acc:.4f} @ thr={best_thr:.4f}")
    print(f"    AUROC:                        {auroc:.4f}")
    print(f"    Avg Precision:                {ap:.4f}")
    print(f"    Confusion Matrix:             {cm}")

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
        "threshold_used": threshold,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-shot herring spawn detection via shoreline morphology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory of PNG images to score or validate",
    )
    parser.add_argument(
        "--labels-json", type=str, default=None,
        help="Path to validation labels JSON file (same format as remoteclip_labels.json)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save output JSON results",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip scoring output, just run validation against labels",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Score threshold for binary prediction (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--hue-min", type=int, default=HUE_MIN,
        help=f"Minimum hue for turquoise threshold (0-255, default: {HUE_MIN})",
    )
    parser.add_argument(
        "--hue-max", type=int, default=HUE_MAX,
        help=f"Maximum hue for turquoise threshold (0-255, default: {HUE_MAX})",
    )
    parser.add_argument(
        "--sat-min", type=int, default=SAT_MIN,
        help=f"Minimum saturation for turquoise threshold (0-255, default: {SAT_MIN})",
    )
    parser.add_argument(
        "--val-min", type=int, default=VAL_MIN,
        help=f"Minimum value/brightness for turquoise threshold (0-255, default: {VAL_MIN})",
    )
    args = parser.parse_args(argv)

    # Build HSV threshold parameters from CLI args
    hsv_params = HSVThresholds(
        hue_min=args.hue_min,
        hue_max=args.hue_max,
        sat_min=args.sat_min,
        val_min=args.val_min,
    )

    # Resolve paths
    repo_root = Path(__file__).resolve().parent.parent
    given = Path(args.image_dir) if args.image_dir else None
    img_dir = repo_root / args.image_dir if given and not given.is_absolute() else given

    if not args.image_dir:
        parser.print_help()
        print("\nERROR: --image-dir is required")
        return 1

    if not img_dir or not img_dir.is_dir():
        print(f"ERROR: Image directory not found: {img_dir}")
        return 1

    # Resolve labels path
    labels_path = None
    if args.labels_json:
        lp = Path(args.labels_json)
        labels_path = repo_root / args.labels_json if not lp.is_absolute() else lp
        if not labels_path.exists():
            print(f"ERROR: Labels file not found: {labels_path}")
            return 1

    if args.validate_only:
        # ----- Validate-only mode -----
        if labels_path is None:
            print("ERROR: --validate-only requires --labels-json")
            return 1
        print("=" * 60)
        print("  Shoreline Morphology — Validation Only")
        print("=" * 60)
        print(f"  {hsv_params}")
        print(f"  Score threshold: {args.threshold}")
        result = validate(
            str(labels_path), str(img_dir),
            threshold=args.threshold, hsv_params=hsv_params,
        )
    else:
        # ----- Scoring mode -----
        print("=" * 60)
        print("  Shoreline Morphology — Scoring")
        print("=" * 60)
        print(f"  {hsv_params}")
        print(f"  Score threshold: {args.threshold}")
        print(f"  Image directory: {img_dir}")
        result = score_directory(str(img_dir), threshold=args.threshold, hsv_params=hsv_params)
        result = {
            "threshold": args.threshold,
            "hsv_thresholds": {
                "hue_min": hsv_params.hue_min,
                "hue_max": hsv_params.hue_max,
                "sat_min": hsv_params.sat_min,
                "val_min": hsv_params.val_min,
            },
            "n_images_scored": len(result),
            "n_positive_predictions": sum(1 for r in result if r["prediction"] == 1),
            "results": result,
        }

        if labels_path is not None:
            print("\n" + "=" * 60)
            print("  Running validation against labels...")
            val_result = validate(
                str(labels_path), str(img_dir),
                threshold=args.threshold, hsv_params=hsv_params,
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
        # Only print summary to stdout, not the full results list
        # (validation already printed its summary)
        if not args.validate_only:
            print(f"\n  Summary: {result['n_images_scored']} images scored, "
                  f"{result['n_positive_predictions']} positive predictions")

    return 0


if __name__ == "__main__":
    sys.exit(main())
