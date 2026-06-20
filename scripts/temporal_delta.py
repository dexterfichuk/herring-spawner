#!/usr/bin/env python3
"""Zero-shot herring spawn detector using temporal paired-image deltas.

The core idea: compare a candidate spawn image against a recent (<14 day prior)
image of the same location.  Herring spawn = sudden appearance of turquoise
(large spectral change).  Sediment plumes, kelp, shallow bottom -> persistent
features (small change).

We have temporal pairs in ``data/candidates_v2/``:
- Root directory has spawn-season images (2037 PNGs).
- ``data/candidates_v2/offseason/`` has 200 off-season images (paired).
- ``data/candidates_v2/temporal_cache/`` has 692 temporally cached images.

Filenames encode location info; same-location images share the same lat/lon
suffix.

Usage::

    # Score all temporal pairs found in data/candidates_v2/
    python scripts/temporal_delta.py \\
        --image-dir data/candidates_v2 \\
        --output-json data/temporal_delta_results.json

    # Validate against human labels (remoteclip format)
    python scripts/temporal_delta.py \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --output-json data/temporal_delta_results.json

    # Validate only (skip per-pair scoring)
    python scripts/temporal_delta.py \\
        --image-dir data/candidates_v2 \\
        --labels-json data/samples/remoteclip_labels.json \\
        --validate-only

Dependencies::

    pip install numpy Pillow scikit-learn
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
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

DEFAULT_PREDICTION_THRESHOLD = 0.0
"""Default threshold for delta score -> binary prediction."""

SHIRSI_THRESHOLD = 0.015
"""SHSI threshold for counting turquoise pixels."""

TURQUOISE_SHSI_THRESHOLD = 0.05
"""SHSI value above which a pixel is considered turquoise (spawn-like)."""

MAX_DAYS_APART = 365
"""Maximum allowed days between images in a temporal pair."""

BRIGHTNESS_MAX = 200
"""Maximum channel value for NIR water mask approx (RGB thumbnails)."""

GREEN_MIN = 6.375
"""Minimum Green value (0-255) for green background suppression."""

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
# Main format: <region>_<date>_scoreX.XX_<lat>_<lon>_<scene_date>.png
#   e.g. qualicum_2024-03-18_score0.01_49.254865_-124.497442_20240318.png
#
# Off-season format: <region>_<lat_enc>__<lon_enc>_off_<date>_<scene>.png
#   e.g. qualicum_49_374865__124_497442_off_2024_07_16_20240716.png
#   Encoding:  '.' -> '_' , '-' -> '_' (producing '__' prefix for negative lon)
#
# Temporal-cache subdir: <region>_<lat_enc>_-<lon_enc>/
#   e.g. tofino_49_054865_-125_766603/
#   Encoding: '.' -> '_' , '-' is preserved


def extract_location_key_main(filename: str) -> str | None:
    """Extract location key from a standard spawn-season filename.

    The location key is ``lat_lon``, e.g. ``49.254865_-124.497442``.

    Format: ``<region>_<date>_scoreX.XX_<lat>_<lon>_<scene_date>.png``
    """
    m = re.search(r"_(-?\d+\.\d+)_(-?\d+\.\d+)_\d{8}\.png$", filename)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return None


def extract_date_main(filename: str) -> str | None:
    """Extract scene date (YYYY-MM-DD) from a spawn-season filename.

    The last 8 digits before ``.png`` are the scene date in YYYYMMDD.
    """
    m = re.search(r"_(\d{8})\.png$", filename)
    if m:
        ds = m.group(1)
        return f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"
    return None


def extract_location_key_offseason(filename: str) -> str | None:
    """Extract location key from an off-season filename.

    Off-season format:
    ``<region>_<lat_enc>__<lon_enc>_off_<date>_<scene_date>.png``

    Encoding details:
    - ``49.374865``  -> ``49_374865``  (dot -> underscore)
    - ``-124.497442`` -> ``__124_497442`` (minus -> underscore, dot -> underscore,
      producing a double underscore prefix)

    Returns location key ``lat_lon``, e.g. ``49.374865_-124.497442``.
    """
    # Negative lon (most common for BC coast)
    m = re.search(r"_(\d+_\d+)__(\d+_\d+)_off_", filename)
    if m:
        lat_enc = m.group(1)  # "49_374865"
        lon_enc = m.group(2)  # "124_497442" (no leading minus)
        lat = lat_enc.replace("_", ".")
        lon = f"-{lon_enc.replace('_', '.')}"
        return f"{lat}_{lon}"
    # Positive lon (rare, but handle it)
    m = re.search(r"_(\d+_\d+)_(\d+_\d+)_off_", filename)
    if m:
        lat_enc = m.group(1)
        lon_enc = m.group(2)
        lat = lat_enc.replace("_", ".")
        lon = lon_enc.replace("_", ".")
        return f"{lat}_{lon}"
    return None


def extract_date_offseason(filename: str) -> str | None:
    """Extract date from an off-season filename.

    Format: ``_off_YYYY_MM_DD_YYYYMMDD.png``
    """
    m = re.search(r"_off_(\d{4})_(\d{2})_(\d{2})_\d{8}\.png$", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def extract_location_key_cache_subdir(subdir_name: str) -> str | None:
    """Extract location key from a temporal-cache subdirectory name.

    Subdir format: ``<region>_<lat_enc>_-<lon_enc>``
    e.g. ``tofino_49_054865_-125_766603``

    Encoding: ``.`` -> ``_``, ``-`` is preserved in the lon.

    Returns location key ``lat_lon``, e.g. ``49.054865_-125.766603``.
    """
    m = re.search(r"_(\d+_\d+)_(-\d+_\d+)$", subdir_name)
    if m:
        lat_enc = m.group(1)
        lon_enc = m.group(2)
        lat = lat_enc.replace("_", ".")
        lon = lon_enc.replace("_", ".")  # - stays as -
        return f"{lat}_{lon}"
    return None


def extract_date_from_scene_id(filename: str) -> str | None:
    """Extract date (YYYY-MM-DD) from the first 8 digits of a filename.

    Used for temporal-cache files whose names start with a scene ID:
    ``20230428T191911_..._lat_lon.png``
    """
    m = re.match(r"(\d{4})(\d{2})(\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


# ---------------------------------------------------------------------------
# Image loading and delta computation
# ---------------------------------------------------------------------------


def load_image(path: str) -> np.ndarray:
    """Load a PNG image as uint8 RGB numpy array (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def compute_shsi_map(image: np.ndarray) -> np.ndarray:
    """Compute per-pixel SHSI = Green - 2*Red on 0-1 normalised values.

    Args:
        image: uint8 array of shape (H, W, 3).

    Returns:
        Float array of shape (H, W) with per-pixel SHSI values.
    """
    normalised = image.astype(np.float32) / 255.0
    r = normalised[..., 0]
    g = normalised[..., 1]
    return g - 2.0 * r


def compute_turquoise_mask(shsi: np.ndarray, threshold: float = TURQUOISE_SHSI_THRESHOLD) -> np.ndarray:
    """Return boolean mask for pixels with SHSI above threshold."""
    return shsi > threshold


def apply_prefilters(image: np.ndarray) -> np.ndarray:
    """Apply NIR water-mask approx and green-background suppression.

    Returns boolean mask where True means the pixel is eligible.
    """
    max_channel = image.max(axis=-1)
    nir_mask = max_channel < BRIGHTNESS_MAX
    g = image[..., 1].astype(np.float32)
    green_mask = g > GREEN_MIN
    return nir_mask & green_mask


def compute_delta_score(img1_path: str, img2_path: str) -> dict:
    """Compute spectral delta between two images of the same location.

    Steps:
    1. Load both images as uint8 numpy arrays.
    2. Resize img2 to img1's dimensions if different.
    3. Per-pixel absolute difference in RGB.
    4. SHSI delta: absolute per-pixel difference in SHSI (Green - 2*Red).
    5. Turquoise appearance: fraction of pixels changing from non-turquoise
       to turquoise between the two images.
    6. Brightness delta: difference in mean brightness.

    Returns dict with all delta metrics.  ``score`` is shsi_delta_mean.
    """
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    # Resize img2 to match img1 if dimensions differ
    if img1.shape != img2.shape:
        img2_pil = Image.open(img2_path).convert("RGB")
        img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.LANCZOS)
        img2 = np.array(img2_pil, dtype=np.uint8)

    h, w = img1.shape[:2]
    total_pixels = h * w

    # ---- 1. RGB delta ----
    rgb_diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    mean_rgb_delta = float(np.mean(rgb_diff))
    median_rgb_delta = float(np.median(rgb_diff))
    max_rgb_delta = float(np.max(rgb_diff))

    r_delta = float(np.mean(rgb_diff[..., 0]))
    g_delta = float(np.mean(rgb_diff[..., 1]))
    b_delta = float(np.mean(rgb_diff[..., 2]))

    # ---- 2. SHSI delta ----
    shsi1 = compute_shsi_map(img1)
    shsi2 = compute_shsi_map(img2)

    # Apply pre-filters to both images
    mask1 = apply_prefilters(img1)
    mask2 = apply_prefilters(img2)
    combined_mask = mask1 & mask2
    n_eligible = int(combined_mask.sum())

    if n_eligible == 0:
        # Fall back to full-image comparison
        combined_mask = np.ones((h, w), dtype=bool)

    shsi_delta_map = np.abs(shsi1 - shsi2)
    shsi_delta_masked = shsi_delta_map[combined_mask]

    shsi_delta_mean = float(np.mean(shsi_delta_masked)) if len(shsi_delta_masked) > 0 else 0.0
    shsi_delta_max = float(np.max(shsi_delta_masked)) if len(shsi_delta_masked) > 0 else 0.0
    shsi_delta_95p = float(np.percentile(shsi_delta_masked, 95)) if len(shsi_delta_masked) > 0 else 0.0

    shsi1_eligible = shsi1[combined_mask]
    shsi2_eligible = shsi2[combined_mask]
    shsi1_mean = float(np.mean(shsi1_eligible)) if len(shsi1_eligible) > 0 else 0.0
    shsi2_mean = float(np.mean(shsi2_eligible)) if len(shsi2_eligible) > 0 else 0.0
    shsi1_95p = float(np.percentile(shsi1_eligible, 95)) if len(shsi1_eligible) > 0 else 0.0
    shsi2_95p = float(np.percentile(shsi2_eligible, 95)) if len(shsi2_eligible) > 0 else 0.0

    # ---- 3. Turquoise change (appearance / disappearance / persistence) ----
    turquoise1 = compute_turquoise_mask(shsi1)
    turquoise2 = compute_turquoise_mask(shsi2)

    new_turquoise = (~turquoise1) & turquoise2
    lost_turquoise = turquoise1 & (~turquoise2)
    persistent_turquoise = turquoise1 & turquoise2

    n_new_turquoise = int(new_turquoise.sum())
    n_lost_turquoise = int(lost_turquoise.sum())
    n_persistent_turquoise = int(persistent_turquoise.sum())

    turquoise_appearance = n_new_turquoise / total_pixels if total_pixels > 0 else 0.0
    turquoise_disappearance = n_lost_turquoise / total_pixels if total_pixels > 0 else 0.0
    turquoise_persistence = n_persistent_turquoise / total_pixels if total_pixels > 0 else 0.0
    net_turquoise_change = turquoise_appearance - turquoise_disappearance

    # ---- 4. Brightness delta ----
    brightness1 = shsi1  # SHSI correlates with spawn brightness
    brightness2 = shsi2
    brightness_delta = float(np.abs(np.mean(brightness1) - np.mean(brightness2)))

    # ---- 5. Score ----
    # shsi_delta_mean is the primary score — high delta = potential spawn event
    score = shsi_delta_mean
    prediction = 1 if score > DEFAULT_PREDICTION_THRESHOLD else 0

    return {
        "mean_rgb_delta": mean_rgb_delta,
        "median_rgb_delta": median_rgb_delta,
        "max_rgb_delta": max_rgb_delta,
        "r_delta": r_delta,
        "g_delta": g_delta,
        "b_delta": b_delta,
        "shsi_delta_mean": shsi_delta_mean,
        "shsi_delta_max": shsi_delta_max,
        "shsi_delta_95p": shsi_delta_95p,
        "shsi1_mean": shsi1_mean,
        "shsi2_mean": shsi2_mean,
        "shsi1_95p": shsi1_95p,
        "shsi2_95p": shsi2_95p,
        "turquoise_appearance": turquoise_appearance,
        "turquoise_disappearance": turquoise_disappearance,
        "turquoise_persistence": turquoise_persistence,
        "net_turquoise_change": net_turquoise_change,
        "n_new_turquoise": n_new_turquoise,
        "n_lost_turquoise": n_lost_turquoise,
        "n_persistent_turquoise": n_persistent_turquoise,
        "brightness_delta": brightness_delta,
        "n_eligible_pixels": n_eligible,
        "total_pixels": total_pixels,
        "score": score,
        "prediction": prediction,
        "prediction_threshold": DEFAULT_PREDICTION_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Single-image fallback (SHSI 95th percentile)
# ---------------------------------------------------------------------------


def fallback_score_single(image_path: str) -> dict:
    """Fallback score for images without temporal pairs.

    Uses SHSI 95th percentile, the best single-image method identified
    in earlier validation.
    """
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)

    total_pixels = rgb.shape[0] * rgb.shape[1]

    mask = apply_prefilters(rgb)
    n_eligible = int(mask.sum())
    frac_eligible = n_eligible / total_pixels if total_pixels > 0 else 0.0

    if n_eligible == 0:
        return {
            "method": "shsi_fallback",
            "score": 0.0,
            "shsi_mean": 0.0,
            "shsi_95p": 0.0,
            "shsi_max": 0.0,
            "frac_eligible": 0.0,
            "frac_positive": 0.0,
            "prediction": 0,
            "image_path": image_path,
        }

    shsi = compute_shsi_map(rgb)
    shsi_eligible = shsi[mask]

    shsi_mean = float(np.mean(shsi_eligible))
    shsi_max = float(np.max(shsi_eligible))
    shsi_95p = float(np.percentile(shsi_eligible, 95))
    frac_positive = float(np.mean(shsi_eligible > SHIRSI_THRESHOLD))

    return {
        "method": "shsi_fallback",
        "score": shsi_95p,
        "shsi_mean": shsi_mean,
        "shsi_95p": shsi_95p,
        "shsi_max": shsi_max,
        "frac_eligible": frac_eligible,
        "frac_positive": frac_positive,
        "prediction": 1 if shsi_95p > DEFAULT_PREDICTION_THRESHOLD else 0,
        "image_path": image_path,
    }


# ---------------------------------------------------------------------------
# Temporal pair finding
# ---------------------------------------------------------------------------


def _index_images(image_dir: str) -> dict[str, list[dict]]:
    """Build a location-key -> [entry, ...] index from an image directory.

    Searches:
    - ``image_dir`` (main-format PNGs)
    - ``image_dir/offseason/`` (off-season PNGs)
    - ``image_dir/temporal_cache/`` (cached multi-date PNGs)
    """
    img_dir = Path(image_dir)
    index: dict[str, list[dict]] = {}

    # Main directory
    for p in sorted(img_dir.glob("*.png")):
        loc = extract_location_key_main(p.name)
        if loc is None:
            continue
        date = extract_date_main(p.name)
        if date is None:
            continue
        index.setdefault(loc, []).append({
            "path": str(p.resolve()),
            "filename": p.name,
            "date": date,
            "source": "main",
        })

    # Off-season subdirectory
    offseason_dir = img_dir / "offseason"
    if offseason_dir.is_dir():
        for p in sorted(offseason_dir.glob("*.png")):
            loc = extract_location_key_offseason(p.name)
            if loc is None:
                continue
            date = extract_date_offseason(p.name)
            if date is None:
                continue
            index.setdefault(loc, []).append({
                "path": str(p.resolve()),
                "filename": p.name,
                "date": date,
                "source": "offseason",
            })

    # Temporal-cache subdirectory
    cache_dir = img_dir / "temporal_cache"
    if cache_dir.is_dir():
        for subdir in sorted(cache_dir.iterdir()):
            if not subdir.is_dir():
                continue
            loc = extract_location_key_cache_subdir(subdir.name)
            if loc is None:
                continue
            for p in sorted(subdir.glob("*.png")):
                date = extract_date_from_scene_id(p.name)
                if date is None:
                    continue
                index.setdefault(loc, []).append({
                    "path": str(p.resolve()),
                    "filename": p.name,
                    "date": date,
                    "source": "cache",
                })

    return index


def find_temporal_pairs(image_dir: str) -> list[dict]:
    """Find temporal pairs in a directory.

    Groups images by location, then pairs all images that are within
    ``MAX_DAYS_APART`` of each other.

    Returns list of dicts with keys:
    ``img1``, ``img2``, ``location_key``, ``date1``, ``date2``,
    ``days_apart``, ``source1``, ``source2``.
    """
    index = _index_images(image_dir)
    pairs: list[dict] = []

    for loc_key, entries in sorted(index.items()):
        if len(entries) < 2:
            continue

        # Sort by date ascending
        entries.sort(key=lambda e: e["date"])

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                d_i = datetime.strptime(entries[i]["date"], "%Y-%m-%d")
                d_j = datetime.strptime(entries[j]["date"], "%Y-%m-%d")
                days_apart = abs((d_j - d_i).days)

                if days_apart <= MAX_DAYS_APART:
                    pairs.append({
                        "img1": entries[i]["path"],
                        "img2": entries[j]["path"],
                        "location_key": loc_key,
                        "date1": entries[i]["date"],
                        "date2": entries[j]["date"],
                        "days_apart": days_apart,
                        "source1": entries[i]["source"],
                        "source2": entries[j]["source"],
                    })

    return pairs


def find_pairs_for_image(image_path: str, image_dir: str) -> list[dict]:
    """Find temporal pairs that include a specific image.

    Useful for validation: for a labeled image, find all other images at
    the same location within ``MAX_DAYS_APART``.
    """
    index = _index_images(image_dir)
    filename = Path(image_path).name

    loc_key = extract_location_key_main(filename)
    if loc_key is None:
        return []

    date_str = extract_date_main(filename)
    if date_str is None:
        return []

    entries = index.get(loc_key, [])
    pairs: list[dict] = []

    for entry in entries:
        if entry["filename"] == filename:
            continue

        d_entry = datetime.strptime(entry["date"], "%Y-%m-%d")
        d_target = datetime.strptime(date_str, "%Y-%m-%d")
        days_apart = abs((d_entry - d_target).days)

        if days_apart <= MAX_DAYS_APART:
            pairs.append({
                "img1": str(Path(image_path).resolve()),
                "img2": entry["path"],
                "location_key": loc_key,
                "date1": date_str,
                "date2": entry["date"],
                "days_apart": days_apart,
                "source1": "main",
                "source2": entry["source"],
            })

    return pairs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_all_pairs(image_dir: str, quiet: bool = False) -> list[dict]:
    """Find all temporal pairs and score them.

    Returns list of score dicts (see :func:`compute_delta_score`) sorted by
    score descending.
    """
    pairs = find_temporal_pairs(image_dir)
    if not quiet:
        print(f"  Found {len(pairs)} temporal pairs")

    results: list[dict] = []
    errors: list[str] = []

    for pair in pairs:
        try:
            delta = compute_delta_score(pair["img1"], pair["img2"])
            result: dict = {
                "img1": pair["img1"],
                "img2": pair["img2"],
                "location_key": pair["location_key"],
                "date1": pair["date1"],
                "date2": pair["date2"],
                "days_apart": pair["days_apart"],
                "source1": pair["source1"],
                "source2": pair["source2"],
            }
            result.update(delta)
            results.append(result)
        except Exception as exc:
            p1 = Path(pair["img1"]).name
            p2 = Path(pair["img2"]).name
            errors.append(f"{p1} <-> {p2}: {exc}")

    if errors and not quiet:
        print(f"  WARNING: {len(errors)} pairs failed to score")
        for err in errors[:5]:
            print(f"    {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def score_all_unpaired(image_dir: str, quiet: bool = False) -> list[dict]:
    """For images without a temporal pair, use SHSI single-image fallback.

    Returns list of score dicts sorted by score descending.
    """
    img_dir = Path(image_dir)
    pngs = sorted(img_dir.glob("*.png"))
    if not quiet:
        print(f"  Found {len(pngs)} PNG images")

    # Build set of location keys that HAVE temporal pairs
    pairs = find_temporal_pairs(image_dir)
    paired_locations: set[str] = set()
    for p in pairs:
        paired_locations.add(p["location_key"])

    results: list[dict] = []
    errors: list[str] = []

    for p in pngs:
        loc = extract_location_key_main(p.name)
        date = extract_date_main(p.name)
        if loc is None or date is None:
            continue
        if loc in paired_locations:
            continue  # This location already has a temporal pair

        try:
            fallback = fallback_score_single(str(p.resolve()))
            result: dict = {
                "filename": p.name,
                "location_key": loc,
                "date": date,
                "method": "shsi_fallback",
            }
            result.update(fallback)
            results.append(result)
        except Exception as exc:
            errors.append(f"{p.name}: {exc}")

    if errors and not quiet:
        print(f"  WARNING: {len(errors)} unpaired images failed to score")
        for err in errors[:5]:
            print(f"    {err}")

    results.sort(key=lambda r: r["score"], reverse=True)
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
    """Validate temporal delta scoring against human labels.

    For each labeled image:
    - If a temporal pair exists at the same location, use the delta score.
    - If not, fall back to SHSI 95th percentile single-image score.

    Labels JSON format (matching ``remoteclip_labels.json``)::

        {"labels": [{"filename": "image.png", "label": 1}, ...]}

    where ``label`` = 1 means positive (spawn) and ``label`` = 0 means
    negative (no spawn).

    Returns dict with accuracy, best_accuracy, best_threshold, auc_roc,
    avg_precision, confusion_matrix, per_sample, and counts.
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

    img_dir = Path(image_dir)
    per_sample: list[dict] = []
    n_temporal = 0
    n_fallback = 0

    for entry in label_entries:
        fname = entry["filename"]
        true_label = entry["label"]
        img_path = img_dir / fname

        if not img_path.exists():
            if not quiet:
                print(f"  WARNING: Image not found: {img_path}")
            continue

        # Try to find temporal pairs for this image
        pairs = find_pairs_for_image(str(img_path), str(img_dir))

        if pairs:
            # Use the pair with the smallest days_apart (closest in time)
            pairs.sort(key=lambda p: p["days_apart"])
            best_pair = pairs[0]

            try:
                delta = compute_delta_score(best_pair["img1"], best_pair["img2"])
                score = delta["score"]
                pred = 1 if score > threshold else 0

                per_sample.append({
                    "filename": fname,
                    "true_label": true_label,
                    "prediction": pred,
                    "score": score,
                    "method": "temporal_delta",
                    "paired_with": Path(best_pair["img2"]).name,
                    "date1": best_pair["date1"],
                    "date2": best_pair["date2"],
                    "days_apart": best_pair["days_apart"],
                    **delta,
                })
                n_temporal += 1
                continue
            except Exception as exc:
                if not quiet:
                    print(f"  WARNING: Delta failed for {fname}: {exc}")

        # Fall back to single-image scoring
        if not quiet:
            print(f"  No temporal pair for {fname}, using SHSI fallback")

        try:
            fallback = fallback_score_single(str(img_path))
            score = fallback["score"]
            pred = 1 if score > threshold else 0

            per_sample.append({
                "filename": fname,
                "true_label": true_label,
                "prediction": pred,
                "score": score,
                "method": "shsi_fallback",
                **fallback,
            })
            n_fallback += 1
        except Exception as exc:
            if not quiet:
                print(f"  WARNING: Fallback failed for {fname}: {exc}")

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

    # ---- Aggregate metrics ----
    y_true = np.array([s["true_label"] for s in per_sample])
    y_score = np.array([s["score"] for s in per_sample])

    n_total = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_total - n_pos

    # Default threshold predictions
    y_pred = (y_score > threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))

    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        cm = confusion_matrix(y_true, y_pred).tolist()
    else:
        cm = [[0, 0], [0, 0]]

    # ---- Best accuracy via threshold sweep ----
    lo = float(y_score.min()) - 0.1
    hi = float(y_score.max()) + 0.1
    thresholds = np.linspace(lo, hi, 201)
    best_acc = 0.0
    best_thr = float(threshold)
    for thr in thresholds:
        thr_pred = (y_score > thr).astype(int)
        thr_acc = accuracy_score(y_true, thr_pred)
        if thr_acc > best_acc:
            best_acc = thr_acc
            best_thr = float(thr)

    # ---- AUROC ----
    auroc = 0.0
    if n_pos > 0 and n_neg > 0 and not np.all(y_score == y_score[0]):
        try:
            auroc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auroc = 0.0

    # ---- Average precision ----
    ap = 0.0
    if n_pos > 0:
        try:
            ap = float(average_precision_score(y_true, y_score))
        except Exception:
            ap = 0.0

    if not quiet:
        print(f"\n  Validation results (threshold={threshold}):")
        print(f"    Total: {n_total}  Pos: {n_pos}  Neg: {n_neg}")
        print(f"    Temporal pairs used: {n_temporal}  Fallback: {n_fallback}")
        print(f"    Accuracy:       {acc:.4f}")
        print(f"    Best accuracy:  {best_acc:.4f} @ thr={best_thr:.4f}")
        print(f"    AUROC:          {auroc:.4f}")
        print(f"    Avg Precision:  {ap:.4f}")
        print(f"    Confusion Mat:  {cm}")

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
        "n_temporal_pairs_used": n_temporal,
        "n_fallback_used": n_fallback,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-shot herring spawn detector using temporal paired-image deltas",
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
        help="Skip scoring all pairs; run validation against labels only",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_PREDICTION_THRESHOLD,
        help=(
            f"Decision threshold for binary prediction "
            f"(default: {DEFAULT_PREDICTION_THRESHOLD})"
        ),
    )
    args = parser.parse_args(argv)

    # ---- Required: --image-dir ----
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

    # ---- Resolve --labels-json ----
    labels_path = None
    if args.labels_json:
        lp = Path(args.labels_json)
        labels_path = repo_root / args.labels_json if not lp.is_absolute() else lp
        if not labels_path.exists():
            print(f"ERROR: Labels file not found: {labels_path}")
            return 1

    # ---- Validate-only mode ----
    if args.validate_only:
        if labels_path is None:
            print("ERROR: --validate-only requires --labels-json")
            return 1

        print("=" * 60)
        print("  Temporal Delta — Validation")
        print("=" * 60)
        result = validate(
            str(labels_path),
            str(img_dir),
            threshold=args.threshold,
        )

    else:
        print("=" * 60)
        print("  Temporal Delta — Herring Spawn Scoring")
        print("=" * 60)

        # Score all temporal pairs
        print("\n--- Scoring temporal pairs ---")
        pair_results = score_all_pairs(str(img_dir))

        # Also score unpaired images (fallback)
        print("\n--- Scoring unpaired images (fallback) ---")
        unpaired_results = score_all_unpaired(str(img_dir), quiet=len(pair_results) > 0)

        n_predicted = sum(1 for r in pair_results if r.get("prediction", 0) == 1)
        n_unpaired_predicted = sum(1 for r in unpaired_results if r.get("prediction", 0) == 1)

        result: dict = {
            "method": "temporal_delta",
            "description": (
                "Temporal paired-image delta scoring. Pairs are images at the "
                "same location from different dates; the delta score is the "
                "mean per-pixel absolute SHSI difference between images."
            ),
            "threshold": args.threshold,
            "n_pairs_scored": len(pair_results),
            "n_unpaired_scored": len(unpaired_results),
            "n_predictions_positive": n_predicted + n_unpaired_predicted,
            "results_pairs": pair_results,
            "results_unpaired": unpaired_results,
        }

        # Also validate if labels are provided
        if labels_path is not None:
            print("\n" + "-" * 60)
            print("  Running validation against labels...")
            val_result = validate(
                str(labels_path),
                str(img_dir),
                threshold=args.threshold,
            )
            result["validation"] = val_result

    # ---- Save output ----
    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = repo_root / args.output_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Results saved to: {out_path}")
    else:
        # Print summary to stdout
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
