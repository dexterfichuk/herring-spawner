#!/usr/bin/env python3
"""Comprehensive 13-region BC herring spawn scan using DINOv2 SubspaceAD.

Scans ALL 13 known BC herring habitat regions across 3 years (2023-2025)
using the champion DINOv2 patch-level SubspaceAD method.

Pipeline:
  1. Train PCA subspace on 126 negative images (one-time)
  2. Generate grid points at 0.01° spacing across all regions
  3. For each year, scan all grid points via GEE
  4. Score each thumbnail with SubspaceAD (patch reconstruction residuals)
  5. Filter aggressively: score > 95th percentile, area > 2%, ≥8 anomalous patches
  6. Generate heatmap + segmentation overlays
  7. Compare against known DFO events
  8. Track multi-year consistency
  9. Generate review.html with all filters

Usage:
    # Dry run to estimate scale
    python scripts/scan_13regions_subspacead.py --dry-run

    # Full scan (interruptible, saves progress)
    python scripts/scan_13regions_subspacead.py --workers 6

    # Serve existing review
    python scripts/scan_13regions_subspacead.py --serve-only --port 8776

Output: data/subspace_13regions/
    review.html          — Interactive review page
    manifest.json        — All candidate metadata
    summary.json         — Scan summary stats
    thumbnails/          — Raw candidate PNGs
    overlays/{name}/     — original.png, heatmap.png, segmentation.png
    dfo_comparison.json  — Comparison with known DFO events
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import threading
import time
import webbrowser
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, date, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------

MODEL_NAME = "dinov2_vits14"
EMBED_DIM = 384
N_PATCHES = 256  # 16 × 16 grid
PATCH_GRID_SIZE = 16

# SubspaceAD defaults
DEFAULT_N_COMPONENTS = 64
ANOMALOUS_PATCH_FRAC = 0.10  # top 10% most anomalous patches
PCA_VARIANCE_TARGET = 0.90

# Scan defaults
DEFAULT_PROJECT = "redd-fish"
DEFAULT_OUTPUT_DIR = Path("data/subspace_13regions")
DEFAULT_NEGATIVE_DIR = Path("data/samples/negative")
DEFAULT_PORT = 8776
DEFAULT_WORKERS = 6
DEFAULT_MAX_CLOUD = 30.0
DEFAULT_GRID_SPACING = 0.01
DEFAULT_MAX_CANDIDATES = 500
DEFAULT_TOP_CANDIDATES = 200
SCAN_START_MONTH = 2  # February
SCAN_END_MONTH = 4    # April
SCAN_YEARS = [2023, 2024, 2025]

# Filters
MIN_SPAWN_AREA_FRAC = 0.02  # 2%
MIN_ANOMALOUS_PATCHES = 8

# ---------------------------------------------------------------------------
# DINOv2 transform
# ---------------------------------------------------------------------------

DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# 13 BC Herring Habitat Regions (plus expansions for comprehensive coverage)
# ---------------------------------------------------------------------------
# Each region has a center point and radius in km. Points are generated
# within the circular buffer at the configured grid spacing.

REGIONS: list[dict[str, Any]] = [
    # === Strait of Georgia (most active spawn region) ===
    {"name": "strait-of-georgia-north", "lat": 49.68, "lon": -124.70, "radius_km": 20},
    {"name": "strait-of-georgia-south", "lat": 49.15, "lon": -123.85, "radius_km": 20},
    {"name": "qualicum",               "lat": 49.35, "lon": -124.45, "radius_km": 15},
    {"name": "nanaimo",                "lat": 49.15, "lon": -123.85, "radius_km": 15},
    {"name": "comox",                  "lat": 49.68, "lon": -124.88, "radius_km": 15},
    {"name": "denman-island",          "lat": 49.55, "lon": -124.80, "radius_km": 10},
    {"name": "nanoose",                "lat": 49.30, "lon": -124.20, "radius_km": 12},
    {"name": "howe-sound",             "lat": 49.50, "lon": -123.30, "radius_km": 15},

    # === West Coast Vancouver Island ===
    {"name": "tofino",                 "lat": 49.15, "lon": -125.90, "radius_km": 15},
    {"name": "ucluelet",               "lat": 48.94, "lon": -125.55, "radius_km": 10},
    {"name": "clayoquot-sound",        "lat": 49.25, "lon": -126.00, "radius_km": 12},
    {"name": "hesquiat-harbour",       "lat": 49.55, "lon": -126.42, "radius_km": 10},

    # === Barkley Sound ===
    {"name": "barkley-sound",          "lat": 48.88, "lon": -125.20, "radius_km": 15},

    # === Nootka Sound ===
    {"name": "nootka-sound",           "lat": 49.60, "lon": -126.60, "radius_km": 15},

    # === Kyuquot Sound ===
    {"name": "kyuquot",                "lat": 50.00, "lon": -127.20, "radius_km": 12},

    # === Quatsino Sound ===
    {"name": "quatsino-sound",         "lat": 50.50, "lon": -128.00, "radius_km": 15},

    # === Johnstone Strait ===
    {"name": "johnstone-strait",       "lat": 50.40, "lon": -126.10, "radius_km": 15},
    {"name": "port-mcneill",           "lat": 50.60, "lon": -127.10, "radius_km": 12},

    # === Queen Charlotte Strait ===
    {"name": "queen-charlotte-strait", "lat": 50.90, "lon": -127.30, "radius_km": 15},

    # === Hecate Strait ===
    {"name": "hecate-strait",          "lat": 53.00, "lon": -129.70, "radius_km": 15},

    # === Central Coast ===
    {"name": "spiller-channel",        "lat": 52.30, "lon": -128.30, "radius_km": 15},
    {"name": "milbanke-sound",         "lat": 52.50, "lon": -128.80, "radius_km": 15},

    # === North Coast / Prince Rupert ===
    {"name": "prince-rupert",          "lat": 54.30, "lon": -130.40, "radius_km": 20},

    # === Haida Gwaii ===
    {"name": "haida-gwaii-south",      "lat": 52.40, "lon": -131.40, "radius_km": 15},
    {"name": "masset-inlet",           "lat": 53.70, "lon": -132.90, "radius_km": 15},
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

PRINT_LOCK = threading.Lock()
MANIFEST_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def html_escape(value: Any) -> str:
    import html
    return html.escape(str(value))


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


# ---------------------------------------------------------------------------
# Grid point generation
# ---------------------------------------------------------------------------

def generate_grid_points(
    regions: list[dict[str, Any]],
    spacing_deg: float,
) -> list[dict[str, Any]]:
    """Generate grid points within each region's circular buffer.

    Args:
        regions: List of region dicts with name, lat, lon, radius_km.
        spacing_deg: Grid spacing in degrees (~0.01° ≈ 1.1 km).

    Returns:
        List of dicts with: region, lat, lon.
    """
    points: list[dict[str, Any]] = []
    total_estimated = 0

    for region in regions:
        lat, lon = region["lat"], region["lon"]
        radius_km = region["radius_km"]
        radius_deg_lat = radius_km / 111.0
        radius_deg_lon = radius_km / (111.0 * math.cos(math.radians(lat)))

        n_steps_lat = max(1, int(2 * radius_deg_lat / spacing_deg))
        n_steps_lon = max(1, int(2 * radius_deg_lon / spacing_deg))

        region_points = 0
        for i in range(n_steps_lat + 1):
            p_lat = lat - radius_deg_lat + i * spacing_deg
            if abs(p_lat - lat) > radius_deg_lat + spacing_deg * 0.5:
                continue
            for j in range(n_steps_lon + 1):
                p_lon = lon - radius_deg_lon + j * spacing_deg
                if abs(p_lon - lon) > radius_deg_lon + spacing_deg * 0.5:
                    continue

                dlat = (p_lat - lat) * 111.0
                dlon = (p_lon - lon) * 111.0 * math.cos(math.radians(lat))
                dist_km = math.sqrt(dlat ** 2 + dlon ** 2)
                if dist_km <= radius_km:
                    points.append({
                        "region": region["name"],
                        "lat": round(p_lat, 6),
                        "lon": round(p_lon, 6),
                    })
                    region_points += 1

        total_estimated += region_points

    return points


# ---------------------------------------------------------------------------
# DINOv2 model loading
# ---------------------------------------------------------------------------

def load_dinov2(device: str) -> torch.nn.Module:
    """Load DINOv2 ViT-S/14, cache globally to avoid reloading."""
    print(f"  Loading DINOv2 model ({MODEL_NAME}) on {device}...")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval()
    model = model.to(device)
    return model


# Global model cache
_dinov2_model: torch.nn.Module | None = None
_dinov2_device: str = ""


def get_dinov2(device: str) -> torch.nn.Module:
    global _dinov2_model, _dinov2_device
    if _dinov2_model is None or _dinov2_device != device:
        _dinov2_model = load_dinov2(device)
        _dinov2_device = device
    return _dinov2_model


# ---------------------------------------------------------------------------
# Patch embedding extraction
# ---------------------------------------------------------------------------

def extract_patch_embeddings_single(
    image_data: bytes | str | Path,
    device: str = "cpu",
) -> np.ndarray:
    """Extract DINOv2 patch tokens from a single image.

    Args:
        image_data: Raw PNG bytes, file path, or PIL Image.
        device: Device for inference.

    Returns:
        np.ndarray of shape (256, 384) — patch tokens.
    """
    if isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
    elif isinstance(image_data, (str, Path)):
        img = Image.open(str(image_data)).convert("RGB")
    else:
        img = image_data

    model = get_dinov2(device)
    tensor = DINO_TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        patch_tokens, _cls_tokens = model.get_intermediate_layers(
            tensor, n=1, reshape=True, return_class_token=True,
        )[0]

    # patch_tokens: [1, 384, 16, 16] -> [256, 384]
    pt = (patch_tokens
          .flatten(2)
          .transpose(1, 2)
          .squeeze(0)
          .cpu()
          .numpy()
          .astype(np.float32))

    return pt


# ---------------------------------------------------------------------------
# PCA training on negatives
# ---------------------------------------------------------------------------

def train_pca_on_negatives(
    negative_dir: str | Path,
    n_components: int = DEFAULT_N_COMPONENTS,
    device: str = "cpu",
    sample_frac: float = 0.15,
) -> dict:
    """Train PCA subspace on DINOv2 patch tokens from negative images.

    Args:
        negative_dir: Directory of negative (non-spawn) PNG images.
        n_components: Number of PCA components.
        device: Device for DINOv2 inference.
        sample_frac: Fraction of patches sampled per image for training.

    Returns:
        dict with 'pca_model', 'n_components', 'n_images_used',
        'n_patches_trained', 'explained_variance_ratio', etc.
    """
    neg_dir = Path(negative_dir)
    pngs = sorted(neg_dir.glob("*.png"))
    print(f"  Found {len(pngs)} negative images in {neg_dir}")

    if not pngs:
        return {"error": "No negative images found"}

    model = get_dinov2(device)
    all_patches: list[np.ndarray] = []
    filenames: list[str] = []

    for p in pngs:
        try:
            pt = extract_patch_embeddings_single(str(p), device=device)
            all_patches.append(pt)
            filenames.append(p.name)
        except Exception as exc:
            print(f"  WARNING: Failed to embed {p.name}: {exc}")

    if not all_patches:
        return {"error": "No patches extracted from negatives"}

    all_patches_arr = np.vstack(all_patches)  # (N*256, 384)
    n_images = len(all_patches)
    n_total_patches = len(all_patches_arr)

    # Sample patches per image to keep memory manageable
    if sample_frac < 1.0:
        sampled: list[np.ndarray] = []
        rng = np.random.RandomState(42)
        for img_idx in range(n_images):
            start = img_idx * N_PATCHES
            end = start + N_PATCHES
            img_patches = all_patches_arr[start:end]
            n_sample = max(1, int(N_PATCHES * sample_frac))
            indices = rng.choice(N_PATCHES, size=n_sample, replace=False)
            sampled.append(img_patches[indices])
        training_patches = np.vstack(sampled)
        print(f"  Sampled {len(training_patches)} patches ({sample_frac:.0%} per image)")
    else:
        training_patches = all_patches_arr
        print(f"  Using all {n_total_patches} patches")

    # Clamp n_components
    max_components = min(len(training_patches) - 1, EMBED_DIM)
    if n_components > max_components:
        n_components = max(1, max_components)
        print(f"  Clamped n_components to {n_components}")

    # Fit PCA
    print(f"  Fitting PCA with {n_components} components on {len(training_patches)} patches...")
    pca = PCA(n_components=n_components, whiten=False, random_state=42)
    pca.fit(training_patches)

    total_var = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance: {total_var:.4f} ({n_components} components)")
    print(f"  Trained on {n_images} negative images ({n_total_patches} total patches)")

    # Compute 95th percentile score on training negatives for filtering
    # Score each image by its top-10% patch residual mean
    neg_scores = []
    for img_idx in range(n_images):
        start = img_idx * N_PATCHES
        end = start + N_PATCHES
        img_patches = all_patches_arr[start:end]
        projected = pca.transform(img_patches)
        reconstructed = pca.inverse_transform(projected)
        residuals = np.mean((img_patches - reconstructed) ** 2, axis=1)
        sorted_res = np.sort(residuals)[::-1]
        n_anom = max(1, int(N_PATCHES * ANOMALOUS_PATCH_FRAC))
        score = float(np.mean(sorted_res[:n_anom]))
        neg_scores.append(score)

    neg_scores_arr = np.array(neg_scores)
    p95 = float(np.percentile(neg_scores_arr, 95))
    print(f"  Negative score distribution: mean={neg_scores_arr.mean():.6f} "
          f"std={neg_scores_arr.std():.6f} p95={p95:.6f}")

    return {
        "pca_model": pca,
        "n_components": n_components,
        "n_images_used": n_images,
        "n_patches_trained": len(training_patches),
        "explained_variance_ratio": total_var,
        "neg_scores": neg_scores,
        "neg_score_p95": p95,
        "neg_score_mean": float(neg_scores_arr.mean()),
        "neg_score_std": float(neg_scores_arr.std()),
        "model_name": MODEL_NAME,
    }


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_patches(patches: np.ndarray, pca_model: PCA) -> dict:
    """Score a set of patch embeddings using PCA reconstruction residuals.

    Args:
        patches: (256, 384) array of patch embeddings.
        pca_model: Fitted PCA model.

    Returns:
        dict with score_top10p, score_mean, score_max, auto_threshold,
        spawn_area_frac, n_spawn_patches, heatmap (16x16), mask (224x224),
        patch_residuals (256,).
    """
    # Project and reconstruct
    projected = pca_model.transform(patches)
    reconstructed = pca_model.inverse_transform(projected)

    # Per-patch MSE residuals
    residuals = np.mean((patches - reconstructed) ** 2, axis=1)  # (256,)

    # Aggregate scores
    sorted_res = np.sort(residuals)[::-1]
    n_anom = max(1, int(N_PATCHES * ANOMALOUS_PATCH_FRAC))
    score_top10p = float(np.mean(sorted_res[:n_anom]))
    score_mean = float(np.mean(residuals))
    score_max = float(np.max(residuals))

    # Auto-threshold: mean + 2*std
    auto_threshold = float(np.mean(residuals) + 2.0 * np.std(residuals))

    # 16x16 heatmap
    patch_grid = residuals.reshape(PATCH_GRID_SIZE, PATCH_GRID_SIZE)

    # Upsample to 224x224
    heatmap_tensor = torch.from_numpy(patch_grid).float().unsqueeze(0).unsqueeze(0)
    heatmap_up = F.interpolate(
        heatmap_tensor, size=(224, 224), mode="bilinear", align_corners=False,
    )
    heatmap_224 = heatmap_up.squeeze().cpu().numpy()  # (224, 224)

    # Binary mask
    mask_224 = (heatmap_224 > auto_threshold).astype(np.float32)
    spawn_area_frac = float(np.mean(mask_224))
    n_spawn_patches = int((patch_grid > auto_threshold).sum())

    return {
        "score_top10p": score_top10p,
        "score_mean": score_mean,
        "score_max": score_max,
        "auto_threshold": auto_threshold,
        "spawn_area_frac": spawn_area_frac,
        "n_spawn_patches": n_spawn_patches,
        "n_patches": N_PATCHES,
        "patch_residuals_16x16": patch_grid,
        "heatmap_224": heatmap_224,
        "mask_224": mask_224,
    }


def passes_filters(
    score_result: dict,
    p95_threshold: float,
) -> bool:
    """Check if a candidate passes all quality filters."""
    if score_result["score_top10p"] <= p95_threshold:
        return False
    if score_result["spawn_area_frac"] < MIN_SPAWN_AREA_FRAC:
        return False
    if score_result["n_spawn_patches"] < MIN_ANOMALOUS_PATCHES:
        return False
    return True


# ---------------------------------------------------------------------------
# Overlay generation
# ---------------------------------------------------------------------------

def make_heatmap_overlay(
    original_img: Image.Image,
    heatmap_224: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Create heatmap overlay: original + jet colormap blended."""
    import matplotlib.cm as cm

    original = original_img.convert("RGB").resize((224, 224))
    h_min, h_max = heatmap_224.min(), heatmap_224.max()
    if h_max - h_min > 1e-12:
        heatmap_norm = (heatmap_224 - h_min) / (h_max - h_min)
    else:
        heatmap_norm = np.zeros_like(heatmap_224)

    jet = cm.get_cmap("jet")
    heatmap_colored = (jet(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_colored, "RGB")
    overlay = Image.blend(original, heatmap_pil, alpha)
    return overlay


def make_segmentation_overlay(
    original_img: Image.Image,
    mask_224: np.ndarray,
    color: tuple = (0, 255, 0),
) -> Image.Image:
    """Create segmentation overlay with green contours."""
    try:
        import cv2
    except ImportError:
        # Fallback: simple green overlay on mask
        original = original_img.convert("RGB").resize((224, 224))
        arr = np.array(original)
        mask_rgb = np.stack([mask_224 * c for c in color], axis=2).astype(np.uint8)
        blended = cv2.addWeighted(arr, 0.7, mask_rgb, 0.3, 0)
        return Image.fromarray(blended)

    original = original_img.convert("RGB").resize((224, 224))
    img_np = np.array(original)
    mask_uint8 = (mask_224 * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    contour_img = img_np.copy()
    cv2.drawContours(contour_img, contours, -1, color, thickness=2)
    return Image.fromarray(contour_img)


# ---------------------------------------------------------------------------
# GEE helpers
# ---------------------------------------------------------------------------

def find_best_scene(
    ee_module: Any,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_cloud: float,
) -> dict[str, Any] | None:
    """Find the single best Sentinel-2 scene for a point and date range."""
    try:
        collection = ee_module.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        point = ee_module.Geometry.Point(lon, lat)

        scenes = (
            collection
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee_module.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        scene_ids = scenes.aggregate_array("system:index").getInfo()
        clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()

        if not scene_ids:
            return None

        best_idx = 0
        sid = scene_ids[best_idx]
        return {
            "scene_id": sid,
            "cloud": float(clouds[best_idx]),
            "date": f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}",
        }
    except Exception as exc:
        with PRINT_LOCK:
            print(f"    GEE search error at ({lat:.4f}, {lon:.4f}): {exc}")
        return None


def download_thumbnail(
    ee_module: Any,
    lat: float,
    lon: float,
    scene_id: str,
) -> bytes | None:
    """Download a 512×512 RGB thumbnail from a Sentinel-2 scene."""
    try:
        scene_img = ee_module.Image(
            f"COPERNICUS/S2_SR_HARMONIZED/{scene_id}"
        )
        rgb = scene_img.select(["B4", "B3", "B2"])
        region = ee_module.Geometry.Point(lon, lat).buffer(1280).bounds()

        url = rgb.getThumbURL({
            "min": 0,
            "max": 3000,
            "region": region,
            "dimensions": 512,
            "format": "png",
        })

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        with PRINT_LOCK:
            print(f"    Download failed for {scene_id}: {exc}")
        return None
    except Exception as exc:
        with PRINT_LOCK:
            print(f"    Thumbnail error for {scene_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Candidate storage
# ---------------------------------------------------------------------------

def save_candidate(
    output_dir: Path,
    png_bytes: bytes,
    info: dict[str, Any],
) -> str:
    """Save a candidate thumbnail and return its filename."""
    region = info["region"]
    date_str = info["date"]
    lat = info["lat"]
    lon = info["lon"]
    scene_id = info["scene_id"]
    scene_short = scene_id[:8] if len(scene_id) >= 8 else scene_id

    fname = f"{region}_{date_str}_score{info['score']:.4f}_{lat}_{lon}_{scene_short}.png"
    fname = "".join(c if c.isalnum() or c in "._-" else "_" for c in fname)

    thumb_dir = output_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    fpath = thumb_dir / fname
    fpath.write_bytes(png_bytes)
    return fname


def update_manifest(output_dir: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Append a candidate to the manifest JSON and return current entries."""
    manifest_path = output_dir / "manifest.json"
    with MANIFEST_LOCK:
        entries: list[dict[str, Any]] = []
        if manifest_path.exists():
            try:
                entries = json.loads(manifest_path.read_text())
                if not isinstance(entries, list):
                    entries = []
            except (json.JSONDecodeError, OSError):
                entries = []
        entries.append(entry)
        manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return entries


def load_manifest(output_dir: Path) -> list[dict[str, Any]]:
    """Load current manifest entries."""
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text())
            if isinstance(entries, list):
                return entries
        except (json.JSONDecodeError, OSError):
            pass
    return []


# ---------------------------------------------------------------------------
# Print progress
# ---------------------------------------------------------------------------

def print_progress(
    idx: int,
    total: int,
    region: str,
    lat: float,
    lon: float,
    status: str,
    elapsed: float,
    year: int | None = None,
) -> None:
    """Print a single progress line."""
    pct = 100.0 * (idx + 1) / total
    year_str = f" {year}" if year else ""
    if idx > 0 and elapsed > 0:
        rate = idx / elapsed
        remaining_s = (total - idx) / rate if rate > 0 else 0
        eta = time.strftime("%H:%M:%S", time.gmtime(remaining_s))
    else:
        eta = "?"
    loc = f"({lat:.4f}, {lon:.4f})"
    with PRINT_LOCK:
        print(f"  [{idx + 1}/{total}] ({pct:.0f}%){year_str} {region} {loc} | {status} | ETA {eta}")


# ---------------------------------------------------------------------------
# Point processing (per year)
# ---------------------------------------------------------------------------

def _process_point_year(
    point: dict[str, Any],
    year: int,
    ee_module: Any,
    pca_model: PCA,
    p95_threshold: float,
    output_dir: Path,
    device: str,
    idx: int,
    total: int,
    start_time: float,
) -> dict[str, Any]:
    """Process a single grid point for a specific year.

    Returns dict with:
        processed, candidate_saved, no_scene, download_error, scoring_error,
        filtered_out, entry (if candidate saved).
    """
    result: dict[str, Any] = {
        "processed": 1,
        "candidate_saved": 0,
        "no_scene": 0,
        "download_error": 0,
        "scoring_error": 0,
        "filtered_out": 0,
        "entry": None,
    }

    scan_start = f"{year}-{SCAN_START_MONTH:02d}-01"
    scan_end = f"{year}-{SCAN_END_MONTH:02d}-30"

    elapsed = time.time() - start_time
    scene_info = find_best_scene(
        ee_module, point["lat"], point["lon"],
        scan_start, scan_end, DEFAULT_MAX_CLOUD,
    )

    if scene_info is None:
        print_progress(idx, total, point["region"], point["lat"], point["lon"],
                       "no scene", elapsed, year)
        result["no_scene"] = 1
        return result

    thumb_bytes = download_thumbnail(ee_module, point["lat"], point["lon"],
                                     scene_info["scene_id"])
    if thumb_bytes is None:
        print_progress(idx, total, point["region"], point["lat"], point["lon"],
                       "download error", elapsed, year)
        result["download_error"] = 1
        return result

    # Extract patch embeddings
    try:
        patches = extract_patch_embeddings_single(thumb_bytes, device=device)
    except Exception as exc:
        print_progress(idx, total, point["region"], point["lat"], point["lon"],
                       f"embed error: {exc}", elapsed, year)
        result["scoring_error"] = 1
        return result

    # Score with SubspaceAD
    try:
        score_result = score_patches(patches, pca_model)
    except Exception as exc:
        print_progress(idx, total, point["region"], point["lat"], point["lon"],
                       f"score error: {exc}", elapsed, year)
        result["scoring_error"] = 1
        return result

    # Apply filters
    if not passes_filters(score_result, p95_threshold):
        print_progress(idx, total, point["region"], point["lat"], point["lon"],
                       f"filtered (score={score_result['score_top10p']:.6f} "
                       f"area={score_result['spawn_area_frac']*100:.1f}%)",
                       elapsed, year)
        result["filtered_out"] = 1
        return result

    # Save candidate
    info = {
        "region": point["region"],
        "lat": point["lat"],
        "lon": point["lon"],
        "date": scene_info["date"],
        "scene_id": scene_info["scene_id"],
        "cloud": scene_info["cloud"],
        "score": round(score_result["score_top10p"], 6),
        "year": year,
    }
    fname = save_candidate(output_dir, thumb_bytes, info)

    entry = {
        **info,
        "thumbnail_path": f"thumbnails/{fname}",
        "score_top10p": score_result["score_top10p"],
        "score_mean": score_result["score_mean"],
        "score_max": score_result["score_max"],
        "auto_threshold": score_result["auto_threshold"],
        "spawn_area_frac": score_result["spawn_area_frac"],
        "n_spawn_patches": score_result["n_spawn_patches"],
        "n_patches": score_result["n_patches"],
    }

    update_manifest(output_dir, entry)

    # Check candidate limit
    manifest = load_manifest(output_dir)
    if len(manifest) > DEFAULT_MAX_CANDIDATES * 2:
        print_progress(idx, total, point["region"], point["lat"], point["lon"],
                       f"reached candidate limit ({len(manifest)})", elapsed, year)
        result["entry"] = entry
        result["candidate_saved"] = 1
        return result

    print_progress(idx, total, point["region"], point["lat"], point["lon"],
                   f"CANDIDATE score={score_result['score_top10p']:.6f} "
                   f"area={score_result['spawn_area_frac']*100:.1f}% {fname}",
                   elapsed, year)

    result["entry"] = entry
    result["candidate_saved"] = 1
    return result


# ---------------------------------------------------------------------------
# Overlay generation (post-process)
# ---------------------------------------------------------------------------

def generate_overlays(
    output_dir: Path,
    device: str,
    pca_model: PCA | None = None,
    pca_meta: dict | None = None,
) -> list[dict[str, Any]]:
    """Generate heatmap + segmentation overlays for all saved candidates.

    Args:
        output_dir: Directory with candidate thumbnails.
        device: Device for DINOv2 inference.
        pca_model: Pre-trained PCA model. If None, trains on negatives.
        pca_meta: PCA metadata dict (optional, for reuse across calls).

    Returns:
        Updated manifest with overlay paths.
    """
    manifest = load_manifest(output_dir)
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Train PCA once (not per image!)
    if pca_model is None:
        print("  Training PCA on negatives (one-time)...")
        train_result = train_pca_on_negatives(
            str(DEFAULT_NEGATIVE_DIR),
            n_components=DEFAULT_N_COMPONENTS,
            device=device,
            sample_frac=0.15,
        )
        if "error" in train_result:
            print(f"  ERROR: PCA training failed: {train_result['error']}")
            return manifest
        pca_model = train_result["pca_model"]
    else:
        print("  Using pre-trained PCA model")

    print(f"\n  Generating overlays for {len(manifest)} candidates...")
    updated_manifest = []

    for idx, entry in enumerate(manifest):
        thumb_path = output_dir / entry["thumbnail_path"]
        if not thumb_path.exists():
            updated_manifest.append(entry)
            continue

        if (idx + 1) % 50 == 0:
            print(f"  Overlay {idx + 1}/{len(manifest)}...")

        stem = thumb_path.stem
        img_overlay_dir = overlays_dir / stem
        img_overlay_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load image
            orig_img = Image.open(thumb_path).convert("RGB")

            # Extract patches and score
            patches = extract_patch_embeddings_single(str(thumb_path),
                                                       device=device)
            score_result = score_patches(patches, pca_model)

            # Save original (resized to 224x224)
            orig_224 = orig_img.resize((224, 224))
            orig_224.save(img_overlay_dir / "original.png")

            # Heatmap overlay
            heat_img = make_heatmap_overlay(orig_img, score_result["heatmap_224"])
            heat_img.save(img_overlay_dir / "heatmap.png")

            # Segmentation overlay
            seg_img = make_segmentation_overlay(orig_img, score_result["mask_224"])
            seg_img.save(img_overlay_dir / "segmentation.png")

            # Update entry with overlay paths
            entry["original_path"] = f"overlays/{stem}/original.png"
            entry["heatmap_path"] = f"overlays/{stem}/heatmap.png"
            entry["segmentation_path"] = f"overlays/{stem}/segmentation.png"

        except Exception as exc:
            print(f"  WARNING: Overlay failed for {stem}: {exc}")
            entry["original_path"] = entry["thumbnail_path"]
            entry["heatmap_path"] = entry["thumbnail_path"]
            entry["segmentation_path"] = entry["thumbnail_path"]

        updated_manifest.append(entry)

    # Re-save manifest with overlay paths
    (output_dir / "manifest.json").write_text(
        json.dumps(updated_manifest, indent=2), encoding="utf-8"
    )
    print(f"  Overlays generated for {len(updated_manifest)} candidates")
    return updated_manifest


# ---------------------------------------------------------------------------
# DFO event comparison
# ---------------------------------------------------------------------------

def load_dfo_events() -> list[dict[str, Any]]:
    """Load known DFO herring spawn events."""
    dfo_files = [
        Path("data/ingressed/dfo_events.json"),
        Path("data/ingressed/manifest.json"),
    ]
    for f in dfo_files:
        if f.exists():
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return data.get("features", data.get("events",
                                   data.get("candidates", [])))
            except (json.JSONDecodeError, OSError):
                pass
    return []


def load_known_events() -> list[dict[str, Any]]:
    """Load all known spawn events from various sources."""
    events: list[dict[str, Any]] = []

    # DFO events
    dfo = load_dfo_events()
    for e in dfo:
        props = e.get("properties", e)
        lat = props.get("Latitude", props.get("lat", props.get("latitude")))
        lon = props.get("Longitude", props.get("lon", props.get("longitude")))
        date_str = props.get("Start", props.get("date", props.get("Date", "")))
        name = props.get("LocationNa", props.get("location_name",
                       props.get("Location", props.get("name", "unknown"))))

        if lat is not None and lon is not None:
            events.append({
                "source": "dfo",
                "name": str(name),
                "lat": float(lat),
                "lon": float(lon),
                "date": str(date_str)[:10] if date_str else "",
            })

    # Positive training samples
    pos_dir = Path("data/samples/positive")
    if pos_dir.exists():
        for p in sorted(pos_dir.glob("*.png")):
            # Parse info from filename
            parts = p.stem.split("_")
            events.append({
                "source": "positive_training",
                "name": parts[0] if parts else "unknown",
                "lat": float(parts[-3]) if len(parts) >= 3 else 0,
                "lon": float(parts[-2]) if len(parts) >= 2 else 0,
                "date": parts[1] if len(parts) >= 2 else "",
                "filename": p.name,
            })

    return events


def compare_with_known_events(
    manifest: list[dict[str, Any]],
    known_events: list[dict[str, Any]],
    distance_km: float = 2.0,
) -> dict[str, Any]:
    """Compare scan candidates against known events.

    Args:
        manifest: List of candidate entries from scan.
        known_events: List of known event dicts with lat, lon, date, source.
        distance_km: Max distance for a match.

    Returns:
        dict with matches, near_matches, stats.
    """
    matches = []
    near_matches = []

    for candidate in manifest:
        c_lat, c_lon = candidate["lat"], candidate["lon"]

        for event in known_events:
            e_lat = event.get("lat", 0)
            e_lon = event.get("lon", 0)
            if e_lat == 0 and e_lon == 0:
                continue

            # Haversine distance (simplified)
            dlat = (c_lat - e_lat) * 111.0
            dlon = (c_lon - e_lon) * 111.0 * math.cos(math.radians((c_lat + e_lat) / 2))
            dist = math.sqrt(dlat ** 2 + dlon ** 2)

            if dist <= distance_km:
                matches.append({
                    "candidate_filename": candidate.get("thumbnail_path", ""),
                    "candidate_score": candidate.get("score_top10p", 0),
                    "candidate_region": candidate.get("region", ""),
                    "candidate_date": candidate.get("date", ""),
                    "event_source": event.get("source", ""),
                    "event_name": event.get("name", ""),
                    "event_date": event.get("date", ""),
                    "distance_km": round(dist, 3),
                })
            elif dist <= distance_km * 2:
                near_matches.append({
                    "candidate_filename": candidate.get("thumbnail_path", ""),
                    "candidate_score": candidate.get("score_top10p", 0),
                    "candidate_region": candidate.get("region", ""),
                    "event_source": event.get("source", ""),
                    "event_name": event.get("name", ""),
                    "distance_km": round(dist, 3),
                })

    return {
        "distance_threshold_km": distance_km,
        "n_matches": len(matches),
        "n_near_matches": len(near_matches),
        "matches": matches,
        "near_matches": near_matches,
    }


# ---------------------------------------------------------------------------
# Multi-year consistency
# ---------------------------------------------------------------------------

def find_multi_year_locations(
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find locations that appear in multiple years.

    Groups by (rounded lat, rounded lon) and finds years with matches.
    """
    location_years: dict[tuple, dict] = defaultdict(lambda: {
        "years": set(),
        "entries": [],
        "lat": 0,
        "lon": 0,
    })

    for entry in manifest:
        # Round to ~100m for grouping
        key = (round(entry["lat"], 3), round(entry["lon"], 3))
        location_years[key]["years"].add(entry.get("year", 0))
        location_years[key]["entries"].append(entry)
        location_years[key]["lat"] = entry["lat"]
        location_years[key]["lon"] = entry["lon"]

    multi_year = []
    for key, data in location_years.items():
        if len(data["years"]) >= 2:
            multi_year.append({
                "lat": data["lat"],
                "lon": data["lon"],
                "years": sorted(data["years"]),
                "n_years": len(data["years"]),
                "n_candidates": len(data["entries"]),
                "max_score": max(e.get("score_top10p", 0) for e in data["entries"]),
                "entries": data["entries"],
            })

    return sorted(multi_year, key=lambda x: -x["n_years"])


# ---------------------------------------------------------------------------
# Review page generation
# ---------------------------------------------------------------------------

def generate_review_page(
    manifest: list[dict[str, Any]],
    output_dir: Path,
    train_meta: dict | None = None,
    dfo_comparison: dict | None = None,
    multi_year_locations: list | None = None,
    top_n: int = DEFAULT_TOP_CANDIDATES,
) -> str:
    """Generate standalone review HTML page.

    Returns path to the HTML file.
    """
    html_path = output_dir / "review.html"

    # Take top N candidates sorted by score
    sorted_manifest = sorted(manifest, key=lambda e: e.get("score_top10p", 0), reverse=True)
    display_manifest = sorted_manifest[:top_n]
    total_candidates = len(manifest)

    # Stats
    high_area = sum(1 for e in display_manifest if e.get("spawn_area_frac", 0) > 0.05)
    max_score = max((e.get("score_top10p", 0) for e in display_manifest), default=0)

    # Region distribution
    region_counts = Counter(e.get("region", "unknown") for e in display_manifest)

    # Training info
    if train_meta:
        train_info = (
            f"PCA: {train_meta.get('n_components', '?')} components &middot; "
            f"Trained on {train_meta.get('n_images_used', '?')} negative images &middot; "
            f"{train_meta.get('n_patches_trained', '?')} patches &middot; "
            f"Var: {train_meta.get('explained_variance_ratio', 0) * 100:.1f}% &middot; "
            f"95th pctl: {train_meta.get('neg_score_p95', 0):.6f}"
        )
    else:
        train_info = "Training info unavailable"

    # Build candidates JSON with overlay paths
    candidates_json = json.dumps(display_manifest)

    # DFO comparison summary
    dfo_html = ""
    if dfo_comparison:
        match_rows = "".join(
            f"<tr><td>{html_escape(m['candidate_region'])}</td>"
            f"<td>{m['candidate_date']}</td>"
            f"<td>{m['event_name']}</td>"
            f"<td>{m['event_source']}</td>"
            f"<td>{m['distance_km']}km</td>"
            f"<td>{m['candidate_score']:.6f}</td></tr>"
            for m in dfo_comparison.get("matches", [])[:20]
        )
        dfo_html = f"""
        <div class="section">
            <h2>DFO Event Comparison</h2>
            <p>{dfo_comparison['n_matches']} direct matches, {dfo_comparison['n_near_matches']} near-matches</p>
            <table><thead><tr><th>Region</th><th>Date</th><th>Event</th><th>Source</th><th>Dist</th><th>Score</th></tr></thead>
            <tbody>{match_rows}</tbody></table>
        </div>
        """

    # Multi-year consistency
    my_html = ""
    if multi_year_locations:
        my_rows = "".join(
            f"<tr><td>{loc['lat']:.4f}, {loc['lon']:.4f}</td>"
            f"<td>{', '.join(str(y) for y in loc['years'])}</td>"
            f"<td>{loc['n_candidates']}</td>"
            f"<td>{loc['max_score']:.6f}</td></tr>"
            for loc in multi_year_locations[:20]
        )
        my_html = f"""
        <div class="section">
            <h2>Multi-Year Consistent Locations</h2>
            <p>{len(multi_year_locations)} locations found in 2+ years</p>
            <table><thead><tr><th>Location</th><th>Years</th><th>Candidates</th><th>Max Score</th></tr></thead>
            <tbody>{my_rows}</tbody></table>
        </div>
        """

    # Region distribution rows
    region_rows = "".join(
        f"<tr><td>{html_escape(r)}</td><td>{c}</td></tr>"
        for r, c in sorted(region_counts.items(), key=lambda x: -x[1])
    )

    # Per-candidate region faceting
    region_facets = "".join(
        f'<option value="{html_escape(r)}">{html_escape(r)} ({c})</option>'
        for r, c in sorted(region_counts.items(), key=lambda x: -x[1])
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>13-Region SubspaceAD Review — Herring Spawn Candidates</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #0d0d1a; color: #ddd; }}
a {{ color: #64B5F6; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* Header */
.header {{ background: #1a1a2e; padding: 12px 24px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #2a2a4e; flex-wrap: wrap; }}
.header h1 {{ font-size: 16px; color: #fff; margin: 0; }}
.header .subtitle {{ font-size: 12px; color: #888; }}
.header .stats {{ margin-left: auto; display: flex; gap: 16px; font-size: 12px; flex-wrap: wrap; }}
.header .stats span {{ color: #888; }}
.header .stats strong {{ color: #fff; }}

/* Controls */
.controls {{ background: #12121e; padding: 10px 24px; border-bottom: 1px solid #2a2a4e; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; font-size: 13px; }}
.controls label {{ color: #888; }}
.controls select, .controls input {{ background: #0a0a14; border: 1px solid #2a2a4e; color: #ddd; padding: 4px 8px; border-radius: 4px; font-size: 13px; }}
.controls .count {{ color: #888; margin-left: auto; }}

/* Sections */
.section {{ margin: 12px 24px; padding: 12px 16px; background: #12121e; border: 1px solid #2a2a4e; border-radius: 8px; }}
.section h2 {{ font-size: 14px; color: #fff; margin: 0 0 8px; }}
.section p {{ font-size: 12px; color: #888; margin: 0 0 8px; }}
.section table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
.section th, .section td {{ padding: 4px 8px; border-bottom: 1px solid #1a1a2e; text-align: left; }}
.section th {{ color: #888; font-weight: 600; }}

/* Train info */
.train-info {{ font-size: 11px; color: #555; background: #0a0a14; padding: 6px 12px; border-radius: 4px; margin: 0 24px; }}

/* Grid */
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; padding: 16px 24px; }}
.card {{ background: #12121e; border: 1px solid #2a2a4e; border-radius: 8px; overflow: hidden; transition: border-color 0.2s; cursor: pointer; }}
.card:hover {{ border-color: #64B5F6; }}
.card.highlight {{ border-color: #4CAF50; box-shadow: 0 0 12px rgba(76,175,80,0.3); }}

.card-header {{ padding: 8px 12px; background: #0a0a14; border-bottom: 1px solid #1a1a2e; display: flex; justify-content: space-between; align-items: center; }}
.card-header .filename {{ font-size: 10px; color: #666; font-family: monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 60%; }}
.card-header .year-badge {{ font-size: 9px; padding: 2px 6px; border-radius: 3px; background: #2a2a4e; color: #888; }}

.card-images {{ display: flex; gap: 0; }}
.card-images .img-wrap {{ flex: 1; position: relative; }}
.card-images img {{ width: 100%; height: auto; display: block; aspect-ratio: 1; object-fit: cover; }}
.card-images .img-label {{ position: absolute; bottom: 0; left: 0; right: 0; text-align: center; font-size: 9px; padding: 2px; background: rgba(0,0,0,0.75); color: #888; text-transform: uppercase; letter-spacing: 0.3px; }}

.card-body {{ padding: 8px 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 11px; }}
.card-body .metric {{ display: flex; justify-content: space-between; padding: 2px 4px; background: #0a0a14; border-radius: 3px; }}
.card-body .metric .label {{ color: #666; }}
.card-body .metric .value {{ color: #ddd; font-weight: 600; font-variant-numeric: tabular-nums; }}
.card-body .metric .value.good {{ color: #4CAF50; }}
.card-body .metric .value.warn {{ color: #FFC107; }}
.card-body .metric .value.bad {{ color: #f44336; }}

.score-bar {{ height: 4px; background: #0a0a14; border-radius: 2px; margin: 0 12px 8px; overflow: hidden; }}
.score-bar .fill {{ height: 100%; border-radius: 2px; transition: width 0.3s; }}

/* Modal */
.modal-overlay {{ display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.8); z-index: 1000; justify-content: center; align-items: center; }}
.modal-overlay.active {{ display: flex; }}
.modal {{ background: #1a1a2e; border: 1px solid #2a2a4e; border-radius: 12px; max-width: 800px; width: 90%; max-height: 90vh; overflow-y: auto; padding: 24px; }}
.modal h2 {{ font-size: 14px; color: #fff; margin-bottom: 12px; word-break: break-all; }}
.modal-images {{ display: flex; gap: 8px; margin-bottom: 16px; }}
.modal-images img {{ width: 33%; aspect-ratio: 1; object-fit: cover; border-radius: 6px; }}
.modal-metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 16px; }}
.modal-metrics .metric {{ background: #0a0a14; border-radius: 6px; padding: 10px; text-align: center; }}
.modal-metrics .metric .label {{ font-size: 10px; color: #666; text-transform: uppercase; }}
.modal-metrics .metric .value {{ font-size: 20px; font-weight: 700; color: #fff; margin-top: 4px; }}
.modal-close {{ float: right; background: #2a2a4e; border: none; color: #ddd; padding: 6px 16px; border-radius: 4px; cursor: pointer; font-size: 13px; }}
.modal-close:hover {{ background: #3a3a5e; }}

/* Empty state */
.empty {{ text-align: center; padding: 60px 24px; color: #555; }}
.empty h2 {{ font-size: 20px; color: #888; margin-bottom: 8px; }}

/* Responsive */
@media (max-width: 700px) {{
    .grid {{ grid-template-columns: 1fr; padding: 12px; }}
    .header {{ padding: 10px 12px; }}
    .controls {{ padding: 8px 12px; }}
    .modal-images {{ flex-direction: column; }}
    .modal-images img {{ width: 100%; }}
    .modal-metrics {{ grid-template-columns: 1fr 1fr; }}
}}
</style>
</head>
<body>

<div class="header">
    <h1>&#9733; 13-Region SubspaceAD Review</h1>
    <span class="subtitle">DINOv2 patch-level anomaly detection &middot; {top_n} of {total_candidates} total</span>
    <div class="stats">
        <span>Displayed: <strong id="statTotal">{top_n}</strong></span>
        <span>High area: <strong id="statHigh" style="color:#4CAF50">{high_area}</strong></span>
        <span>Max score: <strong id="statMaxScore">{max_score:.6f}</strong></span>
    </div>
</div>

<div class="train-info">{train_info}</div>

<div class="controls">
    <label>Sort:</label>
    <select id="sortSelect" onchange="applyFilters()">
        <option value="score">Score descending</option>
        <option value="area">Spawn area descending</option>
        <option value="mean">Mean residual descending</option>
        <option value="year">Year</option>
        <option value="filename">Filename A-Z</option>
    </select>
    <label>Region:</label>
    <select id="regionSelect" onchange="applyFilters()">
        <option value="">All regions</option>
        {region_facets}
    </select>
    <label>Year:</label>
    <select id="yearSelect" onchange="applyFilters()">
        <option value="">All years</option>
        <option value="2023">2023</option>
        <option value="2024">2024</option>
        <option value="2025">2025</option>
    </select>
    <label>Min area:</label>
    <input type="range" id="minArea" min="0" max="100" value="0" oninput="applyFilters()" style="width:100px;">
    <span id="minAreaLabel" style="color:#888;font-size:12px;">0%</span>
    <label>Search:</label>
    <input type="text" id="searchInput" placeholder="Filter filename..." oninput="applyFilters()" style="width:160px;">
    <span class="count" id="filterCount">Showing {top_n}/{top_n}</span>
</div>

{dfo_html}
{my_html}

<div class="section" id="regionDistSection">
    <h2>Region Distribution</h2>
    <table><thead><tr><th>Region</th><th>Candidates (top {top_n})</th></tr></thead>
    <tbody>{region_rows}</tbody></table>
</div>

<div class="grid" id="candidateGrid"></div>
<div class="empty" id="emptyState" style="display:none;">
    <h2>No candidates match filter</h2>
    <p>Try adjusting the search or minimum area threshold.</p>
</div>

<!-- Modal -->
<div class="modal-overlay" id="modalOverlay" onclick="closeModal(event)">
    <div class="modal" id="modalContent">
        <button class="modal-close" onclick="closeModal()">Close</button>
        <h2 id="modalFilename"></h2>
        <div class="modal-images">
            <img id="modalOriginal" alt="Original">
            <img id="modalHeatmap" alt="Heatmap">
            <img id="modalSeg" alt="Segmentation">
        </div>
        <div class="modal-metrics" id="modalMetrics"></div>
    </div>
</div>

<script>
const allCandidates = {candidates_json};

function getScoreColor(score, maxScore) {{
    const ratio = maxScore > 0 ? score / maxScore : 0;
    if (ratio > 0.8) return '#4CAF50';
    if (ratio > 0.4) return '#FFC107';
    return '#f44336';
}}

function getAreaColor(frac) {{
    if (frac > 0.05) return '#4CAF50';
    if (frac > 0.01) return '#FFC107';
    return '#888';
}}

function render() {{
    const sortBy = document.getElementById('sortSelect').value;
    const regionFilter = document.getElementById('regionSelect').value;
    const yearFilter = document.getElementById('yearSelect').value;
    const minArea = parseFloat(document.getElementById('minArea').value) / 100;
    const search = document.getElementById('searchInput').value.toLowerCase();

    let filtered = allCandidates.filter(c => {{
        if (regionFilter && c.region !== regionFilter) return false;
        if (yearFilter && String(c.year) !== yearFilter) return false;
        if (c.spawn_area_frac < minArea) return false;
        if (search && !c.filename?.toLowerCase().includes(search) &&
            !(c.thumbnail_path?.toLowerCase().includes(search))) return false;
        return true;
    }});

    filtered.sort((a, b) => {{
        switch (sortBy) {{
            case 'score': return (b.score_top10p || 0) - (a.score_top10p || 0);
            case 'area': return (b.spawn_area_frac || 0) - (a.spawn_area_frac || 0);
            case 'mean': return (b.score_mean || 0) - (a.score_mean || 0);
            case 'year': return (b.year || 0) - (a.year || 0);
            case 'filename': return (a.thumbnail_path || '').localeCompare(b.thumbnail_path || '');
            default: return 0;
        }}
    }});

    const grid = document.getElementById('candidateGrid');
    const empty = document.getElementById('emptyState');
    document.getElementById('filterCount').textContent = `Showing ${{filtered.length}}/${{allCandidates.length}}`;

    if (filtered.length === 0) {{
        grid.innerHTML = '';
        empty.style.display = 'block';
        return;
    }}
    empty.style.display = 'none';

    const globalMax = Math.max(...allCandidates.map(c => c.score_top10p || 0));

    let html = '';
    for (const c of filtered) {{
        const scorePct = globalMax > 0 ? (c.score_top10p || 0) / globalMax * 100 : 0;
        const isHighlight = c.spawn_area_frac > 0.05;
        const fn = c.thumbnail_path || c.filename || '';
        const origPath = c.original_path || c.thumbnail_path || '';
        const heatPath = c.heatmap_path || c.thumbnail_path || '';
        const segPath = c.segmentation_path || c.thumbnail_path || '';

        html += `
        <div class="card ${{isHighlight ? 'highlight' : ''}}" onclick="openModal('${{fn.replace(/'/g, "\\\\'")}}')">
            <div class="card-header">
                <span class="filename" title="${{fn.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')}}">${{fn.split('/').pop()}}</span>
                <span class="year-badge">${{c.year || '?'}}</span>
            </div>
            <div class="card-images">
                <div class="img-wrap"><img src="${{origPath}}" alt="orig"><div class="img-label">RGB</div></div>
                <div class="img-wrap"><img src="${{heatPath}}" alt="heatmap"><div class="img-label">Heatmap</div></div>
                <div class="img-wrap"><img src="${{segPath}}" alt="seg"><div class="img-label">Seg</div></div>
            </div>
            <div class="score-bar"><div class="fill" style="width:${{scorePct.toFixed(1)}}%;background:${{getScoreColor(c.score_top10p || 0, globalMax)}}"></div></div>
            <div class="card-body">
                <div class="metric"><span class="label">Score</span><span class="value" style="color:${{getScoreColor(c.score_top10p || 0, globalMax)}}">${{(c.score_top10p || 0).toFixed(6)}}</span></div>
                <div class="metric"><span class="label">Area</span><span class="value" style="color:${{getAreaColor(c.spawn_area_frac)}}">${{((c.spawn_area_frac || 0) * 100).toFixed(1)}}%</span></div>
                <div class="metric"><span class="label">Region</span><span class="value">${{c.region || '?'}}</span></div>
                <div class="metric"><span class="label">Patches</span><span class="value">${{c.n_spawn_patches || 0}}/${{c.n_patches || 256}}</span></div>
            </div>
        </div>`; 
    }}
    grid.innerHTML = html;
}}

function applyFilters() {{
    const val = document.getElementById('minArea').value;
    document.getElementById('minAreaLabel').textContent = val + '%';
    render();
}}

function openModal(fn) {{
    const c = allCandidates.find(x => (x.thumbnail_path === fn || x.filename === fn));
    if (!c) return;

    document.getElementById('modalFilename').textContent = c.thumbnail_path || c.filename || '';
    document.getElementById('modalOriginal').src = c.original_path || c.thumbnail_path || '';
    document.getElementById('modalHeatmap').src = c.heatmap_path || c.thumbnail_path || '';
    document.getElementById('modalSeg').src = c.segmentation_path || c.thumbnail_path || '';

    const globalMax = Math.max(...allCandidates.map(x => x.score_top10p || 0));

    document.getElementById('modalMetrics').innerHTML = `
        <div class="metric"><div class="label">Score (top-10%)</div><div class="value" style="color:${{getScoreColor(c.score_top10p || 0, globalMax)}}">${{(c.score_top10p || 0).toFixed(6)}}</div></div>
        <div class="metric"><div class="label">Mean Residual</div><div class="value">${{(c.score_mean || 0).toFixed(6)}}</div></div>
        <div class="metric"><div class="label">Max Residual</div><div class="value">${{(c.score_max || 0).toFixed(6)}}</div></div>
        <div class="metric"><div class="label">Spawn Area</div><div class="value" style="color:${{getAreaColor(c.spawn_area_frac)}}">${{((c.spawn_area_frac || 0) * 100).toFixed(2)}}%</div></div>
        <div class="metric"><div class="label">Anomalous Patches</div><div class="value">${{c.n_spawn_patches || 0}} / ${{c.n_patches || 256}}</div></div>
        <div class="metric"><div class="label">Year</div><div class="value">${{c.year || '?'}}</div></div>
    `;

    document.getElementById('modalOverlay').classList.add('active');
    document.body.style.overflow = 'hidden';
}}

function closeModal(e) {{
    if (e && e.target !== e.currentTarget) return;
    document.getElementById('modalOverlay').classList.remove('active');
    document.body.style.overflow = '';
}}

document.addEventListener('keydown', (e) => {{
    if (e.key === 'Escape') closeModal();
}});

document.addEventListener('DOMContentLoaded', render);
</script>

</body>
</html>"""

    html_path.write_text(html)
    print(f"\n  Review page generated: {html_path}")
    return str(html_path)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format_, *args):
        pass


def start_server(directory: str, port: int) -> HTTPServer:
    os.chdir(directory)
    server = HTTPServer(("0.0.0.0", port), QuietHTTPRequestHandler)
    return server


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary(
    manifest: list[dict[str, Any]],
    points: list[dict[str, Any]],
    train_meta: dict,
    dfo_comparison: dict | None,
    multi_year: list | None,
    elapsed: float,
) -> dict:
    """Generate scan summary."""
    sorted_manifest = sorted(manifest, key=lambda e: e.get("score_top10p", 0), reverse=True)
    top20 = sorted_manifest[:20]

    region_counts = Counter(e.get("region", "unknown") for e in manifest)
    year_counts = Counter(e.get("year", 0) for e in manifest)

    return {
        "scan_parameters": {
            "n_regions": len(REGIONS),
            "n_grid_points": len(points),
            "years": SCAN_YEARS,
            "grid_spacing_deg": DEFAULT_GRID_SPACING,
            "max_cloud_pct": DEFAULT_MAX_CLOUD,
            "min_spawn_area_pct": MIN_SPAWN_AREA_FRAC * 100,
            "min_anomalous_patches": MIN_ANOMALOUS_PATCHES,
        },
        "training": {
            "n_negative_images": train_meta.get("n_images_used", 0),
            "n_pca_components": train_meta.get("n_components", 0),
            "explained_variance_ratio": train_meta.get("explained_variance_ratio", 0),
            "neg_score_p95": train_meta.get("neg_score_p95", 0),
        },
        "results": {
            "total_candidates": len(manifest),
            "top_n_presented": min(DEFAULT_TOP_CANDIDATES, len(manifest)),
            "regions_found": len(region_counts),
            "region_breakdown": dict(region_counts.most_common()),
            "year_breakdown": dict(sorted(year_counts.items())),
            "elapsed_seconds": round(elapsed, 1),
        },
        "dfo_comparison": {
            "direct_matches": dfo_comparison["n_matches"] if dfo_comparison else 0,
            "near_matches": dfo_comparison["n_near_matches"] if dfo_comparison else 0,
        } if dfo_comparison else None,
        "multi_year_locations": len(multi_year) if multi_year else 0,
        "top_20_candidates": [
            {
                "rank": i + 1,
                "region": c.get("region", ""),
                "year": c.get("year", ""),
                "date": c.get("date", ""),
                "lat": c.get("lat", 0),
                "lon": c.get("lon", 0),
                "score": c.get("score_top10p", 0),
                "spawn_area_pct": round(c.get("spawn_area_frac", 0) * 100, 1),
                "n_spawn_patches": c.get("n_spawn_patches", 0),
                "thumbnail": c.get("thumbnail_path", ""),
            }
            for i, c in enumerate(top20)
        ],
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> int:
    """Execute the full scan pipeline."""
    t0 = time.time()
    output_dir = Path(args.output)
    max_candidates = args.max_candidates
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "thumbnails").mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    print(f"  Device: {device}")
    print()

    # ==================================================================
    # Step 1: Generate grid points
    # ==================================================================
    print("=" * 60)
    print("  Step 1: Generating Grid Points")
    print("=" * 60)
    points = generate_grid_points(REGIONS, args.grid_spacing)
    print(f"  Total grid points: {len(points)} across {len(REGIONS)} regions")

    # Region breakdown
    region_counts = Counter(p["region"] for p in points)
    for region, count in region_counts.most_common():
        print(f"    {region}: {count} points")

    # Estimated GEE calls
    est_calls = len(points) * len(SCAN_YEARS)
    est_downloads = est_calls  # Rough estimate (most points have 1 scene)
    est_time_min = est_downloads / (args.workers * 60) * 1.5  # ~1.5s per download
    print(f"\n  Estimated GEE searches: {est_calls}")
    print(f"  Estimated downloads: ~{est_downloads}")
    print(f"  Estimated time: ~{est_time_min:.0f} min with {args.workers} workers")
    print()

    if args.dry_run:
        print("  Dry run complete. Remove --dry-run to execute.")
        return 0

    # ==================================================================
    # Step 2: Train PCA on negatives
    # ==================================================================
    print("=" * 60)
    print("  Step 2: Training PCA Subspace on Negatives")
    print("=" * 60)
    train_meta = train_pca_on_negatives(
        args.negative_dir,
        n_components=args.n_components,
        device=device,
        sample_frac=args.sample_frac,
    )

    if "error" in train_meta:
        print(f"ERROR: PCA training failed: {train_meta['error']}")
        return 1

    pca_model = train_meta["pca_model"]
    p95_threshold = train_meta["neg_score_p95"]
    print(f"  Filter threshold (95th percentile): {p95_threshold:.6f}")
    print()

    # ==================================================================
    # Step 3: Initialize GEE
    # ==================================================================
    print("=" * 60)
    print("  Step 3: Initializing Google Earth Engine")
    print("=" * 60)
    try:
        import ee
        ee.Initialize(project=args.project)
        print("  GEE initialized")
    except Exception as exc:
        print(f"ERROR: GEE initialization failed: {exc}")
        print("  Ensure you are authenticated: earthengine authenticate")
        return 1
    print()

    # ==================================================================
    # Step 4: Scan all years
    # ==================================================================
    print("=" * 60)
    print("  Step 4: Scanning All Regions Across All Years")
    print("=" * 60)

    total_scanned = 0
    total_candidates = 0
    total_no_scene = 0
    total_errors = 0
    total_filtered = 0

    for year_idx, year in enumerate(SCAN_YEARS):
        print(f"\n  --- Year {year} ({year_idx + 1}/{len(SCAN_YEARS)}) ---")
        year_start = time.time()

        stats: dict[str, int] = Counter()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _process_point_year,
                    point, year, ee, pca_model, p95_threshold,
                    output_dir, device, idx, len(points), t0,
                ): idx
                for idx, point in enumerate(points)
            }

            for future in as_completed(futures):
                result = future.result()
                stats["processed"] += result.get("processed", 0)
                stats["candidates"] += result.get("candidate_saved", 0)
                stats["no_scene"] += result.get("no_scene", 0)
                stats["errors"] += result.get("download_error", 0) + result.get("scoring_error", 0)
                stats["filtered"] += result.get("filtered_out", 0)

        # Check if we've hit candidate limit
        manifest = load_manifest(output_dir)
        if len(manifest) >= DEFAULT_MAX_CANDIDATES:
            print(f"\n  Reached candidate limit ({DEFAULT_MAX_CANDIDATES}). Stopping scan.")
            break

        year_elapsed = time.time() - year_start
        total_scanned += stats["processed"]
        total_candidates = len(manifest)
        total_no_scene += stats["no_scene"]
        total_errors += stats["errors"]
        total_filtered += stats["filtered"]

        print(f"\n  --- Year {year} Summary ---")
        print(f"    Processed: {stats['processed']}")
        print(f"    Candidates so far: {total_candidates}")
        print(f"    No scene: {stats['no_scene']}")
        print(f"    Filtered out: {stats['filtered']}")
        print(f"    Errors: {stats['errors']}")
        print(f"    Time: {year_elapsed:.0f}s")

    print(f"\n  Scan complete.")
    print(f"  Total candidates: {total_candidates}")

    manifest = load_manifest(output_dir)

    # ==================================================================
    # Step 5: Generate Overlays
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Step 5: Generating Overlays")
    print("=" * 60)
    manifest = generate_overlays(output_dir, device, pca_model=pca_model)
    print()

    # ==================================================================
    # Step 6: DFO Event Comparison
    # ==================================================================
    print("=" * 60)
    print("  Step 6: Comparing with Known Events")
    print("=" * 60)
    known_events = load_known_events()
    print(f"  Loaded {len(known_events)} known events")
    dfo_comparison = compare_with_known_events(manifest, known_events)
    print(f"  Direct matches: {dfo_comparison['n_matches']}")
    print(f"  Near matches: {dfo_comparison['n_near_matches']}")
    (output_dir / "dfo_comparison.json").write_text(
        json.dumps(dfo_comparison, indent=2)
    )
    print()

    # ==================================================================
    # Step 7: Multi-Year Consistency
    # ==================================================================
    print("=" * 60)
    print("  Step 7: Finding Multi-Year Consistent Locations")
    print("=" * 60)
    multi_year = find_multi_year_locations(manifest)
    print(f"  Found {len(multi_year)} locations with candidates in 2+ years")
    if multi_year:
        for loc in multi_year[:5]:
            print(f"    ({loc['lat']:.4f}, {loc['lon']:.4f}) → years {loc['years']}")
    (output_dir / "multi_year_locations.json").write_text(
        json.dumps(multi_year, indent=2, default=str)
    )
    print()

    # ==================================================================
    # Step 8: Generate Review Page
    # ==================================================================
    print("=" * 60)
    print("  Step 8: Generating Review Page")
    print("=" * 60)
    generate_review_page(
        manifest, output_dir,
        train_meta=train_meta,
        dfo_comparison=dfo_comparison,
        multi_year_locations=multi_year,
        top_n=args.top_candidates,
    )
    print()

    # ==================================================================
    # Step 9: Generate Summary
    # ==================================================================
    print("=" * 60)
    print("  Step 9: Generating Summary")
    print("=" * 60)
    elapsed = time.time() - t0
    summary = generate_summary(
        manifest, points, train_meta,
        dfo_comparison, multi_year, elapsed,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Summary saved to {output_dir / 'summary.json'}")

    # Print top 20
    print(f"\n  Top 20 Candidates:")
    print(f"  {'Rank':<5} {'Region':<22} {'Year':<5} {'Score':<12} {'Area':<8} {'Patches':<8} {'Date':<12}")
    print(f"  {'-'*5} {'-'*22} {'-'*5} {'-'*12} {'-'*8} {'-'*8} {'-'*12}")
    for c in summary["top_20_candidates"]:
        print(f"  {c['rank']:<5} {c['region']:<22} {c['year']:<5} {c['score']:<12.6f} {c['spawn_area_pct']:<7.1f}% {c['n_spawn_patches']:<3}/{256:<3} {c['date']:<12}")

    # Score distribution
    print(f"\n  Total candidates: {len(manifest)}")
    print(f"  Regions found: {len(summary['results']['region_breakdown'])}")
    print(f"  Year breakdown: {summary['results']['year_breakdown']}")
    print(f"  DFO direct matches: {summary['dfo_comparison']['direct_matches']}")
    print(f"  Multi-year locations: {summary['multi_year_locations']}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="13-region BC herring spawn scan using DINOv2 SubspaceAD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--negative-dir", type=Path, default=DEFAULT_NEGATIVE_DIR)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--sample-frac", type=float, default=0.15)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--top-candidates", type=int, default=DEFAULT_TOP_CANDIDATES)
    parser.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dry-run", action="store_true", help="Count points and estimate cost, don't scan")
    parser.add_argument("--serve-only", action="store_true", help="Serve existing review page")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args(argv)
    output_dir = Path(args.output)

    # ==================================================================
    # Serve-only mode
    # ==================================================================
    if args.serve_only:
        if not (output_dir / "review.html").exists():
            print(f"ERROR: No review page found in {output_dir}")
            return 1
        print(f"\n  Serving existing review from {output_dir}")
        print(f"  URL: http://localhost:{args.port}/review.html")
        print(f"  Press Ctrl+C to stop.\n")

        server = start_server(str(output_dir), args.port)
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{args.port}/review.html")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")
        return 0

    # ==================================================================
    # Run pipeline
    # ==================================================================
    exit_code = run_pipeline(args)

    if exit_code == 0:
        # Start server
        print("\n" + "=" * 60)
        print("  Starting Review Server")
        print("=" * 60)
        server = start_server(str(output_dir), args.port)
        print(f"  Serving from {output_dir}")
        print(f"  URL: http://localhost:{args.port}/review.html")
        print(f"  Press Ctrl+C to stop.\n")

        if not args.no_browser:
            webbrowser.open(f"http://localhost:{args.port}/review.html")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
