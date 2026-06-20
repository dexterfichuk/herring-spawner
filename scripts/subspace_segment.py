#!/usr/bin/env python3
"""DINOv2 patch-level SubspaceAD segmentation for herring spawn localisation.

Uses DINOv2 patch tokens to localise spawn regions within an image.

Approach:
  1. Train PCA on patch tokens from negatives (same as patch_subspace_ad.py).
  2. For a candidate image, compute patch-level PCA reconstruction residuals.
  3. Reshape 256 residuals → 16×16 heatmap.
  4. Upsample to 224×224 using bilinear interpolation.
  5. Threshold to create binary spawn mask.
  6. Save overlays: heatmap blended with original, and green contours.

Usage:
    # Train on negatives, segment candidates
    python scripts/subspace_segment.py \\
        --train-dir data/samples/negative \\
        --image-dir data/candidates_knn \\
        --output-dir data/segmented \\
        --save-overlays \\
        --output-json data/segmented/results.json \\
        --device cpu

    # Segment with explicit PCA components and threshold
    python scripts/subspace_segment.py \\
        --train-dir data/samples/negative \\
        --image-dir data/samples/unified \\
        --output-dir data/segmented \\
        --n-components 64 \\
        --threshold 0.0005 \\
        --save-overlays

    # Segment a single image
    python scripts/subspace_segment.py \\
        --train-dir data/samples/negative \\
        --image-dir data/samples/unified \\
        --output-dir data/segmented \\
        --save-overlays \\
        --image data/samples/unified/qualicum_2024-03-18_score0.01_49.254865_-124.497442_20240318.png

Dependencies: torch, torchvision, Pillow, scikit-learn, numpy, tqdm, opencv-python
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
from tqdm import tqdm

from scripts.train_classifier import DINO_TRANSFORM, MODEL_NAME, EMBED_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DINOv2 ViT-S/14 produces a 16x16 grid of patch tokens
N_PATCHES = 256  # 16 * 16
PATCH_GRID_SIZE = 16  # 16x16

# PCA variance ratio target when auto-selecting n_components
AUTO_VARIANCE_TARGET = 0.90

# Default number of PCA components for patch tokens
DEFAULT_N_COMPONENTS = 64

# Fraction of anomalous patches to consider when computing top-k score
ANOMALOUS_PATCH_FRAC = 0.10  # top 10%

# Cache path for extracted patch embeddings
PATCH_EMBEDDINGS_CACHE_PATH = "data/embeddings/patch_subspace_embeddings.npz"


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
    print(f"  Model loaded: {MODEL_NAME} ({EMBED_DIM}-dim embeddings, "
          f"{N_PATCHES} patch tokens)")
    return model


# ---------------------------------------------------------------------------
# Patch embedding extraction
# ---------------------------------------------------------------------------

def extract_patch_embeddings(
    image_dir: str, device: str = "auto",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract DINOv2 patch tokens for all PNGs in a directory.

    Args:
        image_dir: Directory containing PNG images.
        device: 'auto', 'cuda', or 'cpu'.

    Returns:
        (all_patches, filenames_repeated, filenames_unique) where:
        - all_patches: np.ndarray of shape (N*256, 384), dtype float32.
        - filenames_repeated: list of str, each filename repeated 256 times.
        - filenames_unique: list of str, one per successfully embedded image.
    """
    resolved = _resolve_device(device)
    img_dir = Path(image_dir)

    if not img_dir.is_dir():
        print(f"  ERROR: Not a directory: {img_dir}")
        return np.array([]), [], []

    pngs = sorted(img_dir.glob("*.png"))
    if not pngs:
        print(f"  No PNG images found in {image_dir}")
        return np.array([]), [], []

    # Check cache
    cache_path = Path(PATCH_EMBEDDINGS_CACHE_PATH)
    if cache_path.exists():
        print(f"  Loading cached patch embeddings from {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        all_patches = loaded["all_patches"]
        cached_fnames_unique = loaded["filenames_unique"].tolist()
        requested_basenames = {p.name for p in pngs}
        cached_basenames = set(cached_fnames_unique)
        if requested_basenames == cached_basenames:
            cached_fnames_repeated = loaded["filenames_repeated"].tolist()
            print(f"  Loaded {len(cached_fnames_unique)} cached images "
                  f"({len(all_patches)} patch tokens)")
            return all_patches, cached_fnames_repeated, cached_fnames_unique
        else:
            print(f"  Cache mismatch — re-extracting patch embeddings")

    print(f"  Extracting DINOv2 patch embeddings from: {image_dir}")
    model = _load_dinov2_model(resolved)

    all_patches_list: list[np.ndarray] = []
    filenames_repeated: list[str] = []
    filenames_unique: list[str] = []

    for p in tqdm(pngs, desc="Patch embedding", unit="img"):
        try:
            img = Image.open(p).convert("RGB")
            tensor = DINO_TRANSFORM(img).unsqueeze(0).to(resolved)

            with torch.no_grad():
                patch_tokens, cls_tokens = model.get_intermediate_layers(
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

            all_patches_list.append(pt)
            filenames_repeated.extend([p.name] * N_PATCHES)
            filenames_unique.append(p.name)

        except Exception as exc:
            print(f"  WARNING: Failed to embed {p.name}: {exc}")

    if not all_patches_list:
        print("  No patch embeddings extracted successfully.")
        return np.array([]), [], []

    all_patches_arr = np.vstack(all_patches_list).astype(np.float32)
    print(f"  Extracted {len(filenames_unique)} images, "
          f"{len(all_patches_arr)} patch tokens")

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        all_patches=all_patches_arr,
        filenames_repeated=np.array(filenames_repeated, dtype=object),
        filenames_unique=np.array(filenames_unique, dtype=object),
    )
    print(f"  Saved patch embeddings cache to {cache_path}")

    return all_patches_arr, filenames_repeated, filenames_unique


# ---------------------------------------------------------------------------
# PCA training on patch tokens
# ---------------------------------------------------------------------------

def _auto_select_n_components(
    n_samples: int, n_features: int, variance_target: float = AUTO_VARIANCE_TARGET,
) -> int:
    """Automatically select n_components for PCA on patch tokens.

    Args:
        n_samples: Number of training patch tokens.
        n_features: Number of feature dimensions (384 for DINOv2 ViT-S/14).
        variance_target: Ignored — kept for API compatibility.

    Returns:
        int: Number of PCA components to use.
    """
    max_possible = min(n_samples - 1, n_features)
    if max_possible < 1:
        return 1
    suggested = min(DEFAULT_N_COMPONENTS, n_samples // 2, max_possible)
    return max(1, suggested)


def train_patch_pca(
    negative_dir: str, n_components: int = 64, device: str = "auto",
    sample_frac: float = 0.1,
) -> dict:
    """Train PCA on DINOv2 patch tokens from negatives.

    Extracts all patch tokens from negative images, optionally samples a
    fraction of patches per image, then fits PCA.

    Args:
        negative_dir: Directory of negative (non-spawn) PNG images.
        n_components: Number of PCA components (default 64).
        device: 'auto', 'cuda', or 'cpu'.
        sample_frac: Fraction of patches to sample per image for training
            (0.0 to 1.0). Default 0.1 yields ~26 patches per image.

    Returns:
        dict with keys:
        - 'pca_model': fitted PCA object (sklearn.decomposition.PCA)
        - 'mean': ndarray of mean patch embedding
        - 'components': ndarray of PCA components
        - 'explained_variance': ndarray of explained variance per component
        - 'explained_variance_ratio': ndarray of explained variance ratio
        - 'n_components': int
        - 'n_patches_trained': int
        - 'n_images_used': int
    """
    resolved = _resolve_device(device)

    print("=" * 60)
    print("  Subspace Segmentation — Train Patch PCA")
    print("=" * 60)
    print(f"  Negative directory: {negative_dir}")
    print(f"  Device: {resolved}")
    print(f"  Sample fraction: {sample_frac}")
    print(f"  PCA components: {n_components}")

    all_patches, filenames_repeated, filenames_unique = extract_patch_embeddings(
        negative_dir, device=resolved,
    )

    if len(all_patches) == 0:
        print("ERROR: No patch embeddings extracted.")
        return {"error": "No patch embeddings extracted"}

    n_images = len(filenames_unique)
    n_total_patches = len(all_patches)
    print(f"  Total patch tokens available: {n_total_patches} "
          f"from {n_images} images")

    # Sample patches per image
    if sample_frac < 1.0:
        sampled_patches: list[np.ndarray] = []
        rng = np.random.RandomState(42)
        for img_idx in range(n_images):
            start = img_idx * N_PATCHES
            end = start + N_PATCHES
            img_patches = all_patches[start:end]
            n_sample = max(1, int(N_PATCHES * sample_frac))
            indices = rng.choice(N_PATCHES, size=n_sample, replace=False)
            sampled_patches.append(img_patches[indices])
        training_patches = np.vstack(sampled_patches).astype(np.float32)
        print(f"  Sampled {len(training_patches)} patch tokens "
              f"({sample_frac:.1%} per image)")
    else:
        training_patches = all_patches
        print(f"  Using all {n_total_patches} patch tokens for training")

    # Fit PCA
    n_patches_tot, n_features = training_patches.shape

    # Clamp n_components
    max_components = min(n_patches_tot - 1, n_features)
    if n_components > max_components:
        n_components = max(1, max_components)
        print(f"  Clamped n_components to {n_components} "
              f"(limited by data dimensions)")

    print(f"  Fitting PCA with {n_components} components on "
          f"{n_patches_tot} patches...")
    pca = PCA(n_components=n_components, whiten=False, random_state=42)
    pca.fit(training_patches)

    total_var_ratio = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance ratio: {total_var_ratio:.4f} "
          f"(with {n_components} components)")

    print(f"\n  Training complete:")
    print(f"    PCA components:             {pca.n_components_}")
    print(f"    PCA explained variance:     {total_var_ratio:.4f}")
    print(f"    Negative training images:   {n_images}")
    print(f"    Training patch tokens:      {n_patches_tot}")

    return {
        "pca_model": pca,
        "mean": pca.mean_,
        "components": pca.components_,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "n_components": int(n_components),
        "n_patches_trained": n_patches_tot,
        "n_images_used": n_images,
    }


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_image(
    pca_model: PCA, image_path: str, device: str = "auto",
    threshold: float | None = None,
) -> dict:
    """Segment spawn regions in a single image using patch-level PCA residuals.

    Steps:
      1. Extract 256 patch tokens (16x16).
      2. Compute PCA reconstruction residual for each patch.
      3. Reshape to 16x16 heatmap.
      4. Upsample to 224x224 with bilinear interpolation.
      5. Auto-threshold: mean + 2*std of patch residuals, or provided threshold.

    Args:
        pca_model: Fitted PCA object from train_patch_pca().
        image_path: Path to a single PNG image.
        device: 'auto', 'cuda', or 'cpu'.
        threshold: Manual threshold for segmentation. If None, uses
            mean + 2*std of the per-patch residuals.

    Returns:
        dict with keys:
        - 'score': float, mean of top 10% patch residuals.
        - 'score_mean': float, mean residual across all patches.
        - 'score_max': float, maximum patch residual.
        - 'heatmap': np.ndarray (224, 224) float32 of per-pixel anomaly scores.
        - 'mask': np.ndarray (224, 224) float32 binary mask (1 = spawn).
        - 'patch_residuals_16x16': np.ndarray (16, 16) of per-patch residuals.
        - 'auto_threshold': float, the threshold used.
        - 'spawn_area_frac': float, fraction of image classified as spawn.
        - 'n_spawn_patches': int, count of anomalous patches (>threshold).
    """
    resolved = _resolve_device(device)

    # ---- Load and embed image ----
    img = Image.open(image_path).convert("RGB")
    tensor = DINO_TRANSFORM(img).unsqueeze(0).to(resolved)

    dinov2 = _load_dinov2_model(resolved)

    with torch.no_grad():
        patch_tokens, _cls_tokens = dinov2.get_intermediate_layers(
            tensor, n=1, reshape=True, return_class_token=True,
        )[0]

    # patch_tokens: [1, 384, 16, 16] -> [256, 384]
    pt = (patch_tokens
          .flatten(2)
          .transpose(1, 2)
          .squeeze(0)
          .cpu()
          .numpy()
          .astype(np.float32))  # (256, 384)

    # ---- Compute per-patch PCA reconstruction residuals ----
    projected = pca_model.transform(pt)                        # (256, n_components)
    reconstructed = pca_model.inverse_transform(projected)     # (256, 384)

    # Per-patch MSE residual
    residuals = np.mean((pt - reconstructed) ** 2, axis=1)     # (256,)

    # ---- Aggregate scores ----
    sorted_residuals = np.sort(residuals)[::-1]
    n_anom = max(1, int(N_PATCHES * ANOMALOUS_PATCH_FRAC))
    score = float(np.mean(sorted_residuals[:n_anom]))
    score_mean = float(np.mean(residuals))
    score_max = float(np.max(residuals))

    # ---- Reshape to 16x16 heatmap ----
    patch_grid = residuals.reshape(PATCH_GRID_SIZE, PATCH_GRID_SIZE)  # (16, 16)

    # ---- Upsample to 224x224 using bilinear interpolation ----
    heatmap_tensor = torch.from_numpy(patch_grid).float().unsqueeze(0).unsqueeze(0)
    # (1, 1, 16, 16) -> (1, 1, 224, 224)
    heatmap_up = F.interpolate(
        heatmap_tensor, size=(224, 224), mode="bilinear", align_corners=False,
    )
    heatmap = heatmap_up.squeeze().cpu().numpy()  # (224, 224) float32

    # ---- Threshold ----
    if threshold is None:
        auto_threshold = float(np.mean(residuals) + 2.0 * np.std(residuals))
    else:
        auto_threshold = threshold

    mask = (heatmap > auto_threshold).astype(np.float32)
    spawn_area_frac = float(np.mean(mask))
    n_spawn_patches = int((patch_grid > auto_threshold).sum())

    return {
        "score": score,
        "score_mean": score_mean,
        "score_max": score_max,
        "heatmap": heatmap,
        "mask": mask,
        "patch_residuals_16x16": patch_grid,
        "auto_threshold": auto_threshold,
        "spawn_area_frac": spawn_area_frac,
        "n_spawn_patches": n_spawn_patches,
        "n_patches": len(residuals),
    }


# ---------------------------------------------------------------------------
# Directory segmentation
# ---------------------------------------------------------------------------

def segment_directory(
    pca_model: PCA, image_dir: str, device: str = "auto",
    threshold: float | None = None,
) -> list[dict]:
    """Segment all images in a directory.

    Args:
        pca_model: Fitted PCA object from train_patch_pca().
        image_dir: Directory containing PNG images to segment.
        device: 'auto', 'cuda', or 'cpu'.
        threshold: Segmentation threshold (None = auto from residuals).

    Returns:
        list of dicts sorted by score descending, each with keys:
        - 'filename': str
        - 'score': float
        - 'score_mean': float
        - 'score_max': float
        - 'auto_threshold': float
        - 'spawn_area_frac': float
        - 'n_spawn_patches': int
        - 'heatmap': np.ndarray (224, 224), only included if store_heatmaps=True
        - 'mask': np.ndarray (224, 224), only included if store_heatmaps=True
    """
    resolved = _resolve_device(device)
    img_dir = Path(image_dir)

    if not img_dir.is_dir():
        print(f"  ERROR: Not a directory: {image_dir}")
        return []

    pngs = sorted(img_dir.glob("*.png"))
    print(f"  Found {len(pngs)} PNG images in {image_dir}")

    if not pngs:
        print("  No images to segment.")
        return []

    results: list[dict] = []

    for p in tqdm(pngs, desc="Segmenting", unit="img"):
        try:
            seg_result = segment_image(
                pca_model, str(p), device=resolved, threshold=threshold,
            )
            seg_result["filename"] = p.name
            # Remove large arrays from list results (users can regenerate)
            seg_result.pop("heatmap", None)
            seg_result.pop("mask", None)
            seg_result.pop("patch_residuals_16x16", None)
            results.append(seg_result)
        except Exception as exc:
            print(f"  WARNING: Failed to segment {p.name}: {exc}")

    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"  Segmented {len(results)}/{len(pngs)} images successfully")

    if results:
        print(f"  Top 3 spawn area fractions:")
        top3 = sorted(results, key=lambda r: r["spawn_area_frac"], reverse=True)[:3]
        for r in top3:
            print(f"    area={r['spawn_area_frac']:.4f}  score={r['score']:.6f}  "
                  f"{r['filename']}")

    return results


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def save_heatmap_overlay(
    image_path: str, heatmap: np.ndarray, output_path: str,
    alpha: float = 0.5,
) -> str:
    """Create overlay image: original + heatmap blended.

    Uses matplotlib 'jet' colormap for the heatmap visualisation.

    Args:
        image_path: Path to the original image.
        heatmap: 224x224 float array of per-pixel anomaly scores.
        output_path: Where to save the overlay PNG.
        alpha: Blending factor (0 = original only, 1 = heatmap only).

    Returns:
        output_path as str.
    """
    from PIL import Image

    # Load original
    original = Image.open(image_path).convert("RGB").resize((224, 224))

    # Normalize heatmap to [0, 255] for colormap
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-12:
        heatmap_norm = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap_norm = np.zeros_like(heatmap)

    # Apply jet colormap
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    jet = cm.get_cmap("jet")
    heatmap_colored = (jet(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_colored, "RGB")

    # Blend
    overlay = Image.blend(original, heatmap_pil, alpha)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)
    print(f"  Saved heatmap overlay: {out_path}")

    return str(out_path)


def save_segmentation_overlay(
    image_path: str, mask: np.ndarray, output_path: str,
    color: tuple = (0, 255, 0),
) -> str:
    """Create overlay with green contour around spawn regions.

    Draws contours around the binary mask on the original image.

    Args:
        image_path: Path to the original image.
        mask: 224x224 binary array (1 = spawn, 0 = no spawn).
        output_path: Where to save the overlay PNG.
        color: RGB tuple for contour colour (default green).

    Returns:
        output_path as str.
    """
    try:
        import cv2
    except ImportError:
        print("  ERROR: opencv-python required for contour overlays. "
              "Install with: pip install opencv-python")
        return ""

    from PIL import Image

    # Load original and ensure it's 224x224
    original = Image.open(image_path).convert("RGB")
    original_resized = original.resize((224, 224))
    img_np = np.array(original_resized)

    # Convert mask to uint8 (0 or 255)
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    # Draw contours on the original image
    contour_img = img_np.copy()
    cv2.drawContours(contour_img, contours, -1, color, thickness=2)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(contour_img).save(out_path)
    print(f"  Saved segmentation overlay: {out_path}")

    return str(out_path)


def generate_segmentations(
    pca_model: PCA, image_dir: str, output_dir: str,
    device: str = "auto", threshold: float | None = None,
    save_overlays: bool = False,
) -> list[dict]:
    """Segment all images and optionally save overlay images.

    Args:
        pca_model: Fitted PCA object from train_patch_pca().
        image_dir: Directory containing PNG images to segment.
        output_dir: Directory to save overlay images.
        device: 'auto', 'cuda', or 'cpu'.
        threshold: Segmentation threshold (None = auto).
        save_overlays: If True, save heatmap + segmentation overlays.

    Returns:
        list of dicts with per-image metrics (no large arrays).
    """
    resolved = _resolve_device(device)
    out_dir = Path(output_dir)

    img_dir = Path(image_dir)
    if not img_dir.is_dir():
        print(f"  ERROR: Not a directory: {image_dir}")
        return []

    pngs = sorted(img_dir.glob("*.png"))
    print(f"  Found {len(pngs)} PNG images in {image_dir}")

    if not pngs:
        print("  No images to segment.")
        return []

    results: list[dict] = []

    for p in tqdm(pngs, desc="Segmenting", unit="img"):
        try:
            seg_result = segment_image(
                pca_model, str(p), device=resolved, threshold=threshold,
            )

            entry = {
                "filename": p.name,
                "score": seg_result["score"],
                "score_mean": seg_result["score_mean"],
                "score_max": seg_result["score_max"],
                "auto_threshold": seg_result["auto_threshold"],
                "spawn_area_frac": seg_result["spawn_area_frac"],
                "n_spawn_patches": seg_result["n_spawn_patches"],
            }

            # Optionally save overlays
            if save_overlays:
                basename = p.stem
                heatmap_path = str(out_dir / f"{basename}_heatmap.png")
                overlay_path = str(out_dir / f"{basename}_segmentation.png")

                try:
                    save_heatmap_overlay(
                        str(p), seg_result["heatmap"], heatmap_path,
                    )
                    entry["heatmap_overlay_path"] = heatmap_path
                except Exception as exc:
                    print(f"  WARNING: Heatmap overlay failed for {p.name}: {exc}")

                try:
                    save_segmentation_overlay(
                        str(p), seg_result["mask"], overlay_path,
                    )
                    entry["segmentation_overlay_path"] = overlay_path
                except Exception as exc:
                    print(f"  WARNING: Segmentation overlay failed for {p.name}: {exc}")

            results.append(entry)

        except Exception as exc:
            print(f"  WARNING: Failed to segment {p.name}: {exc}")

    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"  Segmented {len(results)}/{len(pngs)} images successfully")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DINOv2 patch-level SubspaceAD segmentation for herring spawn localisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train-dir", type=str, default=None,
        help="Directory of negative (non-spawn) PNG images for training patch PCA",
    )
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory of PNG images to segment",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/segmented",
        help="Directory to save overlay images (default: data/segmented)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save per-image metrics JSON",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image to segment (overrides --image-dir)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Segmentation threshold (default: auto from mean + 2*std of residuals)",
    )
    parser.add_argument(
        "--n-components", type=int, default=64,
        help="Number of PCA components for patch subspace (default: 64)",
    )
    parser.add_argument(
        "--sample-frac", type=float, default=0.1,
        help="Fraction of patches to sample per image for PCA training "
             "(default: 0.1). Use 1.0 for all patches.",
    )
    parser.add_argument(
        "--save-overlays", action="store_true",
        help="Save heatmap + segmentation overlays for each image",
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
    output_dir = _resolve_path(args.output_dir)
    output_json = _resolve_path(args.output_json)

    # ---- Handle single-image mode ----
    single_image_path: str | None = None
    if args.image is not None:
        single_image_path = str(Path(args.image).resolve())
        if not Path(single_image_path).exists():
            print(f"ERROR: Image not found: {single_image_path}")
            return 1

    # ---- Resolve train dir ----
    if train_dir is None:
        default_train = repo_root / "data/samples/negative"
        if default_train.is_dir():
            train_dir = default_train
            print(f"  Using default train dir: {train_dir}")
        else:
            print("ERROR: --train-dir is required")
            return 1

    # ---- Train patch PCA ----
    train_result = train_patch_pca(
        str(train_dir),
        n_components=args.n_components,
        device=resolved,
        sample_frac=args.sample_frac,
    )
    if "error" in train_result:
        print(f"\nERROR: Training failed: {train_result['error']}")
        return 1

    pca_model = train_result["pca_model"]

    # ---- Segment ----
    print("\n" + "=" * 60)
    print("  Segmentation")
    print("=" * 60)

    if single_image_path:
        # Segment single image
        print(f"  Segmenting single image: {single_image_path}")
        seg_result = segment_image(
            pca_model, single_image_path, device=resolved,
            threshold=args.threshold,
        )

        print(f"\n  Segmentation results:")
        print(f"    Score (top-10% patch residual):  {seg_result['score']:.6f}")
        print(f"    Mean patch residual:              {seg_result['score_mean']:.6f}")
        print(f"    Max patch residual:               {seg_result['score_max']:.6f}")
        print(f"    Auto threshold:                   {seg_result['auto_threshold']:.6f}")
        print(f"    Spawn area fraction:              {seg_result['spawn_area_frac']:.4f}")
        print(f"    Anomalous patches:                 {seg_result['n_spawn_patches']}/256")

        # Save overlays
        if args.save_overlays and output_dir is not None:
            basename = Path(single_image_path).stem
            heatmap_path = str(output_dir / f"{basename}_heatmap.png")
            overlay_path = str(output_dir / f"{basename}_segmentation.png")
            try:
                save_heatmap_overlay(
                    single_image_path, seg_result["heatmap"], heatmap_path,
                )
            except Exception as exc:
                print(f"  WARNING: Heatmap overlay failed: {exc}")
            try:
                save_segmentation_overlay(
                    single_image_path, seg_result["mask"], overlay_path,
                )
            except Exception as exc:
                print(f"  WARNING: Segmentation overlay failed: {exc}")

        result = {
            "filename": Path(single_image_path).name,
            "segmentation": {
                "score": seg_result["score"],
                "score_mean": seg_result["score_mean"],
                "score_max": seg_result["score_max"],
                "auto_threshold": seg_result["auto_threshold"],
                "spawn_area_frac": seg_result["spawn_area_frac"],
                "n_spawn_patches": seg_result["n_spawn_patches"],
            },
            "training": {
                "n_components": train_result["n_components"],
                "n_patches_trained": train_result["n_patches_trained"],
                "n_images_used": train_result["n_images_used"],
                "explained_variance_ratio": float(
                    train_result["explained_variance_ratio"].sum()
                ),
            },
        }

    else:
        # Segment directory
        if img_dir is None:
            print("ERROR: --image-dir is required (unless using --image)")
            return 1

        if args.save_overlays and output_dir is not None:
            results = generate_segmentations(
                pca_model, str(img_dir), str(output_dir),
                device=resolved, threshold=args.threshold,
                save_overlays=True,
            )
        else:
            results = segment_directory(
                pca_model, str(img_dir), device=resolved,
                threshold=args.threshold,
            )

        result = {
            "model": f"{MODEL_NAME} + Patch SubspaceAD Segmentation",
            "device": resolved,
            "n_components": train_result["n_components"],
            "n_patches_trained": train_result["n_patches_trained"],
            "n_negative_images": train_result["n_images_used"],
            "explained_variance_ratio": float(
                train_result["explained_variance_ratio"].sum()
            ),
            "threshold_specified": args.threshold,
            "n_images_segmented": len(results),
            "results": results,
        }

        print(f"\n  Total images segmented: {len(results)}")

    # ---- Save output ----
    if args.output_json:
        out_path = Path(str(output_json)) if output_json is not None else \
            repo_root / "data/segmented/results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy arrays to lists for JSON serialisation
        json.dump(result, open(str(out_path), "w"), indent=2, default=str)
        print(f"\n  Results saved to: {out_path}")
    elif single_image_path is None:
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
