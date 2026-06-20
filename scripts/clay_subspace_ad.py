#!/usr/bin/env python3
"""Zero-shot herring spawn detection using CLAY v1.5 + SubspaceAD (PCA reconstruction residual).

CLAY v1.5 is a geospatial foundation model trained on multispectral Sentinel-2
imagery via masked autoencoding. It produces 1024-dim embeddings (large model)
that incorporate spectral, spatial, and temporal context.

Under the SubspaceAD paradigm, PCA is fit on CLAY embeddings from negative (no-spawn)
images to learn the low-dimensional subspace of normal coastal variation. At inference,
anomalies (spawn events) are detected via high PCA reconstruction residuals.

Usage:
    # Train subspace on negatives, validate against human labels
    python scripts/clay_subspace_ad.py --validate-only \\
        --image-dir data/samples/unified \\
        --labels-json data/samples/remoteclip_labels.json \\
        --device cpu

    # Score a directory of candidate images
    python scripts/clay_subspace_ad.py \\
        --train-dir data/samples/negative \\
        --image-dir data/candidates_knn \\
        --output-json data/clay_subspace_results.json

    # Train with explicit component count
    python scripts/clay_subspace_ad.py \\
        --train-dir data/samples/negative \\
        --image-dir data/candidates_knn \\
        --n-components 64 \\
        --output-json data/clay_subspace_results.json

Dependencies: torch, torchvision, Pillow, scikit-learn, numpy, tqdm, claymodel
"""

import argparse
import json
import math
import re
import sys
import warnings
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import torch
import yaml
from box import Box
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from torchvision.transforms import v2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLAY imports
# ---------------------------------------------------------------------------
from claymodel.module import ClayMAEModule

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentinel-2 bands expected by CLAY
BAND_NAMES = ["blue", "green", "red", "nir"]
PLATFORM = "sentinel-2-l2a"

# Image dimensions
SIZE = 256
GSD = 10.0

# Default checkpoint and metadata paths (relative to repo root)
DEFAULT_CKPT = "checkpoints/v1.5/clay-v1.5.ckpt"
DEFAULT_METADATA = "configs/metadata.yaml"

# PCA variance ratio target when auto-selecting n_components
AUTO_VARIANCE_TARGET = 0.90

# Default number of PCA components
DEFAULT_N_COMPONENTS = 32

# Cache path for extracted negative embeddings
EMBEDDINGS_CACHE_DIR = "data/embeddings"
EMBEDDINGS_CACHE_FILE = "clay_subspace_embeddings.npz"

# Default prediction threshold (mean reconstruction residual above this = spawn).
# CLAY embeddings are 1024-dim float vectors — residuals will differ from DINOv2.
# Set to a conservative default; threshold sweep in validation gives best results.
DEFAULT_PREDICTION_THRESHOLD = 0.15

# Filename pattern for our standard thumbnails:
#   {name}_{YYYY-MM-DD}_score{float}_{lat}_{lon}_{YYYYMMDD}.png
FILENAME_PATTERN = re.compile(
    r"(.+)_(\d{4}-\d{2}-\d{2})_score([\d.]+)_(-?[\d.]+)_(-?[\d.]+)_(\d{8})\.(png|tif)$"
)

# Default location/date when filename parsing fails
DEFAULT_LAT = 49.5
DEFAULT_LON = -125.0
DEFAULT_DATE = dt(2024, 3, 15, 12, 0, 0)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu', pass through others."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# CLAY model loading
# ---------------------------------------------------------------------------

CLAY_MODEL_CACHE: dict[str, ClayMAEModule | None] = {}


def load_clay_model(
    device: str = "auto",
    checkpoint_path: str | None = None,
    metadata_path: str | None = None,
) -> ClayMAEModule:
    """Load CLAY v1.5 model from checkpoint.

    Follows the loading pattern from ``scripts/clay_delta_detector.py``.

    Args:
        device: 'auto', 'cuda', or 'cpu'.
        checkpoint_path: Path to ``clay-v1.5.ckpt``.
                         Defaults to ``checkpoints/v1.5/clay-v1.5.ckpt``.
        metadata_path: Path to ``metadata.yaml``.
                       Defaults to ``configs/metadata.yaml``.

    Returns:
        Loaded ``ClayMAEModule`` in eval mode on the requested device.
    """
    resolved = _resolve_device(device)

    if checkpoint_path is None:
        # Resolve relative to repo root
        repo_root = Path(__file__).resolve().parent.parent
        ckpt = repo_root / DEFAULT_CKPT
    else:
        ckpt = Path(checkpoint_path)

    if metadata_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        meta = repo_root / DEFAULT_METADATA
    else:
        meta = Path(metadata_path)

    if not ckpt.exists():
        print(f"  ERROR: CLAY checkpoint not found at {ckpt}")
        print(f"  Download from: https://huggingface.co/made-with-clay/Clay")
        print(f"  Place at: {ckpt}")
        sys.exit(1)

    if not meta.exists():
        print(f"  ERROR: Metadata file not found at {meta}")
        sys.exit(1)

    cache_key = f"{ckpt}:{meta}:{resolved}"
    if cache_key in CLAY_MODEL_CACHE:
        return CLAY_MODEL_CACHE[cache_key]

    print(f"  Loading CLAY v1.5 from checkpoint: {ckpt}")
    model = ClayMAEModule.load_from_checkpoint(
        str(ckpt),
        model_size="large",
        metadata_path=str(meta),
        dolls=[16, 32, 64, 128, 256, 768, 1024],
        doll_weights=[1, 1, 1, 1, 1, 1, 1],
        mask_ratio=0.0,
        shuffle=False,
    )
    model.eval()
    model = model.to(resolved)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  CLAY v1.5 loaded: {n_params:.0f}M params on {resolved}")

    CLAY_MODEL_CACHE[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_location_from_filename(filename: str) -> dict:
    """Extract lat, lon, date from a standard thumbnail filename.

    Returns a dict with keys ``lat``, ``lon``, ``datetime`` (or None for each
    if parsing fails).
    """
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return {"lat": None, "lon": None, "datetime": None}

    name_part, date_str, score_str, lat_str, lon_str, sat_date_str, ext = m.groups()
    try:
        lat = float(lat_str)
        lon = float(lon_str)
        parsed_date = dt.strptime(sat_date_str, "%Y%m%d")
        return {"lat": lat, "lon": lon, "datetime": parsed_date}
    except (ValueError, TypeError):
        return {"lat": None, "lon": None, "datetime": None}


# ---------------------------------------------------------------------------
# Image loading — RGB PNG → 4-band multispectral
# ---------------------------------------------------------------------------

def _load_png_as_multispectral(path: Path, size: int = SIZE) -> np.ndarray:
    """Load an RGB PNG and convert to a 4-band (B, G, R, NIR) float32 array.

    CLAY expects Sentinel-2 surface reflectance values in the range ~0-10000.
    PNG thumbnails are 8-bit (0-255) RGB visualizations. We scale them by a
    heuristic multiplier (8×) to bring values into a range closer to the
    Sentinel-2 SR distribution that CLAY was trained on.

    NIR is approximated as the mean of the three visible bands. This is a
    crude proxy — actual NIR would come from Sentinel-2 B8.

    Returns:
        (4, size, size) float32 array in [B, G, R, NIR_approx] order.
    """
    img = Image.open(path).convert("RGB")
    # Resize to target size
    img_resized = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img_resized, dtype=np.float32)  # (H, W, 3), values 0-255

    # Scale to approximate Sentinel-2 SR range
    # Mean band values from metadata: blue=1105, green=1355, red=1552
    # PNG values 0-255 → multiply by ~8 gives roughly correct magnitude
    SCALE = 8.0
    arr = arr * SCALE

    # Split into bands: PNG stores R, G, B → we want B, G, R
    r = arr[:, :, 0]   # Red
    g = arr[:, :, 1]   # Green
    b = arr[:, :, 2]   # Blue

    # Approximate NIR as mean of visible bands
    nir = (r + g + b) / 3.0

    # Stack into (4, H, W): [blue, green, red, nir_approx]
    multispectral = np.stack([b, g, r, nir], axis=0)
    return multispectral


# ---------------------------------------------------------------------------
# Image loading — GeoTIFF multispectral
# ---------------------------------------------------------------------------

def _load_tif_as_multispectral(path: Path, size: int = SIZE) -> np.ndarray:
    """Load a GeoTIFF with Sentinel-2 bands (B2/B3/B4/B8) as a 4-band array.

    Uses tifffile to read the TIFF. Expects bands in the order:
        B2=blue, B3=green, B4=red, B8=nir
    (matching ``BAND_NAMES = ["blue", "green", "red", "nir"]``).

    Returns:
        (4, size, size) float32 array in [B, G, R, NIR] order,
        or None if loading fails.
    """
    try:
        import tifffile as tiff
    except ImportError:
        print("  WARNING: tifffile not installed. Install with: pip install tifffile")
        return None

    try:
        chip_data = tiff.imread(str(path)).astype(np.float32)
    except Exception as exc:
        print(f"  WARNING: Failed to read {path.name}: {exc}")
        return None

    # tifffile reads as (height, width, bands) — transpose to (bands, height, width)
    if chip_data.ndim == 3:
        chip_data = np.transpose(chip_data, (2, 0, 1))

    # Resize if needed
    if chip_data.shape[-2:] != (size, size):
        try:
            from skimage.transform import resize
            bands_resized = []
            for i in range(chip_data.shape[0]):
                bands_resized.append(
                    resize(chip_data[i], (size, size), preserve_range=True)
                )
            chip_data = np.stack(bands_resized)
        except ImportError:
            print("  WARNING: scikit-image not installed for resize. "
                  "Install with: pip install scikit-image")
            # Fall back to center crop if larger, or pad if smaller
            h, w = chip_data.shape[-2:]
            if h >= size and w >= size:
                cy, cx = h // 2, w // 2
                y1, y2 = cy - size // 2, cy + size // 2
                x1, x2 = cx - size // 2, cx + size // 2
                chip_data = chip_data[:, y1:y2, x1:x2]
            else:
                # Pad to size
                pad_h = max(0, size - h)
                pad_w = max(0, size - w)
                chip_data = np.pad(
                    chip_data,
                    ((0, 0), (pad_h // 2, pad_h - pad_h // 2),
                     (pad_w // 2, pad_w - pad_w // 2)),
                    mode="constant",
                )
                # Then crop
                chip_data = chip_data[:, :size, :size]

    # Ensure 4 bands
    if chip_data.shape[0] != 4:
        print(f"  WARNING: Expected 4 bands, got {chip_data.shape[0]} in {path.name}")
        if chip_data.shape[0] == 3:
            # Add NIR as mean of visible bands
            nir = chip_data.mean(axis=0, keepdims=True)
            chip_data = np.concatenate([chip_data, nir], axis=0)
        else:
            return None

    return chip_data.astype(np.float32)


# ---------------------------------------------------------------------------
# Image loading dispatch
# ---------------------------------------------------------------------------

def load_image_as_multispectral(path: Path, size: int = SIZE) -> np.ndarray | None:
    """Load an image file as a 4-band (B, G, R, NIR) float32 array.

    Dispatches on file extension:
    - ``.png`` — load as RGB and convert via ``_load_png_as_multispectral``.
    - ``.tif`` / ``.tiff`` — load as GeoTIFF via ``_load_tif_as_multispectral``.
    - Other extensions — unsupported, returns None.

    Returns:
        (4, size, size) float32 array, or None on failure.
    """
    ext = path.suffix.lower()
    if ext == ".png":
        return _load_png_as_multispectral(path, size=size)
    elif ext in (".tif", ".tiff"):
        return _load_tif_as_multispectral(path, size=size)
    else:
        return None


# ---------------------------------------------------------------------------
# CLAY normalization helpers
# ---------------------------------------------------------------------------

def _get_metadata(metadata_path: str | Path | None = None) -> Box:
    """Load the CLAY metadata YAML and return a Box for easy access.

    Caches the result to avoid repeated file I/O.
    """
    if not hasattr(_get_metadata, "_cache"):
        if metadata_path is None:
            repo_root = Path(__file__).resolve().parent.parent
            meta_path = repo_root / DEFAULT_METADATA
        else:
            meta_path = Path(metadata_path)

        if not meta_path.exists():
            print(f"  ERROR: Metadata file not found: {meta_path}")
            sys.exit(1)

        _get_metadata._cache = Box(yaml.safe_load(meta_path.read_text()))

    return _get_metadata._cache


def _get_band_stats(metadata: Box | None = None) -> tuple[list[float], list[float], list[float]]:
    """Return (mean, std, wavelength) lists for the standard 4 bands."""
    meta = metadata if metadata is not None else _get_metadata()
    p = meta[PLATFORM]
    means = [p.bands.mean[b] for b in BAND_NAMES]
    stds = [p.bands.std[b] for b in BAND_NAMES]
    waves = [p.bands.wavelength[b] for b in BAND_NAMES]
    return means, stds, waves


def _get_transform(metadata: Box | None = None) -> v2.Compose:
    """Build the normalization transform from metadata."""
    means, stds, _ = _get_band_stats(metadata)
    return v2.Compose([v2.Normalize(mean=means, std=stds)])


def _normalize_ts(dt_obj: dt) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Encode datetime as (sin/cos week), (sin/cos hour) for CLAY."""
    week = dt_obj.isocalendar().week * 2 * math.pi / 52
    hour = dt_obj.hour * 2 * math.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def _normalize_ll(lat: float, lon: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Encode lat/lon as (sin/cos lat), (sin/cos lon) for CLAY."""
    lat_r = lat * math.pi / 180
    lon_r = lon * math.pi / 180
    return (math.sin(lat_r), math.cos(lat_r)), (math.sin(lon_r), math.cos(lon_r))


# ---------------------------------------------------------------------------
# CLAY embedding extraction
# ---------------------------------------------------------------------------

def _compute_clay_embedding(
    model: ClayMAEModule,
    pixels_4band: np.ndarray,
    lat: float,
    lon: float,
    image_datetime: dt,
    device: str,
    metadata: Box | None = None,
) -> np.ndarray | None:
    """Run CLAY encoder on a 4-band chip and return the [CLS] embedding.

    Args:
        model: Loaded ClayMAEModule.
        pixels_4band: (4, H, W) float32 array [B, G, R, NIR].
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        image_datetime: Datetime of the image acquisition.
        device: 'cuda' or 'cpu'.
        metadata: Optional pre-loaded metadata Box.

    Returns:
        1024-dim numpy float32 array (the [CLS] token), or None on failure.
    """
    try:
        means, stds, waves = _get_band_stats(metadata)

        # Normalize pixels using Sentinel-2 band stats
        transform = v2.Compose([v2.Normalize(mean=means, std=stds)])
        pixel_tensor = transform(torch.from_numpy(pixels_4band))  # (4, H, W)

        # Encode time and location
        wn, hn = _normalize_ts(image_datetime)
        ln, lo = _normalize_ll(lat, lon)

        datacube = {
            "platform": PLATFORM,
            "time": torch.tensor(
                np.hstack([wn, hn]), dtype=torch.float32, device=device
            ).unsqueeze(0),
            "latlon": torch.tensor(
                np.hstack([ln, lo]), dtype=torch.float32, device=device
            ).unsqueeze(0),
            "pixels": pixel_tensor.unsqueeze(0).to(device),
            "gsd": torch.tensor(GSD, device=device),
            "waves": torch.tensor(waves, device=device),
        }

        with torch.no_grad():
            unmsk_patch, _, _, _ = model.model.encoder(datacube)

        # [CLS] token is at index 0
        emb = unmsk_patch[:, 0, :].cpu().numpy().flatten()
        return emb
    except Exception as exc:
        print(f"  WARNING: CLAY embedding failed: {exc}")
        return None


def extract_clay_embeddings(
    image_dir: str,
    device: str = "auto",
    cache: bool = True,
    metadata_path: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract CLAY 1024-dim embeddings for all images in a directory.

    Supports both PNG (RGB → 4-band conversion) and TIF (multispectral GeoTIFF)
    files.  Checks for cached embeddings first.

    Args:
        image_dir: Directory containing PNG or TIF images.
        device: 'auto', 'cuda', or 'cpu'.
        cache: Whether to check/save the embeddings cache (default True).
        metadata_path: Optional override for metadata.yaml path.

    Returns:
        (embeddings, filenames) where:
        - embeddings: np.ndarray of shape (N, 1024), dtype float32.
        - filenames: list of str filenames for successfully embedded images.
    """
    resolved = _resolve_device(device)
    img_dir = Path(image_dir)

    if not img_dir.is_dir():
        print(f"  ERROR: Not a directory: {img_dir}")
        return np.array([]), []

    # Collect supported image files
    image_files = sorted(
        list(img_dir.glob("*.png")) + list(img_dir.glob("*.tif")) + list(img_dir.glob("*.tiff"))
    )
    if not image_files:
        print(f"  No PNG/TIF images found in {image_dir}")
        return np.array([]), []

    print(f"  Found {len(image_files)} images in {image_dir}")

    # ---- Check cache ----
    cache_path = Path(EMBEDDINGS_CACHE_DIR) / EMBEDDINGS_CACHE_FILE
    if cache and cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        embeddings = loaded["embeddings"]
        cached_fnames = loaded["filenames"].tolist()
        # Verify cache matches the requested image_dir's basenames
        requested_basenames = {p.name for p in image_files}
        cached_basenames = set(cached_fnames)
        if requested_basenames == cached_basenames:
            print(f"  Loaded {len(cached_fnames)} cached embeddings (shape {embeddings.shape})")
            return embeddings, cached_fnames
        else:
            print(f"  Cache mismatch — re-extracting embeddings")
            print(f"    Requested: {len(requested_basenames)} files, "
                  f"Cached: {len(cached_basenames)} files")

    # ---- Load CLAY model ----
    model = load_clay_model(resolved, metadata_path=metadata_path)
    meta = _get_metadata(metadata_path)

    embeddings: list[np.ndarray] = []
    filenames: list[str] = []

    for p in tqdm(image_files, desc="Extracting CLAY embeddings", unit="img"):
        # Load pixels
        pixels = load_image_as_multispectral(p, size=SIZE)
        if pixels is None:
            print(f"  WARNING: Skipping unsupported file: {p.name}")
            continue

        # Parse location/date from filename, fall back to defaults
        loc = parse_location_from_filename(p.name)
        lat = loc["lat"] if loc["lat"] is not None else DEFAULT_LAT
        lon = loc["lon"] if loc["lon"] is not None else DEFAULT_LON
        img_dt = loc["datetime"] if loc["datetime"] is not None else DEFAULT_DATE

        emb = _compute_clay_embedding(model, pixels, lat, lon, img_dt, resolved, meta)
        if emb is not None:
            embeddings.append(emb)
            filenames.append(p.name)
        else:
            print(f"  WARNING: Failed to embed {p.name}")

    if not embeddings:
        print("  No embeddings extracted successfully.")
        return np.array([]), []

    emb_array = np.stack(embeddings).astype(np.float32)
    print(f"  Extracted {len(filenames)} CLAY embeddings (shape {emb_array.shape})")

    # ---- Save cache ----
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=emb_array,
            filenames=np.array(filenames, dtype=object),
        )
        print(f"  Saved embeddings cache to {cache_path}")

    return emb_array, filenames


# ---------------------------------------------------------------------------
# Subspace training (PCA)
# ---------------------------------------------------------------------------

def _auto_select_n_components(
    n_samples: int,
    n_features: int,
    variance_target: float = AUTO_VARIANCE_TARGET,
) -> int:
    """Automatically select n_components for PCA.

    For anomaly detection we want a *low-dimensional* subspace so that
    anomalies produce meaningful reconstruction residuals. The heuristic:

      1. Start with DEFAULT_N_COMPONENTS (32).
      2. Cap at min(n_samples - 1, n_features) to avoid singular covariance.
      3. Cap at n_samples // 2 to prevent memorizing the training set.

    The user can always override with --n-components.

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


def train_subspace(
    embeddings: np.ndarray,
    n_components: int | None = None,
) -> dict:
    """Fit PCA on normal (negative) embeddings to learn the normal coastal subspace.

    Args:
        embeddings: (N, D) numpy array of CLAY embeddings from negative images.
        n_components: Number of PCA components. None = auto-select.

    Returns:
        dict with keys:
        - 'pca_model': fitted PCA object (sklearn.decomposition.PCA)
        - 'mean': ndarray of mean embedding
        - 'components': ndarray of PCA components
        - 'explained_variance': ndarray of explained variance per component
        - 'explained_variance_ratio': ndarray of explained variance ratio
        - 'n_components': int, number of components used
        - 'n_train': int, number of training samples
    """
    n_samples, n_features = embeddings.shape

    if n_components is None:
        n_components = _auto_select_n_components(n_samples, n_features)
        print(f"  Auto-selected n_components={n_components} "
              f"(from {n_samples} samples x {n_features} features)")

    # Clamp: PCA requires n_components <= min(n_samples, n_features)
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
    }


def train_isolation_forest(
    embeddings: np.ndarray,
    random_state: int = 42,
) -> IsolationForest:
    """Train an IsolationForest on the same embeddings for comparison.

    Args:
        embeddings: (N, D) numpy array of CLAY embeddings.
        random_state: Random seed.

    Returns:
        Fitted IsolationForest model.
    """
    print("  Fitting IsolationForest (n_estimators=100, contamination='auto')...")
    if_model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    if_model.fit(embeddings)
    print(f"  IsolationForest fitted on {len(embeddings)} samples")
    return if_model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_image(
    pca_model: PCA,
    embedding: np.ndarray,
) -> dict:
    """Score a single embedding by its PCA reconstruction residual.

    ``reconstruction = pca.inverse_transform(pca.transform(embedding))``
    ``residual = mean((embedding - reconstruction)^2)``

    Higher residual = more anomalous relative to the normal subspace.

    Args:
        pca_model: Fitted PCA object from train_subspace.
        embedding: 1-D numpy array of shape (D,) — a single CLAY embedding.

    Returns:
        dict with keys:
        - 'score': float, reconstruction residual (higher = more anomalous)
        - 'n_components': int, number of PCA components used
        - 'prediction': 1 if score > DEFAULT_PREDICTION_THRESHOLD else 0
        - 'reconstruction_error_l2': float, L2 reconstruction error
    """
    emb_2d = embedding.reshape(1, -1)
    projected = pca_model.transform(emb_2d)
    reconstructed = pca_model.inverse_transform(projected)

    residual = float(np.mean((emb_2d - reconstructed) ** 2))
    l2_error = float(np.linalg.norm(emb_2d - reconstructed))

    prediction = 1 if residual > DEFAULT_PREDICTION_THRESHOLD else 0

    return {
        "score": residual,
        "n_components": pca_model.n_components_,
        "prediction": prediction,
        "reconstruction_error_l2": l2_error,
    }


def score_directory(
    pca_model: PCA,
    image_dir: str,
    device: str = "auto",
    metadata_path: str | None = None,
) -> list[dict]:
    """Score all images in a directory using trained PCA subspace.

    For each image:
      1. Load and convert to 4-band multispectral format.
      2. Extract CLAY embedding.
      3. Compute PCA reconstruction residual.
      4. Higher residual = more anomalous = more likely spawn.

    Args:
        pca_model: Fitted PCA object from train_subspace.
        image_dir: Directory containing images to score.
        device: 'auto', 'cuda', or 'cpu'.
        metadata_path: Optional override for metadata.yaml path.

    Returns:
        list of dicts sorted by score descending, each with keys:
        - 'filename': str
        - 'score': float
        - 'n_components': int
        - 'prediction': 0 or 1
        - 'reconstruction_error_l2': float
    """
    resolved = _resolve_device(device)
    img_dir = Path(image_dir)

    if not img_dir.is_dir():
        print(f"  ERROR: Not a directory: {image_dir}")
        return []

    # Collect image files
    image_files = sorted(
        list(img_dir.glob("*.png")) + list(img_dir.glob("*.tif")) + list(img_dir.glob("*.tiff"))
    )
    print(f"  Found {len(image_files)} images in {image_dir}")

    if not image_files:
        print("  No images to score.")
        return []

    # ---- Load CLAY model ----
    model = load_clay_model(resolved, metadata_path=metadata_path)
    meta = _get_metadata(metadata_path)

    results: list[dict] = []

    for p in tqdm(image_files, desc="Scoring", unit="img"):
        try:
            pixels = load_image_as_multispectral(p, size=SIZE)
            if pixels is None:
                print(f"  WARNING: Skipping unsupported file: {p.name}")
                continue

            loc = parse_location_from_filename(p.name)
            lat = loc["lat"] if loc["lat"] is not None else DEFAULT_LAT
            lon = loc["lon"] if loc["lon"] is not None else DEFAULT_LON
            img_dt = loc["datetime"] if loc["datetime"] is not None else DEFAULT_DATE

            emb = _compute_clay_embedding(model, pixels, lat, lon, img_dt, resolved, meta)
            if emb is None:
                print(f"  WARNING: CLAY embedding failed for {p.name}")
                continue

            score_result = score_image(pca_model, emb)
            score_result["filename"] = p.name
            results.append(score_result)
        except Exception as exc:
            print(f"  WARNING: Failed to score {p.name}: {exc}")

    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"  Scored {len(results)}/{len(image_files)} images successfully")

    if results:
        print("  Top 3 scores:")
        for r in results[:3]:
            print(f"    {r['score']:.6f}  {r['filename']}  "
                  f"[{'spawn' if r['prediction'] else 'normal'}]")

    return results


# ---------------------------------------------------------------------------
# Validation against human labels
# ---------------------------------------------------------------------------

def validate(
    pca_model: PCA,
    labels_json_path: str,
    image_dir: str,
    device: str = "auto",
    metadata_path: str | None = None,
) -> dict:
    """Validate subspace anomaly detection against human labels.

    Labels JSON format (matching ``subspace_ad.py`` convention)::

        {"labels": [{"filename": "image.png", "label": 1}, ...]}

    where ``label=1`` means positive (spawn), ``label=0`` means negative (no spawn).

    Returns:
        dict with keys:
        accuracy, best_accuracy, best_threshold, auc_roc, avg_precision,
        confusion_matrix, per_sample, n_total, n_pos, n_neg.
    """
    resolved = _resolve_device(device)
    print(f"  Validate device: {resolved}")

    # ---- Load labels ----
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

    # ---- Load CLAY model ----
    model = load_clay_model(resolved, metadata_path=metadata_path)
    meta = _get_metadata(metadata_path)

    img_dir = Path(image_dir)
    per_sample: list[dict] = []

    for entry in tqdm(label_entries, desc="Validating", unit="img"):
        fname = entry["filename"]
        true_label = entry["label"]
        img_path = img_dir / fname

        if not img_path.exists():
            print(f"  WARNING: Image not found: {img_path}")
            continue

        try:
            pixels = load_image_as_multispectral(img_path, size=SIZE)
            if pixels is None:
                print(f"  WARNING: Unsupported file format: {fname}")
                continue

            loc = parse_location_from_filename(fname)
            lat = loc["lat"] if loc["lat"] is not None else DEFAULT_LAT
            lon = loc["lon"] if loc["lon"] is not None else DEFAULT_LON
            img_dt = loc["datetime"] if loc["datetime"] is not None else DEFAULT_DATE

            emb = _compute_clay_embedding(model, pixels, lat, lon, img_dt, resolved, meta)
            if emb is None:
                continue

            score_result = score_image(pca_model, emb)

            per_sample.append({
                "filename": fname,
                "true_label": true_label,
                "prediction": score_result["prediction"],
                "score": score_result["score"],
                "n_components": score_result["n_components"],
                "reconstruction_error_l2": score_result["reconstruction_error_l2"],
            })
        except Exception as exc:
            print(f"  WARNING: Failed to process {fname}: {exc}")

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

    # ---- Aggregate metrics ----
    y_true = np.array([s["true_label"] for s in per_sample])
    y_pred = np.array([s["prediction"] for s in per_sample])
    y_score = np.array([s["score"] for s in per_sample])

    n_total = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_total - n_pos

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Best accuracy via threshold sweep
    if y_score.min() == y_score.max():
        best_acc = acc
        best_thr = float(y_score[0]) if len(y_score) > 0 else 0.0
    else:
        thresholds = np.linspace(
            y_score.min() - 0.1 * max(1.0, abs(y_score.min())),
            y_score.max() + 0.1 * max(1.0, abs(y_score.max())),
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
    print(f"    Best accuracy:         {best_acc:.4f} @ thr={best_thr:.6f}")
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
# Training convenience
# ---------------------------------------------------------------------------

def train_on_negatives(
    negative_dir: str,
    n_components: int | None = None,
    device: str = "auto",
    metadata_path: str | None = None,
) -> dict:
    """Train PCA subspace on all negatives in a directory.

    Extracts CLAY embeddings from all PNGs/TIFs in ``negative_dir``, fits PCA.
    Also trains and returns an IsolationForest on the same embeddings for
    comparison.

    Args:
        negative_dir: Directory of negative (non-spawn) images.
        n_components: Number of PCA components. None = auto-select.
        device: 'auto', 'cuda', or 'cpu'.
        metadata_path: Optional override for metadata.yaml path.

    Returns:
        dict with keys:
        - 'pca_model': fitted PCA object (sklearn.decomposition.PCA)
        - 'pca': the full PCA result dict from train_subspace()
        - 'if_model': fitted IsolationForest model
        - 'n_negative_images': int
        - 'explained_variance_ratio': float (total)
        - 'n_components': int
        - 'embedding_dim': int
    """
    resolved = _resolve_device(device)

    print("=" * 60)
    print("  CLAY SubspaceAD — Train on Negatives")
    print("=" * 60)
    print(f"  Negative directory: {negative_dir}")
    print(f"  Device: {resolved}")

    # Extract CLAY embeddings from negatives
    embeddings, filenames = extract_clay_embeddings(
        negative_dir, device=resolved, cache=True, metadata_path=metadata_path,
    )

    if len(embeddings) == 0:
        print("ERROR: No negative embeddings extracted.")
        return {"error": "No negative embeddings extracted"}

    print(f"  Training on {len(embeddings)} negative images")

    # Train PCA subspace
    pca_result = train_subspace(embeddings, n_components=n_components)
    pca_model = pca_result["pca_model"]

    # Train IsolationForest for comparison
    if_model = train_isolation_forest(embeddings)

    total_var_ratio = float(pca_model.explained_variance_ratio_.sum())

    print(f"\n  Training complete:")
    print(f"    PCA components:             {pca_model.n_components_}")
    print(f"    PCA explained variance:     {total_var_ratio:.4f}")
    print(f"    IsolationForest estimators: {if_model.n_estimators}")
    print(f"    Negative training images:   {len(embeddings)}")
    print(f"    Embedding dimension:        {embeddings.shape[1]}")

    return {
        "pca_model": pca_model,
        "if_model": if_model,
        "n_negative_images": len(embeddings),
        "explained_variance_ratio": pca_result["explained_variance_ratio"],
        "n_components": pca_result["n_components"],
        "embedding_dim": embeddings.shape[1],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-shot herring spawn detection via CLAY v1.5 + SubspaceAD (PCA reconstruction residual)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train-dir", type=str, default=None,
        help="Directory of negative (non-spawn) images for training PCA subspace",
    )
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory of images to score or validate",
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
        "--n-components", type=int, default=None,
        help="Number of PCA components (default: auto-select)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip per-image scoring output, just run validation against labels",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on (default: auto)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to CLAY v1.5 checkpoint file (default: checkpoints/v1.5/clay-v1.5.ckpt)",
    )
    parser.add_argument(
        "--metadata", type=str, default=None,
        help="Path to CLAY metadata.yaml (default: configs/metadata.yaml)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable embedding cache",
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
    ckpt_path = _resolve_path(args.checkpoint)
    meta_path = _resolve_path(args.metadata)

    # Check image dir
    if img_dir is not None and not img_dir.is_dir():
        print(f"ERROR: Image directory not found: {img_dir}")
        return 1

    # Check labels file
    if labels_path is not None and not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_path}")
        return 1

    # Check train dir
    if train_dir is not None and not train_dir.is_dir():
        print(f"ERROR: Train directory not found: {train_dir}")
        return 1

    # ==================================================================
    # Mode 1: Validate-only
    # ==================================================================
    if args.validate_only:
        if args.labels_json is None:
            print("ERROR: --validate-only requires --labels-json")
            return 1
        if img_dir is None:
            print("ERROR: --validate-only requires --image-dir")
            return 1

        # Default to negatives directory for training if --train-dir not given
        if train_dir is None:
            default_train = repo_root / "data/samples/negative"
            if default_train.is_dir():
                train_dir = default_train
                print(f"  Using default train dir: {train_dir}")
            else:
                print("ERROR: --validate-only requires --train-dir or data/samples/negative")
                return 1

        # Train subspace on negatives
        train_result = train_on_negatives(
            str(train_dir),
            n_components=args.n_components,
            device=resolved,
            metadata_path=str(meta_path) if meta_path else None,
        )

        if "error" in train_result:
            print(f"\nERROR: Training failed: {train_result['error']}")
            return 1

        pca_model = train_result["pca_model"]

        # Validate against labels
        print("\n" + "=" * 60)
        print("  Validation")
        print("=" * 60)
        result = validate(
            pca_model,
            str(labels_path),
            str(img_dir),
            device=resolved,
            metadata_path=str(meta_path) if meta_path else None,
        )

        # Attach training metadata
        result["training"] = {
            "train_dir": str(train_dir),
            "n_negative_images": train_result["n_negative_images"],
            "n_components": train_result["n_components"],
            "explained_variance_ratio": [
                float(v) for v in train_result["explained_variance_ratio"]
            ],
            "embedding_dim": train_result["embedding_dim"],
            "model_name": "clay-v1.5-large",
        }

    # ==================================================================
    # Mode 2: Score directory (with optional training)
    # ==================================================================
    else:
        if img_dir is None:
            if train_dir is not None:
                default_img = repo_root / "data/candidates_knn"
                if default_img.is_dir():
                    img_dir = default_img
                    print(f"  Using default image dir: {img_dir}")
                else:
                    print("ERROR: --image-dir is required")
                    return 1
            else:
                print("ERROR: --image-dir or --train-dir is required")
                return 1

        # Train if train-dir provided, or try to load cached
        if train_dir is not None and train_dir.is_dir():
            train_result = train_on_negatives(
                str(train_dir),
                n_components=args.n_components,
                device=resolved,
                metadata_path=str(meta_path) if meta_path else None,
            )
            if "error" in train_result:
                print(f"\nERROR: Training failed: {train_result['error']}")
                return 1
            pca_model = train_result["pca_model"]
        elif args.n_components is not None:
            # Try to load cached embeddings and train from those
            cache_path = Path(EMBEDDINGS_CACHE_DIR) / EMBEDDINGS_CACHE_FILE
            if cache_path.exists():
                print(f"  Loading cached embeddings for PCA training from {cache_path}")
                loaded = np.load(cache_path, allow_pickle=True)
                embeddings = loaded["embeddings"]
                pca_result = train_subspace(embeddings, n_components=args.n_components)
                pca_model = pca_result["pca_model"]
                train_result = {
                    "n_negative_images": len(embeddings),
                    "n_components": pca_result["n_components"],
                    "explained_variance_ratio": float(
                        pca_result["explained_variance_ratio"].sum()
                    ),
                    "embedding_dim": embeddings.shape[1],
                }
            else:
                print("ERROR: --train-dir required (no cached embeddings found)")
                return 1
        else:
            print("ERROR: --train-dir required for training the PCA subspace")
            return 1

        # Score directory
        print("\n" + "=" * 60)
        print("  Scoring")
        print("=" * 60)
        scoring_results = score_directory(
            pca_model,
            str(img_dir),
            device=resolved,
            metadata_path=str(meta_path) if meta_path else None,
        )

        result = {
            "model": "CLAY v1.5 (large) + SubspaceAD (PCA)",
            "device": resolved,
            "n_components": train_result["n_components"],
            "explained_variance_ratio": train_result.get("explained_variance_ratio", []),
            "n_negative_training_images": train_result["n_negative_images"],
            "n_images_scored": len(scoring_results),
            "results": scoring_results,
        }

        # Optionally validate against labels
        if labels_path is not None and labels_path.exists():
            print("\n" + "=" * 60)
            print("  Running validation against labels...")
            val_result = validate(
                pca_model,
                str(labels_path),
                str(img_dir),
                device=resolved,
                metadata_path=str(meta_path) if meta_path else None,
            )
            result["validation"] = val_result

    # ==================================================================
    # Save or print output
    # ==================================================================
    if args.output_json:
        out_path = _resolve_path(args.output_json)
        if out_path is None:
            out_path = repo_root / "data/clay_subspace_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Results saved to: {out_path}")
    elif not args.validate_only:
        # In scoring mode without --output-json, print summary
        print(json.dumps(result, indent=2, default=str))
    else:
        # In validate-only mode, already printed results above
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
