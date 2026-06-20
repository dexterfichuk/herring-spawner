#!/usr/bin/env python3
"""
Honest evaluation of CLAY v1.5 + PCA SubspaceAD on real GeoTIFF chips.

Loads actual Sentinel-2 4-band GeoTIFF chips (B2/B3/B4/B8, surface reflectance),
extracts CLAY embeddings, trains PCA on negative chips, and scores all chips
by PCA reconstruction residual.

Usage:
    python scripts/evaluate_clay_geotiffs.py --chips-dir data/chips --device cpu

Output:
    Per-chip table with scores and rank, plus aggregate metrics (AUROC,
    separation, mean residuals).

References:
    - scripts/clay_subspace_ad.py  (original script with flawed PNG→multispectral)
    - scripts/clay_delta_detector.py  (proper CLAY loading with ClayMAEModule)
    - scripts/run_clay_embeddings.py  (CLAY encoder usage pattern)
"""

import argparse
import json
import math
import sys
import warnings
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import torch
import yaml
from box import Box
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from torchvision.transforms import v2
from claymodel.module import ClayMAEModule

# Suppress noisy tifffile GDAL_NODATA warnings
import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*GDAL_NODATA.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BAND_NAMES = ["blue", "green", "red", "nir"]
PLATFORM = "sentinel-2-l2a"
SIZE = 256
GSD = 10.0
DEFAULT_CKPT = "checkpoints/v1.5/clay-v1.5.ckpt"
DEFAULT_METADATA = "configs/metadata.yaml"

# ---------------------------------------------------------------------------
# Known chip locations (from data/embeddings/metadata.json)
# ---------------------------------------------------------------------------
CHIP_LOCATIONS = {
    "pos-anderson":        (49.646389, -126.468889, "2024-03-16"),
    "pos-breakwater":      (49.135000, -123.683056, "2024-03-18"),
    "pos-fan-island":      (53.905833, -130.739444, "2024-03-22"),
    "pos-qualicum":        (49.355704, -124.456910, "2024-03-18"),
    "pos-salmon":          (48.920000, -125.550000, "2025-02-11"),
    "pos-ucluelet":        (48.942778, -125.546111, "2024-03-18"),
    "neg-qualicum-after":  (49.355704, -124.456910, "2024-03-18"),
    "neg-tree-bluff-pre":  (54.429167, -130.488889, "2024-03-20"),
    "neg-pt2-1":           (50.824935, -126.192928, "2026-04-10"),
    "cand-abrams":         (52.535000, -128.828300, "2024-03-24"),
    "cand-absalom":        (53.849700, -130.602200, "2024-03-22"),
    "cand-alder":          (52.436900, -131.316500, "2024-04-06"),
    "cand-bawden":         (49.290300, -126.016400, "2024-03-16"),
    "cand-big-qualicum":   (49.398900, -124.609100, "2024-03-18"),
    "cand-boca":           (49.619200, -126.626100, "2024-03-16"),
    "cand-bowser":         (49.433300, -124.666700, "2024-03-18"),
    "cand-capelazo":       (49.701400, -124.860000, "2024-03-18"),
    "cand-chetarpe":       (49.245900, -126.009500, "2024-03-16"),
    "cand-skeena":         (54.456600, -130.390700, "2024-03-22"),
    "cand-spiller":        (52.275800, -128.361700, "2024-03-29"),
}

# Map chip file prefix to location key
def _chip_to_key(filename: str) -> str:
    """Extract location key from chip filename.
    
    Handles filenames like: pos-anderson_20240316.tif
    Returns: pos-anderson
    """
    stem = Path(filename).stem  # removes .tif
    # Find where the date starts: after the last word before _2024... or _2025... or _2026...
    # Simpler approach: split on underscore and take all parts before the last one (the date)
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 8:
        return "_".join(parts[:-1])
    # Fall back to full stem
    return stem


def parse_chip_date(filename: str) -> dt:
    """Extract date from chip filename: pos-anderson_20240316.tif → 2024-03-16."""
    stem = Path(filename).stem
    parts = stem.split("_")
    for part in reversed(parts):
        if len(part) == 8 and part.isdigit():
            try:
                return dt.strptime(part, "%Y%m%d")
            except ValueError:
                pass
    return dt(2024, 3, 15, 12, 0, 0)


def get_chip_location(filename: str) -> dict:
    """Look up lat/lon/date for a chip, with fallback defaults."""
    key = _chip_to_key(filename)
    if key in CHIP_LOCATIONS:
        lat, lon, date_str = CHIP_LOCATIONS[key]
        return {
            "lat": lat,
            "lon": lon,
            "datetime": dt.strptime(date_str, "%Y-%m-%d"),
        }
    # Fallback: extract date from filename, use default BC coast location
    date = parse_chip_date(filename)
    print(f"  WARNING: No known location for '{key}', using defaults")
    return {"lat": 50.0, "lon": -126.0, "datetime": date}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# CLAY model loading (same pattern as clay_delta_detector.py)
# ---------------------------------------------------------------------------
def load_clay_model(device: str, 
                    checkpoint_path: str = DEFAULT_CKPT,
                    metadata_path: str = DEFAULT_METADATA) -> ClayMAEModule:
    ckpt = Path(checkpoint_path)
    meta = Path(metadata_path)
    
    if not ckpt.exists():
        print(f"ERROR: CLAY checkpoint not found at {ckpt}")
        print(f"Download from: https://huggingface.co/made-with-clay/Clay")
        sys.exit(1)
    if not meta.exists():
        print(f"ERROR: Metadata not found at {meta}")
        sys.exit(1)
    
    print(f"  Loading CLAY v1.5 from: {ckpt}")
    model = ClayMAEModule.load_from_checkpoint(
        str(ckpt), model_size="large",
        metadata_path=str(meta),
        dolls=[16, 32, 64, 128, 256, 768, 1024],
        doll_weights=[1, 1, 1, 1, 1, 1, 1],
        mask_ratio=0.0, shuffle=False,
    )
    model.eval()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded: {n_params:.0f}M params on {device}")
    return model


# ---------------------------------------------------------------------------
# Load GeoTIFF chip
# ---------------------------------------------------------------------------
def load_geotiff(path: Path) -> np.ndarray | None:
    """Load (4, 256, 256) float32 array from a GeoTIFF chip.
    
    Expects (256, 256, 4) in B2/B3/B4/B8 order from tifffile.
    Transposes to (4, 256, 256) and returns as-is (surface reflectance).
    """
    try:
        import tifffile
        data = tifffile.imread(str(path)).astype(np.float32)
    except Exception as e:
        print(f"  ERROR: Failed to read {path.name}: {e}")
        return None
    
    # tifffile reads as (H, W, C) → transpose to (C, H, W)
    if data.ndim == 3:
        data = np.transpose(data, (2, 0, 1))
    
    # Verify shape
    if data.shape[0] < 4:
        print(f"  ERROR: {path.name} has {data.shape[0]} bands, expected 4")
        return None
    
    # Take first 4 bands (B2, B3, B4, B8)
    data = data[:4, :, :]
    
    # Resize if needed
    if data.shape[-2:] != (SIZE, SIZE):
        from skimage.transform import resize
        bands_resized = []
        for i in range(4):
            bands_resized.append(
                resize(data[i], (SIZE, SIZE), preserve_range=True)
            )
        data = np.stack(bands_resized)
    
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# CLAY embedding extraction (same pattern as clay_delta_detector.py)
# ---------------------------------------------------------------------------
def get_clay_embedding(model: ClayMAEModule, pixels: np.ndarray,
                       lat: float, lon: float, img_dt: dt,
                       device: str, metadata: Box) -> np.ndarray | None:
    """Extract [CLS] embedding from CLAY for a 4-band chip."""
    try:
        # Get band stats from metadata
        p = metadata[PLATFORM]
        means = [p.bands.mean[b] for b in BAND_NAMES]
        stds = [p.bands.std[b] for b in BAND_NAMES]
        waves = [p.bands.wavelength[b] for b in BAND_NAMES]
        
        # Normalize pixels
        transform = v2.Compose([v2.Normalize(mean=means, std=stds)])
        pixel_tensor = transform(torch.from_numpy(pixels))  # (4, H, W)
        
        # Encode time and location
        week = img_dt.isocalendar().week * 2 * math.pi / 52
        hour = img_dt.hour * 2 * math.pi / 24 if hasattr(img_dt, 'hour') else 0
        wn = (math.sin(week), math.cos(week))
        hn = (math.sin(hour), math.cos(hour))
        
        lat_r = lat * math.pi / 180
        lon_r = lon * math.pi / 180
        ln = (math.sin(lat_r), math.cos(lat_r))
        lo = (math.sin(lon_r), math.cos(lon_r))
        
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
    except Exception as e:
        print(f"  ERROR: CLAY embedding failed: {e}")
        return None


# ---------------------------------------------------------------------------
# PCA SubspaceAD
# ---------------------------------------------------------------------------
def train_subspace(embeddings: np.ndarray) -> PCA:
    """Fit PCA on negative (normal) embeddings."""
    n_samples, n_features = embeddings.shape
    n_components = min(2, n_samples - 1, n_features)  # 2 components max for 3 neg samples
    if n_components < 1:
        n_components = 1
    
    print(f"  Fitting PCA with {n_components} component(s) on {n_samples} samples")
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(embeddings)
    
    var_ratio = float(pca.explained_variance_ratio_.sum())
    print(f"  Explained variance: {var_ratio:.4f}")
    return pca


def score_residual(pca: PCA, embedding: np.ndarray) -> float:
    """Compute PCA reconstruction residual (MSE)."""
    emb_2d = embedding.reshape(1, -1)
    projected = pca.transform(emb_2d)
    reconstructed = pca.inverse_transform(projected)
    residual = float(np.mean((emb_2d - reconstructed) ** 2))
    return residual


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLAY v1.5 + PCA SubspaceAD on real GeoTIFF chips"
    )
    parser.add_argument("--chips-dir", default="data/chips",
                        help="Directory with GeoTIFF chips (default: data/chips)")
    parser.add_argument("--device", default="cpu",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for CLAY inference")
    parser.add_argument("--output-json", default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()
    
    device = resolve_device(args.device)
    chips_dir = Path(args.chips_dir)
    
    if not chips_dir.is_dir():
        print(f"ERROR: Chips directory not found: {chips_dir}")
        return 1
    
    # -----------------------------------------------------------------------
    # 1. Load metadata and model
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("CLAY v1.5 + PCA SubspaceAD — GeoTIFF Evaluation")
    print("=" * 65)
    print(f"Chips dir: {chips_dir}")
    print(f"Device:    {device}")
    
    metadata = Box(yaml.safe_load(open(DEFAULT_METADATA)))
    model = load_clay_model(device)
    
    # -----------------------------------------------------------------------
    # 2. Collect chips by label
    # -----------------------------------------------------------------------
    tif_files = sorted(chips_dir.glob("*.tif"))
    print(f"\nFound {len(tif_files)} GeoTIFF chips")
    
    pos_chips = [f for f in tif_files if f.name.startswith("pos-")]
    neg_chips = [f for f in tif_files if f.name.startswith("neg-")]
    cand_chips = [f for f in tif_files if f.name.startswith("cand-")]
    
    print(f"  Positive: {len(pos_chips)}")
    print(f"  Negative: {len(neg_chips)}")
    print(f"  Candidate: {len(cand_chips)}")
    
    if len(neg_chips) < 2:
        print("ERROR: Need at least 2 negative chips for PCA training")
        return 1
    
    # -----------------------------------------------------------------------
    # 3. Extract embeddings for ALL chips
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("Extracting CLAY embeddings...")
    print(f"{'='*65}")
    
    all_results = []  # list of dicts with filename, label, embedding, etc.
    
    for chip_file in tif_files:
        # Determine label
        if chip_file.name.startswith("pos-"):
            label = "positive"
        elif chip_file.name.startswith("neg-"):
            label = "negative"
        elif chip_file.name.startswith("cand-"):
            label = "candidate"
        else:
            print(f"  SKIP: Unknown prefix: {chip_file.name}")
            continue
        
        # Get location
        loc = get_chip_location(chip_file.name)
        
        # Load pixels
        pixels = load_geotiff(chip_file)
        if pixels is None:
            print(f"  SKIP: {chip_file.name} — failed to load")
            continue
        
        print(f"  {chip_file.name:45s} ({label:>9s}) ({loc['lat']:.4f}, {loc['lon']:.4f})", end="")
        
        # Extract embedding
        emb = get_clay_embedding(model, pixels, loc["lat"], loc["lon"],
                                 loc["datetime"], device, metadata)
        if emb is None:
            print(" — embedding FAILED")
            continue
        
        all_results.append({
            "filename": chip_file.name,
            "label": label,
            "embedding": emb,
            "lat": loc["lat"],
            "lon": loc["lon"],
            "datetime": loc["datetime"].isoformat(),
        })
        print(" — OK")
    
    if len(all_results) < 4:
        print(f"ERROR: Too few successful embeddings ({len(all_results)})")
        return 1
    
    # -----------------------------------------------------------------------
    # 4. Train PCA on negative embeddings
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("Training PCA subspace on negatives...")
    print(f"{'='*65}")
    
    neg_embs = np.array([r["embedding"] for r in all_results if r["label"] == "negative"])
    neg_names = [r["filename"] for r in all_results if r["label"] == "negative"]
    print(f"  Training on {len(neg_embs)} negatives: {neg_names}")
    
    if len(neg_embs) < 2:
        print("ERROR: Need ≥2 negative chips for PCA. Aborting.")
        return 1
    
    pca = train_subspace(neg_embs)
    
    # -----------------------------------------------------------------------
    # 5. Score ALL chips by PCA residual
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("Scoring all chips by PCA reconstruction residual...")
    print(f"{'='*65}")
    
    for r in all_results:
        r["residual"] = score_residual(pca, r["embedding"])
    
    # Sort by residual (high = anomalous = potentially spawn)
    all_results.sort(key=lambda r: r["residual"], reverse=True)
    
    # Assign ranks
    for i, r in enumerate(all_results):
        r["rank"] = i + 1
    
    # -----------------------------------------------------------------------
    # 6. Report results
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("RESULTS")
    print(f"{'='*65}")
    
    header = f"{'Chip':45s} {'Label':>10s} {'Residual':>14s} {'Rank':>6s}"
    print(header)
    print("-" * 65)
    
    for r in all_results:
        label_display = r["label"]
        print(f"{r['filename']:45s} {label_display:>10s} {r['residual']:14.8f} {r['rank']:6d}/{len(all_results)}")
    
    # -----------------------------------------------------------------------
    # 7. Aggregate metrics
    # -----------------------------------------------------------------------
    pos_residuals = [r["residual"] for r in all_results if r["label"] == "positive"]
    neg_residuals = [r["residual"] for r in all_results if r["label"] == "negative"]
    cand_residuals = [r["residual"] for r in all_results if r["label"] == "candidate"]
    
    mean_pos = np.mean(pos_residuals) if pos_residuals else 0.0
    mean_neg = np.mean(neg_residuals) if neg_residuals else 0.0
    mean_cand = np.mean(cand_residuals) if cand_residuals else 0.0
    separation = mean_pos - mean_neg
    
    print(f"\n{'='*65}")
    print("AGGREGATE METRICS")
    print(f"{'='*65}")
    print(f"  Mean positive residual:    {mean_pos:.8f}  (n={len(pos_residuals)})")
    print(f"  Mean negative residual:    {mean_neg:.8f}  (n={len(neg_residuals)})")
    print(f"  Mean candidate residual:   {mean_cand:.8f}  (n={len(cand_residuals)})")
    print(f"  Separation (pos - neg):    {separation:+.8f}")
    
    # AUROC for positive vs negative
    if len(pos_residuals) > 0 and len(neg_residuals) > 0:
        y_true = np.array([1] * len(pos_residuals) + [0] * len(neg_residuals))
        y_score = np.array(pos_residuals + neg_residuals)
        try:
            auroc = roc_auc_score(y_true, y_score)
            print(f"  AUROC (pos vs neg):        {auroc:.4f}")
        except Exception as e:
            print(f"  AUROC: could not compute ({e})")
    else:
        print(f"  AUROC: insufficient data (pos={len(pos_residuals)}, neg={len(neg_residuals)})")
    
    # How many positives rank above median?
    all_residuals = [r["residual"] for r in all_results]
    median_res = np.median(all_residuals)
    pos_above_median = sum(1 for v in pos_residuals if v > median_res)
    print(f"  Positives above median:    {pos_above_median}/{len(pos_residuals)}")
    
    # Check if positives rank above ALL negatives
    if neg_residuals:
        max_neg_res = max(neg_residuals)
        pos_above_all_neg = sum(1 for v in pos_residuals if v > max_neg_res)
        print(f"  Positives above ALL negs:  {pos_above_all_neg}/{len(pos_residuals)}")
    
    # Classification accuracy at best threshold
    if len(pos_residuals) > 0 and len(neg_residuals) > 0:
        all_scores = np.array(pos_residuals + neg_residuals)
        all_labels = np.array([1] * len(pos_residuals) + [0] * len(neg_residuals))
        thresholds = np.linspace(all_scores.min() - 0.001, all_scores.max() + 0.001, 201)
        best_acc = 0.0
        best_thr = 0.0
        for thr in thresholds:
            pred = (all_scores > thr).astype(int)
            acc = np.mean(pred == all_labels)
            if acc > best_acc:
                best_acc = acc
                best_thr = thr
        print(f"  Best accuracy:             {best_acc:.4f} @ thr={best_thr:.8f}")
    
    # -----------------------------------------------------------------------
    # 8. Leave-one-out cross-validation for PCA
    # -----------------------------------------------------------------------
    # The PCA above uses ALL 3 negatives for training and scores all
    # chips. But with 3 training samples and 2 PCA components, the
    # model explains 100% of training variance (memorizes). A more
    # honest evaluation holds out one negative at a time.
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("LEAVE-ONE-OUT PCA CROSS-VALIDATION (more honest)")
    print(f"{'='*65}")
    print("  Train PCA on 2 negatives, score held-out negative + all positives")
    print("  Repeat 3 times (once per held-out negative)")
    print(f"{'-'*65}")
    
    neg_indices = [i for i, r in enumerate(all_results) if r["label"] == "negative"]
    pos_indices_loo = [i for i, r in enumerate(all_results) if r["label"] == "positive"]
    
    loo_separations = []
    loo_aurocs = []
    
    for held_out_idx in neg_indices:
        # Identify training and held-out names
        train_indices = [i for i in neg_indices if i != held_out_idx]
        held_name = all_results[held_out_idx]["filename"]
        train_names = [all_results[i]["filename"] for i in train_indices]
        
        # Build training embeddings
        train_embs = np.array([all_results[i]["embedding"] for i in train_indices])
        
        # Train PCA
        n_train = len(train_indices)
        n_comp = min(1, n_train - 1, 1024)  # 1 component for 2 samples
        if n_comp < 1:
            continue
        
        pca_loo = PCA(n_components=n_comp, random_state=42)
        pca_loo.fit(train_embs)
        var_ratio = float(pca_loo.explained_variance_ratio_.sum())
        
        # Score held-out negative + all positives
        held_res = score_residual(pca_loo, all_results[held_out_idx]["embedding"])
        pos_res_loo = [score_residual(pca_loo, all_results[i]["embedding"]) for i in pos_indices_loo]
        
        sep = np.mean(pos_res_loo) - held_res
        loo_separations.append(sep)
        
        # AUROC for this fold
        y_true_loo = np.array([1] * len(pos_res_loo) + [0])
        y_score_loo = np.array(pos_res_loo + [held_res])
        try:
            auc = roc_auc_score(y_true_loo, y_score_loo)
            loo_aurocs.append(auc)
        except Exception:
            loo_aurocs.append(0.5)
        
        print(f"  Train: {train_names} | Hold: {held_name:<35s} | "
              f"PCA var={var_ratio:.2%} | Sep={sep:+.6f} | AUC={auc:.3f}")
    
    if loo_separations:
        print(f"\n  LOO mean separation: {np.mean(loo_separations):+.6f}")
        print(f"  LOO mean AUROC:      {np.mean(loo_aurocs):.4f}")
    
    # -----------------------------------------------------------------------
    # 9. Also: how does single-image CLAY perform? (cosine sim to mean positive)
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("CLAY SINGLE-IMAGE COSINE SIMILARITY (like DINOv2)")
    print(f"{'='*65}")
    
    if len(pos_residuals) >= 2:
        pos_embs = np.array([r["embedding"] for r in all_results if r["label"] == "positive"])
        mean_pos_emb = np.mean(pos_embs, axis=0)
        mean_pos_emb = mean_pos_emb / (np.linalg.norm(mean_pos_emb) + 1e-10)
        
        cosine_results = []
        for r in all_results:
            e = r["embedding"] / (np.linalg.norm(r["embedding"]) + 1e-10)
            cos_sim = float(np.dot(mean_pos_emb, e))
            cosine_results.append((r["filename"], cos_sim, r["label"]))
        
        cosine_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Chip':45s} {'Label':>10s} {'Cos Sim':>10s} {'Rank':>6s}")
        print("-" * 65)
        for i, (fname, cos_sim, label) in enumerate(cosine_results):
            label_display = label
            print(f"{fname:45s} {label_display:>10s} {cos_sim:10.6f} {i+1:6d}/{len(cosine_results)}")
        
        pos_cos = [c for _, c, l in cosine_results if l == "positive"]
        neg_cos = [c for _, c, l in cosine_results if l == "negative"]
        cand_cos = [c for _, c, l in cosine_results if l == "candidate"]
        
        if neg_cos:
            cos_sep = np.mean(pos_cos) - np.mean(neg_cos)
            print(f"\n  Cosine separation (pos - neg): {cos_sep:+.6f}")
            print(f"  Mean pos cosine: {np.mean(pos_cos):.6f}")
            print(f"  Mean neg cosine: {np.mean(neg_cos):.6f}")
            
            # AUROC for cosine similarity
            y_true = np.array([1] * len(pos_cos) + [0] * len(neg_cos))
            y_score = np.array(pos_cos + neg_cos)
            try:
                auroc_cos = roc_auc_score(y_true, y_score)
                print(f"  AUROC (pos vs neg, cosine): {auroc_cos:.4f}")
            except Exception:
                pass
    
    # -----------------------------------------------------------------------
    # 9. Save results if requested
    # -----------------------------------------------------------------------
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable = []
        for r in all_results:
            serializable.append({
                "filename": r["filename"],
                "label": r["label"],
                "residual": r["residual"],
                "rank": r["rank"],
                "lat": r["lat"],
                "lon": r["lon"],
                "datetime": r["datetime"],
            })
        
        output = {
            "model": "CLAY v1.5 large + PCA SubspaceAD",
            "device": device,
            "n_positive": len(pos_residuals),
            "n_negative": len(neg_residuals),
            "n_candidate": len(cand_residuals),
            "n_pca_components": int(pca.n_components_),
            "mean_pos_residual": float(mean_pos),
            "mean_neg_residual": float(mean_neg),
            "mean_cand_residual": float(mean_cand),
            "separation_pos_neg": float(separation),
            "results": serializable,
        }
        
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to: {out_path}")
    
    print(f"\n{'='*65}")
    print("Evaluation complete.")
    print(f"{'='*65}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
