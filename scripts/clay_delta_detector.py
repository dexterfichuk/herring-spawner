#!/usr/bin/env python3
"""
Clay Embedding Delta Detector for Herring Spawn.

Compares spawn-season vs off-season Sentinel-2 GeoTIFF chips at confirmed
spawn and non-spawn locations. Computes Clay v1.5 embedding deltas and
determines whether delta-based detection separates spawn from non-spawn.

Usage:
    python scripts/clay_delta_detector.py

Output:
    data/review/clay_delta_report.html  — interactive analysis report
    data/chips_delta/                    — cached GeoTIFF pairs
"""
import json, math, re, sys, warnings
from datetime import date, timedelta, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import ee, numpy as np, requests, torch, yaml
import tifffile as tiff
from box import Box
from sklearn.decomposition import PCA
from torchvision.transforms import v2
from claymodel.module import ClayMAEModule

# Suppress noisy tifffile GDAL_NODATA warnings
import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*GDAL_NODATA.*")
ee.Initialize(project="redd-fish")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.set_default_device(device)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BANDS = ["B2", "B3", "B4", "B8"]
BAND_NAMES = ["blue", "green", "red", "nir"]
PLATFORM = "sentinel-2-l2a"
SIZE, GSD = 256, 10

ROSE_LABELS = Path("data/candidates_v2/rose_200_labels.json")
CHIPS_DIR = Path("data/chips_delta")
CHIPS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("data/review/clay_delta_report.html")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

OFF_SEASON_CENTER = date(2024, 7, 15)
OFF_SEASON_WINDOW = 14  # ± days

# ---------------------------------------------------------------------------
# Load metadata & model
# ---------------------------------------------------------------------------
metadata = Box(yaml.safe_load(open("configs/metadata.yaml")))
mean = [metadata[PLATFORM].bands.mean[b] for b in BAND_NAMES]
std = [metadata[PLATFORM].bands.std[b] for b in BAND_NAMES]
waves = [metadata[PLATFORM].bands.wavelength[b] for b in BAND_NAMES]
transform = v2.Compose([v2.Normalize(mean=mean, std=std)])

print("Loading Clay v1.5...")
model = ClayMAEModule.load_from_checkpoint(
    "checkpoints/v1.5/clay-v1.5.ckpt", model_size="large",
    metadata_path="configs/metadata.yaml",
    dolls=[16, 32, 64, 128, 256, 768, 1024],
    doll_weights=[1, 1, 1, 1, 1, 1, 1],
    mask_ratio=0.0, shuffle=False,
)
model.eval().to(device)
print(f"Clay loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")


def normalize_ts(d):
    w = d.isocalendar().week * 2 * math.pi / 52
    h = d.hour * 2 * math.pi / 24 if hasattr(d, 'hour') else 0
    return (math.sin(w), math.cos(w)), (math.sin(h), math.cos(h))


def normalize_ll(lat, lon):
    return (math.sin(lat*math.pi/180), math.cos(lat*math.pi/180)), \
           (math.sin(lon*math.pi/180), math.cos(lon*math.pi/180))


# ---------------------------------------------------------------------------
# Parse rose_200_labels.json to get spawn / non-spawn locations
# ---------------------------------------------------------------------------
def parse_rose_entry(entry: dict):
    """Extract lat, lon, date from a rose_200_labels.json entry filename.
    
    Filename format: {name}_{YYYY-MM-DD}_score{float}_{lat}_{lon}_{YYYYMMDD}.png
    Returns (lat, lon, spawn_date) or None if parsing fails.
    """
    fname = entry.get("filename", "")
    pattern = r'(.+)_(\d{4}-\d{2}-\d{2})_score([\d.]+)_(-?[\d.]+)_(-?[\d.]+)_(\d{8})\.png'
    m = re.match(pattern, fname)
    if not m:
        return None
    name_part, date_str, score_str, lat_str, lon_str, sat_date_str = m.groups()
    try:
        lat = float(lat_str)
        lon = float(lon_str)
        spawn_date = date.fromisoformat(date_str)
        return (lat, lon, spawn_date, name_part, float(score_str))
    except (ValueError, TypeError):
        return None


def deduplicate_locations(locations, decimals=3):
    """Remove entries with nearly identical lat/lon, keeping first occurrence."""
    seen = set()
    result = []
    for loc in locations:
        key = (round(loc[0], decimals), round(loc[1], decimals))
        if key not in seen:
            seen.add(key)
            result.append(loc)
    return result


def load_locations():
    """Load spawn and non-spawn locations from rose_200_labels.json."""
    data = json.loads(ROSE_LABELS.read_text())
    
    spawn_entries = [e for e in data if e.get("spawn") is True]
    nonspawn_entries = [e for e in data if e.get("spawn") is False]
    
    spawn_locs = []
    for e in spawn_entries:
        parsed = parse_rose_entry(e)
        if parsed:
            spawn_locs.append(parsed)
    
    nonspawn_locs = []
    for e in nonspawn_entries:
        parsed = parse_rose_entry(e)
        if parsed:
            nonspawn_locs.append(parsed)
    
    # Deduplicate to avoid redundant downloads
    spawn_locs = deduplicate_locations(spawn_locs)
    nonspawn_locs = deduplicate_locations(nonspawn_locs)
    
    # Use up to 10 non-spawn locations, spread geographically
    # Sort by lat to get geographic spread
    nonspawn_locs.sort(key=lambda x: (x[0], x[1]))
    # Pick evenly spaced entries
    if len(nonspawn_locs) > 10:
        indices = np.linspace(0, len(nonspawn_locs)-1, 10, dtype=int)
        nonspawn_locs = [nonspawn_locs[i] for i in indices]
    
    print(f"Loaded {len(spawn_locs)} unique spawn locations, {len(nonspawn_locs)} non-spawn")
    for lat, lon, sd, name, sc in spawn_locs:
        print(f"  SPAWN: {name} @ ({lat:.4f}, {lon:.4f}) on {sd} (score={sc})")
    for lat, lon, sd, name, sc in nonspawn_locs:
        print(f"  NON-SPAWN: {name} @ ({lat:.4f}, {lon:.4f}) on {sd} (score={sc})")
    
    return spawn_locs, nonspawn_locs


# ---------------------------------------------------------------------------
# GEE chip download
# ---------------------------------------------------------------------------
def download_chip(lat: float, lon: float, center_date: date,
                  window_days: int = 7, label: str = "spawn",
                  cache_path: Path | None = None) -> np.ndarray | None:
    """Download a 256x256 GeoTIFF chip (B2/B3/B4/B8) from GEE.
    
    Searches Sentinel-2 within ±window_days of center_date, cloud < 60%.
    Returns (4, 256, 256) float32 array, or None if no scene found.
    """
    start = (center_date - timedelta(days=window_days)).isoformat()
    end = (center_date + timedelta(days=window_days)).isoformat()
    
    scenes = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ee.Geometry.Point(lon, lat))
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60))
        .sort("CLOUDY_PIXEL_PERCENTAGE"))
    
    ids = scenes.aggregate_array("system:index").getInfo()
    if not ids:
        return None
    
    sid = ids[0]
    region = ee.Geometry.Point(lon, lat).buffer(GSD * SIZE / 2).bounds()
    
    try:
        img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{sid}")
        url = img.select(BANDS).getDownloadURL({
            "region": region, "dimensions": [SIZE, SIZE], "format": "GEO_TIFF",
        })
        resp = requests.get(url, timeout=120)
        # Save to cache path if provided, then read
        if cache_path:
            cache_path.write_bytes(resp.content)
            chip_data = tiff.imread(str(cache_path)).astype(np.float32)
        else:
            # Write to temp file then read (tifffile needs file path for BigTIFF)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            chip_data = tiff.imread(tmp_path).astype(np.float32)
            Path(tmp_path).unlink(missing_ok=True)
        # tifffile reads as (height, width, bands), reshape to (bands, height, width)
        if chip_data.ndim == 3:
            chip_data = np.transpose(chip_data, (2, 0, 1))
        if chip_data.shape != (4, SIZE, SIZE):
            from skimage.transform import resize
            chip_data = np.stack([
                resize(chip_data[i], (SIZE, SIZE), preserve_range=True)
                for i in range(chip_data.shape[0])
            ])
        return chip_data
    except Exception as e:
        print(f"  [WARN] Download failed for ({lat:.4f},{lon:.4f}) {label}: {e}")
        return None


def get_or_download_chip(lat: float, lon: float, center_date: date,
                         window_days: int, label: str, cache_key: str) -> tuple:
    """Get chip from cache or download it. Returns (chip_data, actual_date_str)."""
    cache_path = CHIPS_DIR / f"{cache_key}.tif"
    
    if cache_path.exists():
        chip_data = tiff.imread(str(cache_path)).astype(np.float32)
        if chip_data.ndim == 3:
            chip_data = np.transpose(chip_data, (2, 0, 1))
        if chip_data.shape != (4, SIZE, SIZE):
            from skimage.transform import resize
            chip_data = np.stack([
                resize(chip_data[i], (SIZE, SIZE), preserve_range=True)
                for i in range(chip_data.shape[0])
            ])
        return chip_data, "cached"
    
    chip_data = download_chip(lat, lon, center_date, window_days, label, cache_path)
    if chip_data is not None:
        return chip_data, "downloaded"
    
    return None, "failed"


# ---------------------------------------------------------------------------
# Clay embedding
# ---------------------------------------------------------------------------
def get_clay_embedding(lat: float, lon: float, target_date: date,
                       chip_data: np.ndarray) -> np.ndarray | None:
    """Run Clay v1.5 encoder on a chip and return the [CLS] embedding.
    
    Returns 1024-dim vector (for large model) or None on failure.
    """
    if chip_data is None:
        return None
    
    # Normalize pixels
    pixel_tensor = transform(torch.from_numpy(chip_data))
    
    # Encode time and location
    wn, hn = normalize_ts(dt(target_date.year, target_date.month, target_date.day, 12, 0))
    ln, lo = normalize_ll(lat, lon)
    
    datacube = {
        "platform": PLATFORM,
        "time": torch.tensor(np.hstack([wn, hn]), dtype=torch.float32, device=device).unsqueeze(0),
        "latlon": torch.tensor(np.hstack([ln, lo]), dtype=torch.float32, device=device).unsqueeze(0),
        "pixels": pixel_tensor.unsqueeze(0).to(device),
        "gsd": torch.tensor(GSD, device=device),
        "waves": torch.tensor(waves, device=device),
    }
    
    with torch.no_grad():
        unmsk_patch, _, _, _ = model.model.encoder(datacube)
    emb = unmsk_patch[:, 0, :].cpu().numpy().flatten()
    return emb


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_deltas(spawn_deltas: list, nonspawn_deltas: list,
                   spawn_mags: list, nonspawn_mags: list,
                   spawn_info: list, nonspawn_info: list,
                   spawn_embs: list | None = None,
                   nonspawn_embs: list | None = None,
                   spawn_off_embs: list | None = None,
                   nonspawn_off_embs: list | None = None):
    """Compare spawn vs non-spawn delta vectors and generate report."""
    from scipy import stats as sp_stats
    
    print("\n" + "=" * 70)
    print("CLAY EMBEDDING DELTA ANALYSIS")
    print("=" * 70)
    
    spawn_deltas_arr = np.array(spawn_deltas) if len(spawn_deltas) else np.array([])
    nonspawn_deltas_arr = np.array(nonspawn_deltas) if len(nonspawn_deltas) else np.array([])
    all_deltas = np.vstack([spawn_deltas_arr, nonspawn_deltas_arr]) if len(spawn_deltas) and len(nonspawn_deltas) else (spawn_deltas_arr if len(spawn_deltas) else nonspawn_deltas_arr)
    
    all_mags = np.array(spawn_mags + nonspawn_mags)
    n_spawn = len(spawn_mags)
    n_nonspawn = len(nonspawn_mags)
    
    # Basic stats
    spawn_mean_mag = float(np.mean(spawn_mags)) if n_spawn else 0
    spawn_std_mag = float(np.std(spawn_mags)) if n_spawn else 0
    nonspawn_mean_mag = float(np.mean(nonspawn_mags)) if n_nonspawn else 0
    nonspawn_std_mag = float(np.std(nonspawn_mags)) if n_nonspawn else 0
    
    print(f"\nSpawn delta magnitude: mean={spawn_mean_mag:.4f} ± {spawn_std_mag:.4f} (n={n_spawn})")
    print(f"Non-spawn delta magnitude: mean={nonspawn_mean_mag:.4f} ± {nonspawn_std_mag:.4f} (n={n_nonspawn})")
    
    if n_spawn and n_nonspawn:
        separation_mag = spawn_mean_mag - nonspawn_mean_mag
        print(f"Separation in magnitude (spawn - non-spawn): {separation_mag:.4f}")
        t_stat, p_val = sp_stats.ttest_ind(spawn_mags, nonspawn_mags, equal_var=False)
        print(f"Welch t-test (magnitude): t={t_stat:.3f}, p={p_val:.4f}")
    else:
        separation_mag = 0
        t_stat, p_val = 0, 1.0
    
    # --- Cosine similarity to mean spawn delta ---
    # This is analogous to DINOv2's approach: how aligned is each delta
    # with the "typical spawn delta" direction?
    delta_cos_scores = []
    if n_spawn and n_nonspawn and len(all_deltas) >= 2:
        mean_spawn_delta = np.mean(spawn_deltas_arr, axis=0)
        mean_spawn_delta_norm = mean_spawn_delta / (np.linalg.norm(mean_spawn_delta) + 1e-10)
        
        for d in spawn_deltas_arr:
            d_norm = d / (np.linalg.norm(d) + 1e-10)
            cos_sim = float(np.dot(mean_spawn_delta_norm, d_norm))
            delta_cos_scores.append(("spawn", cos_sim))
        for d in nonspawn_deltas_arr:
            d_norm = d / (np.linalg.norm(d) + 1e-10)
            cos_sim = float(np.dot(mean_spawn_delta_norm, d_norm))
            delta_cos_scores.append(("non-spawn", cos_sim))
        
        spawn_cos = [s for l, s in delta_cos_scores if l == "spawn"]
        nonspawn_cos = [s for l, s in delta_cos_scores if l == "non-spawn"]
        cos_separation = float(np.mean(spawn_cos) - np.mean(nonspawn_cos))
        print(f"\nDelta cosine similarity to mean spawn delta:")
        print(f"  Spawn cos: {np.mean(spawn_cos):.4f} ± {np.std(spawn_cos):.4f}")
        print(f"  Non-spawn cos: {np.mean(nonspawn_cos):.4f} ± {np.std(nonspawn_cos):.4f}")
        print(f"  Separation: {cos_separation:+.4f}")
        if len(spawn_cos) >= 2 and len(nonspawn_cos) >= 2:
            t2, p2 = sp_stats.ttest_ind(spawn_cos, nonspawn_cos, equal_var=False)
            print(f"  Welch t-test: t={t2:.3f}, p={p2:.4f}")
    else:
        spawn_cos = []
        nonspawn_cos = []
        cos_separation = 0
    
    # --- Clay single-image (no delta) classification for comparison ---
    # Use spawn-season embeddings only, similar to DINOv2 approach
    clay_single_separation = 0
    clay_single_info = {}
    if spawn_embs and nonspawn_embs and len(spawn_embs) >= 2 and len(nonspawn_embs) >= 2:
        pos_mean = np.mean(spawn_embs, axis=0)
        pos_mean = pos_mean / (np.linalg.norm(pos_mean) + 1e-10)
        neg_mean = np.mean(nonspawn_embs, axis=0)
        neg_mean = neg_mean / (np.linalg.norm(neg_mean) + 1e-10)
        
        def score_single(emb):
            e = emb / (np.linalg.norm(emb) + 1e-10)
            return float(np.dot(pos_mean, e)) - float(np.dot(neg_mean, e))
        
        spawn_scores = [score_single(e) for e in spawn_embs]
        nonspawn_scores = [score_single(e) for e in nonspawn_embs]
        clay_single_separation = float(np.mean(spawn_scores) - np.mean(nonspawn_scores))
        print(f"\nClay single-image (spawn-season only, like DINOv2):")
        print(f"  Spawn score: {np.mean(spawn_scores):.4f} ± {np.std(spawn_scores):.4f}")
        print(f"  Non-spawn score: {np.mean(nonspawn_scores):.4f} ± {np.std(nonspawn_scores):.4f}")
        print(f"  Separation: {clay_single_separation:+.4f}")
        clay_single_info = {
            "spawn_scores": spawn_scores,
            "nonspawn_scores": nonspawn_scores,
            "separation": clay_single_separation,
        }
    
    # PCA on all delta vectors
    if len(all_deltas) >= 3:
        pca = PCA(n_components=min(3, len(all_deltas)))
        pcs = pca.fit_transform(all_deltas)
        var_explained = pca.explained_variance_ratio_.tolist()
        print(f"\nPCA of delta vectors:")
        for i, vr in enumerate(var_explained):
            print(f"  PC{i+1}: {vr:.3f} ({vr*100:.1f}%)")
        
        pc1_coeffs = np.abs(pca.components_[0])
        top_dim_indices = np.argsort(pc1_coeffs)[-10:][::-1].tolist()
    else:
        pcs = np.zeros((len(all_deltas), 2))
        var_explained = [0, 0]
        top_dim_indices = []
    
    # Summary
    print(f"\nTotal spawn locations processed: {n_spawn}/{len(spawn_info)}")
    print(f"Total non-spawn locations processed: {n_nonspawn}/{len(nonspawn_info)}")
    
    return {
        "spawn_mags": spawn_mags,
        "nonspawn_mags": nonspawn_mags,
        "spawn_cos": spawn_cos,
        "nonspawn_cos": nonspawn_cos,
        "cos_separation": cos_separation,
        "spawn_info": spawn_info,
        "nonspawn_info": nonspawn_info,
        "all_deltas": all_deltas,
        "pcs": pcs,
        "var_explained": var_explained,
        "top_dim_indices": top_dim_indices,
        "separation_mag": separation_mag,
        "clay_single_separation": clay_single_separation,
        "clay_single_info": clay_single_info,
        "t_stat": float(t_stat),
        "p_val": float(p_val),
    }


# ---------------------------------------------------------------------------
# HTML Report Generation
# ---------------------------------------------------------------------------
def generate_report(stats: dict, spawn_deltas_list, nonspawn_deltas_list):
    """Generate interactive HTML report with Plotly."""
    spawn_mags = stats["spawn_mags"]
    nonspawn_mags = stats["nonspawn_mags"]
    spawn_cos = stats.get("spawn_cos", [])
    nonspawn_cos = stats.get("nonspawn_cos", [])
    spawn_info = stats["spawn_info"]
    nonspawn_info = stats["nonspawn_info"]
    pcs = stats["pcs"]
    var_explained = stats["var_explained"]
    clay_single_info = stats.get("clay_single_info", {})
    
    # Build labels and hover texts for PCA
    pca_labels = ["spawn"] * len(spawn_deltas_list) + ["non-spawn"] * len(nonspawn_deltas_list)
    pca_hover = []
    for lat, lon, name in spawn_info:
        pca_hover.append(f"Spawn: {name}<br>({lat:.4f}, {lon:.4f})")
    for lat, lon, name in nonspawn_info:
        pca_hover.append(f"Non-spawn: {name}<br>({lat:.4f}, {lon:.4f})")
    
    # Box plot data for magnitude comparison
    box_data_json = json.dumps({
        "spawn": [float(m) for m in spawn_mags],
        "nonspawn": [float(m) for m in nonspawn_mags],
    })
    
    # Box plot data for cosine similarity comparison
    cos_box_json = json.dumps({
        "spawn": [float(c) for c in spawn_cos],
        "nonspawn": [float(c) for c in nonspawn_cos],
    }) if spawn_cos and nonspawn_cos else json.dumps({"spawn": [], "nonspawn": []})
    
    # PCA scatter data
    pca_data_json = json.dumps([
        {"pc1": float(pcs[i, 0]), "pc2": float(pcs[i, 1]), "label": pca_labels[i], "hover": pca_hover[i]}
        for i in range(len(pca_labels))
    ])
    
    # Delta magnitude per location
    loc_entries = []
    for i, (lat, lon, name) in enumerate(spawn_info):
        if i < len(spawn_mags):
            loc_entries.append({"name": name, "lat": lat, "lon": lon, "delta_mag": float(spawn_mags[i]), "type": "spawn"})
    for i, (lat, lon, name) in enumerate(nonspawn_info):
        if i < len(nonspawn_mags):
            loc_entries.append({"name": name, "lat": lat, "lon": lon, "delta_mag": float(nonspawn_mags[i]), "type": "non-spawn"})
    
    loc_data_json = json.dumps(loc_entries)
    
    # DINOv2 comparison reference
    dinov2_separation = 0.0607  # from AGENTS.md
    dinov2_accuracy = 88.9
    sep_mag = stats["separation_mag"]
    cos_sep = stats["cos_separation"]
    clay_single_sep = stats["clay_single_separation"]
    p_val = stats["p_val"]
    
    # Compute best accuracy from delta magnitudes
    best_acc_str = ""
    if len(spawn_mags) > 0 and len(nonspawn_mags) > 0:
        overall_mags = np.array(spawn_mags + nonspawn_mags)
        overall_labels = np.array(["spawn"] * len(spawn_mags) + ["non-spawn"] * len(nonspawn_mags))
        thresholds = np.linspace(min(overall_mags), max(overall_mags), 100)
        best_acc = 0
        best_thresh = 0
        for thresh in thresholds:
            pred = np.where(overall_mags > thresh, "spawn", "non-spawn")
            acc = np.mean(pred == overall_labels)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        spawn_above = sum(1 for m in spawn_mags if m > best_thresh)
        nonspawn_below = sum(1 for m in nonspawn_mags if m <= best_thresh)
        best_acc_str = f"""
<h3>Delta Magnitude Discrimination</h3>
<ul>
    <li><strong>Optimal threshold:</strong> {best_thresh:.4f}</li>
    <li><strong>Accuracy at threshold:</strong> {best_acc*100:.1f}% ({spawn_above + nonspawn_below}/{len(spawn_mags) + len(nonspawn_mags)})</li>
    <li><strong>Spawn above threshold:</strong> {spawn_above}/{len(spawn_mags)}</li>
    <li><strong>Non-spawn below threshold:</strong> {nonspawn_below}/{len(nonspawn_mags)}</li>
</ul>"""
    
    # Determine best approach
    approaches = [
        ("Clay delta (L2 norm)", sep_mag),
        ("Clay delta (cosine sim)", cos_sep),
        ("Clay single-image", clay_single_sep),
        ("DINOv2 single-image", dinov2_separation),
    ]
    best = max(approaches, key=lambda x: x[1])
    
    html = f"""<!DOCTYPE html>
<html><head><title>Clay Embedding Delta Report</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px; max-width: 1100px; margin: auto; background: #fafafa; color: #222; }}
h1 {{ color: #1a3a5c; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #2c5f8a; margin-top: 30px; }}
.plot {{ background: white; border-radius: 8px; padding: 15px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
th, td {{ padding: 10px 14px; border-bottom: 1px solid #eee; text-align: left; }}
th {{ background: #2c5f8a; color: white; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
.highlight {{ background: #fff3cd; padding: 2px 6px; border-radius: 4px; }}
.summary-box {{ background: white; border-left: 4px solid #3498db; padding: 15px 20px; margin: 20px 0; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
.badge-spawn {{ background: #27ae60; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; }}
.badge-nonspawn {{ background: #e74c3c; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; }}
.metric {{ font-size: 1.6em; font-weight: bold; color: #2c5f8a; }}
.metric-label {{ font-size: 0.85em; color: #666; }}
.metric-row {{ display: flex; gap: 30px; flex-wrap: wrap; margin: 20px 0; }}
.metric-card {{ background: white; padding: 20px 25px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); flex: 1; min-width: 180px; text-align: center; }}
</style></head><body>
<h1>Clay v1.5 Embedding Delta Analysis</h1>
<p>Comparing spawn-season vs off-season Sentinel-2 chips at confirmed spawn and non-spawn locations.</p>

<div class="summary-box">
<h3>Method</h3>
<p>For each location, we download <strong>spawn-season</strong> (March 2024, ±7 days) and <strong>off-season</strong> (July 2024, ±14 days) GeoTIFF chips (B2/B3/B4/B8, 256×256 at 10m GSD). Both chips are embedded through Clay v1.5 (large model), producing 1024-dim vectors. The <strong>delta vector</strong> = spawn_emb − off_emb captures spectral change between seasons. Spawn locations should show larger/more directional deltas than non-spawn locations.</p>
</div>

<h2>Results Summary</h2>
<div class="metric-row">
    <div class="metric-card">
        <div class="metric">{np.mean(spawn_mags):.4f}</div>
        <div class="metric-label">Mean Spawn Δ Mag</div>
    </div>
    <div class="metric-card">
        <div class="metric">{np.mean(nonspawn_mags):.4f}</div>
        <div class="metric-label">Mean Non-spawn Δ Mag</div>
    </div>
    <div class="metric-card">
        <div class="metric" style="color: {'#27ae60' if sep_mag > 0 else '#e74c3c'}">{sep_mag:+.4f}</div>
        <div class="metric-label">Δ Mag Separation</div>
    </div>
    <div class="metric-card">
        <div class="metric">{p_val:.4f}</div>
        <div class="metric-label">t-test p-value</div>
    </div>
    <div class="metric-card">
        <div class="metric">{cos_sep:+.4f}</div>
        <div class="metric-label">Δ Cosine Separation</div>
    </div>
</div>

<h2>Method Comparison</h2>
<div class="summary-box">
<table>
<tr><th>Method</th><th>Separation</th><th>Description</th></tr>
<tr><td><strong>Clay delta (L2 norm)</strong></td><td>{sep_mag:+.4f}</td><td>L2 norm of (spawn_emb − off_emb)</td></tr>
<tr><td><strong>Clay delta (cosine sim)</strong></td><td>{cos_sep:+.4f}</td><td>Cosine sim of delta to mean spawn delta</td></tr>
<tr><td><strong>Clay single-image</strong></td><td>{clay_single_sep:+.4f}</td><td>Same as DINOv2: spawn-season emb only, pos−neg similarity</td></tr>
<tr><td><strong>DINOv2 single-image</strong></td><td>{dinov2_separation:+.4f}</td><td>From AGENTS.md: cosine sim to mean pos − mean neg (88.9% acc)</td></tr>
<tr style="background: #d4edda;"><td><strong>🥇 Best: {best[0]}</strong></td><td><strong>{best[1]:+.4f}</strong></td><td></td></tr>
</table>
</div>

<h2>Delta Magnitude Comparison</h2>
<div class="plot" id="boxplot"></div>

<h2>Delta Cosine Similarity to Mean Spawn Delta</h2>
<div class="plot" id="cosplot"></div>
<p>How aligned is each delta vector with the "typical spawn delta" direction? Higher is more spawn-like.</p>

<h2>PCA of Delta Vectors</h2>
<div class="plot" id="pcascatter"></div>
<p>PCA on all delta vectors. Good separation would show spawn (green) and non-spawn (red) in distinct clusters.</p>

<h2>Per-Location Delta Magnitudes</h2>
<div class="plot" id="barchart"></div>

{best_acc_str}

<table>
<tr><th>Location</th><th>Type</th><th>Lat</th><th>Lon</th><th>Delta Magnitude</th><th>Delta Cos (to mean)</th></tr>
"""
    for entry in sorted(loc_entries, key=lambda x: x["delta_mag"], reverse=True):
        badge = '<span class="badge-spawn">spawn</span>' if entry["type"] == "spawn" else '<span class="badge-nonspawn">non-spawn</span>'
        html += f"""<tr><td>{entry['name']}</td><td>{badge}</td><td>{entry['lat']:.4f}</td><td>{entry['lon']:.4f}</td><td>{entry['delta_mag']:.4f}</td><td>—</td></tr>
"""
    
    html += f"""</table>

<h2>Next Steps</h2>
<ul>
    <li>Increase spawn sample size beyond 6 points — more locations would improve statistical power</li>
    <li>Try delta cosine similarity to mean spawn delta as an alternative scoring metric</li>
    <li>Investigate whether certain Clay embedding dimensions correspond to NIR/turbidity spectral signatures</li>
    <li>Compare with actual per-band spectral indices (NDVI, NDWI) for the same chips</li>
    <li>Collect more off-season imagery from multiple summers to reduce seasonal noise</li>
</ul>

<h2>Limitations</h2>
<ul>
    <li>Small spawn sample: only {len(spawn_mags)} unique confirmed spawn locations processed</li>
    <li>Cloud cover and scene availability limit chip quality — some locations use suboptimal scenes</li>
    <li>Off-season window (July 2024) may also have turbidity from other sources (sediment, plankton blooms)</li>
    <li>Clay embedding dimensions are not directly interpretable as spectral bands</li>
    <li>Non-spawn locations from rose_200_labels.json are DINOv2-based predictions, not ground truth</li>
</ul>

<script>
// Box plot - magnitude
var boxData = {box_data_json};
var boxTrace1 = {{
    y: boxData.spawn, type: 'box', name: 'Spawn', marker: {{color: '#27ae60'}},
    boxmean: 'sd', hovertemplate: 'Spawn delta: %{{y:.4f}}<extra></extra>'
}};
var boxTrace2 = {{
    y: boxData.nonspawn, type: 'box', name: 'Non-spawn', marker: {{color: '#e74c3c'}},
    boxmean: 'sd', hovertemplate: 'Non-spawn delta: %{{y:.4f}}<extra></extra>'
}};
var boxLayout = {{
    title: 'Delta Magnitude: Spawn vs Non-Spawn',
    yaxis: {{title: '||Δ embedding|| (L2 norm)'}},
    width: 700, height: 450,
    plot_bgcolor: 'rgba(0,0,0,0)', paper_bgcolor: 'rgba(0,0,0,0)',
    margin: {{l: 60, r: 30, t: 50, b: 60}},
    font: {{family: '-apple-system, BlinkMacSystemFont, sans-serif'}}
}};
Plotly.newPlot('boxplot', [boxTrace1, boxTrace2], boxLayout, {{responsive: true}});

// Box plot - cosine similarity
var cosData = {cos_box_json};
var cosTrace1 = {{
    y: cosData.spawn, type: 'box', name: 'Spawn', marker: {{color: '#27ae60'}},
    boxmean: 'sd', hovertemplate: 'Spawn delta cos: %{{y:.4f}}<extra></extra>'
}};
var cosTrace2 = {{
    y: cosData.nonspawn, type: 'box', name: 'Non-spawn', marker: {{color: '#e74c3c'}},
    boxmean: 'sd', hovertemplate: 'Non-spawn delta cos: %{{y:.4f}}<extra></extra>'
}};
var cosLayout = {{
    title: 'Delta Cosine Similarity to Mean Spawn Delta',
    yaxis: {{title: 'cos(Δ, mean_spawn_Δ)'}},
    width: 700, height: 450,
    plot_bgcolor: 'rgba(0,0,0,0)', paper_bgcolor: 'rgba(0,0,0,0)',
    margin: {{l: 60, r: 30, t: 50, b: 60}},
    font: {{family: '-apple-system, BlinkMacSystemFont, sans-serif'}}
}};
Plotly.newPlot('cosplot', [cosTrace1, cosTrace2], cosLayout, {{responsive: true}});

// PCA scatter
var pcaData = {pca_data_json};
var spawnPca = pcaData.filter(function(d) {{ return d.label === 'spawn'; }});
var nonspawnPca = pcaData.filter(function(d) {{ return d.label === 'non-spawn'; }});
var pcaTrace1 = {{
    x: spawnPca.map(function(d) {{ return d.pc1; }}),
    y: spawnPca.map(function(d) {{ return d.pc2; }}),
    mode: 'markers', type: 'scatter', name: 'Spawn',
    text: spawnPca.map(function(d) {{ return d.hover; }}),
    hovertemplate: '%{{text}}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>',
    marker: {{size: 14, color: '#27ae60', symbol: 'circle'}}
}};
var pcaTrace2 = {{
    x: nonspawnPca.map(function(d) {{ return d.pc1; }}),
    y: nonspawnPca.map(function(d) {{ return d.pc2; }}),
    mode: 'markers', type: 'scatter', name: 'Non-spawn',
    text: nonspawnPca.map(function(d) {{ return d.hover; }}),
    hovertemplate: '%{{text}}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>',
    marker: {{size: 14, color: '#e74c3c', symbol: 'square'}}
}};
var pcaLayout = {{
    title: 'PCA of Delta Vectors (PC1 vs PC2)',
    xaxis: {{title: 'PC1 ({var_explained[0]*100:.1f}%)'}},
    yaxis: {{title: 'PC2 ({var_explained[1]*100:.1f}%)'}},
    width: 700, height: 500,
    plot_bgcolor: 'rgba(0,0,0,0)', paper_bgcolor: 'rgba(0,0,0,0)',
    margin: {{l: 60, r: 30, t: 50, b: 60}},
    font: {{family: '-apple-system, BlinkMacSystemFont, sans-serif'}},
    legend: {{x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)'}}
}};
Plotly.newPlot('pcascatter', [pcaTrace1, pcaTrace2], pcaLayout, {{responsive: true}});

// Bar chart
var locData = {loc_data_json};
var locNames = locData.map(function(d) {{ return d.name; }});
var locMags = locData.map(function(d) {{ return d.delta_mag; }});
var locColors = locData.map(function(d) {{ return d.type === 'spawn' ? '#27ae60' : '#e74c3c'; }});
var barTrace = {{
    x: locNames, y: locMags, type: 'bar',
    marker: {{color: locColors}},
    text: locData.map(function(d) {{ return d.type + ': ' + d.delta_mag.toFixed(4); }}),
    hovertemplate: '%{{text}}<extra></extra>',
}};
var barLayout = {{
    title: 'Delta Magnitude by Location',
    yaxis: {{title: '||Δ embedding||'}},
    xaxis: {{title: 'Location', tickangle: -45}},
    width: 900, height: 450,
    plot_bgcolor: 'rgba(0,0,0,0)', paper_bgcolor: 'rgba(0,0,0,0)',
    margin: {{l: 60, r: 30, t: 50, b: 120}},
    font: {{family: '-apple-system, BlinkMacSystemFont, sans-serif'}}
}};
Plotly.newPlot('barchart', [barTrace], barLayout, {{responsive: true}});
</script>

</body></html>"""
    
    REPORT_PATH.write_text(html)
    print(f"\nReport saved: file://{REPORT_PATH.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CLAY EMBEDDING DELTA DETECTOR")
    print("=" * 70)
    
    # 1. Load locations
    spawn_locs, nonspawn_locs = load_locations()
    
    if len(spawn_locs) == 0 or len(nonspawn_locs) == 0:
        print("ERROR: Not enough locations found. Check rose_200_labels.json format.")
        sys.exit(1)
    
    # 2. Process locations
    spawn_deltas, spawn_mags, spawn_info = [], [], []
    spawn_embeds, spawn_off_embeds = [], []  # for single-image comparison
    nonspawn_deltas, nonspawn_mags, nonspawn_info = [], [], []
    nonspawn_embeds, nonspawn_off_embeds = [], []
    
    for idx, (lat, lon, spawn_date, name, score) in enumerate(spawn_locs):
        print(f"\n[{idx+1}/{len(spawn_locs)}] SPAWN: {name} ({lat:.4f}, {lon:.4f})")
        
        cache_key = f"{name}_spawn_{spawn_date.isoformat()}_{lat}_{lon}".replace(".", "_").replace("-", "_")
        spawn_cache_key = f"{cache_key}_mar"
        off_cache_key = f"{cache_key}_jul"
        
        # Spawn-season chip: around the known spawn date
        spawn_chip, spawn_status = get_or_download_chip(
            lat, lon, spawn_date, window_days=7, label="spawn",
            cache_key=spawn_cache_key)
        print(f"  Spawn chip: {spawn_status}")
        
        # Off-season chip: July 2024
        off_chip, off_status = get_or_download_chip(
            lat, lon, OFF_SEASON_CENTER, window_days=OFF_SEASON_WINDOW, label="off",
            cache_key=off_cache_key)
        print(f"  Off chip: {off_status}")
        
        if spawn_chip is None or off_chip is None:
            print(f"  SKIP: missing chip data")
            continue
        
        # Get embeddings
        emb_spawn = get_clay_embedding(lat, lon, spawn_date, spawn_chip)
        emb_off = get_clay_embedding(lat, lon, OFF_SEASON_CENTER, off_chip)
        
        if emb_spawn is None or emb_off is None:
            print(f"  SKIP: embedding failed")
            continue
        
        # Compute delta
        delta = emb_spawn - emb_off
        delta_mag = float(np.linalg.norm(delta))
        
        spawn_deltas.append(delta)
        spawn_mags.append(delta_mag)
        spawn_info.append((lat, lon, name))
        spawn_embeds.append(emb_spawn)
        spawn_off_embeds.append(emb_off)
        print(f"  Delta magnitude: {delta_mag:.4f}")
    
    for idx, (lat, lon, ref_date, name, score) in enumerate(nonspawn_locs):
        print(f"\n[{idx+1}/{len(nonspawn_locs)}] NON-SPAWN: {name} ({lat:.4f}, {lon:.4f})")
        
        cache_key = f"{name}_nonspawn_{lat}_{lon}".replace(".", "_").replace("-", "_")
        spawn_cache_key = f"{cache_key}_mar"
        off_cache_key = f"{cache_key}_jul"
        
        # "Spawn-season" chip for non-spawn: March 15, 2024
        spawn_chip, spawn_status = get_or_download_chip(
            lat, lon, date(2024, 3, 15), window_days=7, label="nonspawn_mar",
            cache_key=spawn_cache_key)
        print(f"  March chip: {spawn_status}")
        
        # Off-season chip: July 2024
        off_chip, off_status = get_or_download_chip(
            lat, lon, OFF_SEASON_CENTER, window_days=OFF_SEASON_WINDOW, label="off",
            cache_key=off_cache_key)
        print(f"  July chip: {off_status}")
        
        if spawn_chip is None or off_chip is None:
            print(f"  SKIP: missing chip data")
            continue
        
        emb_spawn = get_clay_embedding(lat, lon, date(2024, 3, 15), spawn_chip)
        emb_off = get_clay_embedding(lat, lon, OFF_SEASON_CENTER, off_chip)
        
        if emb_spawn is None or emb_off is None:
            print(f"  SKIP: embedding failed")
            continue
        
        delta = emb_spawn - emb_off
        delta_mag = float(np.linalg.norm(delta))
        
        nonspawn_deltas.append(delta)
        nonspawn_mags.append(delta_mag)
        nonspawn_info.append((lat, lon, name))
        nonspawn_embeds.append(emb_spawn)
        nonspawn_off_embeds.append(emb_off)
        print(f"  Delta magnitude: {delta_mag:.4f}")
    
    # 3. Analyze
    print("\n" + "=" * 70)
    stats = analyze_deltas(
        spawn_deltas, nonspawn_deltas,
        spawn_mags, nonspawn_mags,
        spawn_info, nonspawn_info,
        spawn_embs=spawn_embeds if spawn_embeds else None,
        nonspawn_embs=nonspawn_embeds if nonspawn_embeds else None)
    
    # 4. Generate report
    generate_report(stats, spawn_deltas, nonspawn_deltas)
    
    # 5. Save data for reproducibility
    out = {
        "spawn_delta_mags": spawn_mags,
        "nonspawn_delta_mags": nonspawn_mags,
        "spawn_info": [{"lat": s[0], "lon": s[1], "name": s[2]} for s in spawn_info],
        "nonspawn_info": [{"lat": s[0], "lon": s[1], "name": s[2]} for s in nonspawn_info],
        "separation_mag": stats["separation_mag"],
        "cos_separation": stats["cos_separation"],
        "clay_single_separation": stats["clay_single_separation"],
        "p_val": stats["p_val"],
    }
    Path("data/embeddings/clay_delta_results.json").write_text(json.dumps(out, indent=2))
    print(f"Data saved to data/embeddings/clay_delta_results.json")


if __name__ == "__main__":
    main()
