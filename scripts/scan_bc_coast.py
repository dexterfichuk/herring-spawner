#!/usr/bin/env python3
"""Scan BC coastline for new herring spawn events using DINOv2 scoring.

Generates grid points in defined herring habitat regions, searches Sentinel-2
via Google Earth Engine during spawn season (Feb-Apr), downloads RGB thumbnails,
scores against trained DINOv2 reference vectors, and saves candidates above threshold.

Usage:
    python scripts/scan_bc_coast.py \\
        --output data/candidates \\
        --threshold 0.0 \\
        --start 2024-02-01 \\
        --end 2024-04-30 \\
        --max-cloud 50 \\
        --grid-spacing 0.01

Candidates are saved as PNGs in the output directory. Non-candidates are never
stored to disk. A manifest.json tracks all saved candidates.
"""
import argparse
import hashlib
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# GEE is imported lazily — it can fail if not authenticated, so we only init
# it after CLI parsing and early validation pass.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DINOv2 configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "dinov2_vits14"
EMBED_DIM = 384

# ---------------------------------------------------------------------------
# Herring habitat regions (sheltered bays/inlets along BC coast)
# ---------------------------------------------------------------------------
REGIONS: list[dict[str, Any]] = [
    # Strait of Georgia (most active spawn region)
    {"name": "qualicum", "lat": 49.35, "lon": -124.45, "radius_km": 15},
    {"name": "nanaimo", "lat": 49.15, "lon": -123.85, "radius_km": 15},
    {"name": "comox", "lat": 49.68, "lon": -124.88, "radius_km": 15},
    {"name": "denman-island", "lat": 49.55, "lon": -124.80, "radius_km": 10},
    # WCVI
    {"name": "tofino", "lat": 49.15, "lon": -125.90, "radius_km": 15},
    {"name": "ucluelet", "lat": 48.94, "lon": -125.55, "radius_km": 10},
    {"name": "nootka-sound", "lat": 49.60, "lon": -126.60, "radius_km": 15},
    {"name": "quatsino-sound", "lat": 50.50, "lon": -128.00, "radius_km": 15},
    # Central Coast
    {"name": "spiller-channel", "lat": 52.30, "lon": -128.30, "radius_km": 15},
    {"name": "milbanke-sound", "lat": 52.50, "lon": -128.80, "radius_km": 15},
    # North Coast
    {"name": "prince-rupert", "lat": 54.30, "lon": -130.40, "radius_km": 20},
    # Haida Gwaii
    {"name": "haida-gwaii-south", "lat": 52.40, "lon": -131.40, "radius_km": 15},
    {"name": "masset-inlet", "lat": 53.70, "lon": -132.90, "radius_km": 15},
]

# ---------------------------------------------------------------------------
# Image transform for DINOv2
# ---------------------------------------------------------------------------
DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ===================================================================
# Reference vector computation
# ===================================================================

def _samples_hash(pos_dir: Path, neg_dir: Path) -> str:
    """Return a deterministic hash of the sample directory contents for cache invalidation."""
    pos_files = sorted(pos_dir.glob("*.png"))
    neg_files = sorted(neg_dir.glob("*.png"))
    
    hasher = hashlib.md5()
    hasher.update(f"{len(pos_files)}-{len(neg_files)}".encode())
    for p in pos_files + neg_files:
        try:
            mtime = p.stat().st_mtime_ns
            hasher.update(f"|{p.name}|{mtime}".encode())
        except OSError:
            hasher.update(f"|{p.name}".encode())
    
    return hasher.hexdigest()


def compute_reference_vectors(
    pos_dir: Path,
    neg_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute mean positive and negative embedding vectors from labeled samples.
    
    Returns:
        (mean_pos, mean_neg, error_log) where error_log contains any sample names
        that failed to load.
    """
    pos_embs: list[np.ndarray] = []
    neg_embs: list[np.ndarray] = []
    errors: list[str] = []

    for _label, search_dir, dest_list in [
        ("positive", pos_dir, pos_embs),
        ("negative", neg_dir, neg_embs),
    ]:
        paths = sorted(search_dir.glob("*.png"))
        if not paths:
            print(f"  WARNING: No {_label} samples found in {search_dir}")
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                tensor = DINO_TRANSFORM(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(tensor)
                emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
                dest_list.append(emb)
            except Exception as exc:
                errors.append(f"{p.name}: {exc}")

    if not pos_embs:
        raise RuntimeError(
            f"No valid positive samples found in {pos_dir}. "
            "Cannot compute reference vectors."
        )
    if not neg_embs:
        raise RuntimeError(
            f"No valid negative samples found in {neg_dir}. "
            "Cannot compute reference vectors."
        )

    mean_pos = np.mean(pos_embs, axis=0)
    mean_pos = mean_pos / np.linalg.norm(mean_pos)
    mean_neg = np.mean(neg_embs, axis=0)
    mean_neg = mean_neg / np.linalg.norm(mean_neg)

    # Compute separation on training set
    pos_scores = [float(np.dot(mean_pos, e) - np.dot(mean_neg, e)) for e in pos_embs]
    neg_scores = [float(np.dot(mean_pos, e) - np.dot(mean_neg, e)) for e in neg_embs]
    separation = float(np.mean(pos_scores) - np.mean(neg_scores))

    print(f"  Reference vectors computed: {len(pos_embs)} positive, {len(neg_embs)} negative")
    print(f"  Training separation: {separation:.4f}")
    if errors:
        print(f"  WARNING: {len(errors)} samples failed to load (see errors list in return value)")
        for err in errors:
            print(f"    - {err}")

    return mean_pos, mean_neg, errors


def load_or_compute_reference_vectors(
    pos_dir: Path,
    neg_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached reference vectors or compute them from scratch."""
    current_hash = _samples_hash(pos_dir, neg_dir)
    cache_meta = cache_path.with_suffix(".meta.json")

    # Try loading from cache
    if cache_path.exists() and cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text())
            if meta.get("hash") == current_hash:
                data = np.load(cache_path)
                mean_pos = data["mean_pos"]
                mean_neg = data["mean_neg"]
                print(f"  Loaded cached reference vectors from {cache_path}")
                print(f"  Training separation (from cache): {meta.get('separation', 'N/A')}")
                return mean_pos, mean_neg
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            print(f"  Cache invalid ({exc}), recomputing...")

    # Compute from scratch
    mean_pos, mean_neg, errors = compute_reference_vectors(pos_dir, neg_dir, model, device)

    # Compute separation
    pos_embs: list[np.ndarray] = []
    neg_embs: list[np.ndarray] = []
    for _lb, search_dir, dest_list in [
        ("positive", pos_dir, pos_embs),
        ("negative", neg_dir, neg_embs),
    ]:
        for p in sorted(search_dir.glob("*.png")):
            try:
                img = Image.open(p).convert("RGB")
                tensor = DINO_TRANSFORM(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(tensor)
                emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
                dest_list.append(emb)
            except Exception:
                pass

    pos_scores = []
    if pos_embs:
        pos_scores = [float(np.dot(mean_pos, e) - np.dot(mean_neg, e)) for e in pos_embs]
    neg_scores = []
    if neg_embs:
        neg_scores = [float(np.dot(mean_pos, e) - np.dot(mean_neg, e)) for e in neg_embs]

    separation = 0.0
    if pos_scores and neg_scores:
        separation = float(np.mean(pos_scores) - np.mean(neg_scores))

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, mean_pos=mean_pos, mean_neg=mean_neg)
    cache_meta.write_text(
        json.dumps({"hash": current_hash, "separation": round(separation, 4)}, indent=2)
    )
    print(f"  Cached reference vectors to {cache_path}")

    return mean_pos, mean_neg


# ===================================================================
# Grid point generation
# ===================================================================

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

        # Determine number of steps — avoid degenerate grids
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

                # Distance check: keep points within circular buffer
                dlat = (p_lat - lat) * 111.0
                dlon = (p_lon - lon) * 111.0 * math.cos(math.radians(lat))
                dist_km = math.sqrt(dlat**2 + dlon**2)
                if dist_km <= radius_km:
                    points.append({
                        "region": region["name"],
                        "lat": round(p_lat, 6),
                        "lon": round(p_lon, 6),
                    })
                    region_points += 1

        total_estimated += region_points
        print(f"  {region['name']}: {region_points} grid points")

    return points


# ===================================================================
# GEE scene search
# ===================================================================

def find_best_scene(
    ee_module: Any,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_cloud: float,
) -> dict[str, Any] | None:
    """Find the single best Sentinel-2 scene for a point.
    
    Returns dict with keys: scene_id, cloud, date, or None if no scene found.
    """
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
            "lat": lat,
            "lon": lon,
        }
    except Exception as exc:
        print(f"    GEE search error at ({lat:.4f}, {lon:.4f}): {exc}")
        return None


# ===================================================================
# Thumbnail download
# ===================================================================

def download_thumbnail(
    ee_module: Any,
    lat: float,
    lon: float,
    scene_id: str,
) -> bytes | None:
    """Download a 512×512 RGB thumbnail from a Sentinel-2 scene.
    
    Returns raw PNG bytes, or None on failure.
    """
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
        print(f"    Download failed for {scene_id}: {exc}")
        return None
    except Exception as exc:
        print(f"    Thumbnail error for {scene_id}: {exc}")
        return None


# ===================================================================
# DINOv2 scoring
# ===================================================================

def score_thumbnail(
    png_bytes: bytes,
    model: torch.nn.Module,
    device: torch.device,
    mean_pos: np.ndarray,
    mean_neg: np.ndarray,
) -> float | None:
    """Compute spawn score for a PNG thumbnail.
    
    Score = cosine_similarity(embedding, mean_pos) - cosine_similarity(embedding, mean_neg)
    
    Returns score or None if image processing fails.
    """
    try:
        img = Image.open(io_bytes(png_bytes)).convert("RGB")
        tensor = DINO_TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(tensor)
        emb = F.normalize(emb, dim=1).cpu().numpy().flatten()

        pos_sim = float(np.dot(mean_pos, emb))
        neg_sim = float(np.dot(mean_neg, emb))
        return pos_sim - neg_sim
    except Exception as exc:
        print(f"    Scoring error: {exc}")
        return None


def io_bytes(data: bytes) -> "Image.Image":
    """Helper to create PIL Image from bytes without a temp file."""
    import io
    return io.BytesIO(data)


# ===================================================================
# Candidate storage
# ===================================================================

def save_candidate(
    output_dir: Path,
    png_bytes: bytes,
    info: dict[str, Any],
    score: float,
) -> str:
    """Save a candidate thumbnail and return its relative path.
    
    Filename format: {region}_{date}_score{score:.2f}_{scene_id_short}.png
    """
    region = info["region"]
    date = info["date"]
    scene_id = info["scene_id"]
    scene_short = scene_id[:8] if len(scene_id) >= 8 else scene_id

    fname = f"{region}_{date}_score{score:.2f}_{scene_short}.png"
    # Sanitize filename — replace any problematic characters
    fname = "".join(c if c.isalnum() or c in "._-" else "_" for c in fname)

    fpath = output_dir / fname
    fpath.write_bytes(png_bytes)
    return fname


def update_manifest(
    output_dir: Path,
    entry: dict[str, Any],
) -> None:
    """Append a candidate entry to the manifest JSON file."""
    manifest_path = output_dir / "manifest.json"
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


# ===================================================================
# Concurrent point processing
# ===================================================================
_stats_lock = threading.Lock()


def process_point(
    point: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    mean_pos: np.ndarray,
    mean_neg: np.ndarray,
    ee_module: Any,
    idx: int,
    total: int,
) -> dict[str, int]:
    """Process a single grid point: search, download, score, save/discard."""
    result = {"processed": 1, "candidates": 0, "no_scene": 0, "download_errors": 0, "low_score": 0}

    scene_info = find_best_scene(
        ee_module, point["lat"], point["lon"], args.start, args.end, args.max_cloud,
    )
    if scene_info is None:
        with _stats_lock:
            print_progress(idx, total, point["region"], point["lat"], point["lon"],
                          "no scene", 0)
        result["no_scene"] = 1
        return result

    thumb_bytes = download_thumbnail(ee_module, point["lat"], point["lon"], scene_info["scene_id"])
    if thumb_bytes is None:
        with _stats_lock:
            print_progress(idx, total, point["region"], point["lat"], point["lon"],
                          "download error", 0)
        result["download_errors"] = 1
        return result

    score = score_thumbnail(thumb_bytes, model, device, mean_pos, mean_neg)
    if score is None:
        with _stats_lock:
            print_progress(idx, total, point["region"], point["lat"], point["lon"],
                          "scoring error", 0)
        result["download_errors"] = 1
        return result

    if score > args.threshold:
        info = {
            "region": point["region"],
            "lat": point["lat"],
            "lon": point["lon"],
            "date": scene_info["date"],
            "scene_id": scene_info["scene_id"],
            "cloud": scene_info["cloud"],
            "score": round(score, 4),
        }
        fname = save_candidate(output_dir, thumb_bytes, info, score)
        with _stats_lock:
            update_manifest(output_dir, {**info, "thumbnail_path": fname})
            print_progress(idx, total, point["region"], point["lat"], point["lon"],
                          f"CANDIDATE score={score:.4f} {fname}", 0)
        result["candidates"] = 1
    else:
        with _stats_lock:
            print_progress(idx, total, point["region"], point["lat"], point["lon"],
                          f"below threshold ({score:.4f})", 0)
        result["low_score"] = 1

    return result


# ===================================================================
# Progress display
# ===================================================================

def print_progress(
    idx: int,
    total: int,
    region: str,
    lat: float,
    lon: float,
    status: str,
    elapsed: float,
) -> None:
    """Print a single progress line."""
    pct = 100.0 * (idx + 1) / total
    if idx > 0:
        rate = idx / elapsed
        remaining_s = (total - idx) / rate if rate > 0 else 0
        eta = time.strftime("%H:%M:%S", time.gmtime(remaining_s))
    else:
        eta = "?"
    loc = f"({lat:.4f}, {lon:.4f})"
    print(f"  [{idx + 1}/{total}] ({pct:.0f}%) {region} {loc} | {status} | ETA {eta}")


# ===================================================================
# Main CLI
# ===================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan BC coastline for herring spawn using DINOv2 scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        default="data/candidates",
        help="Output directory for candidates and manifest (default: data/candidates)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Score threshold; points above this are saved as candidates (default: 0.0)",
    )
    parser.add_argument(
        "--start",
        default="2024-02-01",
        help="Start date YYYY-MM-DD (default: 2024-02-01)",
    )
    parser.add_argument(
        "--end",
        default="2024-04-30",
        help="End date YYYY-MM-DD (default: 2024-04-30)",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=50,
        help="Maximum cloud percentage (default: 50)",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=0.01,
        help="Spacing between grid points in degrees (default: 0.01 ~1.1 km)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent threads for scene search and download (default: 8)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute reference vectors from scratch (ignore cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate grid points and report counts, but don't call GEE or DINOv2",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns 0 on success, 1 on error."""
    args = parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paths for labeled samples
    repo_root = Path(__file__).resolve().parent.parent
    pos_dir = repo_root / "data" / "samples" / "positive"
    neg_dir = repo_root / "data" / "samples" / "negative"
    cache_path = repo_root / "data" / "embeddings" / "reference_vectors.npz"

    # ------------------------------------------------------------------
    # 1. Generate grid points
    # ------------------------------------------------------------------
    print("\n=== Generating grid points ===")
    points = generate_grid_points(REGIONS, args.grid_spacing)
    print(f"  Total grid points: {len(points)}")

    if not points:
        print("ERROR: No grid points generated. Check region definitions and spacing.")
        return 1

    # ------------------------------------------------------------------
    # 2. Load DINOv2 model
    # ------------------------------------------------------------------
    print("\n=== Loading DINOv2 model ===")
    print(f"  Model: {MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    try:
        model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
        model.eval()
        model = model.to(device)
    except Exception as exc:
        print(f"ERROR: Failed to load DINOv2 model: {exc}")
        print("  Ensure torch and torchvision are installed.")
        return 1

    # ------------------------------------------------------------------
    # 3. Compute/loda reference vectors
    # ------------------------------------------------------------------
    print("\n=== Computing reference vectors ===")
    if not pos_dir.exists():
        print(f"ERROR: Positive samples directory not found: {pos_dir}")
        return 1
    if not neg_dir.exists():
        print(f"ERROR: Negative samples directory not found: {neg_dir}")
        return 1

    try:
        if args.no_cache:
            mean_pos, mean_neg, _ = compute_reference_vectors(pos_dir, neg_dir, model, device)
        else:
            mean_pos, mean_neg = load_or_compute_reference_vectors(
                pos_dir, neg_dir, model, device, cache_path
            )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    # ------------------------------------------------------------------
    # 4. Initialize GEE
    # ------------------------------------------------------------------
    print("\n=== Initializing Google Earth Engine ===")
    try:
        import ee  # noqa: F811
        ee.Initialize(project="redd-fish")
        print("  GEE initialized (project: redd-fish)")
    except Exception as exc:
        print(f"ERROR: GEE initialization failed: {exc}")
        print("  Ensure you are authenticated: earthengine authenticate")
        return 1

    # ------------------------------------------------------------------
    # Dry run: report and exit
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n=== Dry run complete ===")
        print(f"  Regions: {len(REGIONS)}")
        print(f"  Grid points: {len(points)}")
        print(f"  Grid spacing: {args.grid_spacing}° ({args.grid_spacing * 111:.1f} km)")
        print(f"  Date range: {args.start} to {args.end}")
        print(f"  Max cloud: {args.max_cloud}%")
        print(f"  Score threshold: {args.threshold}")
        print(f"  Output: {output_dir}")
        print("\nTo run for real, omit --dry-run")
        return 0

    # ------------------------------------------------------------------
    # 5. Process points concurrently
    # ------------------------------------------------------------------
    print(f"\n=== Scanning {len(points)} grid points with {args.workers} workers ===")
    print(f"  Date range: {args.start} to {args.end}")
    print(f"  Max cloud: {args.max_cloud}%")
    print(f"  Score threshold: {args.threshold}")
    print(f"  Output: {output_dir.resolve()}")
    print()

    processed = 0
    candidates = 0
    no_scene = 0
    download_errors = 0
    low_score = 0

    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_point, point, args, output_dir, model, device,
                    mean_pos, mean_neg, ee, idx, len(points),
                ): idx
                for idx, point in enumerate(points)
            }

            for future in as_completed(futures):
                result = future.result()
                processed += result["processed"]
                candidates += result["candidates"]
                no_scene += result["no_scene"]
                download_errors += result["download_errors"]
                low_score += result["low_score"]

    except KeyboardInterrupt:
        print("\n\nInterrupted! Partial results saved.")
    except Exception as exc:
        print(f"\n\nError during scan: {exc}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0

    print(f"\n{'='*60}")
    print("  Scan complete")
    print(f"  {'='*60}")
    print(f"  Total grid points:         {len(points)}")
    print(f"  Processed:                 {processed}")
    print(f"  Candidates saved:          {candidates}")
    print(f"  No suitable scene:         {no_scene}")
    print(f"  Download/score errors:     {download_errors}")
    print(f"  Below threshold:           {low_score}")
    print(f"  {'='*60}")
    print(f"  Elapsed time:              {elapsed:.1f}s")
    print(f"  Processing rate:           {rate:.1f} points/s")
    print(f"  Candidates in:             {output_dir.resolve()}")
    print(f"  Manifest:                  {output_dir / 'manifest.json'}")

    if candidates > 0:
        print("\n  To review candidates:")
        print(f"    ls -la {output_dir / '*.png'} | wc -l")
        print(f"    python -m http.server 8766 --directory {output_dir.parent}")
    else:
        print("\n  No candidates found. Try lowering --threshold or adjusting date range.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
