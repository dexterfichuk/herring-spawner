#!/usr/bin/env python3
"""
Batch temporal validation for all high-scoring BC coast candidates.

For each unique (lat, lon) location with spawn_score > 0.3, searches
Sentinel-2 for additional scenes ±7 days from the candidate date,
downloads RGB thumbnails, scores with the improved detector model,
and classifies as super_positive / positive / not_spawn.

Usage:
    python scripts/validate_temporal_candidates.py [--workers 8] [--score-threshold 0.3]

Outputs:
    data/candidates_v2/temporal_results.json
    data/candidates_v2/temporal_review.html
"""

import base64
import io
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.improved_detector import (
    ImprovedDetector,
    extract_features_from_bytes,
    DINO_TRANSFORM,
    download_thumbnail,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MANIFEST_PATH = PROJECT_ROOT / "data" / "candidates_v2" / "manifest.json"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "improved_model.pkl"
CANDIDATES_DIR = PROJECT_ROOT / "data" / "candidates_v2"
CACHE_DIR = PROJECT_ROOT / "data" / "candidates_v2" / "temporal_cache"
OUTPUT_RESULTS = PROJECT_ROOT / "data" / "candidates_v2" / "temporal_results.json"
OUTPUT_HTML = PROJECT_ROOT / "data" / "candidates_v2" / "temporal_review.html"

SEARCH_DAYS = 7           # ±days around candidate date
MAX_CLOUD = 50            # max cloud pct
SCORE_THRESHOLD = 0.2     # score above this = spawn visible on a date
MIN_CANDIDATE_SCORE = 0.3 # only process candidates above this
WORKERS = 8

# Web app
WEBAPP_URL = "http://localhost:5050"
LABEL_API = f"{WEBAPP_URL}/api/candidates/%d/label"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest() -> list[dict[str, Any]]:
    """Load all candidates from manifest.json."""
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def get_best_per_location(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by (lat, lon) rounded to 4dp, keep the best-scoring candidate per location."""
    best: dict[tuple, dict[str, Any]] = {}
    for c in candidates:
        key = (round(c["lat"], 4), round(c["lon"], 4))
        if key not in best or c["score"] > best[key]["score"]:
            best[key] = c
    return list(best.values())


def db_candidate_id(lat: float, lon: float) -> int | None:
    """Look up candidate DB id for a lat/lon."""
    import sqlite3
    db_path = PROJECT_ROOT / "data" / "webapp" / "herring.db"
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(
        "SELECT id, user_label FROM candidates WHERE ABS(lat - ?) < 0.001 AND ABS(lon - ?) < 0.001",
        (lat, lon),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def find_scenes_for_location(
    ee_module: Any,
    lat: float,
    lon: float,
    center_date: str,
    days_window: int = SEARCH_DAYS,
    max_cloud: float = MAX_CLOUD,
) -> list[dict[str, Any]]:
    """Find all Sentinel-2 scenes at a location within ±days_window of center_date."""
    center = datetime.strptime(center_date, "%Y-%m-%d")
    start = (center - timedelta(days=days_window)).strftime("%Y-%m-%d")
    end = (center + timedelta(days=days_window)).strftime("%Y-%m-%d")

    collection = ee_module.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    point = ee_module.Geometry.Point(lon, lat)

    scenes = (
        collection
        .filterBounds(point)
        .filterDate(start, end)
        .filter(ee_module.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )

    try:
        scene_ids = scenes.aggregate_array("system:index").getInfo()
        clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        dates_raw = scenes.aggregate_array("system:time_start").getInfo()
    except Exception as exc:
        print(f"    GEE query error at ({lat:.4f}, {lon:.4f}): {exc}")
        return []

    results = []
    for i in range(len(scene_ids)):
        sid = scene_ids[i]
        date_str = f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}" if len(sid) >= 8 else ""
        ts = dates_raw[i] if i < len(dates_raw) else 0
        results.append({
            "scene_id": sid,
            "cloud": float(clouds[i]) if i < len(clouds) else 100,
            "date": date_str,
            "timestamp_ms": ts,
            "lat": lat,
            "lon": lon,
        })

    # Sort by date ascending
    results.sort(key=lambda x: x["date"])

    # Deduplicate: keep the lowest-cloud scene per unique date
    seen_dates: dict[str, dict[str, Any]] = {}
    for r in results:
        d = r["date"]
        if d not in seen_dates or r["cloud"] < seen_dates[d]["cloud"]:
            seen_dates[d] = r
    results = list(seen_dates.values())
    results.sort(key=lambda x: x["date"])

    return results


def score_thumbnail_bytes(
    png_bytes: bytes,
    detector: ImprovedDetector,
    dinov2_model: torch.nn.Module,
    device: torch.device,
) -> float | None:
    """Score a PNG thumbnail using the improved detector's SVM."""
    try:
        feats = extract_features_from_bytes(png_bytes, dinov2_model, device)
        if feats is None:
            return None
        return detector.score_svm(feats.dinov2_embedding)
    except Exception as exc:
        print(f"    Scoring error: {exc}")
        return None


def classify(dated_scores: list[dict[str, Any]], threshold: float = SCORE_THRESHOLD) -> str:
    """Classify based on scores across multiple dates.

    Returns 'super_positive', 'positive', or 'not_spawn'.
    """
    valid = [d for d in dated_scores if d.get("score") is not None]
    if not valid:
        return "not_spawn"

    above = [d for d in valid if d["score"] > threshold]
    dates_above = sorted(set(d["date"] for d in above))

    if len(dates_above) >= 2:
        return "super_positive"
    if len(dates_above) == 1:
        return "positive"
    return "not_spawn"


def update_label_via_api(candidate_id: int, label: str) -> bool:
    """Update a candidate's label via the web app API."""
    try:
        resp = requests.post(LABEL_API % candidate_id, json={"label": label}, timeout=10)
        return resp.status_code == 200 and resp.json().get("status") == "ok"
    except Exception as exc:
        print(f"    API error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_location(
    candidate: dict[str, Any],
    detector: ImprovedDetector,
    dinov2_model: torch.nn.Module,
    device: torch.device,
    ee_module: Any,
) -> dict[str, Any]:
    """Process one unique location: find scenes, download, score, classify."""
    lat = candidate["lat"]
    lon = candidate["lon"]
    region = candidate["region"]
    date = candidate["date"]
    orig_score = candidate["score"]
    scene_id = candidate["scene_id"]
    thumb_rel = candidate["thumbnail_path"]

    db_id = db_candidate_id(lat, lon)

    result: dict[str, Any] = {
        "region": region,
        "lat": lat,
        "lon": lon,
        "date": date,
        "original_score": orig_score,
        "original_scene_id": scene_id,
        "candidate_id": db_id,
        "classification": "not_spawn",
        "dated_scores": [],
    }

    print(f"\n  {region} ({lat:.4f}, {lon:.4f}) — orig score {orig_score:.4f} on {date}")

    # 1. Find additional scenes
    print(f"    Searching ±{SEARCH_DAYS}d from {date}...")
    scenes = find_scenes_for_location(ee_module, lat, lon, date)
    print(f"    Found {len(scenes)} scenes")

    if not scenes:
        # Just the original, nothing to compare against
        result["dated_scores"].append({
            "date": date,
            "score": orig_score,
            "scene_id": scene_id,
            "cloud": candidate.get("cloud", 0),
            "is_original": True,
            "thumbnail_b64": "",
        })
        result["classification"] = "positive"
        return result

    # 2. Cache dir for this location
    loc_cache = CACHE_DIR / f"{region}_{lat}_{lon}".replace(".", "_")
    loc_cache.mkdir(parents=True, exist_ok=True)

    def process_scene(scene: dict[str, Any]) -> dict[str, Any]:
        date_str = scene["date"]
        sid = scene["scene_id"]
        is_original = (date_str == date)
        cache_key = f"{sid}_{lat}_{lon}".replace(".", "_")
        cache_path = loc_cache / f"{cache_key}.png"

        # Check cache first
        if cache_path.exists():
            png_bytes = cache_path.read_bytes()
        elif is_original and thumb_rel:
            thumb_path = CANDIDATES_DIR / thumb_rel
            if thumb_path.exists():
                png_bytes = thumb_path.read_bytes()
                # Also cache it
                cache_path.write_bytes(png_bytes)
            else:
                png_bytes = download_thumbnail(ee_module, lat, lon, sid)
                if png_bytes:
                    cache_path.write_bytes(png_bytes)
        else:
            png_bytes = download_thumbnail(ee_module, lat, lon, sid)
            if png_bytes:
                cache_path.write_bytes(png_bytes)

        if png_bytes is None:
            return {
                "date": date_str,
                "score": None,
                "scene_id": sid,
                "cloud": scene["cloud"],
                "is_original": is_original,
                "thumbnail_b64": "",
            }

        score = score_thumbnail_bytes(png_bytes, detector, dinov2_model, device)

        return {
            "date": date_str,
            "score": score if score is not None else (orig_score if is_original else None),
            "scene_id": sid,
            "cloud": scene["cloud"],
            "is_original": is_original,
            "thumbnail_b64": base64.b64encode(png_bytes).decode("ascii") if png_bytes else "",
        }

    # Process scenes in parallel
    dated_scores: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_scene, s): s for s in scenes}
        for future in as_completed(futures):
            try:
                ds = future.result()
                dated_scores.append(ds)
            except Exception as exc:
                scene = futures[future]
                dated_scores.append({
                    "date": scene["date"],
                    "score": None,
                    "scene_id": scene["scene_id"],
                    "cloud": scene["cloud"],
                    "is_original": False,
                    "thumbnail_b64": "",
                })
                print(f"    Error {scene['date']}: {exc}")

    dated_scores.sort(key=lambda x: x["date"])
    result["dated_scores"] = dated_scores

    # 3. Classify
    classification = classify(dated_scores)
    result["classification"] = classification
    print(f"    -> {classification}")

    # Print score summary
    for ds in dated_scores:
        sc = f"{ds['score']:.4f}" if ds["score"] is not None else "N/A"
        orig = " (orig)" if ds.get("is_original") else ""
        print(f"      {ds['date']}: score={sc}{orig}")

    # 4. Update web app label for super_positives and positives
    if classification in ("super_positive", "positive") and db_id is not None:
        ok = update_label_via_api(db_id, "spawn")
        if ok:
            print(f"    Updated candidate #{db_id} label -> spawn")

    return result


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_html_report(results: list[dict[str, Any]]) -> str:
    """Generate a temporal validation HTML report."""
    counts = {"super_positive": 0, "positive": 0, "not_spawn": 0}
    for r in results:
        cls = r.get("classification", "not_spawn")
        if cls in counts:
            counts[cls] += 1

    cards = []
    for r in results:
        region = r.get("region", "?")
        classification = r.get("classification", "not_spawn")
        lat = r.get("lat", 0)
        lon = r.get("lon", 0)
        orig_score = r.get("original_score", 0)
        date = r.get("date", "?")
        db_id = r.get("candidate_id")

        badge_class = {
            "super_positive": "badge-super",
            "positive": "badge-positive",
            "not_spawn": "badge-none",
        }.get(classification, "badge-none")

        dates_html = ""
        for ds in r.get("dated_scores", []):
            score = ds.get("score")
            date_str = ds.get("date", "?")
            is_original = ds.get("is_original", False)
            sid = ds.get("scene_id", "")
            cloud = ds.get("cloud", 0)
            thumb_b64 = ds.get("thumbnail_b64", "")

            if score is None:
                score_class = "score-none"
                score_label = "N/A"
            elif score > 0.2:
                score_class = "score-good"
                score_label = f"{score:.3f}"
            elif score > 0.1:
                score_class = "score-mid"
                score_label = f"{score:.3f}"
            else:
                score_class = "score-low"
                score_label = f"{score:.3f}"

            orig_tag = '<span class="orig">original</span>' if is_original else ""
            img_html = f'<img src="data:image/png;base64,{thumb_b64}" alt="{date_str}" loading="lazy">' if thumb_b64 else '<div class="no-img">no thumbnail</div>'

            dates_html += f"""
            <div class="date-card">
                {img_html}
                <div class="date-info">
                    <div class="dh"><strong>{date_str}</strong> {orig_tag}</div>
                    <div class="dm"><span class="{score_class}">Score: {score_label}</span> <span class="cloudp">{cloud:.0f}%</span></div>
                    <div class="dsid">{sid[:50]}</div>
                </div>
            </div>"""

        n_dates = len(r.get("dated_scores", []))
        n_above = sum(1 for d in r.get("dated_scores", []) if d.get("score") is not None and d["score"] > 0.2)

        cards.append(f"""
        <div class="card">
            <div class="ch">
                <h2>{region}</h2>
                <span class="{badge_class}">{classification}</span>
            </div>
            <div class="cm">
                <span>{lat:.4f}, {lon:.4f}</span>
                <span>Date: {date}</span>
                <span>Score: {orig_score:.4f}</span>
                <span>Candidate #{db_id or "?"}</span>
                <span>{n_dates} dates &middot; {n_above} above threshold</span>
            </div>
            <div class="dg">{dates_html}</div>
        </div>""")

    cards_html = "\n".join(cards)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Temporal Validation — BC Coast Candidates</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #0d1117; color: #c9d1d9; }}
.bar {{ background: #161b22; padding: 16px 24px; position: sticky; top: 0; z-index: 99; border-bottom: 1px solid #30363d; }}
.bar h1 {{ margin: 0; font-size: 20px; color: #f0f6fc; }}
.bar .sub {{ font-size: 13px; color: #8b949e; margin-top: 4px; }}
.summary {{ background: #161b22; margin: 16px; padding: 16px 20px; border-radius: 8px; border: 1px solid #30363d; }}
.sg {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; }}
.sg > div {{ text-align: center; padding: 10px 20px; background: #0d1117; border-radius: 6px; border: 1px solid #30363d; flex: 1; min-width: 100px; }}
.sg .n {{ font-size: 26px; font-weight: 700; }}
.sg .l {{ font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }}
.nsuper {{ color: #3fb950; }}
.npos {{ color: #58a6ff; }}
.nnone {{ color: #8b949e; }}

.events {{ padding: 0 16px 16px; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 16px; overflow: hidden; }}
.ch {{ display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-bottom: 1px solid #21262d; }}
.ch h2 {{ margin: 0; font-size: 15px; color: #f0f6fc; }}
.cm {{ display: flex; gap: 16px; padding: 8px 16px; font-size: 11px; color: #8b949e; border-bottom: 1px solid #21262d; flex-wrap: wrap; }}
.dg {{ display: flex; gap: 10px; padding: 12px 16px; overflow-x: auto; }}
.date-card {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; min-width: 190px; max-width: 230px; flex-shrink: 0; }}
.date-card img {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; }}
.no-img {{ width: 100%; aspect-ratio: 1/1; display: flex; align-items: center; justify-content: center; color: #484f58; font-size: 11px; background: #0d1117; }}
.date-info {{ padding: 8px; }}
.dh {{ font-size: 11px; margin-bottom: 4px; display: flex; align-items: center; gap: 4px; }}
.dh strong {{ color: #f0f6fc; }}
.dm {{ display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 4px; font-size: 10px; }}
.dsid {{ font-size: 9px; color: #484f58; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
.badge-super {{ background: #0f2d1a; color: #3fb950; border: 1px solid #3fb950; }}
.badge-positive {{ background: #0c2d48; color: #58a6ff; border: 1px solid #58a6ff; }}
.badge-none {{ background: #1c1c1c; color: #8b949e; border: 1px solid #30363d; }}
.orig {{ display: inline-block; padding: 1px 5px; border-radius: 4px; font-size: 9px; background: #1c1c1c; color: #8b949e; border: 1px solid #30363d; }}
.score-good {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; background: #0f2d1a; color: #3fb950; font-weight: 600; }}
.score-mid {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; background: #2d1f0a; color: #d29922; font-weight: 600; }}
.score-low {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; background: #3d1212; color: #f85149; font-weight: 600; }}
.score-none {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; background: #1c1c1c; color: #484f58; }}
.cloudp {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; background: #161b22; color: #8b949e; }}
</style>
</head>
<body>
<div class="bar">
    <h1>&#x1f41f; Temporal Validation — BC Coast Candidates</h1>
    <div class="sub">Validated {len(results)} unique locations with spawn_score &gt; 0.3 &middot; Searching &plusmn;{SEARCH_DAYS}d</div>
</div>
<div class="summary">
    <strong>Summary</strong>
    <div class="sg">
        <div><div class="n nsuper">{counts["super_positive"]}</div><div class="l">Super Positive (2+ dates)</div></div>
        <div><div class="n npos">{counts["positive"]}</div><div class="l">Positive (1 date)</div></div>
        <div><div class="n nnone">{counts["not_spawn"]}</div><div class="l">Not Spawn (0 dates)</div></div>
        <div><div class="n" style="color:#f0f6fc">{len(results)}</div><div class="l">Total Locations</div></div>
    </div>
</div>
<div class="events">
{cards_html}
</div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("  Temporal Validation — BC Coast Candidates")
    print("=" * 60)

    # 1. Load and filter candidates
    print("\n=== Loading candidates ===")
    all_candidates = load_manifest()
    print(f"  Total: {len(all_candidates)}")
    high_score = [c for c in all_candidates if c["score"] > MIN_CANDIDATE_SCORE]
    print(f"  Score > {MIN_CANDIDATE_SCORE}: {len(high_score)}")
    locations = get_best_per_location(high_score)
    print(f"  Unique locations: {len(locations)}")

    if not locations:
        print("  No locations to process.")
        return 0

    # 2. Load models
    print("\n=== Loading DINOv2 model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    try:
        dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        dinov2_model.eval()
        dinov2_model = dinov2_model.to(device)
    except Exception as exc:
        print(f"ERROR: Failed to load DINOv2: {exc}")
        return 1

    print("\n=== Loading improved detector ===")
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return 1
    try:
        detector = ImprovedDetector.load(MODEL_PATH)
        stats = detector.training_stats
        print(f"  Loaded model: SVM kernel={detector.svm.kernel if detector.svm else 'N/A'}")
        print(f"  Training: {stats.get('n_train', '?')} samples, accuracy={stats.get('full_accuracy', '?'):.4f}")
    except Exception as exc:
        print(f"ERROR: Failed to load model: {exc}")
        return 1

    # 3. Initialize GEE
    print("\n=== Initializing GEE ===")
    try:
        import ee
        ee.Initialize(project="redd-fish")
        print("  GEE initialized (project: redd-fish)")
    except Exception as exc:
        print(f"ERROR: GEE initialization failed: {exc}")
        return 1

    # 4. Process each location
    print(f"\n=== Processing {len(locations)} locations with {WORKERS} workers ===")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    start_time = time.time()

    for i, loc in enumerate(locations):
        print(f"\n--- Location {i+1}/{len(locations)} ---")
        try:
            result = process_location(loc, detector, dinov2_model, device, ee)
            results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()
            results.append({
                "region": loc.get("region", "?"),
                "lat": loc.get("lat", 0),
                "lon": loc.get("lon", 0),
                "date": loc.get("date", ""),
                "original_score": loc.get("score", 0),
                "candidate_id": None,
                "classification": "not_spawn",
                "dated_scores": [],
                "error": str(exc),
            })

        # Save results periodically
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(locations) - i - 1) / rate if rate > 0 else 0
            print(f"\n  [{i+1}/{len(locations)}] {elapsed/60:.1f}min elapsed, ~{remaining/60:.1f}min remaining")
            OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_RESULTS, "w") as f:
                json.dump(results, f, indent=2, default=str)

    # 5. Write final results
    print("\n=== Writing results ===")
    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {OUTPUT_RESULTS}")

    # 6. Generate HTML report
    print("\n=== Generating HTML report ===")
    html = generate_html_report(results)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"  Report: file://{OUTPUT_HTML.resolve()}")

    # 7. Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  Temporal Validation Complete")
    print("=" * 60)
    counts = {"super_positive": 0, "positive": 0, "not_spawn": 0}
    for r in results:
        cls = r.get("classification", "not_spawn")
        if cls in counts:
            counts[cls] += 1
    print(f"  Super Positive (2+ dates):  {counts['super_positive']}")
    print(f"  Positive (1 date):           {counts['positive']}")
    print(f"  Not Spawn:                   {counts['not_spawn']}")
    print(f"  Total:                       {len(results)}")
    print(f"  Time:                        {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Rate:                        {len(results)/elapsed:.2f} loc/s")
    print(f"  Report: file://{OUTPUT_HTML.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
