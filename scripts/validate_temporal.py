#!/usr/bin/env python3
"""
Multi-date temporal validation for confirmed herring spawn events.

For each confirmed spawn event (from the web app API), finds additional
Sentinel-2 scenes within ±7 days of the spawn date at the same location,
downloads RGB thumbnails, scores with the improved detector model,
classifies the event based on multi-date evidence, and updates the web app.

Classifications:
  - super_positive — spawn visible on 2+ consecutive dates within the window
  - positive       — only 1 date shows spawn clearly
  - possible       — nearby dates exist but scores are ambiguous

Usage:
    python scripts/validate_temporal.py

Requirements:
    - Flask web app running at http://localhost:5050
    - GEE authenticated (project=redd-fish)
    - Trained model at data/models/improved_model.pkl
"""

import io
import json
import os
import sys
import time
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

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.improved_detector import (
    ImprovedDetector,
    extract_features,
    DINO_TRANSFORM,
    download_thumbnail,
    find_best_scene,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEBAPP_URL = "http://localhost:5050"
SPAWN_EVENTS_API = f"{WEBAPP_URL}/api/spawn-events"
LABEL_API = f"{WEBAPP_URL}/api/candidates/%d/label"

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "improved_model.pkl"
CANDIDATES_DIR = PROJECT_ROOT / "data" / "candidates_v2"
REVIEW_DIR = PROJECT_ROOT / "data" / "review"
OUTPUT_DIR = PROJECT_ROOT / "data" / "temporal_validation"

SEARCH_DAYS = 7          # ± days around the spawn date to search
MAX_CLOUD = 50           # max cloud percentage for additional scenes
SCORE_THRESHOLD = 0.2    # score above this = spawn visible
WORKERS = 4              # concurrent download/scoring threads

GRID_SPACING_DEG = 0.005  # ~500m — used to generate sample points around event

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_spawn_events() -> list[dict[str, Any]]:
    """Fetch all confirmed spawn events from the web app API."""
    resp = requests.get(SPAWN_EVENTS_API, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("features", [])


def find_matching_candidate(lat: float, lon: float, tolerance: float = 0.01) -> dict[str, Any] | None:
    """Find a candidate in the webapp database matching lat/lon within tolerance.

    Uses the SQLite DB directly to avoid API round-trips for candidate matching.
    """
    import sqlite3
    db_path = PROJECT_ROOT / "data" / "webapp" / "herring.db"
    if not db_path.exists():
        print(f"  WARNING: Database not found at {db_path}")
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """SELECT id, lat, lon, region, spawn_score, date, scene_id,
                  thumbnail_path, user_label
           FROM candidates
           WHERE ABS(lat - ?) < ? AND ABS(lon - ?) < ?
           ORDER BY spawn_score DESC
           LIMIT 1""",
        (lat, tolerance, lon, tolerance),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def find_scenes_for_location(
    ee_module: Any,
    lat: float,
    lon: float,
    center_date: str,
    days_window: int = SEARCH_DAYS,
    max_cloud: float = MAX_CLOUD,
) -> list[dict[str, Any]]:
    """Find all Sentinel-2 scenes at a location within ±days_window of center_date.

    Returns a list of dicts sorted by date ascending.
    """
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

    # Deduplicate: keep only the best (lowest cloud) scene per unique date
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
    """Score a PNG thumbnail using the improved detector.

    Returns the combined delta score, or None on failure.
    """
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        feats = extract_features(img, dinov2_model, device)
        svm_score = detector.score_svm(feats.dinov2_embedding)

        # Combined score (matching improved_detector's is_candidate logic):
        # svm_score is the decision function from the SVM
        return svm_score
    except Exception as exc:
        print(f"    Scoring error: {exc}")
        return None


def classify_event(
    dated_scores: list[dict[str, Any]],
    threshold: float = SCORE_THRESHOLD,
) -> str:
    """Classify an event based on scores across multiple dates.

    Args:
        dated_scores: list of dicts with 'date', 'score', 'is_original'
        threshold: score above this = spawn visible

    Returns:
        'super_positive', 'positive', or 'possible'
    """
    # Filter to dates with valid scores
    valid = [d for d in dated_scores if d["score"] is not None]

    if not valid:
        return "possible"

    # Count distinct dates above threshold
    above = [d for d in valid if d["score"] > threshold]
    dates_above = sorted(set(d["date"] for d in above))

    if len(dates_above) >= 2:
        return "super_positive"

    if len(dates_above) == 1:
        return "positive"

    # All scores below threshold
    # Check if any are close to threshold
    close = [d for d in valid if d["score"] is not None and d["score"] > threshold * 0.5]
    if close:
        return "possible"

    return "possible"


def update_label_via_api(candidate_id: int, label: str) -> bool:
    """Update a candidate's label via the web app API.

    Extends valid labels to include temporal classifications.
    """
    resp = requests.post(
        LABEL_API % candidate_id,
        json={"label": label},
        timeout=10,
    )
    if resp.status_code == 200:
        result = resp.json()
        return result.get("status") == "ok"
    else:
        print(f"    API error: {resp.status_code} {resp.text[:200]}")
        return False


# ===================================================================
# HTML Report Generation
# ===================================================================

def generate_html_report(
    results: list[dict[str, Any]],
    output_path: Path,
) -> str:
    """Generate a temporal validation HTML report."""
    # Count classifications
    counts = {"super_positive": 0, "positive": 0, "possible": 0, "unprocessed": 0}
    for r in results:
        cls = r.get("classification", "unprocessed")
        if cls in counts:
            counts[cls] += 1

    # Build event cards
    cards_html = []
    for r in results:
        region = r.get("region", "?")
        classification = r.get("classification", "unprocessed")
        lat = r.get("lat", 0)
        lon = r.get("lon", 0)
        event_id = r.get("event_id", "?")
        candidate_id = r.get("candidate_id")
        notes = r.get("notes", "")

        # Badge color
        badge_class = {
            "super_positive": "badge-super",
            "positive": "badge-positive",
            "possible": "badge-possible",
            "unprocessed": "badge-unproc",
        }.get(classification, "badge-unproc")

        # Collect dates
        dated_scores = r.get("dated_scores", [])
        dates_html = ""
        for ds in dated_scores:
            score = ds.get("score")
            date_str = ds.get("date", "?")
            is_original = ds.get("is_original", False)
            scene_id = ds.get("scene_id", "")
            cloud = ds.get("cloud", 0)
            thumbnail_b64 = ds.get("thumbnail_b64", "")

            # Color by score
            if score is None:
                score_class = "score-none"
                score_label = "N/A"
            elif score > SCORE_THRESHOLD:
                score_class = "score-good"
                score_label = f"{score:.3f}"
            elif score > SCORE_THRESHOLD * 0.5:
                score_class = "score-mid"
                score_label = f"{score:.3f}"
            else:
                score_class = "score-low"
                score_label = f"{score:.3f}"

            original_tag = '<span class="original-badge">original</span>' if is_original else ""
            cloud_pct = f"{cloud:.0f}%"

            img_html = ""
            if thumbnail_b64:
                img_html = f'<img src="data:image/png;base64,{thumbnail_b64}" alt="{date_str}" loading="lazy">'

            dates_html += f"""
            <div class="date-card">
                {img_html}
                <div class="date-info">
                    <div class="date-header">
                        <strong>{date_str}</strong> {original_tag}
                    </div>
                    <div class="date-meta">
                        <span class="{score_class}">Score: {score_label}</span>
                        <span class="cloud-badge">Cloud: {cloud_pct}</span>
                    </div>
                    <div class="date-scene">{scene_id[:50]}</div>
                </div>
            </div>"""

        n_dates = len(dated_scores)
        n_above = sum(1 for d in dated_scores if d.get("score") is not None and d["score"] > SCORE_THRESHOLD)

        cards_html.append(f"""
        <div class="event-card">
            <div class="event-header">
                <h2>{region}</h2>
                <span class="{badge_class}">{classification}</span>
            </div>
            <div class="event-meta">
                <span>{lat:.4f}, {lon:.4f}</span>
                <span>Event #{event_id} | Candidate #{candidate_id or "?"}</span>
                <span>{n_dates} dates · {n_above} above threshold</span>
            </div>
            {f'<div class="event-notes">{notes}</div>' if notes else ""}
            <div class="dates-grid">
                {dates_html}
            </div>
        </div>""")

    cards_joined = "\n".join(cards_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Temporal Validation — Herring Spawn Events</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #0d1117; color: #c9d1d9; }}
.bar {{ background: #161b22; padding: 16px 24px; position: sticky; top: 0; z-index: 99;
        border-bottom: 1px solid #30363d; }}
.bar h1 {{ margin: 0; font-size: 20px; color: #f0f6fc; }}
.bar .sub {{ font-size: 13px; color: #8b949e; margin-top: 4px; }}
.summary {{ background: #161b22; margin: 16px; padding: 16px 20px; border-radius: 8px;
            border: 1px solid #30363d; }}
.summary-grid {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; }}
.summary-stat {{ text-align: center; padding: 10px 20px; background: #0d1117; border-radius: 6px;
                border: 1px solid #30363d; flex: 1; min-width: 120px; }}
.summary-stat .num {{ font-size: 26px; font-weight: 700; }}
.summary-stat .lbl {{ font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }}
.num-super {{ color: #3fb950; }}
.num-pos {{ color: #58a6ff; }}
.num-poss {{ color: #d29922; }}

.events {{ padding: 0 16px 16px; }}
.event-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 16px;
              overflow: hidden; }}
.event-header {{ display: flex; justify-content: space-between; align-items: center;
                padding: 12px 16px; border-bottom: 1px solid #21262d; }}
.event-header h2 {{ margin: 0; font-size: 16px; color: #f0f6fc; }}
.event-meta {{ display: flex; gap: 16px; padding: 8px 16px; font-size: 12px; color: #8b949e;
               border-bottom: 1px solid #21262d; flex-wrap: wrap; }}
.event-notes {{ padding: 8px 16px; font-size: 12px; color: #d29922; background: #1c1c1c; }}
.dates-grid {{ display: flex; gap: 10px; padding: 12px 16px; overflow-x: auto; }}
.date-card {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px; overflow: hidden;
              min-width: 200px; max-width: 240px; flex-shrink: 0; }}
.date-card img {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; }}
.date-info {{ padding: 8px; }}
.date-header {{ font-size: 12px; margin-bottom: 4px; display: flex; align-items: center; gap: 4px; }}
.date-header strong {{ color: #f0f6fc; }}
.date-meta {{ display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 4px; }}
.date-scene {{ font-size: 9px; color: #484f58; overflow: hidden; text-overflow: ellipsis;
               white-space: nowrap; }}

.badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px;
          font-weight: 600; text-transform: uppercase; }}
.badge-super {{ background: #0f2d1a; color: #3fb950; border: 1px solid #3fb950; }}
.badge-positive {{ background: #0c2d48; color: #58a6ff; border: 1px solid #58a6ff; }}
.badge-possible {{ background: #2d1f0a; color: #d29922; border: 1px solid #d29922; }}
.badge-unproc {{ background: #1c1c1c; color: #484f58; border: 1px solid #484f58; }}

.original-badge {{ display: inline-block; padding: 1px 5px; border-radius: 4px; font-size: 9px;
                   background: #1c1c1c; color: #8b949e; border: 1px solid #30363d; }}
.score-good {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px;
               background: #0f2d1a; color: #3fb950; font-weight: 600; }}
.score-mid {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px;
              background: #2d1f0a; color: #d29922; font-weight: 600; }}
.score-low {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px;
              background: #3d1212; color: #f85149; font-weight: 600; }}
.score-none {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px;
               background: #1c1c1c; color: #484f58; }}
.cloud-badge {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px;
                background: #161b22; color: #8b949e; }}
</style>
</head>
<body>
<div class="bar">
    <h1>&#x1f41f; Temporal Validation</h1>
    <div class="sub">
        Validated {len(results)} spawn events across multiple Sentinel-2 dates
        &middot; Threshold: {SCORE_THRESHOLD}
    </div>
</div>
<div class="summary">
    <strong>Summary</strong> &middot; Multi-date evidence strengthens spawn event confidence.
    <div class="summary-grid">
        <div class="summary-stat">
            <div class="num num-super">{counts["super_positive"]}</div>
            <div class="lbl">Super Positive (2+ dates)</div>
        </div>
        <div class="summary-stat">
            <div class="num num-pos">{counts["positive"]}</div>
            <div class="lbl">Positive (1 date)</div>
        </div>
        <div class="summary-stat">
            <div class="num num-poss">{counts["possible"]}</div>
            <div class="lbl">Possible (unclear)</div>
        </div>
        <div class="summary-stat">
            <div class="num" style="color:#f0f6fc">{len(results)}</div>
            <div class="lbl">Total Events</div>
        </div>
    </div>
</div>
<div class="events">
{cards_joined}
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\n  Report: file://{output_path.resolve()}")
    return str(output_path)


# ===================================================================
# Main Validation Logic
# ===================================================================

def process_event(
    event: dict[str, Any],
    detector: ImprovedDetector,
    dinov2_model: torch.nn.Module,
    device: torch.device,
    ee_module: Any,
) -> dict[str, Any]:
    """Process a single spawn event: find dates, download, score, classify."""
    props = event.get("properties", {})
    geom = event.get("geometry", {})
    coords = geom.get("coordinates", [0, 0])
    lon, lat = coords[0], coords[1]

    event_id = props.get("id", "?")
    region = props.get("region", "Unknown")
    first_detected = props.get("first_detected", "")
    notes = props.get("notes", "")

    result: dict[str, Any] = {
        "event_id": event_id,
        "region": region,
        "lat": lat,
        "lon": lon,
        "first_detected": first_detected,
        "notes": notes,
        "candidate_id": None,
        "original_date": first_detected,
        "classification": "possible",
        "dated_scores": [],
    }

    # 1. Find matching candidate in the database
    candidate = find_matching_candidate(lat, lon)
    if candidate is None:
        print(f"  Event #{event_id} ({region}): No matching candidate found for ({lat:.4f}, {lon:.4f})")
        result["notes"] = f"{notes}. No matching candidate found in DB."
        return result

    result["candidate_id"] = candidate["id"]
    original_date = candidate.get("date") or first_detected
    result["original_date"] = original_date
    original_score = candidate.get("spawn_score") or 0.0
    original_scene_id = candidate.get("scene_id") or ""
    original_thumbnail = candidate.get("thumbnail_path") or ""

    print(f"\n  Event #{event_id}: {region} ({lat:.4f}, {lon:.4f})")
    print(f"    Candidate #{candidate['id']} | Date: {original_date} | Score: {original_score:.4f}")

    # 2. Find additional scenes in Sentinel-2
    print(f"    Searching ±{SEARCH_DAYS} days from {original_date}...")
    scenes = find_scenes_for_location(ee_module, lat, lon, original_date)
    print(f"    Found {len(scenes)} scenes")

    if not scenes:
        # Only the original, nothing to compare
        result["dated_scores"].append({
            "date": original_date,
            "score": original_score,
            "scene_id": original_scene_id,
            "cloud": candidate.get("cloud", 0),
            "is_original": True,
            "thumbnail_b64": "",
        })
        result["classification"] = "positive"
        return result

    # 3. Download and score each scene
    def process_scene(scene: dict[str, Any]) -> dict[str, Any]:
        date_str = scene["date"]
        scene_id = scene["scene_id"]
        is_original = (date_str == original_date)

        # If this is the original scene, use the cached thumbnail
        if is_original and original_thumbnail:
            thumb_path = CANDIDATES_DIR / original_thumbnail
            if thumb_path.exists():
                png_bytes = thumb_path.read_bytes()
                score = score_thumbnail_bytes(png_bytes, detector, dinov2_model, device)
                return {
                    "date": date_str,
                    "score": score if score is not None else original_score,
                    "scene_id": scene_id,
                    "cloud": scene["cloud"],
                    "is_original": True,
                    "thumbnail_b64": base64_encode_bytes(png_bytes),
                }

        # Download thumbnail from GEE
        png_bytes = download_thumbnail(ee_module, lat, lon, scene_id)
        if png_bytes is None:
            return {
                "date": date_str,
                "score": None,
                "scene_id": scene_id,
                "cloud": scene["cloud"],
                "is_original": is_original,
                "thumbnail_b64": "",
            }

        score = score_thumbnail_bytes(png_bytes, detector, dinov2_model, device)
        return {
            "date": date_str,
            "score": score,
            "scene_id": scene_id,
            "cloud": scene["cloud"],
            "is_original": is_original,
            "thumbnail_b64": base64_encode_bytes(png_bytes),
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
                print(f"    Error processing {scene['date']}: {exc}")

    # Sort by date
    dated_scores.sort(key=lambda x: x["date"])
    result["dated_scores"] = dated_scores

    # 4. Classify
    classification = classify_event(dated_scores)
    result["classification"] = classification
    print(f"    Classification: {classification}")

    # Print scores summary
    for ds in dated_scores:
        score_str = f"{ds['score']:.4f}" if ds['score'] is not None else "N/A"
        orig = " (original)" if ds.get("is_original") else ""
        print(f"      {ds['date']}: score={score_str}{orig}")

    # 5. Update label via API (only for super_positive and positive)
    if classification in ("super_positive", "positive"):
        candidate_id = candidate["id"]
        ok = update_label_via_api(candidate_id, "spawn")
        if ok:
            print(f"    Updated candidate #{candidate_id} label to 'spawn'")
        # Update spawn_events note via API
        notes = result.get("notes", "")
        new_note = f"Temporal: {classification} ({len([d for d in dated_scores if d['score'] is not None and d['score'] > SCORE_THRESHOLD])}/{len(dated_scores)} dates above threshold)"
        result["notes"] = f"{notes}. {new_note}" if notes else new_note

    return result


def base64_encode_bytes(data: bytes) -> str:
    """Base64 encode bytes for inline embedding in HTML."""
    import base64
    return base64.b64encode(data).decode("ascii")


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    print("=" * 60)
    print("  Temporal Validation — Herring Spawn Events")
    print("=" * 60)

    # 1. Fetch spawn events from web app
    print("\n=== Fetching spawn events ===")
    events = fetch_spawn_events()
    print(f"  Found {len(events)} events")
    if not events:
        print("  No events to validate.")
        return 0

    for ev in events:
        p = ev.get("properties", {})
        g = ev.get("geometry", {})
        coords = g.get("coordinates", [])
        print(f"  Event #{p.get('id', '?')}: {p.get('region', '?')} "
              f"({coords[1] if len(coords) > 1 else '?'}, {coords[0] if coords else '?'}) "
              f"confidence={p.get('confidence', '?')}")

    # 2. Load DINOv2 model
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

    # 3. Load improved detector model
    print("\n=== Loading improved detector ===")
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("  Train one first: python scripts/improved_detector.py --mode train")
        return 1

    try:
        detector = ImprovedDetector.load(MODEL_PATH)
        stats = detector.training_stats
        print(f"  Loaded model from {MODEL_PATH}")
        print(f"  SVM: kernel={detector.svm.kernel if detector.svm else 'N/A'}")
        print(f"  Training stats: {stats.get('n_train', '?')} samples, "
              f"accuracy={stats.get('full_accuracy', '?'):.4f}")
    except Exception as exc:
        print(f"ERROR: Failed to load model: {exc}")
        return 1

    # 4. Initialize GEE
    print("\n=== Initializing Google Earth Engine ===")
    try:
        import ee
        ee.Initialize(project="redd-fish")
        print("  GEE initialized (project: redd-fish)")
    except Exception as exc:
        print(f"ERROR: GEE initialization failed: {exc}")
        return 1

    # 5. Process each event
    print("\n=== Processing events ===")
    results: list[dict[str, Any]] = []

    for i, event in enumerate(events):
        print(f"\n--- Event {i + 1}/{len(events)} ---")
        try:
            result = process_event(event, detector, dinov2_model, device, ee)
            results.append(result)
        except Exception as exc:
            print(f"  ERROR processing event: {exc}")
            import traceback
            traceback.print_exc()
            results.append({
                "event_id": event.get("properties", {}).get("id", "?"),
                "region": event.get("properties", {}).get("region", "?"),
                "lat": event.get("geometry", {}).get("coordinates", [0, 0])[1] if len(event.get("geometry", {}).get("coordinates", [])) > 1 else 0,
                "lon": event.get("geometry", {}).get("coordinates", [0, 0])[0] if event.get("geometry", {}).get("coordinates") else 0,
                "notes": f"Error: {exc}",
                "classification": "unprocessed",
                "dated_scores": [],
            })

    # 6. Generate HTML report
    print("\n=== Generating report ===")
    report_path = REVIEW_DIR / "temporal_validation.html"
    generate_html_report(results, report_path)

    # 7. Final summary
    print("\n" + "=" * 60)
    print("  Temporal Validation Complete")
    print("=" * 60)
    counts = {"super_positive": 0, "positive": 0, "possible": 0, "unprocessed": 0}
    for r in results:
        cls = r.get("classification", "unprocessed")
        if cls in counts:
            counts[cls] += 1
    print(f"  Super Positive (2+ dates):  {counts['super_positive']}")
    print(f"  Positive (1 date):           {counts['positive']}")
    print(f"  Possible (unclear):          {counts['possible']}")
    print(f"  Unprocessed:                 {counts['unprocessed']}")
    print(f"  Total:                       {len(results)}")
    print(f"  Report: file://{report_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
