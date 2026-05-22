"""Fetch real RGB thumbnails from Google Earth Engine for all known spawn events.

Usage:
    python scripts/fetch_gee_thumbnails.py

Requires: earthengine-api, requests (authenticated to GEE project "redd-fish")

Output:
    - PNG thumbnails in data/review/
    - Updates data/interim/gee_search_results.json with thumbnail metadata
"""
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import ee
import requests
from shapely.geometry import shape

from herring_spawner.config import Settings

PROJECT = Settings().gee_project  # "redd-fish"
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
MAX_CLOUD = 30  # maximum cloud percentage for thumbnails
SEARCH_DAYS = 14  # days before/after event to search
THUMB_DIMENSIONS = 512

# ---------------------------------------------------------------------------
# 1. Load events
# ---------------------------------------------------------------------------
ee.Initialize(project=PROJECT)

geo_path = Path(Settings().interim_dir / "events.geojson")
if not geo_path.exists():
    print(f"ERROR: {geo_path} not found. Run scripts/build_event_catalog.py first.")
    sys.exit(1)

payload = json.loads(geo_path.read_text())
events = []
for feat in payload["features"]:
    p = feat["properties"]
    eid = p.get("event_id", "")
    start_str = p.get("start_date")
    source_type = p.get("source_type", "")

    # Skip track events -- only process known spawn events
    if source_type == "track":
        continue
    if not start_str:
        continue

    start_dt = date.fromisoformat(start_str)
    end_str = p.get("end_date")
    end_dt = date.fromisoformat(end_str) if end_str else start_dt
    geom = shape(feat["geometry"])

    events.append({
        "id": eid,
        "start": start_dt,
        "end": end_dt,
        "geom": geom,
        "source": p.get("source", ""),
    })

events.sort(key=lambda e: e["id"])
print(f"Loaded {len(events)} events with start_date (tracks excluded)")

# ---------------------------------------------------------------------------
# 2. Search GEE for scenes
# ---------------------------------------------------------------------------
collection = ee.ImageCollection(COLLECTION)
all_results = []

for event in events:
    search_start = (event["start"] - timedelta(days=SEARCH_DAYS)).isoformat()
    search_end = (event["end"] + timedelta(days=SEARCH_DAYS)).isoformat()
    minx, miny, maxx, maxy = event["geom"].buffer(0.015).bounds

    scenes = (
        collection
        .filterBounds(ee.Geometry.Rectangle(minx, miny, maxx, maxy))
        .filterDate(search_start, search_end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 70))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    ids = scenes.aggregate_array("system:index").getInfo()
    clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()

    for sid, cld in zip(ids, clouds):
        all_results.append({
            "event_id": event["id"],
            "scene_id": sid,
            "date": f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}",
            "cloud": cld,
            "bounds": (minx, miny, maxx, maxy),
        })

    print(f"  {event['id']}: {len(ids)} scenes found")

# ---------------------------------------------------------------------------
# 3. Download RGB thumbnails for best scenes (cloud < 30%, nearest to spawn)
# ---------------------------------------------------------------------------
review_dir = Settings().review_dir
review_dir.mkdir(parents=True, exist_ok=True)

review_rows = []
for r in all_results:
    if r["cloud"] >= MAX_CLOUD:
        continue

    scene_date = date.fromisoformat(r["date"])

    # Find event start date
    event_start = None
    for e in events:
        if e["id"] == r["event_id"]:
            event_start = e["start"]
            break
    if event_start is None:
        continue

    days_diff = abs((scene_date - event_start).days)
    if days_diff > SEARCH_DAYS:
        continue

    minx, miny, maxx, maxy = r["bounds"]
    scene_img = ee.Image(f"{COLLECTION}/{r['scene_id']}")
    rgb = scene_img.select(["B4", "B3", "B2"])

    url = rgb.getThumbURL({
        "min": 0, "max": 3000,
        "region": ee.Geometry.Rectangle(minx, miny, maxx, maxy),
        "dimensions": THUMB_DIMENSIONS,
        "format": "png",
    })

    name = f"{r['event_id']}_{r['date']}_{r['scene_id'][:8]}.png"
    thumb_path = review_dir / name

    # Skip if already downloaded
    if thumb_path.exists():
        print(f"  EXISTS: {name}")
    else:
        print(f"  FETCH: {name} (cloud={r['cloud']:.1f}%, {days_diff}d from spawn)")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        thumb_path.write_bytes(resp.content)

    review_rows.append({
        "chip_id": r["scene_id"],
        "event_id": r["event_id"],
        "acquired": r["date"],
        "thumbnail_path": name,
        "cloud": r["cloud"],
        "days_from_spawn": days_diff,
        "review_label": "unknown",
    })

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
by_event = defaultdict(list)
for r in review_rows:
    by_event[r["event_id"]].append(r)

print(f"\n{'='*60}")
print(f"Downloaded {len(review_rows)} thumbnails for {len(by_event)} events")
for eid, thumbs in sorted(by_event.items()):
    for t in sorted(thumbs, key=lambda x: x["acquired"]):
        print(f"  {eid}: {t['acquired']} | cloud={t['cloud']:.1f}% | {t['days_from_spawn']}d from spawn")

# ---------------------------------------------------------------------------
# 5. Write updated search results
# ---------------------------------------------------------------------------
output_path = Settings().interim_dir / "gee_search_results.json"
existing = {}
if output_path.exists():
    existing = json.loads(output_path.read_text())

output = {
    "results": all_results,
    "thumbnails": review_rows,
}
output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"\nResults written to {output_path}")
print(f"Review thumbnails in: {review_dir}")
