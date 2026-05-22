"""Run GEE scene search across all known spawn events by year."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import ee
import requests
from shapely.geometry import shape

ee.Initialize(project="redd-fish")

payload = json.loads(Path("data/interim/events.geojson").read_text())
events = []
for feat in payload["features"]:
    p = feat["properties"]
    if not p.get("start_date"):
        continue
    events.append({
        "id": p["event_id"],
        "start": date.fromisoformat(p["start_date"]),
        "end": date.fromisoformat(p["end_date"]) if p.get("end_date") else date.fromisoformat(p["start_date"]),
        "geom": shape(feat["geometry"]),
        "source": p["source"],
    })

collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
all_results = []

for event in events:
    start = (event["start"] - timedelta(days=14)).isoformat()
    end = (event["end"] + timedelta(days=14)).isoformat()
    minx, miny, maxx, maxy = event["geom"].buffer(0.015).bounds

    scenes = (
        collection
        .filterBounds(ee.Geometry.Rectangle(minx, miny, maxx, maxy))
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 70))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    ids = scenes.aggregate_array("system:index").getInfo()
    clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()

    for sid, cloud in zip(ids, clouds):
        all_results.append({
            "event_id": event["id"],
            "scene_id": sid,
            "date": f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}",
            "cloud": cloud,
            "bounds": (minx, miny, maxx, maxy),
        })
    print(f"{event['id']}: {len(ids)} scenes")

Path("data/exports").mkdir(parents=True, exist_ok=True)
Path("data/review").mkdir(parents=True, exist_ok=True)

# Export thumbnails for scenes within ±7 days of spawn with cloud < 30%
review_rows = []
for r in all_results:
    if r["cloud"] >= 30:
        continue
    scene_date = date.fromisoformat(r["date"])
    event_start = next(e["start"] for e in events if e["id"] == r["event_id"])
    days_diff = abs((scene_date - event_start).days)
    if days_diff > 14:
        continue

    minx, miny, maxx, maxy = r["bounds"]
    scene_img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{r['scene_id']}")
    rgb = scene_img.select(["B4", "B3", "B2"])

    url = rgb.getThumbURL({
        "min": 0, "max": 3000,
        "region": ee.Geometry.Rectangle(minx, miny, maxx, maxy),
        "dimensions": 512, "format": "png",
    })

    name = f"{r['event_id']}_{r['date']}_{r['scene_id'][:8]}.png"
    thumb_path = f"data/review/{name}"
    resp = requests.get(url, timeout=60)
    with open(thumb_path, "wb") as f:
        f.write(resp.content)

    review_rows.append({
        "chip_id": r["scene_id"],
        "event_id": r["event_id"],
        "acquired": r["date"],
        "thumbnail_path": name,
        "cloud": r["cloud"],
        "days_from_spawn": days_diff,
        "review_label": "unknown",
    })

# Group by event for summary
from collections import defaultdict
by_event = defaultdict(list)
for r in review_rows:
    by_event[r["event_id"]].append(r)

print(f"\n=== {len(review_rows)} thumbnails exported ===")
for eid, thumbs in sorted(by_event.items()):
    print(f"\n{eid}:")
    for t in sorted(thumbs, key=lambda x: x["acquired"]):
        print(f"  {t['acquired']} | cloud={t['cloud']:.1f}% | {t['days_from_spawn']}d from spawn")

if review_rows:
    from herring_spawner.review.static import write_review_page
    write_review_page(review_rows, Path("data/review/review.html"))
    print(f"\nReview: file://{Path('data/review/review.html').resolve()}")

Path("data/interim/gee_search_results.json").write_text(
    json.dumps({"results": all_results, "thumbnails": review_rows}, indent=2), encoding="utf-8"
)
