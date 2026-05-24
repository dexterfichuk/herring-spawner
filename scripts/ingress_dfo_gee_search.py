#!/usr/bin/env python3
"""Search Sentinel-2 for DFO spawn events and build a review page.

Workflow:
  1. Load data/ingressed/dfo_events.json
  2. Filter to 2016-2025 events with valid coordinates
  3. Deduplicate by date + location
  4. Rank by spawn_length_m descending and keep the top 200
  5. Search Sentinel-2 (GEE project redd-fish) within ±14 days, cloud < 30%
  6. Download the best thumbnail per event into data/ingressed/thumbnails/
  7. Generate data/ingressed/review.html and a manifest
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Iterable

import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "data" / "ingressed" / "dfo_events.json"
OUTPUT_DIR = REPO_ROOT / "data" / "ingressed"
THUMB_DIR = OUTPUT_DIR / "thumbnails"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
REVIEW_HTML_PATH = OUTPUT_DIR / "review.html"

PROJECT = "redd-fish"
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
YEAR_MIN = 2016
YEAR_MAX = 2025
SEARCH_DAYS = 14
MAX_CLOUD = 30
TOP_N = 200
THUMB_DIMENSIONS = 512
REGION_BUFFER_METERS = 1280


@dataclass(frozen=True)
class DfoEvent:
    event_id: str
    start_date: date
    end_date: date
    location: str
    lon: float
    lat: float
    spawn_length_m: float
    spawn_width_m: float
    source: str


def _as_float(value) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, str) and not value.strip():
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _parse_date(value) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _normalize_location(value: str | None) -> str:
    if not value:
        return "unknown"
    return " ".join(str(value).strip().lower().split())


def _event_key(event: DfoEvent) -> tuple:
    location_key = _normalize_location(event.location)
    if location_key == "unknown":
        return (event.start_date.isoformat(), round(event.lat, 6), round(event.lon, 6))
    return (event.start_date.isoformat(), location_key)


def load_and_rank_events(path: Path, top_n: int) -> list[DfoEvent]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    deduped: dict[tuple, DfoEvent] = {}

    for item in payload:
        start_date = _parse_date(item.get("start_date"))
        if start_date is None or not (YEAR_MIN <= start_date.year <= YEAR_MAX):
            continue

        geom = item.get("geometry") or {}
        coords = geom.get("coordinates") if geom.get("type") == "Point" else None
        if not coords or len(coords) < 2:
            continue

        lon = _as_float(coords[0])
        lat = _as_float(coords[1])
        if lon is None or lat is None:
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue

        props = item.get("properties") or {}
        location = str(props.get("location") or "unknown")
        spawn_length = _as_float(props.get("spawn_length_m")) or 0.0
        spawn_width = _as_float(props.get("spawn_width_m")) or 0.0
        end_date = _parse_date(item.get("end_date")) or start_date

        event = DfoEvent(
            event_id=str(item.get("event_id") or ""),
            start_date=start_date,
            end_date=end_date,
            location=location,
            lon=lon,
            lat=lat,
            spawn_length_m=spawn_length,
            spawn_width_m=spawn_width,
            source=str(item.get("source") or ""),
        )

        key = _event_key(event)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = event
            continue

        candidate_rank = (event.spawn_length_m, event.spawn_width_m, event.event_id)
        existing_rank = (existing.spawn_length_m, existing.spawn_width_m, existing.event_id)
        if candidate_rank > existing_rank:
            deduped[key] = event

    ranked = sorted(
        deduped.values(),
        key=lambda e: (-e.spawn_length_m, -e.spawn_width_m, e.start_date.isoformat(), _normalize_location(e.location), e.event_id),
    )
    return ranked[:top_n]


def _scene_date(scene_id: str) -> date:
    return date(int(scene_id[0:4]), int(scene_id[4:6]), int(scene_id[6:8]))


def _scene_date_from_millis(value) -> date | None:
    try:
        if value is None:
            return None
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).date()
    except (TypeError, ValueError, OverflowError):
        return None


def search_best_scene(event: DfoEvent, collection, ee, search_days: int, max_cloud: float) -> dict | None:
    search_start = (event.start_date - timedelta(days=search_days)).isoformat()
    search_end = (event.end_date + timedelta(days=search_days)).isoformat()
    region = ee.Geometry.Point(event.lon, event.lat).buffer(REGION_BUFFER_METERS).bounds()

    scenes = (
        collection.filterBounds(region)
        .filterDate(search_start, search_end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    scene_ids = scenes.aggregate_array("system:index").getInfo() or []
    clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo() or []
    time_starts = scenes.aggregate_array("system:time_start").getInfo() or []

    if not scene_ids:
        return None

    best = None
    for scene_id, cloud, millis in zip(scene_ids, clouds, time_starts):
        scene_date = _scene_date_from_millis(millis) or _scene_date(scene_id)
        days_from_spawn = abs((scene_date - event.start_date).days)
        candidate = {
            "scene_id": scene_id,
            "scene_date": scene_date.isoformat(),
            "cloud": float(cloud),
            "days_from_spawn": days_from_spawn,
        }
        sort_key = (candidate["cloud"], candidate["days_from_spawn"], candidate["scene_date"], candidate["scene_id"])
        if best is None or sort_key < best[0]:
            best = (sort_key, candidate)

    return best[1] if best else None


def download_thumbnail(event: DfoEvent, scene: dict, collection, ee, thumb_dir: Path) -> tuple[str, bool]:
    region = ee.Geometry.Point(event.lon, event.lat).buffer(REGION_BUFFER_METERS).bounds()
    scene_img = ee.Image(f"{COLLECTION}/{scene['scene_id']}")
    rgb = scene_img.select(["B4", "B3", "B2"])
    url = rgb.getThumbURL(
        {
            "min": 0,
            "max": 3000,
            "region": region,
            "dimensions": THUMB_DIMENSIONS,
            "format": "png",
        }
    )

    filename = f"{event.event_id}_{scene['scene_date']}_{scene['scene_id'][:8]}.png"
    thumb_path = thumb_dir / filename
    if thumb_path.exists():
        return filename, False

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    thumb_path.write_bytes(resp.content)
    return filename, True


def build_review_html(
    rows: list[dict],
    output_path: Path,
    top_n: int,
    search_days: int,
    max_cloud: float,
) -> None:
    rows_sorted = sorted(rows, key=lambda r: (-r["spawn_length_m"], -r["spawn_width_m"], r["start_date"], r["location"], r["event_id"]))
    cards = []
    for row in rows_sorted:
        cloud_text = f"cloud {row['cloud']:.1f}%" if row.get("cloud") is not None else "no clear scene"
        date_text = row.get("scene_date") or "no scene"
        image_html = (
            f'<img src="thumbnails/{escape(row["thumbnail_path"])}" alt="{escape(row["event_id"])}">'
            if row.get("thumbnail_path")
            else '<div class="placeholder">No clear scene</div>'
        )
        cards.append(
            f"""
            <article class="card">
              {image_html}
              <div class="body">
                <div class="title">{escape(row['location'])}</div>
                <div class="meta">{escape(row['start_date'])} · {row['spawn_length_m']:.0f} m × {row['spawn_width_m']:.0f} m</div>
                <div class="meta">{escape(date_text)} · {escape(cloud_text)} · {row.get('days_from_spawn', 0) if row.get('days_from_spawn') is not None else '—'}d</div>
                <div class="meta mono">{escape(row['event_id'])}</div>
              </div>
            </article>
            """
        )

    summary = {
        "selected_events": len(rows_sorted),
        "clear_events": sum(1 for row in rows_sorted if row.get("scene_id")),
        "thumbnails": sum(1 for row in rows_sorted if row.get("thumbnail_path")),
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DFO GEE Search Review</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #f4f6fb; color: #1a1a2e; }}
    header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 1.25rem 1.5rem; }}
    .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; padding: 1rem 1.5rem; }}
    .stat {{ background: white; border-radius: 10px; padding: 0.75rem 1rem; min-width: 150px; box-shadow: 0 1px 6px rgba(0,0,0,.08); }}
    .stat strong {{ display: block; font-size: 1.6rem; }}
    .stat span {{ font-size: .75rem; color: #666; text-transform: uppercase; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; padding: 0 1.5rem 1.5rem; }}
    .card {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,.08); }}
    .card img, .placeholder {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; background: #e9ecf4; }}
    .placeholder {{ display: grid; place-items: center; color: #667; font-size: .9rem; }}
    .body {{ padding: .85rem .9rem 1rem; }}
    .title {{ font-weight: 700; margin-bottom: .25rem; }}
    .meta {{ font-size: .8rem; color: #666; margin-top: .2rem; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-all; }}
  </style>
</head>
<body>
  <header>
    <h1>DFO GEE Search Review</h1>
    <div>Top {top_n} deduplicated events from {YEAR_MIN}–{YEAR_MAX} · cloud &lt; {max_cloud}% · ±{search_days} days</div>
  </header>
  <section class="stats">
    <div class="stat"><strong>{summary['selected_events']}</strong><span>selected events</span></div>
    <div class="stat"><strong>{summary['clear_events']}</strong><span>clear scenes</span></div>
    <div class="stat"><strong>{summary['thumbnails']}</strong><span>thumbnails downloaded</span></div>
  </section>
  <main class="grid">
    {''.join(cards) if cards else '<p style="padding:1rem 1.5rem">No thumbnails available.</p>'}
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--search-days", type=int, default=SEARCH_DAYS)
    parser.add_argument("--max-cloud", type=float, default=MAX_CLOUD)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = args.output_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    events = load_and_rank_events(args.input, args.top_n)

    try:
        import ee
    except ImportError as error:  # pragma: no cover - environment issue
        raise RuntimeError("earthengine-api is required to query Sentinel-2") from error

    ee.Initialize(project=PROJECT)
    collection = ee.ImageCollection(COLLECTION)

    rows: list[dict] = []
    clear_events = 0
    downloaded = 0
    for event in events:
        scene = search_best_scene(event, collection, ee, args.search_days, args.max_cloud)
        if scene is None:
            rows.append(
                {
                    "event_id": event.event_id,
                    "start_date": event.start_date.isoformat(),
                    "location": event.location,
                    "spawn_length_m": event.spawn_length_m,
                    "spawn_width_m": event.spawn_width_m,
                    "scene_id": None,
                    "scene_date": None,
                    "cloud": None,
                    "days_from_spawn": None,
                    "thumbnail_path": None,
                }
            )
            continue

        clear_events += 1
        thumb_name, was_downloaded = download_thumbnail(event, scene, collection, ee, thumb_dir)
        downloaded += int(was_downloaded)
        rows.append(
            {
                "event_id": event.event_id,
                "start_date": event.start_date.isoformat(),
                "location": event.location,
                "spawn_length_m": event.spawn_length_m,
                "spawn_width_m": event.spawn_width_m,
                "scene_id": scene["scene_id"],
                "scene_date": scene["scene_date"],
                "cloud": scene["cloud"],
                "days_from_spawn": scene["days_from_spawn"],
                "thumbnail_path": thumb_name,
            }
        )

    manifest = {
        "project": PROJECT,
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "selected_events": len(events),
        "clear_events": clear_events,
        "thumbnails_downloaded": downloaded,
        "thumbnails": rows,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    build_review_html(rows, REVIEW_HTML_PATH, args.top_n, args.search_days, args.max_cloud)

    print(f"Selected events: {len(events)}")
    print(f"Clear scenes: {clear_events}")
    print(f"Thumbnails downloaded: {downloaded}")
    print(f"Manifest: file://{MANIFEST_PATH.resolve()}")
    print(f"Review: file://{REVIEW_HTML_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
