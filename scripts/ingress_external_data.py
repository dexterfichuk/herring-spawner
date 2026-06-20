#!/usr/bin/env python3
"""Ingest external herring spawn datasets and build a review page.

Reads the DFO spawn index, Alaska survey data, and optional Washington data
from the local Downloads folder, normalizes them into the project's Event model,
writes per-source JSON files, and fetches Sentinel-2 thumbnails for dated point
events into a combined review page.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import timedelta
from html import escape
from pathlib import Path
from typing import Callable

import requests

from herring_spawner.config import Settings
from herring_spawner.datasets.alaska import load_alaska_csv
from herring_spawner.datasets.dfo import load_dfo_csv
from herring_spawner.datasets.washington import load_washington_csv
from herring_spawner.models import Event

DEFAULT_DOWNLOADS_DIR = Path("/Users/dexterfichuk/Downloads")
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
THUMB_DIMENSIONS = 512


def wait_for_file(path: Path, retries: int = 12, delay_seconds: int = 10) -> bool:
    for attempt in range(retries + 1):
        if path.exists():
            return True
        if attempt < retries:
            time.sleep(delay_seconds)
    return False


def discover_washington_csv(downloads_dir: Path) -> Path | None:
    candidates = [
        downloads_dir / "washington_herring_surveys.csv",
        downloads_dir / "wdfw_herring_surveys.csv",
        downloads_dir / "washington_herring.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for pattern in ("*wash*.csv", "*WDFW*.csv", "*wdfw*.csv"):
        matches = sorted(downloads_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def serialize_events(events: list[Event], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([event.to_record() for event in events], indent=2),
        encoding="utf-8",
    )


def build_review_page(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["event_id"]].append(row)

    source_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        source_counts[row["source_type"]] += 1

    cards = []
    for event_id in sorted(grouped):
        event_rows = sorted(grouped[event_id], key=lambda row: row["acquired"])
        first = event_rows[0]
        thumbs = "".join(
            f"""
            <figure class="thumb">
              <img src="{escape(row['thumbnail_path'])}" alt="{escape(row['scene_id'])}">
              <figcaption>{escape(row['acquired'])} · cloud {row['cloud']:.1f}%</figcaption>
            </figure>
            """
            for row in event_rows
        )
        cards.append(
            f"""
            <section class="event-card">
              <header>
                <h2>{escape(event_id)}</h2>
                <div class="meta">{escape(first['source_type'])} · {escape(first['location'])} · {escape(first['start_date'])}</div>
                <div class="meta">{len(event_rows)} thumbnails</div>
              </header>
              <div class="thumb-grid">{thumbs}</div>
            </section>
            """
        )

    summary = "".join(
        f"<div class='stat'><strong>{count}</strong><span>{escape(source)}</span></div>"
        for source, count in sorted(source_counts.items())
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>External Herring Data Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #f5f6fa; color: #1a1a2e; }}
    .header {{ padding: 1.25rem 1.5rem; background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; }}
    .stats {{ display: flex; gap: 1rem; padding: 1rem 1.5rem; flex-wrap: wrap; }}
    .stat {{ background: white; border-radius: 10px; padding: 0.8rem 1rem; min-width: 120px; box-shadow: 0 1px 6px rgba(0,0,0,0.08); }}
    .stat strong {{ display: block; font-size: 1.6rem; }}
    .stat span {{ font-size: 0.75rem; color: #666; text-transform: uppercase; }}
    .content {{ padding: 0 1.5rem 1.5rem; display: grid; gap: 1rem; }}
    .event-card {{ background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }}
    .event-card header {{ padding: 1rem 1.1rem; border-bottom: 1px solid #eee; }}
    .event-card h2 {{ margin: 0 0 0.4rem; font-size: 1rem; }}
    .meta {{ font-size: 0.85rem; color: #666; margin-top: 0.2rem; }}
    .thumb-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 0; }}
    .thumb {{ margin: 0; border-right: 1px solid #f0f0f0; border-top: 1px solid #f0f0f0; padding: 0.75rem; }}
    .thumb img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; border-radius: 8px; border: 1px solid #ddd; }}
    .thumb figcaption {{ margin-top: 0.4rem; font-size: 0.78rem; color: #666; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>External Herring Data Review</h1>
    <p>DFO, Alaska, and optional Washington event ingestion results.</p>
  </div>
  <div class="stats">{summary}</div>
  <main class="content">
    {''.join(cards) if cards else '<p>No thumbnails were downloaded.</p>'}
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _load_source_events(
    name: str,
    path: Path | None,
    loader: Callable[[Path], list[Event]],
    output_dir: Path,
) -> list[Event]:
    if path is None:
        events: list[Event] = []
    else:
        events = loader(path)

    serialize_events(events, output_dir / f"{name}_events.json")
    return events


def _scene_date(scene_id: str) -> str:
    return f"{scene_id[:4]}-{scene_id[4:6]}-{scene_id[6:8]}"


def download_thumbnails(
    events: list[Event],
    output_dir: Path,
    project: str,
    search_days: int,
    max_cloud: float,
    max_scenes_per_event: int,
) -> list[dict]:
    try:
        import ee
    except ImportError as error:  # pragma: no cover - environment issue
        raise RuntimeError("earthengine-api is required for thumbnail downloads") from error

    ee.Initialize(project=project)
    collection = ee.ImageCollection(COLLECTION)
    thumbnail_dir = output_dir / "thumbnails"
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for event in events:
        if event.start_date is None or event.geometry.geom_type != "Point":
            continue

        end_date = event.end_date or event.start_date
        search_start = (event.start_date - timedelta(days=search_days)).isoformat()
        search_end = (end_date + timedelta(days=search_days)).isoformat()
        lon, lat = event.geometry.x, event.geometry.y
        region = ee.Geometry.Point(lon, lat).buffer(1280).bounds()

        scenes = (
            collection.filterBounds(region)
            .filterDate(search_start, search_end)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        scene_ids = scenes.aggregate_array("system:index").getInfo() or []
        clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo() or []

        for scene_id, cloud in list(zip(scene_ids, clouds))[:max_scenes_per_event]:
            scene = ee.Image(f"{COLLECTION}/{scene_id}")
            rgb = scene.select(["B4", "B3", "B2"])
            url = rgb.getThumbURL(
                {
                    "min": 0,
                    "max": 3000,
                    "region": region,
                    "dimensions": THUMB_DIMENSIONS,
                    "format": "png",
                }
            )

            scene_day = _scene_date(scene_id)
            filename = f"{event.event_id}_{scene_day}_{scene_id[:8]}.png"
            thumb_path = thumbnail_dir / filename
            if not thumb_path.exists():
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                thumb_path.write_bytes(resp.content)

            rows.append(
                {
                    "event_id": event.event_id,
                    "source_type": event.source_type.value,
                    "location": event.properties.get("location", "unknown"),
                    "start_date": event.start_date.isoformat(),
                    "acquired": scene_day,
                    "scene_id": scene_id,
                    "cloud": float(cloud),
                    "thumbnail_path": f"thumbnails/{filename}",
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads-dir", type=Path, default=DEFAULT_DOWNLOADS_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ingressed"))
    parser.add_argument("--project", default=Settings().gee_project)
    parser.add_argument("--search-days", type=int, default=14)
    parser.add_argument("--max-cloud", type=float, default=50)
    parser.add_argument("--max-scenes-per-event", type=int, default=2)
    parser.add_argument("--wait-retries", type=int, default=12)
    parser.add_argument("--wait-seconds", type=int, default=10)
    args = parser.parse_args()

    dfo_path = args.downloads_dir / "dfo_herring_spawn_index.csv"
    alaska_path = args.downloads_dir / "alaska_herring_surveys.csv"
    washington_path = discover_washington_csv(args.downloads_dir)

    for path in [dfo_path, alaska_path]:
        if path.exists():
            continue
        if not wait_for_file(path, retries=args.wait_retries, delay_seconds=args.wait_seconds):
            print(f"WARNING: {path.name} not found; skipping")

    if washington_path is not None and not washington_path.exists():
        washington_path = None

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_events = {
        "dfo": _load_source_events("dfo", dfo_path if dfo_path.exists() else None, load_dfo_csv, output_dir),
        "alaska": _load_source_events(
            "alaska", alaska_path if alaska_path.exists() else None, load_alaska_csv, output_dir
        ),
        "washington": _load_source_events(
            "washington",
            washington_path if washington_path is not None and washington_path.exists() else None,
            load_washington_csv,
            output_dir,
        ),
    }

    all_events = [event for events in source_events.values() for event in events]
    thumbnails = download_thumbnails(
        events=all_events,
        output_dir=output_dir,
        project=args.project,
        search_days=args.search_days,
        max_cloud=args.max_cloud,
        max_scenes_per_event=args.max_scenes_per_event,
    )

    manifest = {
        "sources": {name: len(events) for name, events in source_events.items()},
        "thumbnail_count": len(thumbnails),
        "thumbnails": thumbnails,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    build_review_page(thumbnails, output_dir / "review.html")

    print("Ingest summary:")
    for source_name, events in source_events.items():
        print(f"  {source_name}: {len(events)} events")
    print(f"  thumbnails: {len(thumbnails)}")
    print(f"  review: file://{(output_dir / 'review.html').resolve()}")


if __name__ == "__main__":
    main()
