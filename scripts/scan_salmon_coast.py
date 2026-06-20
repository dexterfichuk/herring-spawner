#!/usr/bin/env python3
"""Focused temporal scan of the Salmon Coast Research Station area.

Scans the requested Salmon Coast regions with a 0.01° grid, searches Sentinel-2
imagery in the requested Feb-May spawn window, scores the best scene with the trained
`data/models/improved_model.pkl` detector, and temporally validates any point
with score > 0.3 using ±7 day scenes.

Outputs:
  - data/candidates_salmon_coast/manifest.json
  - data/candidates_salmon_coast/review.html
  - data/candidates_salmon_coast/candidates.geojson
  - candidate thumbnails only (never non-candidate imagery)
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.improved_detector import (  # noqa: E402
    ImprovedDetector,
    download_thumbnail,
    extract_features_from_bytes,
)

STATION_LAT = 50.746
STATION_LON = -126.498

REGIONS: list[dict[str, Any]] = [
    {"name": "salmn-coast-base", "lat": 50.746, "lon": -126.498, "radius_km": 5},
    {"name": "broughton-arch", "lat": 50.75, "lon": -126.50, "radius_km": 20},
    {"name": "echo-bay", "lat": 50.77, "lon": -126.60, "radius_km": 8},
    {"name": "gilford-island", "lat": 50.70, "lon": -126.55, "radius_km": 10},
    {"name": "knight-inlet", "lat": 50.85, "lon": -126.00, "radius_km": 12},
    {"name": "tribune-channel", "lat": 50.65, "lon": -126.35, "radius_km": 8},
    {"name": "kingcome-inlet", "lat": 50.95, "lon": -126.25, "radius_km": 8},
    {"name": "wachlis-village", "lat": 50.65, "lon": -126.15, "radius_km": 8},
]

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "improved_model.pkl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "candidates_salmon_coast"
DEFAULT_START = "2024-02-01"
DEFAULT_END = "2024-05-31"
DEFAULT_GRID_SPACING = 0.01
DEFAULT_THRESHOLD = 0.3
DEFAULT_SEARCH_DAYS = 7
DEFAULT_MAX_CLOUD = 50

LOG_LINE_RE = re.compile(
    r"^\s*\[(?P<idx>\d+)/(?:\d+)\]\s+\([^)]*\)\s+"
    r"(?P<region>[a-z0-9\-]+)\s+\((?P<lat>-?\d+\.\d+),\s+(?P<lon>-?\d+\.\d+)\)\s+\|"
)

_print_lock = threading.Lock()


def resolve_scan_config(
    *,
    year: int | None,
    output: Path | None,
    start: str | None,
    end: str | None,
) -> tuple[Path, str, str]:
    if year is not None:
        default_start = f"{year}-02-01"
        default_end = f"{year}-05-31"
        default_output = Path(f"{DEFAULT_OUTPUT}_{year}")
    else:
        default_start = DEFAULT_START
        default_end = DEFAULT_END
        default_output = DEFAULT_OUTPUT

    resolved_output = output if output is not None else default_output
    resolved_start = start if start is not None else default_start
    resolved_end = end if end is not None else default_end
    return resolved_output, resolved_start, resolved_end


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def log_path(output_dir: Path) -> Path:
    return output_dir / "scan.log"


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path(output_dir), "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def point_key(region: str, lat: float, lon: float) -> tuple[str, float, float]:
    return region, round(lat, 4), round(lon, 4)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def generate_grid_points(regions: list[dict[str, Any]], spacing_deg: float) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for region in regions:
        lat = float(region["lat"])
        lon = float(region["lon"])
        radius_km = float(region["radius_km"])
        radius_deg_lat = radius_km / 111.0
        radius_deg_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
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
                dlat = (p_lat - lat) * 111.0
                dlon = (p_lon - lon) * 111.0 * math.cos(math.radians(lat))
                if math.sqrt(dlat**2 + dlon**2) <= radius_km:
                    points.append({"region": region["name"], "lat": round(p_lat, 6), "lon": round(p_lon, 6)})
                    region_points += 1

        print(f"  {region['name']}: {region_points} grid points")
    return points


def load_processed_points(log_file: Path) -> set[tuple[str, float, float]]:
    processed: set[tuple[str, float, float]] = set()
    if not log_file.exists():
        return processed
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LOG_LINE_RE.match(line)
        if not match:
            continue
        processed.add(
            point_key(
                match.group("region"),
                float(match.group("lat")),
                float(match.group("lon")),
            )
        )
    return processed


def find_best_scene(
    ee_module: Any,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_cloud: float,
) -> dict[str, Any] | None:
    try:
        collection = ee_module.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        point = ee_module.Geometry.Point(lon, lat)
        scenes = (
            collection.filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee_module.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        scene_ids = scenes.aggregate_array("system:index").getInfo()
        clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        if not scene_ids:
            return None
        sid = scene_ids[0]
        return {
            "scene_id": sid,
            "cloud": float(clouds[0]),
            "date": f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}",
            "lat": lat,
            "lon": lon,
        }
    except Exception as exc:
        print(f"    GEE search error at ({lat:.4f}, {lon:.4f}): {exc}")
        return None


def find_scenes_for_location(
    ee_module: Any,
    lat: float,
    lon: float,
    center_date: str,
    days_window: int,
    max_cloud: float,
) -> list[dict[str, Any]]:
    from datetime import datetime, timedelta

    center = datetime.strptime(center_date, "%Y-%m-%d")
    start = (center - timedelta(days=days_window)).strftime("%Y-%m-%d")
    end = (center + timedelta(days=days_window)).strftime("%Y-%m-%d")

    collection = ee_module.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    point = ee_module.Geometry.Point(lon, lat)
    scenes = (
        collection.filterBounds(point)
        .filterDate(start, end)
        .filter(ee_module.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )
    try:
        scene_ids = scenes.aggregate_array("system:index").getInfo()
        clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        times = scenes.aggregate_array("system:time_start").getInfo()
    except Exception as exc:
        print(f"    GEE query error at ({lat:.4f}, {lon:.4f}): {exc}")
        return []

    results: list[dict[str, Any]] = []
    for idx, sid in enumerate(scene_ids):
        date = f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}" if len(sid) >= 8 else ""
        results.append(
            {
                "scene_id": sid,
                "cloud": float(clouds[idx]) if idx < len(clouds) else 100.0,
                "date": date,
                "timestamp_ms": times[idx] if idx < len(times) else 0,
                "lat": lat,
                "lon": lon,
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for item in results:
        d = item["date"]
        if d not in deduped or item["cloud"] < deduped[d]["cloud"]:
            deduped[d] = item

    return sorted(deduped.values(), key=lambda x: x["date"])


def score_thumbnail_bytes(
    png_bytes: bytes,
    detector: ImprovedDetector,
    dinov2_model: torch.nn.Module,
    device: torch.device,
) -> float | None:
    feats = extract_features_from_bytes(png_bytes, dinov2_model, device)
    if feats is None:
        return None
    return detector.score_svm(feats.dinov2_embedding)


def classify(dated_scores: list[dict[str, Any]], threshold: float) -> str:
    valid = [d for d in dated_scores if d.get("score") is not None]
    above_dates = sorted({d["date"] for d in valid if d["score"] > threshold})
    if len(above_dates) >= 2:
        return "super_positive"
    if len(above_dates) == 1:
        return "positive"
    return "not_spawn"


def sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def save_candidate_thumbnail(output_dir: Path, point: dict[str, Any], scene: dict[str, Any], score: float, png_bytes: bytes) -> str:
    region = point["region"]
    date = scene["date"]
    scene_short = scene["scene_id"][:8] if len(scene["scene_id"]) >= 8 else scene["scene_id"]
    filename = sanitize_filename(
        f"{region}_{date}_score{score:.2f}_{point['lat']:.5f}_{point['lon']:.5f}_{scene_short}.png"
    )
    (output_dir / filename).write_bytes(png_bytes)
    return filename


def temporal_validate(
    candidate: dict[str, Any],
    detector: ImprovedDetector,
    dinov2_model: torch.nn.Module,
    device: torch.device,
    ee_module: Any,
    output_dir: Path,
    search_days: int,
    max_cloud: float,
) -> dict[str, Any]:
    lat = candidate["lat"]
    lon = candidate["lon"]
    date = candidate["date"]
    region = candidate["region"]
    original_bytes = candidate.pop("_original_bytes")

    cache_dir = output_dir / "temporal_cache" / f"{region}_{lat}_{lon}".replace(".", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)

    scenes = find_scenes_for_location(ee_module, lat, lon, date, search_days, max_cloud)
    dated_scores: list[dict[str, Any]] = []

    for scene in scenes:
        sid = scene["scene_id"]
        scene_date = scene["date"]
        is_original = scene_date == date and sid == candidate["scene_id"]
        cache_path = cache_dir / f"{sid}_{lat:.5f}_{lon:.5f}.png"

        if is_original:
            png_bytes = original_bytes
            if not cache_path.exists():
                cache_path.write_bytes(original_bytes)
        elif cache_path.exists():
            png_bytes = cache_path.read_bytes()
        else:
            png_bytes = download_thumbnail(ee_module, lat, lon, sid)
            if png_bytes is not None:
                cache_path.write_bytes(png_bytes)

        score = None
        if png_bytes is not None:
            score = score_thumbnail_bytes(png_bytes, detector, dinov2_model, device)

        dated_scores.append(
            {
                "date": scene_date,
                "scene_id": sid,
                "cloud": scene["cloud"],
                "score": score,
                "is_original": is_original,
            }
        )

    dated_scores.sort(key=lambda x: x["date"])
    cls = classify(dated_scores, threshold=candidate["threshold"])
    above = sorted({d["date"] for d in dated_scores if d.get("score") is not None and d["score"] > candidate["threshold"]})

    candidate["classification"] = cls
    candidate["n_dates_above"] = len(above)
    candidate["n_dates_total"] = len(dated_scores)
    candidate["dated_scores"] = dated_scores
    return candidate


def process_point(
    point: dict[str, Any],
    detector: ImprovedDetector,
    dinov2_model: torch.nn.Module,
    device: torch.device,
    ee_module: Any,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    lat = point["lat"]
    lon = point["lon"]
    region = point["region"]

    scene = find_best_scene(ee_module, lat, lon, args.start, args.end, args.max_cloud)
    if scene is None:
        return None

    png_bytes = download_thumbnail(ee_module, lat, lon, scene["scene_id"])
    if png_bytes is None:
        return None

    score = score_thumbnail_bytes(png_bytes, detector, dinov2_model, device)
    if score is None or score <= args.threshold:
        return None

    candidate = {
        "region": region,
        "lat": lat,
        "lon": lon,
        "date": scene["date"],
        "scene_id": scene["scene_id"],
        "cloud": scene["cloud"],
        "score": round(float(score), 6),
        "threshold": args.threshold,
        "thumbnail_path": save_candidate_thumbnail(output_dir, point, scene, float(score), png_bytes),
        "_original_bytes": png_bytes,
    }

    print(f"\n    >>> candidate {region} ({lat:.4f}, {lon:.4f}) score={score:.4f}; validating ±{args.search_days}d")
    return temporal_validate(candidate, detector, dinov2_model, device, ee_module, output_dir, args.search_days, args.max_cloud)


def candidate_priority(entry: dict[str, Any]) -> tuple[int, float, str, str]:
    cls_order = {"super_positive": 0, "positive": 1, "not_spawn": 2, "unknown": 3}
    return cls_order.get(entry.get("classification", "unknown"), 3), -float(entry.get("score", 0)), str(entry.get("date", "")), str(entry.get("region", ""))


def merge_candidates(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[float, float], dict[str, Any]] = {}
    for entry in existing + new:
        key = (round(float(entry.get("lat", 0.0)), 4), round(float(entry.get("lon", 0.0)), 4))
        if key not in merged or float(entry.get("score", 0)) > float(merged[key].get("score", 0)):
            merged[key] = entry
    return sorted(merged.values(), key=candidate_priority)


def build_geojson(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    features = []
    for c in candidates:
        props = {k: v for k, v in c.items() if k != "_original_bytes"}
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]},
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def render_review_page(
    candidates: list[dict[str, Any]],
    output_dir: Path,
    stats: dict[str, Any],
    start_date: str,
    end_date: str,
) -> str:
    sections = ["super_positive", "positive", "not_spawn"]
    cards: list[str] = []
    for cls in sections:
        for c in [x for x in candidates if x.get("classification") == cls]:
            thumb = html.escape(str(c.get("thumbnail_path", "")))
            dated_rows = []
            for ds in c.get("dated_scores", []):
                score = ds.get("score")
                score_txt = "N/A" if score is None else f"{float(score):.3f}"
                mark = "★" if ds.get("is_original") else ""
                dated_rows.append(
                    f"<tr><td>{html.escape(str(ds.get('date', '')))}</td><td>{mark}</td><td>{html.escape(score_txt)}</td><td>{float(ds.get('cloud', 0)):.0f}%</td><td>{html.escape(str(ds.get('scene_id', '')))}</td></tr>"
                )
            cards.append(
                f"""
                <article class="card {cls}">
                  <header>
                    <div><strong>{html.escape(str(c.get('region', '')))}</strong></div>
                    <div class="badge {cls}">{cls}</div>
                  </header>
                  <div class="meta">
                    <span>{c.get('lat', 0):.4f}, {c.get('lon', 0):.4f}</span>
                    <span>{html.escape(str(c.get('date', '')))}</span>
                    <span>score {float(c.get('score', 0)):.3f}</span>
                    <span>{int(c.get('n_dates_above', 0))} above / {int(c.get('n_dates_total', 0))} total</span>
                  </div>
                  <div class="body">
                    <img src="{thumb}" alt="{html.escape(str(c.get('region', '')))}" loading="lazy">
                    <table>
                      <thead><tr><th>Date</th><th></th><th>Score</th><th>Cloud</th><th>Scene</th></tr></thead>
                      <tbody>{''.join(dated_rows)}</tbody>
                    </table>
                  </div>
                </article>
                """
            )

    region_rows = "".join(
        f"<tr><td>{html.escape(str(r['name']))}</td><td>{r['lat']}</td><td>{r['lon']}</td><td>{r['radius_km']} km</td></tr>"
        for r in REGIONS
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Salmon Coast Candidates</title>
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0d1117; color: #c9d1d9; }}
  .bar {{ position: sticky; top: 0; background: #161b22; border-bottom: 1px solid #30363d; padding: 16px 20px; z-index: 10; }}
  .bar h1 {{ margin: 0; font-size: 20px; color: #f0f6fc; }}
  .bar .sub {{ margin-top: 4px; color: #8b949e; font-size: 13px; }}
  .summary, .regions {{ margin: 16px; padding: 16px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; }}
  .counts {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .count {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; min-width: 120px; }}
  .count .n {{ font-size: 24px; font-weight: 700; }}
  .count .l {{ font-size: 10px; text-transform: uppercase; letter-spacing: .08em; color: #8b949e; }}
  .cards {{ padding: 0 16px 16px; }}
  .card {{ margin-bottom: 16px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  .card header {{ display: flex; justify-content: space-between; gap: 8px; padding: 12px 14px; border-bottom: 1px solid #21262d; }}
  .badge {{ border-radius: 999px; padding: 3px 10px; font-size: 11px; text-transform: uppercase; border: 1px solid #30363d; }}
  .badge.super_positive {{ color: #3fb950; border-color: #3fb950; background: #0f2d1a; }}
  .badge.positive {{ color: #58a6ff; border-color: #58a6ff; background: #0c2d48; }}
  .badge.not_spawn {{ color: #8b949e; background: #1c1c1c; }}
  .meta {{ display: flex; flex-wrap: wrap; gap: 14px; padding: 8px 14px; font-size: 11px; color: #8b949e; border-bottom: 1px solid #21262d; }}
  .body {{ display: grid; grid-template-columns: 220px 1fr; gap: 12px; padding: 12px 14px; align-items: start; }}
  img {{ width: 220px; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 6px; border: 1px solid #30363d; background: #0d1117; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid #21262d; text-align: left; }}
  th {{ color: #8b949e; font-size: 10px; text-transform: uppercase; letter-spacing: .06em; }}
  @media (max-width: 900px) {{ .body {{ grid-template-columns: 1fr; }} img {{ width: 100%; }} }}
</style>
</head>
<body>
  <div class="bar">
    <h1>Salmon Coast — focused herring spawn scan</h1>
    <div class="sub">super_positives first · {start_date} to {end_date} · grid {DEFAULT_GRID_SPACING}° · threshold {DEFAULT_THRESHOLD}</div>
  </div>
  <div class="summary">
    <div class="counts">
      <div class="count"><div class="n">{stats['super_positive']}</div><div class="l">Super positive</div></div>
      <div class="count"><div class="n">{stats['positive']}</div><div class="l">Positive</div></div>
      <div class="count"><div class="n">{stats['not_spawn']}</div><div class="l">Not spawn</div></div>
      <div class="count"><div class="n">{len(candidates)}</div><div class="l">Total</div></div>
    </div>
  </div>
  <div class="regions">
    <table>
      <thead><tr><th>Region</th><th>Lat</th><th>Lon</th><th>Radius</th></tr></thead>
      <tbody>{region_rows}</tbody>
    </table>
  </div>
  <div class="cards">{''.join(cards)}</div>
</body>
</html>"""


def write_outputs(
    output_dir: Path,
    candidates: list[dict[str, Any]],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    geojson_path = output_dir / "candidates.geojson"
    html_path = output_dir / "review.html"

    compact = []
    for c in candidates:
        entry = dict(c)
        entry.pop("_original_bytes", None)
        compact.append(entry)

    manifest_path.write_text(json.dumps(compact, indent=2), encoding="utf-8")
    geojson_path.write_text(json.dumps(build_geojson(compact), indent=2), encoding="utf-8")

    counts = {"super_positive": 0, "positive": 0, "not_spawn": 0}
    for c in compact:
        counts[c.get("classification", "not_spawn")] = counts.get(c.get("classification", "not_spawn"), 0) + 1

    html_path.write_text(render_review_page(compact, output_dir, counts, start_date, end_date), encoding="utf-8")
    return {"manifest": manifest_path, "geojson": geojson_path, "html": html_path, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Focused Salmon Coast herring spawn scan")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--search-days", type=int, default=DEFAULT_SEARCH_DAYS)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    output_dir, args.start, args.end = resolve_scan_config(
        year=args.year,
        output=args.output,
        start=args.start,
        end=args.end,
    )
    setup_logging(output_dir)

    print("=" * 72)
    print("  Salmon Coast — focused herring spawn scan")
    print("=" * 72)

    points = generate_grid_points(REGIONS, args.grid_spacing)
    print(f"  Total grid points: {len(points)}")

    processed = set()
    if not args.no_resume:
        processed = load_processed_points(log_path(output_dir))
        if processed:
            before = len(points)
            points = [p for p in points if point_key(p["region"], p["lat"], p["lon"]) not in processed]
            print(f"  Resume: skipping {before - len(points)} already-logged points")

    existing_candidates = load_json(output_dir / "manifest.json", [])
    if not points and existing_candidates:
        merged = merge_candidates(existing_candidates, [])
        outputs = write_outputs(output_dir, merged, args.start, args.end)
        print(f"  Nothing left to scan; rebuilt outputs from existing manifest")
        print(f"  Super positives: {outputs['counts']['super_positive']}")
        print(f"  Positives:       {outputs['counts']['positive']}")
        print(f"  Near station:    {sum(1 for c in merged if c['region'] == 'salmn-coast-base')}")
        print(f"  Manifest: {outputs['manifest']}")
        print(f"  GeoJSON:  {outputs['geojson']}")
        print(f"  Review:   {outputs['html']}")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Torch device: {device}")

    print("\n=== Loading models ===")
    try:
        dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        dinov2_model.eval().to(device)
    except Exception as exc:
        print(f"ERROR: failed to load DINOv2: {exc}")
        return 1

    if not MODEL_PATH.exists():
        print(f"ERROR: missing model: {MODEL_PATH}")
        return 1
    try:
        detector = ImprovedDetector.load(MODEL_PATH)
    except Exception as exc:
        print(f"ERROR: failed to load improved model: {exc}")
        return 1

    print("\n=== Initializing GEE ===")
    try:
        import ee

        ee.Initialize(project="redd-fish")
    except Exception as exc:
        print(f"ERROR: GEE init failed: {exc}")
        return 1

    print(f"\n=== Scanning {len(points)} grid points with {args.workers} workers ===")
    print(f"  Date range: {args.start} to {args.end}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Cloud max:  {args.max_cloud}%")
    print(f"  Temporal:   ±{args.search_days} days")
    print(f"  Output:     {output_dir.resolve()}")

    start_time = time.time()
    new_candidates: list[dict[str, Any]] = []

    def progress(idx: int, total: int, region: str, lat: float, lon: float, status: str) -> None:
        elapsed = max(time.time() - start_time, 0.001)
        pct = 100.0 * (idx + 1) / max(total, 1)
        rate = (idx + 1) / elapsed
        remaining = max(total - (idx + 1), 0) / rate if rate > 0 else 0
        eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
        print(f"  [{idx + 1}/{total}] ({pct:.0f}%) {region} ({lat:.4f}, {lon:.4f}) | {status} | ETA {eta}")

    lock = threading.Lock()

    def worker(idx: int, point: dict[str, Any]) -> dict[str, Any] | None:
        try:
            result = process_point(point, detector, dinov2_model, device, ee, output_dir, args)
            with lock:
                if result is None:
                    progress(idx, len(points), point["region"], point["lat"], point["lon"], f"below {args.threshold}")
                else:
                    progress(idx, len(points), point["region"], point["lat"], point["lon"], f"CANDIDATE {result['classification']} {result['score']:.4f}")
            return result
        except Exception as exc:
            with lock:
                progress(idx, len(points), point["region"], point["lat"], point["lon"], f"error: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, idx, point): idx for idx, point in enumerate(points)}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                new_candidates.append(result)

    merged = merge_candidates(existing_candidates, new_candidates)
    outputs = write_outputs(output_dir, merged, args.start, args.end)

    near_station = [c for c in merged if haversine_km(c["lat"], c["lon"], STATION_LAT, STATION_LON) <= 5.0]

    elapsed = time.time() - start_time
    print("\n=== Scan complete ===")
    print(f"  Grid points scanned: {len(points)}")
    print(f"  Candidates total:    {len(merged)}")
    print(f"  Super positives:     {outputs['counts']['super_positive']}")
    print(f"  Positives:           {outputs['counts']['positive']}")
    print(f"  Near station:        {len(near_station)}")
    print(f"  Time:                {elapsed/60:.1f} min")
    print(f"  Manifest:            {outputs['manifest']}")
    print(f"  GeoJSON:             {outputs['geojson']}")
    print(f"  Review:              {outputs['html']}")

    if near_station:
        best = sorted(near_station, key=lambda c: (-float(c.get("score", 0)), c.get("date", "")))[0]
        print(
            "  Nearest detections:  "
            f"{best['region']} {best['date']} score={float(best['score']):.3f} "
            f"({best['lat']:.4f}, {best['lon']:.4f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
