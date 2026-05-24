#!/usr/bin/env python3
"""KNN-based BC coast scan for herring spawn candidates.

This pipeline ingests Strait of Georgia records plus the existing labeled
candidate sets, builds a 3-NN classifier in DINOv2 embedding space, scans the
13 BC habitat regions at 0.02° spacing, and stores only majority-spawn
candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.knn_detector import DINO_TRANSFORM, _load_image_embedding, _pick_device, match_candidate_image
from scripts.scan_bc_coast import (
    REGIONS,
    download_thumbnail,
    find_best_scene,
    generate_grid_points,
    print_progress,
    save_candidate,
    update_manifest,
)


MODEL_NAME = "dinov2_vits14"
DEFAULT_PROJECT = "redd-fish"
DEFAULT_LABELS = Path("/Users/dexterfichuk/Downloads/herring-labels-v2.json")
DEFAULT_ROSE_REVIEW = Path("data/candidates_v2/rose_super_review.json")
DEFAULT_MODE = "bc"
DEFAULT_SOG_FILES = [
    Path("/Users/dexterfichuk/Downloads/sog_data/spawn_index_part1.geojson"),
    Path("/Users/dexterfichuk/Downloads/sog_data/spawn_index_part2.geojson"),
    Path("/Users/dexterfichuk/Downloads/sog_data/spawn_index_part3.geojson"),
]
DEFAULT_SOG_OUTPUT = Path("data/sog_candidates")
DEFAULT_NEGATIVE_DIR = Path("data/samples/negative")
DEFAULT_OUTPUT_DIR = Path("data/candidates_knn")
DEFAULT_INGRESSED_DIR = Path("data/ingressed/thumbnails")
DEFAULT_INGRESSED_SOG_OUTPUT = Path("data/ingressed/sog_events.json")
DEFAULT_START = "2024-02-01"
DEFAULT_END = "2024-05-31"
DEFAULT_MAX_CLOUD = 50.0
DEFAULT_GRID_SPACING = 0.02
DEFAULT_WORKERS = 6
DEFAULT_K = 3
DEFAULT_SOG_SEARCH_DAYS = 14
DEFAULT_SOG_THUMBNAILS_PER_RECORD = 2
SOG_YEAR_MIN = 2016
SOG_YEAR_MAX = 2021


MODEL_LOCK = threading.Lock()
MANIFEST_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _parse_yyyy_mm_dd(text: str) -> str | None:
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return None


def _parse_sog_date(start_value: str) -> str | None:
    if not isinstance(start_value, str) or len(start_value) < 8:
        return None
    try:
        parsed = datetime.strptime(start_value[:8], "%Y%m%d")
    except ValueError:
        return None
    return parsed.date().isoformat()


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def _parse_date_like(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    if len(text) >= 8 and text[:8].isdigit():
        return _parse_sog_date(text)
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


def _scene_date(scene_id: str) -> str | None:
    if len(scene_id) < 8 or not scene_id[:8].isdigit():
        return None
    return f"{scene_id[:4]}-{scene_id[4:6]}-{scene_id[6:8]}"


def _scene_date_from_millis(value: Any) -> str | None:
    try:
        if value is None:
            return None
        return datetime.fromtimestamp(float(value) / 1000.0).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def load_sog_records(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load and rank Strait of Georgia spawn index records."""
    records: list[dict[str, Any]] = []
    summary: Counter[str] = Counter()

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = payload.get("features", [])
        if not isinstance(features, list):
            raise ValueError(f"{path} does not contain a features list")

        for feature in features:
            summary["raw_features"] += 1
            props = feature.get("properties") or {}

            year = props.get("Year")
            if not isinstance(year, int) or not (SOG_YEAR_MIN <= year <= SOG_YEAR_MAX):
                continue

            lon = _as_float(props.get("Longitude"))
            lat = _as_float(props.get("Latitude"))
            if lon is None or lat is None:
                continue
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                continue

            start_date = _parse_date_like(props.get("Start"))
            end_date = _parse_date_like(props.get("End_") or props.get("End"))
            target_date = start_date or end_date
            if target_date is None:
                continue

            location_name = str(props.get("LocationNa") or props.get("LocationCo") or "unknown").strip() or "unknown"
            region = str(props.get("Region") or "").strip() or None
            combined_si = _as_float(props.get("CombinedSI")) or 0.0
            object_id = props.get("OBJECTID", props.get("FID", feature.get("id")))

            records.append(
                {
                    "id": f"sog:{_slugify(location_name)}:{target_date}:{object_id}",
                    "year": year,
                    "region": region,
                    "location_name": location_name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "start_date": start_date,
                    "end_date": end_date,
                    "target_date": target_date,
                    "combined_si": combined_si,
                    "object_id": object_id,
                    "source_path": str(path),
                }
            )
            summary["kept_records"] += 1

    records.sort(
        key=lambda row: (
            -float(row.get("combined_si", 0.0) or 0.0),
            int(row.get("year", 0)),
            str(row.get("region") or ""),
            str(row.get("location_name") or ""),
            str(row.get("id") or ""),
        )
    )
    return records, dict(summary)


def _scene_sort_key(scene: dict[str, Any], target_date: str) -> tuple[float, int, str, str]:
    scene_date = scene.get("date") or "9999-12-31"
    try:
        target = date.fromisoformat(target_date)
        scene_dt = date.fromisoformat(str(scene_date))
        distance = abs((scene_dt - target).days)
    except ValueError:
        distance = 9999
    cloud = float(scene.get("cloud", 1000.0))
    return (cloud, distance, str(scene_date), str(scene.get("scene_id") or ""))


def search_sog_scenes(
    ee_module: Any,
    record: dict[str, Any],
    search_days: int = DEFAULT_SOG_SEARCH_DAYS,
    max_cloud: float = DEFAULT_MAX_CLOUD,
    limit: int = DEFAULT_SOG_THUMBNAILS_PER_RECORD,
) -> list[dict[str, Any]]:
    target_date = str(record["target_date"])
    target = date.fromisoformat(target_date)
    search_start = (target - timedelta(days=search_days)).isoformat()
    search_end = (target + timedelta(days=search_days + 1)).isoformat()

    region = ee_module.Geometry.Point(record["longitude"], record["latitude"]).buffer(1280).bounds()
    collection = ee_module.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    scenes = (
        collection.filterBounds(region)
        .filterDate(search_start, search_end)
        .filter(ee_module.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    scene_ids = scenes.aggregate_array("system:index").getInfo() or []
    clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo() or []
    time_starts = scenes.aggregate_array("system:time_start").getInfo() or []
    candidates: list[dict[str, Any]] = []

    for scene_id, cloud, millis in zip(scene_ids, clouds, time_starts):
        scene_date = _scene_date_from_millis(millis) or _scene_date(str(scene_id))
        if scene_date is None:
            continue
        candidates.append(
            {
                "scene_id": str(scene_id),
                "cloud": float(cloud),
                "date": scene_date,
            }
        )

    candidates.sort(key=lambda scene: _scene_sort_key(scene, target_date))

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for scene in candidates:
        if scene["scene_id"] in seen:
            continue
        seen.add(scene["scene_id"])
        unique.append(scene)
        if len(unique) >= limit:
            break

    return unique


def build_sog_review_html(entries: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    cards = []
    for row in sorted(
        entries,
        key=lambda item: (
            -float(item.get("knn_score", item.get("score", 0.0)) or 0.0),
            -float(item.get("combined_si", 0.0) or 0.0),
            str(item.get("region") or ""),
            str(item.get("date") or ""),
            str(item.get("thumbnail_path") or ""),
        ),
    ):
        lat = float(row.get("latitude", row.get("lat", 0.0)) or 0.0)
        lon = float(row.get("longitude", row.get("lon", 0.0)) or 0.0)
        cards.append(
            f"""
            <article class="card">
              <img src="thumbnails/{html_escape(row['thumbnail_path'])}" alt="sog thumbnail">
              <div class="meta"><strong>{html_escape(row['location_name'])}</strong> · {html_escape(row.get('region') or 'unknown')}</div>
              <div class="meta">date {html_escape(row['date'])} · cloud {float(row['cloud']):.1f}%</div>
              <div class="meta">CombinedSI {float(row['combined_si']):.2f} · KNN score {float(row['knn_score']):.2f} · votes {int(row['spawn_votes'])}/{int(row['k'])}</div>
              <div class="meta">({lat:.4f}, {lon:.4f})</div>
            </article>
            """
        )

    top_regions = summary.get("top_regions", {})
    region_rows = "".join(
        f"<tr><td>{html_escape(region)}</td><td>{count}</td></tr>"
        for region, count in sorted(top_regions.items(), key=lambda item: (-item[1], item[0]))
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SoG spawn candidate review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #f5f6fa; color: #1f2937; }}
    header {{ background: linear-gradient(135deg, #111827, #0f172a); color: white; padding: 24px; }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .stat {{ background: white; border-radius: 12px; padding: 14px 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .value {{ font-size: 28px; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 14px; }}
    .card {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .card img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; }}
    .meta {{ padding: 0 14px 8px; font-size: 13px; color: #374151; }}
    table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-top: 12px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
    th {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <header>
    <h1>SoG spawn candidate review</h1>
    <p>Top Strait of Georgia records from 2016–2021 · ±{DEFAULT_SOG_SEARCH_DAYS} days · cloud &lt; {DEFAULT_MAX_CLOUD}% · 2 thumbnails per record</p>
  </header>
  <main>
    <section class="stats">
      <div class="stat"><div class="label">Records</div><div class="value">{summary['record_count']}</div></div>
      <div class="stat"><div class="label">Thumbnails</div><div class="value">{summary['thumbnail_count']}</div></div>
      <div class="stat"><div class="label">Regions</div><div class="value">{len(top_regions)}</div></div>
      <div class="stat"><div class="label">Runtime</div><div class="value">{float(summary.get('elapsed_seconds', 0.0)):.1f}s</div></div>
    </section>

    <h2>Top regions</h2>
    <table><thead><tr><th>Region</th><th>Thumbnails</th></tr></thead><tbody>{region_rows}</tbody></table>

    <h2>Candidates</h2>
    <section class="grid">{''.join(cards)}</section>
  </main>
</body>
</html>"""


def _load_png_files(dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in dirs:
        if directory.exists():
            files.extend(sorted(directory.glob("*.png")))
    return files


def _find_exact_thumbnail(filename: str, search_dirs: list[Path]) -> Path | None:
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def _find_candidate_thumbnail(candidate_id: str, search_dirs: list[Path]) -> Path | None:
    candidate_dir = next((d for d in search_dirs if d.name == "candidates_v2"), None)
    if candidate_dir is not None:
        match = match_candidate_image(candidate_dir, candidate_id)
        if match is not None:
            return match
    for directory in search_dirs:
        for path in sorted(directory.glob("*.png")):
            if path.name == candidate_id or path.stem == candidate_id:
                return path
    return None


def _find_sog_thumbnail(event: dict[str, Any], search_dirs: list[Path], file_index: list[Path]) -> Path | None:
    slug = _slugify(str(event.get("location_name", "")))
    start_date = str(event.get("start_date", ""))
    compact = start_date.replace("-", "")
    if not slug or not start_date:
        return None

    exact_hits: list[Path] = []
    fuzzy_hits: list[Path] = []
    for path in file_index:
        stem = path.stem.lower()
        if slug not in stem:
            continue
        if start_date in stem or compact in stem:
            exact_hits.append(path)
        else:
            fuzzy_hits.append(path)

    if exact_hits:
        return sorted(exact_hits)[0]
    if fuzzy_hits:
        return sorted(fuzzy_hits)[0]
    return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_geojson_features(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    features = payload.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"{path} does not contain a features list")
    return features


def ingest_sog_records(paths: list[Path], output_path: Path, search_dirs: list[Path]) -> tuple[list[dict[str, Any]], dict[int, int], int]:
    records: list[dict[str, Any]] = []
    by_year: Counter[int] = Counter()
    total = 0

    for source_path in paths:
        for feature in _load_geojson_features(source_path):
            total += 1
            props = feature.get("properties", {})
            year = props.get("Year")
            lon = props.get("Longitude")
            lat = props.get("Latitude")
            start = props.get("Start")
            end = props.get("End_")

            if not isinstance(year, int) or not (2016 <= year <= 2025):
                continue
            if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
                continue
            if not (-180 <= float(lon) <= 180 and -90 <= float(lat) <= 90):
                continue

            start_date = _parse_sog_date(str(start))
            if start_date is None:
                continue
            if _parse_yyyy_mm_dd(str(end)) is None and str(end) not in {"", "NA", "None"}:
                continue

            location_name = str(props.get("LocationNa", "")).strip() or "unknown"
            object_id = props.get("OBJECTID", props.get("FID", feature.get("id")))
            record = {
                "id": f"sog:{_slugify(location_name)}:{start_date}:{object_id}",
                "source": "sog",
                "year": year,
                "longitude": float(lon),
                "latitude": float(lat),
                "start_date": start_date,
                "end_date": None if str(end) in {"", "NA", "None"} else _parse_yyyy_mm_dd(str(end)),
                "region": str(props.get("Region", "")).strip() or None,
                "location_name": location_name,
                "combined_si": props.get("CombinedSI"),
                "object_id": object_id,
                "thumbnail_path": None,
            }
            records.append(record)
            by_year[year] += 1

    _write_json(output_path, records)
    return records, dict(sorted(by_year.items())), total


def build_training_records(
    labels_path: Path,
    rose_review_path: Path,
    sog_records: list[dict[str, Any]],
    negative_dir: Path,
    search_dirs: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    summary: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []

    labels = json.loads(labels_path.read_text()).get("items", [])
    spawn_items = [item for item in labels if item.get("label") == "spawn"]
    nospawn_items = [item for item in labels if item.get("label") == "nospawn"]

    for item in spawn_items + nospawn_items:
        label = str(item.get("label", "")).strip().lower()
        candidate_id = str(item.get("id", ""))
        image_path = _find_candidate_thumbnail(candidate_id, search_dirs)
        if image_path is None:
            summary[f"skipped_{label}"] += 1
            continue
        records.append(
            {
                "id": candidate_id,
                "source": "user_labels",
                "label": label,
                "label_int": 1 if label == "spawn" else 0,
                "image_path": str(image_path),
                "thumbnail_path": image_path.name,
            }
        )
        summary[f"matched_{label}"] += 1

    rose = json.loads(rose_review_path.read_text())
    rose_spawns = [row for row in rose if row.get("classification") == "spawn"]
    for idx, row in enumerate(rose_spawns):
        filename = str(row.get("filename", ""))
        image_path = _find_exact_thumbnail(filename, search_dirs)
        if image_path is None:
            summary["skipped_rose_spawn"] += 1
            continue
        records.append(
            {
                "id": f"rose:{idx}:{filename}",
                "source": "rose_verified",
                "label": "spawn",
                "label_int": 1,
                "image_path": str(image_path),
                "thumbnail_path": image_path.name,
                "notes": row.get("notes"),
            }
        )
        summary["matched_rose_spawn"] += 1

    for path in sorted(negative_dir.glob("*.png")):
        records.append(
            {
                "id": f"negative:{path.stem}",
                "source": "negative_samples",
                "label": "nospawn",
                "label_int": 0,
                "image_path": str(path),
                "thumbnail_path": path.name,
            }
        )
    summary["negative_samples"] = len(list(sorted(negative_dir.glob("*.png"))))

    for sog in sog_records:
        image_path = _find_sog_thumbnail(sog, search_dirs, _load_png_files(search_dirs))
        if image_path is None:
            summary["skipped_sog"] += 1
            continue
        record = {
            "id": sog["id"],
            "source": "sog",
            "label": "spawn",
            "label_int": 1,
            "image_path": str(image_path),
            "thumbnail_path": image_path.name,
            "location_name": sog.get("location_name"),
            "start_date": sog.get("start_date"),
        }
        records.append(record)
        summary["matched_sog"] += 1

    return records, dict(summary)


def _records_hash(records: list[dict[str, Any]]) -> str:
    hasher = hashlib.md5()
    for record in records:
        path = Path(record["image_path"])
        stat = path.stat()
        hasher.update(f"{record['id']}|{record['label']}|{path}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return hasher.hexdigest()


def _load_or_compute_embeddings(
    records: list[dict[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    cache_path: Path,
) -> np.ndarray:
    cache_meta = cache_path.with_suffix(".meta.json")
    current_hash = _records_hash(records)

    if cache_path.exists() and cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text())
            if meta.get("hash") == current_hash:
                data = np.load(cache_path, allow_pickle=False)
                if list(data["paths"]) == [record["image_path"] for record in records]:
                    return data["embeddings"]
        except Exception:
            pass

    embeddings: list[np.ndarray] = []
    for record in records:
        path = Path(record["image_path"])
        embeddings.append(_load_image_embedding(model, path, device))

    emb_arr = np.asarray(embeddings, dtype=float)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=emb_arr,
        paths=np.asarray([record["image_path"] for record in records], dtype=object),
    )
    cache_meta.write_text(json.dumps({"hash": current_hash, "count": len(records)}, indent=2), encoding="utf-8")
    return emb_arr


class KnnIndex:
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, records: list[dict[str, Any]], k: int = 3):
        self.embeddings = np.asarray([row / max(np.linalg.norm(row), 1e-12) for row in embeddings], dtype=float)
        self.labels = np.asarray(labels, dtype=int)
        self.records = records
        self.k = max(1, min(int(k), len(self.labels)))

    def predict(self, embedding: np.ndarray) -> dict[str, Any]:
        query = embedding / max(float(np.linalg.norm(embedding)), 1e-12)
        similarities = self.embeddings @ query
        idx = np.argpartition(similarities, -self.k)[-self.k:]
        idx = idx[np.argsort(similarities[idx])[::-1]]
        neighbor_labels = self.labels[idx]
        spawn_votes = int(np.sum(neighbor_labels == 1))
        prediction = int(spawn_votes > (self.k / 2))
        neighbors = []
        for neighbor_idx in idx:
            neighbor = self.records[int(neighbor_idx)]
            neighbors.append(
                {
                    "id": neighbor["id"],
                    "label": neighbor["label"],
                    "image_path": neighbor["image_path"],
                    "source": neighbor["source"],
                    "similarity": float(similarities[neighbor_idx]),
                }
            )
        return {
            "prediction": prediction,
            "spawn_votes": spawn_votes,
            "vote_fraction": spawn_votes / float(self.k),
            "neighbor_indices": idx.tolist(),
            "neighbors": neighbors,
        }


def _build_knn_index(
    positive_dir: Path,
    negative_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    k: int,
) -> tuple[KnnIndex, dict[str, int]]:
    records: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    summary: Counter[str] = Counter()

    for label_name, directory, label_int in (
        ("spawn", positive_dir, 1),
        ("nospawn", negative_dir, 0),
    ):
        for path in sorted(directory.glob("*.png")):
            try:
                embedding = _load_image_embedding(model, path, device)
            except Exception:
                summary[f"skipped_{label_name}"] += 1
                continue
            embeddings.append(embedding)
            labels.append(label_int)
            records.append(
                {
                    "id": path.stem,
                    "label": label_name,
                    "image_path": str(path),
                    "source": f"{label_name}_samples",
                }
            )
            summary[f"matched_{label_name}"] += 1

    if not embeddings:
        raise RuntimeError("No labeled images were available for KNN scoring")

    knn_index = KnnIndex(np.asarray(embeddings, dtype=float), np.asarray(labels, dtype=int), records, k=k)
    summary["training_size"] = len(records)
    return knn_index, dict(summary)


def _process_sog_record(
    record: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    knn_index: KnnIndex,
    model: torch.nn.Module,
    device: torch.device,
    ee_module: Any,
    idx: int,
    total: int,
) -> dict[str, Any]:
    result = {
        "processed": 1,
        "candidates": 0,
        "no_scene": 0,
        "download_errors": 0,
        "scored": 0,
        "entries": [],
    }

    try:
        scenes = search_sog_scenes(
            ee_module,
            record,
            search_days=getattr(args, "sog_search_days", DEFAULT_SOG_SEARCH_DAYS),
            max_cloud=getattr(args, "max_cloud", DEFAULT_MAX_CLOUD),
            limit=getattr(args, "sog_thumbnails_per_record", DEFAULT_SOG_THUMBNAILS_PER_RECORD),
        )
    except Exception as exc:
        with PRINT_LOCK:
            print_progress(idx, total, record.get("location_name", "unknown"), record["latitude"], record["longitude"], f"search error: {exc}", 0)
        result["download_errors"] = 1
        return result

    if not scenes:
        with PRINT_LOCK:
            print_progress(idx, total, record.get("location_name", "unknown"), record["latitude"], record["longitude"], "no scene", 0)
        result["no_scene"] = 1
        return result

    for rank, scene_info in enumerate(scenes, start=1):
        thumb_bytes = download_thumbnail(ee_module, record["latitude"], record["longitude"], scene_info["scene_id"])
        if thumb_bytes is None:
            with PRINT_LOCK:
                print_progress(idx, total, record.get("location_name", "unknown"), record["latitude"], record["longitude"], f"download error {rank}", 0)
            result["download_errors"] += 1
            continue

        try:
            embedding = _embedding_from_png_bytes(model, device, thumb_bytes)
        except Exception:
            with PRINT_LOCK:
                print_progress(idx, total, record.get("location_name", "unknown"), record["latitude"], record["longitude"], f"embedding error {rank}", 0)
            result["download_errors"] += 1
            continue

        vote = knn_index.predict(embedding)
        thumb_dir = output_dir / "thumbnails"
        info = {
            "region": record.get("region") or record.get("location_name") or "unknown",
            "location_name": record.get("location_name") or "unknown",
            "lat": record["latitude"],
            "lon": record["longitude"],
            "date": scene_info["date"],
            "scene_id": scene_info["scene_id"],
            "cloud": scene_info["cloud"],
            "score": round(float(vote["vote_fraction"]), 4),
            "spawn_votes": int(vote["spawn_votes"]),
            "k": int(knn_index.k),
        }
        fname = save_candidate(thumb_dir, thumb_bytes, info, float(vote["vote_fraction"]))
        entry = {
            "record_id": record["id"],
            "record_rank": rank,
            "year": record["year"],
            "region": info["region"],
            "location_name": info["location_name"],
            "latitude": record["latitude"],
            "longitude": record["longitude"],
            "combined_si": float(record.get("combined_si", 0.0) or 0.0),
            "target_date": record["target_date"],
            "date": scene_info["date"],
            "scene_id": scene_info["scene_id"],
            "cloud": scene_info["cloud"],
            "score": float(vote["vote_fraction"]),
            "knn_score": float(vote["vote_fraction"]),
            "spawn_votes": int(vote["spawn_votes"]),
            "k": int(knn_index.k),
            "thumbnail_path": fname,
            "neighbors": vote["neighbors"],
        }
        with MANIFEST_LOCK:
            update_manifest(output_dir, entry)
        result["entries"].append(entry)
        result["scored"] += 1
        result["candidates"] += 1
        with PRINT_LOCK:
            print_progress(idx, total, record.get("location_name", "unknown"), record["latitude"], record["longitude"], f"saved {rank}/{len(scenes)} {fname}", 0)

    return result


def run_sog_mode(args: argparse.Namespace) -> int:
    t0 = time.time()
    output_dir = args.output if args.output != DEFAULT_OUTPUT_DIR else DEFAULT_SOG_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "thumbnails").mkdir(parents=True, exist_ok=True)

    sog_records, ingest_summary = load_sog_records(args.sog_files)
    if not sog_records:
        raise RuntimeError("No SoG records matched the requested filters")

    print(f"Loaded {len(sog_records)} SoG records from {ingest_summary.get('raw_features', 0)} features")
    print(f"SoG ingest summary: {ingest_summary}")

    device = _pick_device()
    try:
        model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError(f"Failed to load DINOv2 model {MODEL_NAME}: {exc}") from exc
    model.eval().to(device)

    knn_index, knn_summary = _build_knn_index(
        args.positive_dir,
        args.negative_dir,
        model,
        device,
        args.k,
    )
    print(f"KNN training summary: {knn_summary}")

    try:
        import ee
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("earthengine-api is required for SoG mode") from exc

    ee.Initialize(project=args.project)

    stats = Counter()
    entries: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [
            executor.submit(_process_sog_record, record, args, output_dir, knn_index, model, device, ee, idx, len(sog_records))
            for idx, record in enumerate(sog_records)
        ]
        for future in as_completed(futures):
            result = future.result()
            stats.update({k: v for k, v in result.items() if isinstance(v, int)})
            entries.extend(result.get("entries", []))

    top_regions = Counter(str(row.get("region") or "unknown") for row in entries)
    elapsed = time.time() - t0
    summary = {
        "record_count": len(sog_records),
        "thumbnail_count": len(entries),
        "records_with_scenes": len({row["record_id"] for row in entries}),
        "top_regions": dict(top_regions),
        "processed": int(stats["processed"]),
        "no_scene": int(stats["no_scene"]),
        "download_errors": int(stats["download_errors"]),
        "scored": int(stats["scored"]),
        "elapsed_seconds": elapsed,
        "ingest_summary": ingest_summary,
        "knn_summary": knn_summary,
    }

    _write_json(output_dir / "manifest.json", entries)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "review.html").write_text(build_sog_review_html(entries, summary), encoding="utf-8")

    print(f"Records: {len(sog_records)}")
    print(f"Thumbnails downloaded: {len(entries)}")
    print(f"Top regions: {dict(top_regions.most_common(5))}")
    print(f"Review page: file://{(output_dir / 'review.html').resolve()}")
    print(f"Processing time: {elapsed:.1f}s")
    return 0


def _embedding_from_png_bytes(model: torch.nn.Module, device: torch.device, png_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    tensor = DINO_TRANSFORM(image).unsqueeze(0).to(device)
    with MODEL_LOCK:
        with torch.no_grad():
            emb = model(tensor)
    return F.normalize(emb, dim=1).cpu().numpy().flatten().astype(float)


def build_review_html(entries: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    cards = []
    for row in sorted(entries, key=lambda item: (item.get("region", ""), -float(item.get("score", 0.0)))):
        neighbor_html = "".join(
            f"<li>{html_escape(n['label'])} · {html_escape(Path(n['image_path']).name)} · {n['similarity']:.3f}</li>"
            for n in row.get("neighbors", [])
        )
        cards.append(
            f"""
            <article class="card">
              <img src="{html_escape(row['thumbnail_path'])}" alt="candidate">
              <div class="meta"><strong>{html_escape(row['region'])}</strong> · {html_escape(row['date'])}</div>
              <div class="meta">score {row['score']:.2f} · votes {row['spawn_votes']}/{row['k']} · cloud {row['cloud']:.1f}%</div>
              <div class="meta">({row['lat']:.4f}, {row['lon']:.4f})</div>
              <details>
                <summary>nearest neighbors</summary>
                <ul>{neighbor_html}</ul>
              </details>
            </article>
            """
        )

    region_rows = "".join(
        f"<tr><td>{html_escape(region)}</td><td>{count}</td></tr>"
        for region, count in sorted(summary.get("candidate_regions", {}).items(), key=lambda item: (-item[1], item[0]))
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KNN BC Coast Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #f5f6fa; color: #1f2937; }}
    header {{ background: linear-gradient(135deg, #111827, #0f172a); color: white; padding: 24px; }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .stat {{ background: white; border-radius: 12px; padding: 14px 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .value {{ font-size: 28px; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 14px; }}
    .card {{ background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .card img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; }}
    .meta {{ padding: 0 14px 8px; font-size: 13px; color: #374151; }}
    details {{ padding: 0 14px 14px; font-size: 12px; color: #4b5563; }}
    table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-top: 12px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
    th {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <header>
    <h1>KNN BC Coast Review</h1>
    <p>Majority-vote DINOv2 KNN candidates from the 2024 Feb-May BC scan.</p>
  </header>
  <main>
    <section class="stats">
      <div class="stat"><div class="label">Training images</div><div class="value">{summary['training_size']}</div></div>
      <div class="stat"><div class="label">Candidates</div><div class="value">{summary['candidate_count']}</div></div>
      <div class="stat"><div class="label">Scanned points</div><div class="value">{summary['points_scanned']}</div></div>
      <div class="stat"><div class="label">Runtime</div><div class="value">{summary['elapsed_seconds']:.1f}s</div></div>
    </section>

    <h2>Top regions</h2>
    <table><thead><tr><th>Region</th><th>Candidates</th></tr></thead><tbody>{region_rows}</tbody></table>

    <h2>Candidates</h2>
    <section class="grid">{''.join(cards)}</section>
  </main>
</body>
</html>"""


def html_escape(value: Any) -> str:
    import html

    return html.escape(str(value))


def _process_point(
    point: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    knn_index: KnnIndex,
    model: torch.nn.Module,
    device: torch.device,
    ee_module: Any,
    idx: int,
    total: int,
) -> dict[str, int]:
    result = {"processed": 1, "candidates": 0, "no_scene": 0, "download_errors": 0, "low_score": 0}

    scene_info = find_best_scene(ee_module, point["lat"], point["lon"], args.start, args.end, args.max_cloud)
    if scene_info is None:
        with PRINT_LOCK:
            print_progress(idx, total, point["region"], point["lat"], point["lon"], "no scene", 0)
        result["no_scene"] = 1
        return result

    thumb_bytes = download_thumbnail(ee_module, point["lat"], point["lon"], scene_info["scene_id"])
    if thumb_bytes is None:
        with PRINT_LOCK:
            print_progress(idx, total, point["region"], point["lat"], point["lon"], "download error", 0)
        result["download_errors"] = 1
        return result

    try:
        embedding = _embedding_from_png_bytes(model, device, thumb_bytes)
    except Exception:
        with PRINT_LOCK:
            print_progress(idx, total, point["region"], point["lat"], point["lon"], "embedding error", 0)
        result["download_errors"] = 1
        return result

    vote = knn_index.predict(embedding)
    if vote["prediction"] != 1:
        with PRINT_LOCK:
            print_progress(idx, total, point["region"], point["lat"], point["lon"], f"below threshold ({vote['vote_fraction']:.2f})", 0)
        result["low_score"] = 1
        return result

    info = {
        "region": point["region"],
        "lat": point["lat"],
        "lon": point["lon"],
        "date": scene_info["date"],
        "scene_id": scene_info["scene_id"],
        "cloud": scene_info["cloud"],
        "score": round(float(vote["vote_fraction"]), 4),
        "spawn_votes": int(vote["spawn_votes"]),
        "k": int(knn_index.k),
    }
    fname = save_candidate(output_dir, thumb_bytes, info, float(vote["vote_fraction"]))
    entry = {
        **info,
        "thumbnail_path": fname,
        "neighbors": vote["neighbors"],
    }
    with MANIFEST_LOCK:
        update_manifest(output_dir, entry)
    with PRINT_LOCK:
        print_progress(idx, total, point["region"], point["lat"], point["lon"], f"CANDIDATE votes={vote['spawn_votes']}/{knn_index.k} {fname}", 0)
    result["candidates"] = 1
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bc", "sog"], default=DEFAULT_MODE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--rose-review", type=Path, default=DEFAULT_ROSE_REVIEW)
    parser.add_argument("--sog-files", type=Path, nargs="+", default=DEFAULT_SOG_FILES)
    parser.add_argument("--negative-dir", type=Path, default=DEFAULT_NEGATIVE_DIR)
    parser.add_argument("--positive-dir", type=Path, default=Path("data/samples/positive"))
    parser.add_argument("--ingressed-dir", type=Path, default=DEFAULT_INGRESSED_DIR)
    parser.add_argument("--sog-output", type=Path, default=DEFAULT_INGRESSED_SOG_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD)
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--sog-search-days", type=int, default=DEFAULT_SOG_SEARCH_DAYS)
    parser.add_argument("--sog-thumbnails-per-record", type=int, default=DEFAULT_SOG_THUMBNAILS_PER_RECORD)
    args = parser.parse_args(argv)

    if args.mode == "sog":
        return run_sog_mode(args)

    t0 = time.time()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    search_dirs = [Path("data/candidates_v2"), args.ingressed_dir]

    sog_records, sog_year_counts, sog_raw_count = ingest_sog_records(args.sog_files, args.sog_output, search_dirs)
    print(f"Ingested {len(sog_records)} SoG records from {sog_raw_count} features")
    print(f"SoG by year: {sog_year_counts}")

    training_records, build_summary = build_training_records(
        args.labels,
        args.rose_review,
        sog_records,
        args.negative_dir,
        search_dirs,
    )

    if not training_records:
        raise RuntimeError("No training records could be matched to thumbnails")

    labels = np.asarray([record["label_int"] for record in training_records], dtype=int)
    print(f"Training set size: {len(training_records)}")
    print(f"Training breakdown: spawn={int(np.sum(labels == 1))}, nospawn={int(np.sum(labels == 0))}")
    print(f"Match summary: {dict(build_summary)}")

    training_manifest = {
        "summary": {
            "training_size": len(training_records),
            "spawn_count": int(np.sum(labels == 1)),
            "nospawn_count": int(np.sum(labels == 0)),
            "sog_ingested": len(sog_records),
            "sog_raw_features": sog_raw_count,
            "sog_by_year": sog_year_counts,
            "match_summary": dict(build_summary),
        },
        "records": training_records,
    }
    _write_json(args.output / "training_manifest.json", training_manifest)

    device = _pick_device()
    try:
        model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError(f"Failed to load DINOv2 model {MODEL_NAME}: {exc}") from exc
    model.eval().to(device)

    cache_path = Path("data/embeddings/knn_training_embeddings.npz")
    embeddings = _load_or_compute_embeddings(training_records, model, device, cache_path)
    knn_index = KnnIndex(embeddings, labels, training_records, k=args.k)

    try:
        import ee
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("earthengine-api is required for the scan") from exc

    ee.Initialize(project=args.project)

    points = generate_grid_points(REGIONS, args.grid_spacing)
    print(f"Generated {len(points)} grid points across {len(REGIONS)} regions")

    stats = Counter()
    candidate_regions: Counter[str] = Counter()
    candidate_entries: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [
            executor.submit(_process_point, point, args, output_dir, knn_index, model, device, ee, idx, len(points))
            for idx, point in enumerate(points)
        ]
        for future in as_completed(futures):
            result = future.result()
            stats.update(result)

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        candidate_entries = json.loads(manifest_path.read_text())
    else:
        candidate_entries = []

    for row in candidate_entries:
        candidate_regions[str(row.get("region", "unknown"))] += 1

    elapsed = time.time() - t0
    summary = {
        "training_size": len(training_records),
        "points_scanned": len(points),
        "candidates": int(stats["candidates"]),
        "candidate_count": len(candidate_entries),
        "candidate_regions": dict(candidate_regions),
        "processed": int(stats["processed"]),
        "no_scene": int(stats["no_scene"]),
        "download_errors": int(stats["download_errors"]),
        "low_score": int(stats["low_score"]),
        "elapsed_seconds": elapsed,
    }

    _write_json(output_dir / "summary.json", summary)
    review_html = build_review_html(candidate_entries, summary)
    (output_dir / "review.html").write_text(review_html, encoding="utf-8")

    print(f"Candidates found: {len(candidate_entries)}")
    print(f"Top regions: {dict(candidate_regions.most_common(5))}")
    print(f"Processing time: {elapsed:.1f}s")
    print(f"Review page: file://{(output_dir / 'review.html').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
