#!/usr/bin/env python3
"""Environmental-matched temporal scoring for herring spawn detection.

The matcher groups scenes by approximate sun position and tide conditions,
then scores each scene as an embedding outlier inside its environmental group.
It is designed to work from the canonical golden set on disk, with best-effort
Earth Engine fallback for missing local thumbnails.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.knn_detector import DINO_TRANSFORM, _pick_device
from scripts.temporal_detector import (
    extract_location_key,
    extract_region_key,
    parse_acquisition_datetime,
    parse_location_from_filename,
    parse_observation_date,
)

MODEL_NAME = "dinov2_vits14"
DEFAULT_MANIFEST = Path("data/samples/training_manifest.json")
DEFAULT_POSITIVE_DIR = Path("data/samples/positive")
DEFAULT_NEGATIVE_DIR = Path("data/samples/negative")
DEFAULT_OUTPUT_DIR = Path("data/environmental_matching")
DEFAULT_LANDSAT_DIR = Path("data/landsat_embeddings")
DEFAULT_SUN_ELEVATION_STEP = 5.0
DEFAULT_SUN_AZIMUTH_STEP = 15.0
DEFAULT_TIDE_STEP_ACTUAL = 0.5
DEFAULT_TIDE_STEP_PROXY = 1.0


@dataclass(frozen=True)
class Sample:
    filename: str
    label: int
    image_path: Path
    region_key: str
    location_key: str
    observation_date: date
    acquisition_dt: datetime
    lat: float | None = None
    lon: float | None = None


@dataclass
class Observation:
    sample: Sample
    embedding: np.ndarray | None = None
    sun_elevation: float | None = None
    sun_azimuth: float | None = None
    tide_height: float | None = None
    tide_source: str = "utc_proxy"
    source: str = "local"


def _safe_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _candidate_path(
    manifest_path: Path, positive_dir: Path, negative_dir: Path, filename: str, positive: bool
) -> Path:
    subdir = positive_dir if positive else negative_dir
    candidates = [
        subdir / filename,
        manifest_path.parent / ("positive" if positive else "negative") / filename,
        manifest_path.parent / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_local_samples(manifest_path: Path, positive_dir: Path, negative_dir: Path) -> list[Sample]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples: list[Sample] = []
    seen: set[str] = set()

    for filename in manifest.get("positives", []):
        if filename in seen:
            continue
        seen.add(filename)
        path = _candidate_path(manifest_path, positive_dir, negative_dir, filename, True)
        samples.append(_build_sample(path, 1))

    for filename in manifest.get("rejected", []):
        if filename in seen:
            continue
        seen.add(filename)
        path = _candidate_path(manifest_path, positive_dir, negative_dir, filename, False)
        samples.append(_build_sample(path, 0))

    return samples


def _build_sample(path: Path, label: int) -> Sample:
    filename = path.name
    region_key = extract_region_key(filename)
    location_key, lat, lon = extract_location_key(filename)
    obs_date = parse_observation_date(filename)
    if obs_date is None and path.exists():
        obs_date = date.fromtimestamp(path.stat().st_mtime)
    if obs_date is None:
        obs_date = date.today()
    acq_dt = parse_acquisition_datetime(filename, obs_date) or datetime.combine(
        obs_date, dtime(12, 0, 0)
    )
    return Sample(
        filename=filename,
        label=label,
        image_path=path,
        region_key=region_key,
        location_key=location_key,
        observation_date=obs_date,
        acquisition_dt=acq_dt,
        lat=lat,
        lon=lon,
    )


def approximate_solar_position(
    acquisition_dt: datetime, lat: float | None, lon: float | None
) -> tuple[float, float]:
    """Approximate solar elevation/azimuth in degrees using NOAA-style geometry."""

    if acquisition_dt.tzinfo is not None:
        acquisition_dt = acquisition_dt.astimezone(UTC).replace(tzinfo=None)

    lat = float(lat if lat is not None else 49.0)
    lon = float(lon if lon is not None else -123.0)
    day_of_year = acquisition_dt.timetuple().tm_yday
    hour = acquisition_dt.hour + acquisition_dt.minute / 60.0 + acquisition_dt.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    eq_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    true_solar_minutes = hour * 60.0 + eq_time + 4.0 * lon
    hour_angle_deg = (true_solar_minutes / 4.0) - 180.0
    hour_angle = math.radians(hour_angle_deg)
    lat_rad = math.radians(lat)
    cos_zenith = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(
        hour_angle
    )
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    elevation = 90.0 - math.degrees(zenith)

    az = math.degrees(
        math.atan2(
            math.sin(hour_angle),
            math.cos(hour_angle) * math.sin(lat_rad) - math.tan(decl) * math.cos(lat_rad),
        )
    )
    azimuth = (az + 180.0) % 360.0
    return elevation, azimuth


def _round_value(value: float | None, step: float) -> float | None:
    if value is None:
        return None
    return round(value / step) * step


def _round_azimuth(value: float | None, step: float) -> float | None:
    if value is None:
        return None
    return (round(value / step) * step) % 360.0


def environmental_bucket_key(
    sun_elevation: float | None,
    sun_azimuth: float | None,
    tide_height: float | None,
    *,
    tide_source: str,
    sun_elevation_step: float = DEFAULT_SUN_ELEVATION_STEP,
    sun_azimuth_step: float = DEFAULT_SUN_AZIMUTH_STEP,
    tide_step_actual: float = DEFAULT_TIDE_STEP_ACTUAL,
    tide_step_proxy: float = DEFAULT_TIDE_STEP_PROXY,
) -> tuple[Any, ...]:
    tide_step = tide_step_proxy if tide_source == "utc_proxy" else tide_step_actual
    tide_bucket = _round_value(tide_height, tide_step)
    return (
        _round_value(sun_elevation, sun_elevation_step),
        _round_azimuth(sun_azimuth, sun_azimuth_step),
        tide_bucket,
        tide_source,
    )


def _environment_distance(
    left: Observation,
    right: Observation,
    *,
    sun_elevation_step: float = DEFAULT_SUN_ELEVATION_STEP,
    sun_azimuth_step: float = DEFAULT_SUN_AZIMUTH_STEP,
    tide_step_actual: float = DEFAULT_TIDE_STEP_ACTUAL,
    tide_step_proxy: float = DEFAULT_TIDE_STEP_PROXY,
) -> float:
    tide_step = tide_step_proxy if left.tide_source == "utc_proxy" else tide_step_actual
    tide_step = max(tide_step, 1e-6)
    az_diff = abs(float(left.sun_azimuth or 0.0) - float(right.sun_azimuth or 0.0))
    az_diff = min(az_diff, 360.0 - az_diff)
    return (
        abs(float(left.sun_elevation or 0.0) - float(right.sun_elevation or 0.0))
        / sun_elevation_step
        + az_diff / sun_azimuth_step
        + abs(float(left.tide_height or 0.0) - float(right.tide_height or 0.0)) / tide_step
    )


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        return embedding
    return embedding / norm


def _distance_score(
    value: np.ndarray, baseline: Sequence[np.ndarray]
) -> tuple[float, dict[str, float]]:
    if not baseline:
        return 0.0, {"baseline_count": 0.0, "baseline_std_distance": 0.0, "value_distance": 0.0}
    base = np.vstack(baseline)
    centroid = base.mean(axis=0)
    distances = np.linalg.norm(base - centroid, axis=1)
    mean_dist = float(distances.mean())
    std_dist = float(distances.std())
    value_dist = float(np.linalg.norm(np.asarray(value, dtype=float) - centroid))
    if std_dist < 1e-6:
        score = value_dist
    else:
        score = (value_dist - mean_dist) / std_dist
    return score, {
        "baseline_count": float(len(baseline)),
        "baseline_mean_distance": mean_dist,
        "baseline_std_distance": std_dist,
        "value_distance": value_dist,
    }


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.0) -> dict[str, Any]:
    preds = (scores >= threshold).astype(int)
    metrics: dict[str, Any] = {"accuracy": float(accuracy_score(y_true, preds))}
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, scores))
        metrics["ap"] = float(average_precision_score(y_true, scores))
    else:
        metrics["auroc"] = None
        metrics["ap"] = None

    thresholds = np.unique(scores)
    if len(thresholds) == 0:
        metrics.update(
            {
                "best_threshold": 0.0,
                "best_accuracy": 0.0,
                "best_balanced_accuracy": 0.0,
                "best_f1": 0.0,
            }
        )
        return metrics

    best_acc = (-1.0, 0.0)
    best_bal = (-1.0, 0.0)
    best_f1 = (-1.0, 0.0)
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        acc = float(accuracy_score(y_true, pred))
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        tn = float(np.sum((pred == 0) & (y_true == 0)))
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        fn = float(np.sum((pred == 0) & (y_true == 1)))
        tpr = tp / max(1.0, tp + fn)
        tnr = tn / max(1.0, tn + fp)
        bal = 0.5 * (tpr + tnr)
        f1 = (2.0 * tp) / max(1.0, 2.0 * tp + fp + fn)
        if acc > best_acc[0]:
            best_acc = (acc, float(thr))
        if bal > best_bal[0]:
            best_bal = (bal, float(thr))
        if f1 > best_f1[0]:
            best_f1 = (f1, float(thr))

    metrics.update(
        {
            "best_threshold": best_f1[1],
            "best_accuracy": best_acc[0],
            "best_threshold_accuracy": best_acc[1],
            "best_balanced_accuracy": best_bal[0],
            "best_threshold_balanced_accuracy": best_bal[1],
            "best_f1": best_f1[0],
            "best_threshold_f1": best_f1[1],
        }
    )
    return metrics


def _group_observations(observations: Sequence[Observation]) -> dict[str, list[Observation]]:
    groups: dict[str, list[Observation]] = defaultdict(list)
    for obs in observations:
        groups[obs.sample.location_key].append(obs)
    for group in groups.values():
        group.sort(key=lambda obs: (obs.sample.acquisition_dt, obs.sample.filename))
    return groups


def _observation_row(obs: Observation) -> dict[str, Any]:
    return {
        "filename": obs.sample.filename,
        "location_key": obs.sample.location_key,
        "region_key": obs.sample.region_key,
        "label": obs.sample.label,
        "date": obs.sample.observation_date.isoformat(),
        "sun_elevation": obs.sun_elevation,
        "sun_azimuth": obs.sun_azimuth,
        "tide_height": obs.tide_height,
        "tide_source": obs.tide_source,
        "source": obs.source,
    }


def _baseline_for_observation(
    obs: Observation,
    group: Sequence[Observation],
    all_observations: Sequence[Observation],
) -> tuple[list[np.ndarray], str, int]:
    exact_bucket = environmental_bucket_key(
        obs.sun_elevation,
        obs.sun_azimuth,
        obs.tide_height,
        tide_source=obs.tide_source,
    )
    exact_pool = [
        other
        for other in group
        if environmental_bucket_key(
            other.sun_elevation,
            other.sun_azimuth,
            other.tide_height,
            tide_source=other.tide_source,
        )
        == exact_bucket
    ]
    if len(exact_pool) >= 3:
        return (
            [other.embedding for other in exact_pool if other.embedding is not None],
            "exact_group",
            len(exact_pool),
        )

    nearby = [
        other
        for other in group
        if other.sample.filename != obs.sample.filename and _environment_distance(obs, other) <= 2.5
    ]
    if len(nearby) >= 3:
        return (
            [other.embedding for other in nearby if other.embedding is not None],
            "nearby_group",
            len(nearby),
        )

    location_pool = [other for other in group if other.sample.filename != obs.sample.filename]
    if len(location_pool) >= 2:
        return (
            [other.embedding for other in location_pool if other.embedding is not None],
            "location_group",
            len(location_pool) + 1,
        )

    global_pool = [
        other for other in all_observations if other.sample.filename != obs.sample.filename
    ]
    return (
        [other.embedding for other in global_pool if other.embedding is not None],
        "global_group",
        len(global_pool) + 1,
    )


def score_environmental_outliers(
    observations: Sequence[Observation],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = _group_observations(observations)

    for _location_key, group in groups.items():
        for obs in group:
            row = _observation_row(obs)
            if obs.embedding is None:
                row.update({"score": None, "evaluated": False, "reason": "missing_embedding"})
                rows.append(row)
                continue

            baseline, baseline_scope, group_size = _baseline_for_observation(
                obs, group, observations
            )
            if not baseline:
                row.update(
                    {
                        "score": 0.0,
                        "evaluated": False,
                        "reason": "no_baseline",
                        "group_size": group_size,
                        "baseline_scope": baseline_scope,
                    }
                )
                rows.append(row)
                continue

            score, stats = _distance_score(np.asarray(obs.embedding, dtype=float), baseline)
            row.update(
                {
                    "score": float(score),
                    "evaluated": True,
                    "reason": baseline_scope,
                    "group_size": group_size,
                    "baseline_scope": baseline_scope,
                    **stats,
                }
            )
            rows.append(row)

    scored = [row for row in rows if row.get("evaluated") and row.get("score") is not None]
    y = (
        np.asarray([int(row["label"]) for row in scored], dtype=int)
        if scored
        else np.zeros(0, dtype=int)
    )
    scores = (
        np.asarray([float(row["score"]) for row in scored], dtype=float)
        if scored
        else np.zeros(0, dtype=float)
    )

    summary = {
        "n_rows": len(rows),
        "n_evaluated": len(scored),
        "evaluated_count": len(scored),
        "n_positive": int(sum(int(row["label"]) for row in rows)),
        "n_negative": int(sum(1 - int(row["label"]) for row in rows)),
        "evaluated_positive_count": int(sum(1 for row in scored if int(row["label"]) == 1)),
        "evaluated_negative_count": int(sum(1 for row in scored if int(row["label"]) == 0)),
        "evaluated_fraction": float(len(scored) / max(1, len(rows))),
        "metrics": _metrics(y, scores)
        if len(scored)
        else {"accuracy": 0.0, "auroc": None, "ap": None},
        "location_count": len(groups),
        "average_scenes_per_location": float(len(rows) / max(1, len(groups))),
    }
    return rows, summary


def benchmark_environmental_matcher(
    observations: Sequence[Observation],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, scene_summary = score_environmental_outliers(observations)

    location_rows: list[dict[str, Any]] = []
    for location_key, group in _group_observations(observations).items():
        scored = [
            row
            for row in rows
            if row["location_key"] == location_key and row.get("score") is not None
        ]
        if not scored:
            continue
        best = max(scored, key=lambda row: float(row["score"]))
        location_rows.append(
            {
                "location_key": location_key,
                "label": int(max(int(obs.sample.label) for obs in group)),
                "score": float(best["score"]),
            }
        )

    loc_y = (
        np.asarray([row["label"] for row in location_rows], dtype=int)
        if location_rows
        else np.zeros(0, dtype=int)
    )
    loc_scores = (
        np.asarray([row["score"] for row in location_rows], dtype=float)
        if location_rows
        else np.zeros(0, dtype=float)
    )

    summary = {
        **scene_summary,
        "location_metrics": _metrics(loc_y, loc_scores)
        if len(location_rows)
        else {"accuracy": 0.0, "auroc": None, "ap": None},
        "location_row_count": len(location_rows),
    }
    return rows, summary


def _load_image_bytes(
    sample: Sample, allow_gee_fallback: bool, ee_project: str, max_cloud: float
) -> tuple[bytes | None, str]:
    if sample.image_path.exists():
        source = "supplement" if sample.filename.startswith(("L8_", "L9_")) else "local"
        return sample.image_path.read_bytes(), source

    if not allow_gee_fallback or sample.lat is None or sample.lon is None:
        return None, "missing"

    try:
        from scripts.scan_bc_coast import download_thumbnail as gee_download_thumbnail
        from scripts.scan_bc_coast import find_best_scene as gee_find_best_scene
    except Exception:
        return None, "gee_unavailable"

    try:
        import ee

        ee.Initialize(project=ee_project)
    except Exception:
        return None, "gee_unavailable"

    start = (sample.observation_date - timedelta(days=14)).isoformat()
    end = (sample.observation_date + timedelta(days=14)).isoformat()
    scene = gee_find_best_scene(ee, sample.lat, sample.lon, start, end, max_cloud)
    if scene is None:
        return None, "gee_no_scene"
    thumb = gee_download_thumbnail(ee, sample.lat, sample.lon, scene["scene_id"])
    if thumb is None:
        return None, "gee_download_failed"
    return thumb, "gee"


def _embed_image(model: torch.nn.Module, device: torch.device, png_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    tensor = DINO_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    return F.normalize(emb, dim=1).cpu().numpy().flatten().astype(float)


def build_observations(
    samples: Sequence[Sample],
    model: torch.nn.Module | None,
    device: torch.device | None,
    *,
    tide_provider: Any | None = None,
    allow_gee_fallback: bool = True,
    ee_project: str = "redd-fish",
    max_cloud: float = 50.0,
) -> list[Observation]:
    observations: list[Observation] = []
    tide_provider = tide_provider or UtcProxyTideProvider()

    for sample in samples:
        png_bytes, source = _load_image_bytes(sample, allow_gee_fallback, ee_project, max_cloud)
        if png_bytes is None:
            continue
        embedding = None
        if model is not None and device is not None:
            embedding = _embed_image(model, device, png_bytes)

        sun_elevation, sun_azimuth = approximate_solar_position(
            sample.acquisition_dt,
            sample.lat,
            sample.lon,
        )
        try:
            tide_height, tide_source = tide_provider.estimate(
                sample.lat, sample.lon, sample.acquisition_dt
            )
        except Exception:
            tide_height, tide_source = UtcProxyTideProvider().estimate(
                sample.lat, sample.lon, sample.acquisition_dt
            )

        observations.append(
            Observation(
                sample=sample,
                embedding=embedding,
                sun_elevation=sun_elevation,
                sun_azimuth=sun_azimuth,
                tide_height=tide_height,
                tide_source=tide_source,
                source=source,
            )
        )
    return observations


def load_landsat_observations(
    landsat_dir: Path,
    positive_locations: Sequence[dict[str, Any]],
) -> list[Observation]:
    """Load pre-computed Landsat 8/9 DINOv2 embeddings as Observations.

    These are .npy files created by download_hls_thumbnails.py.
    Each file contains a 384-dim DINOv2 embedding extracted from a 256x256
    Landsat thumbnail. The PNG was discarded after embedding to save disk.
    """
    observations: list[Observation] = []
    landsat_dir = Path(landsat_dir)

    if not landsat_dir.exists():
        return observations

    for npy_file in sorted(landsat_dir.rglob("*.npy")):
        png_name = npy_file.stem + ".png"
        parsed = parse_location_from_filename(png_name)
        if parsed is None:
            continue

        lat = float(parsed["lat"])
        lon = float(parsed["lon"])

        # Match against known positive locations
        label = 0
        for ploc in positive_locations:
            if abs(lat - float(ploc["lat"])) < 0.001 and abs(
                lon - float(ploc["lon"])
            ) < 0.001:
                label = 1
                break

        if label != 1:
            continue

        obs_date = parse_observation_date(png_name)
        if obs_date is None:
            continue

        acq_dt = parse_acquisition_datetime(png_name, obs_date) or datetime.combine(
            obs_date, dtime(12, 0, 0)
        )

        embedding = np.load(npy_file).astype(np.float64)
        location_key, _, _ = extract_location_key(png_name)
        region_key = extract_region_key(png_name)

        sample = Sample(
            filename=npy_file.name,
            label=label,
            image_path=npy_file,
            region_key=region_key,
            location_key=location_key,
            observation_date=obs_date,
            acquisition_dt=acq_dt,
            lat=lat,
            lon=lon,
        )

        sun_elev, sun_az = approximate_solar_position(acq_dt, lat, lon)
        tide_provider = UtcProxyTideProvider()
        tide_height, tide_source = tide_provider.estimate(lat, lon, acq_dt)

        observations.append(
            Observation(
                sample=sample,
                embedding=embedding,
                sun_elevation=sun_elev,
                sun_azimuth=sun_az,
                tide_height=tide_height,
                tide_source=tide_source,
                source="landsat_embedding",
            )
        )

    return observations


class UtcProxyTideProvider:
    def estimate(
        self, lat: float | None, lon: float | None, acquisition_dt: datetime
    ) -> tuple[float, str]:
        return (
            acquisition_dt.hour + acquisition_dt.minute / 60.0 + acquisition_dt.second / 3600.0,
            "utc_proxy",
        )


class NoaaCoopsTideProvider:
    """Best-effort NOAA CO-OPS tide prediction provider.

    This uses the nearest tide station and CO-OPS predictions when network access
    is available. If anything fails, the caller should fall back to UTC proxy.
    """

    def __init__(self) -> None:
        self._stations: list[dict[str, Any]] | None = None
        self._station_cache: dict[tuple[str, date], list[tuple[datetime, float]]] = {}

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2.0) ** 2
        )
        return 2.0 * r * math.asin(math.sqrt(a))

    def _load_stations(self) -> list[dict[str, Any]]:
        if self._stations is not None:
            return self._stations
        import urllib.request

        with urllib.request.urlopen(
            "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json",
            timeout=30,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        stations = payload.get("stations", [])
        self._stations = [
            {
                "id": station.get("id") or station.get("stationid"),
                "name": station.get("name", ""),
                "lat": _safe_float(str(station.get("lat", ""))),
                "lon": _safe_float(str(station.get("lng", station.get("lon", "")))),
            }
            for station in stations
        ]
        self._stations = [
            s for s in self._stations if s["id"] and s["lat"] is not None and s["lon"] is not None
        ]
        return self._stations

    def _nearest_station(self, lat: float, lon: float) -> dict[str, Any]:
        stations = self._load_stations()
        return min(
            stations,
            key=lambda station: self._haversine_km(
                lat, lon, float(station["lat"]), float(station["lon"])
            ),
        )

    def _load_predictions(self, station_id: str, day: date) -> list[tuple[datetime, float]]:
        key = (station_id, day)
        cached = self._station_cache.get(key)
        if cached is not None:
            return cached

        import urllib.parse
        import urllib.request

        begin = day.isoformat()
        end = (day + timedelta(days=1)).isoformat()
        query = urllib.parse.urlencode(
            {
                "product": "predictions",
                "application": "herring-spawner",
                "begin_date": begin,
                "end_date": end,
                "station": station_id,
                "datum": "MLLW",
                "interval": "h",
                "time_zone": "gmt",
                "units": "metric",
                "format": "json",
            }
        )
        url = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?{query}"
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        predictions = []
        for row in payload.get("predictions", []):
            when = datetime.strptime(row["t"], "%Y-%m-%d %H:%M")
            value = float(row["v"])
            predictions.append((when, value))
        self._station_cache[key] = predictions
        return predictions

    def estimate(
        self, lat: float | None, lon: float | None, acquisition_dt: datetime
    ) -> tuple[float, str]:
        if lat is None or lon is None:
            raise RuntimeError("NOAA tide prediction requires coordinates")
        station = self._nearest_station(lat, lon)
        predictions = self._load_predictions(str(station["id"]), acquisition_dt.date())
        if not predictions:
            raise RuntimeError("NOAA tide predictions unavailable")
        best_when, best_value = min(predictions, key=lambda item: abs(item[0] - acquisition_dt))
        return best_value, f"noaa:{station['id']}"


def _build_review_html(
    title: str, summary: dict[str, Any], rows: Sequence[dict[str, Any]], output_dir: Path
) -> str:
    scored = sorted(rows, key=lambda row: float(row.get("score") or -1e9), reverse=True)
    body: list[str] = []
    for row in scored:
        img_path = Path(str(row.get("image_path", "")))
        rel = ""
        if img_path.exists():
            try:
                rel = os.path.relpath(img_path, output_dir)
            except ValueError:
                rel = img_path.as_posix()
        score = row.get("score")
        score_text = "n/a" if score is None else f"{float(score):.4f}"
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('filename', '')))}</td>"
            f"<td>{html.escape(str(row.get('location_key', '')))}</td>"
            f"<td>{row.get('label', '')}</td>"
            f"<td>{score_text}</td>"
            f"<td>{html.escape(str(row.get('reason', '')))}</td>"
            f"<td>{html.escape(str(row.get('sun_elevation', '')))}</td>"
            f"<td>{html.escape(str(row.get('sun_azimuth', '')))}</td>"
            f"<td>{html.escape(str(row.get('tide_height', '')))}</td>"
            f"<td>{html.escape(str(row.get('tide_source', '')))}</td>"
            f"<td>{html.escape(rel)}</td>"
            "</tr>"
        )

    metrics = summary.get("metrics", {})
    location_metrics = summary.get("location_metrics", {})
    n_rows = summary.get("n_rows", 0)
    n_evaluated = summary.get("n_evaluated", 0)
    n_pos = summary.get("evaluated_positive_count", 0)
    n_neg = summary.get("evaluated_negative_count", 0)
    accuracy = metrics.get("accuracy", "n/a")
    auroc = metrics.get("auroc", "n/a")
    ap = metrics.get("ap", "n/a")
    location_accuracy = location_metrics.get("accuracy", "n/a")
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif;",
        "background:#0f111a;color:#eee;margin:0;padding:16px;}",
        ".summary{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 18px;}",
        ".stat{background:#161a24;border:1px solid #24293a;padding:10px 12px;",
        "border-radius:8px;min-width:140px;}",
        "table{width:100%;border-collapse:collapse;font-size:13px;}",
        "th,td{border-bottom:1px solid #2b2f3a;padding:8px;text-align:left;",
        "vertical-align:top;}",
        "th{position:sticky;top:0;background:#161a24;}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<div class='summary'>",
        f"<div class='stat'><div>Rows</div><strong>{n_rows}</strong></div>",
        f"<div class='stat'><div>Evaluated</div><strong>{n_evaluated}</strong></div>",
        f"<div class='stat'><div>Pos evaluated</div><strong>{n_pos}</strong></div>",
        f"<div class='stat'><div>Neg evaluated</div><strong>{n_neg}</strong></div>",
        f"<div class='stat'><div>Accuracy</div><strong>{accuracy}</strong></div>",
        f"<div class='stat'><div>AUROC</div><strong>{auroc}</strong></div>",
        f"<div class='stat'><div>AP</div><strong>{ap}</strong></div>",
        f"<div class='stat'><div>Loc acc</div><strong>{location_accuracy}</strong></div>",
        "</div>",
        "<table><thead><tr><th>File</th><th>Location</th><th>Label</th>",
        "<th>Score</th><th>Reason</th><th>Sun elev</th><th>Sun azim</th>",
        "<th>Tide</th><th>Tide src</th><th>Path</th></tr></thead><tbody>",
        "".join(body),
        "</tbody></table>",
        "</body></html>",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fieldnames})


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _build_model(device: torch.device) -> torch.nn.Module:
    return torch.hub.load("facebookresearch/dinov2", MODEL_NAME).eval().to(device)


def run_benchmark(args: argparse.Namespace) -> int:
    # Load base samples from manifest
    samples = load_local_samples(args.manifest, args.positive_dir, args.negative_dir)
    samples = [sample for sample in samples if sample.image_path.exists()]

    print(f"Base samples loaded: {len(samples)} "
          f"({sum(1 for s in samples if s.label == 1)} positive, "
          f"{sum(1 for s in samples if s.label == 0)} negative)")

    # Build positive location index for Landsat matching
    pos_locations: list[dict[str, Any]] = [
        {"lat": s.lat, "lon": s.lon, "name": s.location_key}
        for s in samples
        if s.label == 1 and s.lat is not None and s.lon is not None
    ]

    # Report per-location scene counts (S2 only)
    from collections import Counter
    loc_counts = Counter(s.location_key for s in samples)
    scenes_before = sum(loc_counts.values())
    locations_before = len(loc_counts)
    avg_before = scenes_before / max(1, locations_before)
    print(f"S2 scenes: {scenes_before} across {locations_before} locations "
          f"({avg_before:.1f} avg/loc)")

    model = None
    device = None
    if any(not sample.image_path.exists() for sample in samples):
        device = _pick_device() if args.device == "auto" else torch.device(args.device)
        print(f"Device: {device}")
        model = _build_model(device)

    if model is None and args.require_embeddings:
        device = torch.device(args.device) if args.device != "auto" else torch.device("cpu")
        print(f"Device: {device}")
        model = _build_model(device)

    if model is None and args.device != "none":
        device = torch.device(args.device) if args.device != "auto" else torch.device("cpu")
        print(f"Device: {device}")
        model = _build_model(device)

    tide_provider: Any = UtcProxyTideProvider()
    try:
        tide_provider = NoaaCoopsTideProvider()
        _ = tide_provider._load_stations()
    except Exception:
        tide_provider = UtcProxyTideProvider()

    observations = build_observations(
        samples,
        model,
        device,
        tide_provider=tide_provider,
        allow_gee_fallback=args.allow_gee_fallback,
        ee_project=args.ee_project,
        max_cloud=args.max_cloud,
    )

    # Load pre-computed Landsat embeddings if directory exists
    landsat_obs: list[Observation] = []
    if args.landsat_dir and args.landsat_dir.exists():
        landsat_obs = load_landsat_observations(args.landsat_dir, pos_locations)
        observations.extend(landsat_obs)
        print(f"Landsat observations loaded: {len(landsat_obs)} "
              f"(from {args.hls_dir})")

    rows, summary = benchmark_environmental_matcher(observations)

    # Add Landsat stats to summary
    n_supp = sum(1 for obs in observations if obs.source == "landsat_embedding")
    summary["landsat_observations"] = n_supp
    summary["supplementary_comparison"] = {
        "note": "Run with --hls-dir to include Landsat, or --no-hls for S2-only"
    }
    # Report per-location scene counts
    from collections import Counter
    loc_counts = Counter(s.location_key for s in samples)
    n_s2 = sum(loc_counts.values())
    n_locs = len(loc_counts)
    print(f"\nScene summary: S2={n_s2}, Landsat={n_supp}, "
          f"locations={n_locs}, "
          f"avg={(n_s2 + n_supp) / max(1, n_locs):.1f}/loc (S2+Landsat)")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "scores.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    review_html = _build_review_html("Environmental Matcher", summary, rows, output_dir)
    (output_dir / "review.html").write_text(review_html, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Review page: file://{(output_dir / 'review.html').resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Environmental-matched temporal herring spawn detection"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--positive-dir", type=Path, default=DEFAULT_POSITIVE_DIR)
    parser.add_argument("--negative-dir", type=Path, default=DEFAULT_NEGATIVE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, mps, auto, or none")
    parser.add_argument("--allow-gee-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ee-project", type=str, default="redd-fish")
    parser.add_argument("--max-cloud", type=float, default=50.0)
    parser.add_argument(
        "--require-embeddings", action="store_true", help="force DINOv2 embedding extraction"
    )
    parser.add_argument(
        "--landsat-dir", type=Path, default=DEFAULT_LANDSAT_DIR,
        help="Landsat embedding directory (default: data/landsat_embeddings)",
    )
    parser.add_argument(
        "--no-landsat", action="store_true",
        help="Disable Landsat augmentation",
    )
    # Alias for backward compatibility
    parser.add_argument("--hls-dir", type=Path, default=None,
                        help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_landsat:
        args.landsat_dir = None
    elif args.hls_dir is not None:
        args.landsat_dir = args.hls_dir  # Alias support
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
