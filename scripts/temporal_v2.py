#!/usr/bin/env python3
"""Temporal v2 herring spawn detection for sparse Sentinel-2 coverage.

This script favors local thumbnails in ``data/samples/positive`` and
``data/samples/negative``. If a sample is missing locally, it can fall back to
Earth Engine on demand.

Modes
-----
outlier
    Sliding-window embedding outlier detection.
yoy
    Year-over-year March/April deviation scoring.
spectral
    RGB-fallback spectral-temporal ratio scoring.
trajectory
    Trajectory-shape classifier over seasonal embedding traces.
cloud_fusion
    Cloud-weighted multi-image fusion score.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.knn_detector import DINO_TRANSFORM, _pick_device
from scripts.scan_bc_coast import download_thumbnail as gee_download_thumbnail
from scripts.scan_bc_coast import find_best_scene as gee_find_best_scene
from scripts.temporal_detector import (
    extract_location_key,
    extract_region_key,
    load_hls_samples,
    parse_acquisition_datetime,
    parse_location_from_filename,
    parse_observation_date,
)


MODEL_NAME = "dinov2_vits14"
DEFAULT_MANIFEST = Path("data/samples/training_manifest.json")
DEFAULT_POSITIVE_DIR = Path("data/samples/positive")
DEFAULT_NEGATIVE_DIR = Path("data/samples/negative")
DEFAULT_OUTPUT_DIR = Path("data/temporal_v2")
DEFAULT_LANDSAT_DIR = Path("data/landsat_embeddings")
TARGET_MONTHS = {3, 4}
BASELINE_MONTHS = {1, 2}
TRAJECTORY_MIN_POINTS = 5
TRAJECTORY_FALLBACK_MIN_POINTS = 4


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
    bands: dict[str, float] = field(default_factory=dict)
    cloud_fraction: float | None = None
    source: str = "local"


def _safe_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _cloud_fraction_from_filename(filename: str) -> float | None:
    m = re.search(r"(?:^|[_-])(cld|cloud)(\d+(?:\.\d+)?)", filename)
    if not m:
        return 0.0
    value = float(m.group(2))
    if value > 1.0:
        return min(1.0, value / 100.0)
    return max(0.0, min(1.0, value))


def load_local_samples(manifest_path: Path, positive_dir: Path, negative_dir: Path) -> list[Sample]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples: list[Sample] = []
    seen: set[str] = set()

    def _candidate_path(filename: str, positive: bool) -> Path:
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

    for filename in manifest.get("positives", []):
        if filename in seen:
            continue
        seen.add(filename)
        path = _candidate_path(filename, True)
        samples.append(_build_sample(path, 1))

    for filename in manifest.get("rejected", []):
        if filename in seen:
            continue
        seen.add(filename)
        path = _candidate_path(filename, False)
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
    acq_dt = parse_acquisition_datetime(filename, obs_date) or datetime.combine(obs_date, dtime(12, 0, 0))
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


def _load_image_bytes(sample: Sample, allow_gee_fallback: bool, ee_project: str, max_cloud: float) -> tuple[bytes | None, str]:
    if sample.image_path.exists():
        source = "supplement" if sample.filename.startswith(("L8_", "L9_")) else "local"
        return sample.image_path.read_bytes(), source

    if not allow_gee_fallback or sample.lat is None or sample.lon is None:
        return None, "missing"

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


def _image_to_bands(image: Image.Image) -> dict[str, float]:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    means = array.mean(axis=(0, 1))
    return {"red": float(means[0]), "green": float(means[1]), "blue": float(means[2])}


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
    allow_gee_fallback: bool = True,
    ee_project: str = "redd-fish",
    max_cloud: float = 50.0,
) -> list[Observation]:
    observations: list[Observation] = []
    for sample in samples:
        png_bytes, source = _load_image_bytes(sample, allow_gee_fallback, ee_project, max_cloud)
        if png_bytes is None:
            continue
        with Image.open(io.BytesIO(png_bytes)) as image_file:
            bands = _image_to_bands(image_file)
        embedding = None
        if model is not None and device is not None:
            embedding = _embed_image(model, device, png_bytes)
        observations.append(
            Observation(
                sample=sample,
                embedding=embedding,
                bands=bands,
                cloud_fraction=_cloud_fraction_from_filename(sample.filename),
                source=source,
            )
        )
    return observations


def load_landsat_observations(
    landsat_dir: Path,
    positive_locations: Sequence[dict[str, Any]],
) -> list[Observation]:
    """Load pre-computed Landsat 8/9 DINOv2 embeddings as Observations.

    Each .npy file contains a 384-dim DINOv2 embedding extracted from a
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

        observations.append(
            Observation(
                sample=sample,
                embedding=embedding,
                source="landsat_embedding",
            )
        )

    return observations


def _group_observations(observations: Sequence[Observation]) -> dict[str, list[Observation]]:
    groups: dict[str, list[Observation]] = defaultdict(list)
    for obs in observations:
        groups[obs.sample.location_key].append(obs)
    for group in groups.values():
        group.sort(key=lambda o: (o.sample.acquisition_dt, o.sample.filename))
    return groups


def _row_template(obs: Observation) -> dict[str, Any]:
    return {
        "filename": obs.sample.filename,
        "location_key": obs.sample.location_key,
        "region_key": obs.sample.region_key,
        "label": obs.sample.label,
        "date": obs.sample.observation_date.isoformat(),
        "source": obs.source,
        "cloud_fraction": obs.cloud_fraction,
    }


def _series_stats(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 1.0
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, max(std, 1e-6)


def _distance_score(value: np.ndarray, baseline: Sequence[np.ndarray]) -> tuple[float | None, dict[str, float]]:
    if not baseline:
        return None, {"baseline_count": 0.0}
    base = np.vstack(baseline)
    centroid = base.mean(axis=0)
    distances = np.linalg.norm(base - centroid, axis=1)
    mean_dist, std_dist = _series_stats(distances)
    value_dist = float(np.linalg.norm(np.asarray(value, dtype=float) - centroid))
    score = (value_dist - mean_dist) / std_dist
    return score, {"baseline_count": float(len(baseline)), "baseline_mean_distance": mean_dist, "baseline_std_distance": std_dist, "value_distance": value_dist}


def _spectral_features(obs: Observation) -> dict[str, float]:
    bands = obs.bands
    red = float(bands.get("red", 0.0))
    green = float(bands.get("green", 0.0))
    blue = float(bands.get("blue", 0.0))
    coastal = float(bands.get("coastal", blue))
    nir = float(bands.get("nir", max(red, green)))
    eps = 1e-6
    return {
        "blue_green_ratio": blue / (green + eps),
        "coastal_proxy": coastal,
        "ndwi_proxy": (green - nir) / (green + nir + eps),
        "whiteness": 1.0 - float(np.std([red, green, blue])),
    }


def _score_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("evaluated") and row.get("score") is not None]
    if not scored:
        return {
            "accuracy": 0.0,
            "auroc": None,
            "ap": None,
            "best_threshold": 0.0,
            "best_accuracy": 0.0,
            "best_balanced_accuracy": 0.0,
            "best_f1": 0.0,
        }

    y = np.asarray([int(row["label"]) for row in scored], dtype=int)
    scores = np.asarray([float(row["score"]) for row in scored], dtype=float)
    preds = (scores >= 0.0).astype(int)
    metrics: dict[str, Any] = {"accuracy": float(accuracy_score(y, preds))}
    if len(np.unique(y)) > 1:
        metrics["auroc"] = float(roc_auc_score(y, scores))
        metrics["ap"] = float(average_precision_score(y, scores))
    else:
        metrics["auroc"] = None
        metrics["ap"] = None

    thresholds = np.unique(scores)
    best_acc = (-1.0, 0.0)
    best_bal = (-1.0, 0.0)
    best_f1 = (-1.0, 0.0)
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        acc = float(accuracy_score(y, pred))
        tp = float(np.sum((pred == 1) & (y == 1)))
        tn = float(np.sum((pred == 0) & (y == 0)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1)))
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


def _score_outlier(observations: Sequence[Observation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_baseline = [obs.embedding for obs in observations if obs.embedding is not None]
    for group in _group_observations(observations).values():
        baseline = [obs.embedding for obs in group if obs.embedding is not None and obs.sample.observation_date.month in BASELINE_MONTHS]
        if not baseline:
            baseline = [obs.embedding for obs in group if obs.embedding is not None and obs.sample.observation_date.month not in TARGET_MONTHS]
        baseline = [emb for emb in baseline if emb is not None]
        for obs in group:
            row = _row_template(obs)
            if obs.embedding is None or obs.sample.observation_date.month not in TARGET_MONTHS or not baseline:
                leave_one_out = [other.embedding for other in group if other.embedding is not None and other.sample.filename != obs.sample.filename]
                leave_one_out = [emb for emb in leave_one_out if emb is not None]
                if not leave_one_out:
                    leave_one_out = [emb for emb in global_baseline if emb is not None and not np.array_equal(emb, obs.embedding)]
                if not leave_one_out:
                    row.update({"score": None, "evaluated": False, "reason": "outside_window_or_no_baseline"})
                    rows.append(row)
                    continue
                score, stats = _distance_score(obs.embedding, leave_one_out)
                row.update({"score": float(score) if score is not None else None, "evaluated": score is not None, "reason": "leave_one_out_fallback", **stats})
                rows.append(row)
                continue
            score, stats = _distance_score(obs.embedding, baseline)
            row.update({"score": float(score) if score is not None else None, "evaluated": score is not None, "reason": "ok", **stats})
            rows.append(row)
    return rows


def _score_yoy(observations: Sequence[Observation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in _group_observations(observations).values():
        for obs in group:
            row = _row_template(obs)
            if obs.embedding is None or obs.sample.observation_date.month not in TARGET_MONTHS:
                row.update({"score": None, "evaluated": False, "reason": "outside_window_or_missing_embedding"})
                rows.append(row)
                continue
            same_month_other_years = [
                other.embedding
                for other in group
                if other.embedding is not None
                and other.sample.observation_date.month == obs.sample.observation_date.month
                and other.sample.observation_date.year != obs.sample.observation_date.year
            ]
            if not same_month_other_years:
                same_month_other_years = [
                    other.embedding
                    for other in group
                    if other.embedding is not None
                    and other.sample.observation_date.month in TARGET_MONTHS
                    and other.sample.observation_date.year != obs.sample.observation_date.year
                ]
            same_month_other_years = [emb for emb in same_month_other_years if emb is not None]
            if not same_month_other_years:
                row.update({"score": None, "evaluated": False, "reason": "no_multi_year_context"})
                rows.append(row)
                continue
            score, stats = _distance_score(obs.embedding, same_month_other_years)
            row.update({"score": float(score) if score is not None else None, "evaluated": score is not None, "reason": "ok", **stats})
            rows.append(row)
    return rows


def _score_spectral(observations: Sequence[Observation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in _group_observations(observations).values():
        baseline_feats = [
            _spectral_features(obs)
            for obs in group
            if obs.sample.observation_date.month in BASELINE_MONTHS
        ]
        if not baseline_feats:
            baseline_feats = [_spectral_features(obs) for obs in group if obs.sample.observation_date.month not in TARGET_MONTHS]
        if not baseline_feats:
            baseline_feats = [_spectral_features(obs) for obs in group]

        base = {k: np.asarray([feat[k] for feat in baseline_feats], dtype=float) for k in baseline_feats[0].keys()}
        base_stats = {k: _series_stats(v) for k, v in base.items()}

        for obs in group:
            row = _row_template(obs)
            feats = _spectral_features(obs)
            if not feats:
                row.update({"score": None, "evaluated": False, "reason": "no_spectral_features"})
                rows.append(row)
                continue
            z_bg = (feats["blue_green_ratio"] - base_stats["blue_green_ratio"][0]) / base_stats["blue_green_ratio"][1]
            z_coastal = (feats["coastal_proxy"] - base_stats["coastal_proxy"][0]) / base_stats["coastal_proxy"][1]
            z_ndwi = (feats["ndwi_proxy"] - base_stats["ndwi_proxy"][0]) / base_stats["ndwi_proxy"][1]
            z_white = (feats["whiteness"] - base_stats["whiteness"][0]) / base_stats["whiteness"][1]
            score = z_bg + z_coastal - z_ndwi + 0.5 * z_white
            row.update({"score": float(score), "evaluated": True, "reason": "rgb_fallback", **feats})
            rows.append(row)
    return rows


def _trajectory_features(group: Sequence[Observation], min_points: int = TRAJECTORY_MIN_POINTS) -> dict[str, float] | None:
    embeddings = [obs.embedding for obs in group if obs.embedding is not None]
    if len(embeddings) < min_points:
        return None
    X = np.vstack(embeddings)
    X = X - X.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        series = X @ vt[0]
    except np.linalg.LinAlgError:
        series = np.linalg.norm(X, axis=1)

    series = np.asarray(series, dtype=float)
    mean = float(series.mean())
    std = float(series.std())
    rng = float(series.max() - series.min())
    variance = float(series.var())
    peak_count = int(np.sum((series[1:-1] > series[:-2]) & (series[1:-1] > series[2:]) & (series[1:-1] > mean + 2.0 * std))) if len(series) >= 3 else 0
    peak_month = int(group[int(np.argmax(series))].sample.observation_date.month)

    def _autocorr(day_lag: int) -> float:
        pairs: list[tuple[float, float]] = []
        for i, left in enumerate(group):
            for j, right in enumerate(group):
                if j <= i:
                    continue
                delta = abs((right.sample.observation_date - left.sample.observation_date).days)
                if abs(delta - day_lag) <= 3:
                    pairs.append((float(series[i]), float(series[j])))
        if len(pairs) < 2:
            return 0.0
        left_vals = np.asarray([p[0] for p in pairs], dtype=float)
        right_vals = np.asarray([p[1] for p in pairs], dtype=float)
        if np.std(left_vals) < 1e-6 or np.std(right_vals) < 1e-6:
            return 0.0
        return float(np.corrcoef(left_vals, right_vals)[0, 1])

    return {
        "variance": variance,
        "range": rng,
        "autocorr_7": _autocorr(7),
        "autocorr_14": _autocorr(14),
        "peak_count_2sigma": float(peak_count),
        "peak_month": float(peak_month),
        "n_points": float(len(series)),
        "span_days": float((group[-1].sample.observation_date - group[0].sample.observation_date).days),
    }


def _score_trajectory(observations: Sequence[Observation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = _group_observations(observations)
    def _collect_feature_rows(min_points: int) -> tuple[list[dict[str, Any]], list[str]]:
        collected_rows: list[dict[str, Any]] = []
        collected_groups: list[str] = []
        for location_key, group in groups.items():
            features = _trajectory_features(group, min_points=min_points)
            if features is None:
                continue
            collected_rows.append({**features, "label": int(max(obs.sample.label for obs in group)), "location_key": location_key})
            collected_groups.append(location_key)
        return collected_rows, collected_groups

    feature_rows, feature_groups = _collect_feature_rows(TRAJECTORY_MIN_POINTS)
    fallback_used = False
    if feature_rows and len(np.unique([row["label"] for row in feature_rows])) < 2:
        feature_rows, feature_groups = _collect_feature_rows(TRAJECTORY_FALLBACK_MIN_POINTS)
        fallback_used = True
    elif not feature_rows:
        feature_rows, feature_groups = _collect_feature_rows(TRAJECTORY_FALLBACK_MIN_POINTS)
        fallback_used = True

    if not feature_rows:
        for group in groups.values():
            for obs in group:
                row = _row_template(obs)
                row.update({"score": None, "evaluated": False, "reason": "too_few_points"})
                rows.append(row)
        return rows

    X = np.asarray([[row[k] for k in ("variance", "range", "autocorr_7", "autocorr_14", "peak_count_2sigma", "peak_month", "n_points", "span_days")] for row in feature_rows], dtype=float)
    y = np.asarray([int(row["label"]) for row in feature_rows], dtype=int)

    if len(np.unique(y)) < 2:
        scores = np.zeros(len(y), dtype=float)
    else:
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", gamma="scale"))
        counts = np.bincount(y)
        min_class = int(counts[counts > 0].min()) if np.any(counts > 0) else 0
        if len(y) < 4 or min_class < 2:
            clf.fit(X, y)
            scores = clf.decision_function(X)
        else:
            cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)
            scores = cross_val_predict(clf, X, y, cv=cv, method="decision_function")

    score_map = {location: float(score) for location, score in zip(feature_groups, scores)}
    for group in groups.values():
        location_score = score_map.get(group[0].sample.location_key)
        features = _trajectory_features(group, min_points=TRAJECTORY_FALLBACK_MIN_POINTS if fallback_used else TRAJECTORY_MIN_POINTS)
        for obs in group:
            row = _row_template(obs)
            if location_score is None or features is None:
                row.update({"score": None, "evaluated": False, "reason": "too_few_points"})
            else:
                row.update({"score": float(location_score), "evaluated": True, "reason": "fallback_min_points" if fallback_used else "ok", **features})
            rows.append(row)
    return rows


def _score_cloud_fusion(observations: Sequence[Observation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in _group_observations(observations).values():
        embeddings = [obs.embedding for obs in group if obs.embedding is not None]
        if len(embeddings) < 2:
            for obs in group:
                row = _row_template(obs)
                row.update({"score": None, "evaluated": False, "reason": "too_few_points"})
                rows.append(row)
            continue

        weights = np.asarray([max(0.05, 1.0 - float(obs.cloud_fraction or 0.0)) for obs in group], dtype=float)
        base_mask = np.asarray([obs.sample.observation_date.month in BASELINE_MONTHS for obs in group], dtype=bool)
        if not np.any(base_mask):
            base_mask = np.asarray([idx < max(1, len(group) // 3) for idx in range(len(group))], dtype=bool)
        spawn_mask = np.asarray([obs.sample.observation_date.month in TARGET_MONTHS for obs in group], dtype=bool)
        if not np.any(spawn_mask):
            spawn_mask = ~base_mask

        base_embs = np.vstack([emb for emb, keep in zip(embeddings, base_mask) if keep])
        spawn_list = [emb for emb, keep in zip(embeddings, spawn_mask) if keep]
        if not spawn_list:
            spawn_list = embeddings
        spawn_embs = np.vstack(spawn_list)
        base_weights = weights[base_mask]
        spawn_weights = weights[spawn_mask]
        if len(spawn_weights) == 0:
            spawn_weights = weights
        base_centroid = np.average(base_embs, axis=0, weights=base_weights[: len(base_embs)])
        spawn_centroid = np.average(spawn_embs, axis=0, weights=spawn_weights[: len(spawn_embs)])
        centroid_gap = float(np.linalg.norm(spawn_centroid - base_centroid))
        peak_gap = float(max((float(np.linalg.norm(emb - base_centroid)) * float(weight) for emb, weight in zip(embeddings, weights)), default=0.0))
        group_score = 0.5 * centroid_gap + 0.5 * peak_gap

        for obs in group:
            row = _row_template(obs)
            row.update({"score": group_score, "evaluated": True, "reason": "cloud_weighted_fusion", "cloud_weight": max(0.05, 1.0 - float(obs.cloud_fraction or 0.0))})
            rows.append(row)
    return rows


def score_mode(mode: str, observations: Sequence[Observation]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode == "outlier":
        rows = _score_outlier(observations)
    elif mode == "yoy":
        rows = _score_yoy(observations)
    elif mode == "spectral":
        rows = _score_spectral(observations)
    elif mode == "trajectory":
        rows = _score_trajectory(observations)
    elif mode == "cloud_fusion":
        rows = _score_cloud_fusion(observations)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    scored = [row for row in rows if row.get("evaluated") and row.get("score") is not None]
    summary = {
        "mode": mode,
        "n_rows": len(rows),
        "n_evaluated": len(scored),
        "n_positive": int(sum(int(row["label"]) for row in rows)),
        "n_negative": int(sum(1 - int(row["label"]) for row in rows)),
        "evaluated_positive_count": int(sum(1 for row in scored if int(row["label"]) == 1)),
        "evaluated_negative_count": int(sum(1 for row in scored if int(row["label"]) == 0)),
        "evaluated_fraction": float(len(scored) / max(1, len(rows))),
        "metrics": _score_rows(rows),
    }
    return rows, summary


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


def _build_review_html(mode: str, summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    scored = sorted(rows, key=lambda row: float(row["score"]) if row.get("score") is not None else -1e9, reverse=True)
    cards: list[str] = []
    for row in scored:
        score = row.get("score")
        score_text = "n/a" if score is None else f"{float(score):.4f}"
        cards.append(
            f"<tr><td>{html.escape(str(row.get('filename','')))}</td>"
            f"<td>{html.escape(str(row.get('location_key','')))}</td>"
            f"<td>{row.get('label','')}</td>"
            f"<td>{score_text}</td>"
            f"<td>{'yes' if row.get('evaluated') else 'no'}</td>"
            f"<td>{html.escape(str(row.get('reason','')))}</td>"
            f"<td>{html.escape(str(row.get('date','')))}</td>"
            f"<td>{html.escape(str(row.get('source','')))}</td></tr>"
        )
    metrics = summary.get("metrics", {})
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Temporal V2 {html.escape(mode)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif;background:#0f111a;color:#eee;margin:0;padding:16px;}}
.summary{{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 18px;}}
.stat{{background:#161a24;border:1px solid #24293a;padding:10px 12px;border-radius:8px;min-width:140px;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
th,td{{border-bottom:1px solid #2b2f3a;padding:8px;text-align:left;vertical-align:top;}}
th{{position:sticky;top:0;background:#161a24;}}
</style></head><body>
<h1>Temporal V2 — {html.escape(mode)}</h1>
<div class='summary'>
<div class='stat'><div>Rows</div><strong>{summary.get('n_rows',0)}</strong></div>
<div class='stat'><div>Evaluated</div><strong>{summary.get('n_evaluated',0)}</strong></div>
<div class='stat'><div>Pos evaluated</div><strong>{summary.get('evaluated_positive_count',0)}</strong></div>
<div class='stat'><div>Neg evaluated</div><strong>{summary.get('evaluated_negative_count',0)}</strong></div>
<div class='stat'><div>Accuracy</div><strong>{metrics.get('accuracy','n/a')}</strong></div>
<div class='stat'><div>AUROC</div><strong>{metrics.get('auroc','n/a')}</strong></div>
<div class='stat'><div>AP</div><strong>{metrics.get('ap','n/a')}</strong></div>
</div>
<table><thead><tr><th>File</th><th>Location</th><th>Label</th><th>Score</th><th>Eval</th><th>Reason</th><th>Date</th><th>Source</th></tr></thead><tbody>{''.join(cards)}</tbody></table>
</body></html>"""


def _run_mode(mode: str, args: argparse.Namespace) -> int:
    samples = load_local_samples(args.manifest, args.positive_dir, args.negative_dir)
    print(f"Base samples loaded: {len(samples)}")

    # Build positive location index for Landsat matching
    pos_locations: list[dict[str, Any]] = []
    for s in samples:
        if s.label == 1 and s.lat is not None and s.lon is not None:
            pos_locations.append({"lat": s.lat, "lon": s.lon})

    device = None
    model = None
    needs_embeddings = mode in {"outlier", "yoy", "trajectory", "cloud_fusion"}
    if needs_embeddings and any(not sample.image_path.exists() for sample in samples):
        device = _pick_device()
        print(f"Device: {device}")
        model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME).eval().to(device)

    observations = build_observations(
        samples,
        model if needs_embeddings else None,
        device,
        allow_gee_fallback=args.allow_gee_fallback,
        ee_project=args.ee_project,
        max_cloud=args.max_cloud,
    )
    if needs_embeddings and model is None and any(obs.embedding is None for obs in observations):
        device = _pick_device()
        print(f"Device: {device}")
        model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME).eval().to(device)
        observations = build_observations(
            samples,
            model,
            device,
            allow_gee_fallback=args.allow_gee_fallback,
            ee_project=args.ee_project,
            max_cloud=args.max_cloud,
        )

    # Load pre-computed Landsat embeddings if directory exists
    landsat_obs: list[Observation] = []
    if args.landsat_dir and args.landsat_dir.exists():
        landsat_obs = load_landsat_observations(args.landsat_dir, pos_locations)
        observations.extend(landsat_obs)
        print(f"Landsat observations loaded: {len(landsat_obs)} (from {args.landsat_dir})")

    n_hls = sum(1 for obs in observations if obs.source == "landsat_embedding")

    rows, summary = score_mode(mode, observations)
    summary["landsat_observations"] = n_hls

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "scores.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "review.html").write_text(_build_review_html(mode, summary, rows), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Review page: file://{(output_dir / 'review.html').resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal v2 herring spawn detection")
    parser.add_argument("--mode", required=True, choices=["outlier", "yoy", "spectral", "trajectory", "cloud_fusion"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--positive-dir", type=Path, default=DEFAULT_POSITIVE_DIR)
    parser.add_argument("--negative-dir", type=Path, default=DEFAULT_NEGATIVE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-gee-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ee-project", type=str, default="redd-fish")
    parser.add_argument("--max-cloud", type=float, default=50.0)
    parser.add_argument(
        "--landsat-dir", type=Path, default=DEFAULT_LANDSAT_DIR,
        help="Landsat embedding directory (default: data/landsat_embeddings)",
    )
    parser.add_argument(
        "--no-landsat", action="store_true",
        help="Disable Landsat augmentation",
    )
    parser.add_argument("--hls-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_landsat:
        args.landsat_dir = None
    elif args.hls_dir is not None:
        args.landsat_dir = args.hls_dir
    return _run_mode(args.mode, args)


if __name__ == "__main__":
    sys.exit(main())
