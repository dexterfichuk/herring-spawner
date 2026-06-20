#!/usr/bin/env python3
"""Temporal herring spawn detector with tide-aware image matching.

Modes
-----
triplet
    Before/during/after DINOv2 delta scoring with a reversal check.
timeseries
    Intra-season trajectory scoring and plotting.
scan
    Score candidate locations with the best temporal model.

The implementation is deliberately CPU-friendly and works from the golden set
files on disk. If exact tide data is unavailable, acquisition time is used as a
proxy so that scenes are still tide-matched approximately by local time of day.
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
from dataclasses import dataclass
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
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.delta_detector import parse_location_from_filename
from scripts.knn_detector import DINO_TRANSFORM, _pick_device


MODEL_NAME = "dinov2_vits14"
DEFAULT_MANIFEST = Path("data/samples/training_manifest.json")
DEFAULT_POSITIVE_DIR = Path("data/samples/positive")
DEFAULT_NEGATIVE_DIR = Path("data/samples/negative")
DEFAULT_TRIPLET_OUT = Path("data/temporal_triplet")
DEFAULT_SERIES_OUT = Path("data/temporal_timeseries")
DEFAULT_SCAN_DIR = Path("data/candidates_knn")
DEFAULT_REVERSAL_THRESHOLD = 0.75
DEFAULT_TIDE_TOLERANCE = 1.0


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


def _safe_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_observation_date(text: str) -> date | None:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{8})", text)
    if m:
        return date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8]))
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def parse_acquisition_datetime(text: str, fallback_date: date | None = None) -> datetime | None:
    """Parse acquisition datetime from a scene ID or filename.

    Supports ``YYYYMMDDTHHMMSS`` and ``T192019``-style fragments. If only a
    date is available, returns noon local solar time as a proxy.
    """
    m = re.search(r"(\d{8})T(\d{6})", text)
    if m:
        y, mo, d = int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8])
        hh, mm, ss = int(m.group(2)[:2]), int(m.group(2)[2:4]), int(m.group(2)[4:6])
        return datetime(y, mo, d, hh, mm, ss)

    m = re.search(r"T(\d{6})", text)
    if m:
        day = fallback_date or parse_observation_date(text)
        if day is not None:
            hh, mm, ss = int(m.group(1)[:2]), int(m.group(1)[2:4]), int(m.group(1)[4:6])
            return datetime.combine(day, dtime(hh, mm, ss))

    day = fallback_date or parse_observation_date(text)
    if day is not None:
        return datetime.combine(day, dtime(12, 0, 0))
    return None


def extract_region_key(filename: str) -> str:
    m = re.search(r"^(.*?)(?:_\d{4}-\d{2}-\d{2}|_\d{8})", filename)
    if m:
        return m.group(1).strip("_") or Path(filename).stem
    return Path(filename).stem


def extract_location_key(filename: str) -> tuple[str, float | None, float | None]:
    parsed = parse_location_from_filename(filename)
    if parsed is not None:
        lat = round(float(parsed["lat"]), 6)
        lon = round(float(parsed["lon"]), 6)
        return f"coords:{lat:.6f}_{lon:.6f}", float(parsed["lat"]), float(parsed["lon"])
    region_key = extract_region_key(filename)
    return f"region:{region_key}", None, None


def _candidate_image_path(root: Path, filename: str, positive: bool) -> Path:
    subdir = root / ("positive" if positive else "negative")
    return subdir / filename


def load_golden_samples(manifest_path: Path, positive_dir: Path, negative_dir: Path) -> list[Sample]:
    manifest = json.loads(manifest_path.read_text())
    samples: list[Sample] = []
    seen: set[str] = set()

    for filename in manifest.get("positives", []):
        if filename in seen:
            continue
        seen.add(filename)
        path = _candidate_image_path(manifest_path.parent, filename, True)
        if not path.exists():
            path = positive_dir / filename
        if not path.exists():
            continue
        samples.append(_build_sample(path, 1))

    for filename in manifest.get("rejected", []):
        if filename in seen:
            continue
        seen.add(filename)
        path = _candidate_image_path(manifest_path.parent, filename, False)
        if not path.exists():
            path = negative_dir / filename
        if not path.exists():
            continue
        samples.append(_build_sample(path, 0))

    return samples


def load_hls_samples(
    hls_dir: Path,
    positive_locations: Sequence[dict[str, Any]] | None = None,
    include_labels: set[int] | None = None,
) -> list[Sample]:
    """Load HLS thumbnails as additional samples, matched to known positive locations.

    Args:
        hls_dir: Root of the HLS thumbnail directory (contains */*.png).
        positive_locations: List of dicts with 'lat', 'lon' keys for
            known positive coordinates. If None, builds from manifest.
        include_labels: Set of label values to include (None = all).

    Returns:
        List of Sample objects for HLS thumbnails.
    """
    if include_labels is None:
        include_labels = {1}

    samples: list[Sample] = []
    seen_files: set[str] = set()

    hls_dir = Path(hls_dir)
    if not hls_dir.exists():
        return samples

    # Build a fast proximity index from positive locations
    pos_index: list[dict[str, Any]] = []
    if positive_locations:
        pos_index = list(positive_locations)
    else:
        manifest_path = hls_dir.parent / "samples" / "training_manifest.json"
        if manifest_path.exists():
            manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
            for fname in manifest_json.get("positives", []):
                parsed = parse_location_from_filename(fname)
                if parsed:
                    pos_index.append({
                        "lat": float(parsed["lat"]),
                        "lon": float(parsed["lon"]),
                    })

    for hls_file in sorted(hls_dir.rglob("*.png")):
        if hls_file.name in seen_files:
            continue
        seen_files.add(hls_file.name)

        parsed = parse_location_from_filename(hls_file.name)
        if parsed is None:
            continue

        lat = float(parsed["lat"])
        lon = float(parsed["lon"])

        # Match against known positive locations (within ~100m ~ 0.001 deg)
        label = 0
        for ploc in pos_index:
            if abs(lat - float(ploc["lat"])) < 0.001 and abs(
                lon - float(ploc["lon"])
            ) < 0.001:
                label = 1
                break

        if label not in include_labels:
            continue

        obs_date = parse_observation_date(hls_file.name)
        if obs_date is None:
            continue

        location_key, _, _ = extract_location_key(hls_file.name)
        region_key = extract_region_key(hls_file.name)
        acq_dt = parse_acquisition_datetime(hls_file.name, obs_date) or datetime.combine(
            obs_date, dtime(12, 0, 0)
        )

        samples.append(
            Sample(
                filename=hls_file.name,
                label=label,
                image_path=hls_file,
                region_key=region_key,
                location_key=location_key,
                observation_date=obs_date,
                acquisition_dt=acq_dt,
                lat=lat,
                lon=lon,
            )
        )

    return samples


def load_scan_samples(scan_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for path in sorted(scan_dir.glob("*.png")):
        filename = path.name
        region_key = extract_region_key(filename)
        location_key, lat, lon = extract_location_key(filename)
        obs_date = parse_observation_date(filename) or date.fromtimestamp(path.stat().st_mtime)
        acquisition_dt = parse_acquisition_datetime(filename, obs_date) or datetime.combine(obs_date, dtime(12, 0, 0))
        samples.append(
            Sample(
                filename=filename,
                label=-1,
                image_path=path,
                region_key=region_key,
                location_key=location_key,
                observation_date=obs_date,
                acquisition_dt=acquisition_dt,
                lat=lat,
                lon=lon,
            )
        )
    return samples


def load_embedding(model: torch.nn.Module, device: torch.device, image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
    tensor = DINO_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    return F.normalize(emb, dim=1).cpu().numpy().flatten().astype(float)


def embed_samples(samples: Sequence[Sample], model: torch.nn.Module, device: torch.device) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for sample in samples:
        cache[sample.filename] = load_embedding(model, device, sample.image_path)
    return cache


def tide_proxy(acquisition_dt: datetime | None) -> float:
    if acquisition_dt is None:
        return 0.0
    return acquisition_dt.hour + acquisition_dt.minute / 60.0 + acquisition_dt.second / 3600.0


def estimate_tide_height(sample: Sample) -> tuple[float, str]:
    # Best-effort exact tide providers would slot in here. In this repo,
    # fallback proxy matching is the reliable path.
    return tide_proxy(sample.acquisition_dt), "utc_proxy"


def select_tide_matched_scene(
    scenes: Sequence[dict[str, Any]],
    target_tide_height: float,
    tide_tolerance: float = DEFAULT_TIDE_TOLERANCE,
) -> dict[str, Any] | None:
    if not scenes:
        return None

    def sort_key(scene: dict[str, Any]) -> tuple[float, float, float]:
        tide_height = float(scene.get("tide_height", 0.0))
        cloud = float(scene.get("cloud", 100.0))
        tide_diff = abs(tide_height - target_tide_height)
        within = 0.0 if tide_diff <= tide_tolerance else 1.0
        return (within, tide_diff, cloud)

    return sorted(scenes, key=sort_key)[0]


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_triplet_features(
    before_emb: np.ndarray,
    during_emb: np.ndarray,
    after_emb: np.ndarray,
    tide_before: float | None = None,
    tide_during: float | None = None,
    tide_after: float | None = None,
    reversal_threshold: float = DEFAULT_REVERSAL_THRESHOLD,
) -> dict[str, Any]:
    spawn_score = _l2(during_emb, before_emb)
    temporal_score = _l2(during_emb, after_emb)
    recovery_score = _l2(after_emb, before_emb)
    reversal_ok = temporal_score >= reversal_threshold
    tide_values = [v for v in [tide_before, tide_during, tide_after] if v is not None]
    tide_span = float(max(tide_values) - min(tide_values)) if tide_values else 0.0
    tide_match_score = 1.0 / (1.0 + tide_span)
    combined = spawn_score * float(reversal_ok) * tide_match_score
    return {
        "spawn_score": spawn_score,
        "temporal_score": temporal_score,
        "recovery_score": recovery_score,
        "tide_span": tide_span,
        "tide_match_score": tide_match_score,
        "reversal_ok": reversal_ok,
        "combined_score": combined,
        "feature_vector": np.array(
            [spawn_score, temporal_score, recovery_score, tide_span, tide_match_score],
            dtype=float,
        ),
    }


def compute_timeseries_features(
    embeddings: Sequence[np.ndarray],
    dates: Sequence[str],
    tides: Sequence[float],
) -> dict[str, Any]:
    if not embeddings:
        return {
            "peak_distance": 0.0,
            "rise_rate": 0.0,
            "fall_rate": 0.0,
            "bump_score": 0.0,
            "seasonal_contrast": 0.0,
            "peak_index": -1,
            "distance_series": [],
        }

    baseline = np.asarray(embeddings[0], dtype=float)
    series = np.asarray([_l2(np.asarray(emb, dtype=float), baseline) for emb in embeddings], dtype=float)
    peak_index = int(np.argmax(series))
    peak_distance = float(series[peak_index])
    seasonal_contrast = float(series.max() - series.min())

    def _span_days(a: str, b: str) -> float:
        da = datetime.fromisoformat(a)
        db = datetime.fromisoformat(b)
        return max(1.0, abs((db - da).days))

    rise_days = _span_days(dates[0], dates[peak_index]) if peak_index > 0 else 1.0
    fall_days = _span_days(dates[peak_index], dates[-1]) if peak_index < len(series) - 1 else 1.0
    rise_rate = float((series[peak_index] - series[0]) / rise_days)
    fall_rate = float((series[peak_index] - series[-1]) / fall_days)
    bump_score = float(max(0.0, rise_rate) * max(0.0, fall_rate) * (1.0 + peak_distance))

    tide_span = float(max(tides) - min(tides)) if tides else 0.0
    return {
        "peak_distance": peak_distance,
        "rise_rate": rise_rate,
        "fall_rate": fall_rate,
        "bump_score": bump_score,
        "seasonal_contrast": seasonal_contrast,
        "peak_index": peak_index,
        "distance_series": series.tolist(),
        "tide_span": tide_span,
    }


def _group_samples(samples: Sequence[Sample]) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.location_key, []).append(sample)
    for group in groups.values():
        group.sort(key=lambda s: (s.acquisition_dt, s.filename))
    return groups


def _fit_predict_scores(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> np.ndarray:
    if len(np.unique(y)) < 2:
        return np.zeros(len(y), dtype=float)

    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", gamma="scale"))
    counts = np.bincount(y.astype(int))
    min_class = int(counts[counts > 0].min()) if np.any(counts > 0) else 0
    if len(y) < 4 or min_class < 2:
        model.fit(X, y)
        return model.decision_function(X)

    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_predict(model, X, y, cv=cv, method="decision_function")


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.0) -> dict[str, Any]:
    preds = (scores >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, scores))
        metrics["ap"] = float(average_precision_score(y_true, scores))
    else:
        metrics["auroc"] = None
        metrics["ap"] = None
    return metrics


def _threshold_sweep(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    thresholds = np.unique(scores)
    if len(thresholds) == 0:
        return {"best_threshold": 0.0, "best_accuracy": 0.0, "best_balanced_accuracy": 0.0, "best_f1": 0.0}

    best = {"best_accuracy": -1.0, "best_balanced_accuracy": -1.0, "best_f1": -1.0}
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        acc = float(accuracy_score(y_true, pred))
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        tn = float(np.sum((pred == 0) & (y_true == 0)))
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        fn = float(np.sum((pred == 0) & (y_true == 1)))
        tpr = tp / max(1.0, tp + fn)
        tnr = tn / max(1.0, tn + fp)
        bal_acc = 0.5 * (tpr + tnr)
        f1 = (2.0 * tp) / max(1.0, 2.0 * tp + fp + fn)
        if acc > best["best_accuracy"]:
            best["best_accuracy"] = acc
            best["best_threshold_accuracy"] = float(thr)
        if bal_acc > best["best_balanced_accuracy"]:
            best["best_balanced_accuracy"] = bal_acc
            best["best_threshold_balanced_accuracy"] = float(thr)
        if f1 > best["best_f1"]:
            best["best_f1"] = f1
            best["best_threshold_f1"] = float(thr)
    best["best_threshold"] = best.get("best_threshold_f1", 0.0)
    return best


def _train_image_svm(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return _fit_predict_scores(embeddings, labels)


def _train_triplet_svm(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return _fit_predict_scores(features, labels)


def _choose_triplet_for_center(
    samples: Sequence[Sample],
    embeddings: dict[str, np.ndarray],
    center_index: int,
    tide_tolerance: float,
) -> tuple[Sample, Sample, Sample, dict[str, Any]] | None:
    center = samples[center_index]
    before_candidates = [s for s in samples[:center_index]]
    after_candidates = [s for s in samples[center_index + 1 :]]
    if not before_candidates or not after_candidates:
        return None

    center_tide, tide_method = estimate_tide_height(center)
    scored_before: list[dict[str, Any]] = []
    for sample in before_candidates:
        tide_height, _ = estimate_tide_height(sample)
        scored_before.append({"sample": sample, "tide_height": tide_height, "cloud": 0.0})
    scored_after: list[dict[str, Any]] = []
    for sample in after_candidates:
        tide_height, _ = estimate_tide_height(sample)
        scored_after.append({"sample": sample, "tide_height": tide_height, "cloud": 0.0})

    chosen_before = select_tide_matched_scene(scored_before, center_tide, tide_tolerance)
    chosen_after = select_tide_matched_scene(scored_after, center_tide, tide_tolerance)
    if chosen_before is None or chosen_after is None:
        return None

    before = chosen_before["sample"]
    after = chosen_after["sample"]
    if before.filename not in embeddings or center.filename not in embeddings or after.filename not in embeddings:
        return None

    triplet_info = compute_triplet_features(
        embeddings[before.filename],
        embeddings[center.filename],
        embeddings[after.filename],
        tide_before=chosen_before["tide_height"],
        tide_during=center_tide,
        tide_after=chosen_after["tide_height"],
    )
    triplet_info["tide_method"] = tide_method
    return before, center, after, triplet_info


def build_triplet_rows(samples: Sequence[Sample], embeddings: dict[str, np.ndarray], tide_tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for location_key, group in _group_samples(samples).items():
        if len(group) < 3:
            continue
        for idx in range(1, len(group) - 1):
            triplet = _choose_triplet_for_center(group, embeddings, idx, tide_tolerance)
            if triplet is None:
                continue
            before, center, after, features = triplet
            rows.append(
                {
                    "location_key": location_key,
                    "label": center.label,
                    "before_file": before.filename,
                    "during_file": center.filename,
                    "after_file": after.filename,
                    "before_date": before.observation_date.isoformat(),
                    "during_date": center.observation_date.isoformat(),
                    "after_date": after.observation_date.isoformat(),
                    "before_tide": estimate_tide_height(before)[0],
                    "during_tide": estimate_tide_height(center)[0],
                    "after_tide": estimate_tide_height(after)[0],
                    **features,
                }
            )
    return rows


def build_series_rows(samples: Sequence[Sample], embeddings: dict[str, np.ndarray], tide_tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for location_key, group in _group_samples(samples).items():
        if not group:
            continue
        group_embeddings = [embeddings[s.filename] for s in group if s.filename in embeddings]
        if not group_embeddings:
            continue
        dates = [s.observation_date.isoformat() for s in group]
        tides = [estimate_tide_height(s)[0] for s in group]
        features = compute_timeseries_features(group_embeddings, dates, tides)
        rise_fall = bool(features["peak_index"] not in {-1, 0, len(group_embeddings) - 1} and features["rise_rate"] > 0 and features["fall_rate"] > 0)
        rows.append(
            {
                "location_key": location_key,
                "label": int(max(s.label for s in group)),
                "n_points": len(group_embeddings),
                "first_date": group[0].observation_date.isoformat(),
                "last_date": group[-1].observation_date.isoformat(),
                "tide_span": features["tide_span"],
                "rise_fall": rise_fall,
                **features,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(row.get(k)) for k in fieldnames})


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _build_review_html(title: str, summary: dict[str, Any], rows: Sequence[dict[str, Any]], output_dir: Path) -> str:
    cards: list[str] = []
    for row in rows:
        thumb = row.get("during_path") or row.get("image_path") or ""
        rel = ""
        if thumb:
            thumb_path = Path(str(thumb))
            if thumb_path.exists():
                try:
                    rel = os.path.relpath(thumb_path, output_dir)
                except ValueError:
                    rel = thumb_path.as_posix()
        score = float(row.get("triplet_score", row.get("score", row.get("combined_score", 0.0))) or 0.0)
        label = row.get("label", "?")
        cards.append(
            f"<tr><td>{html.escape(str(row.get('location_key','')))}</td><td>{label}</td>"
            f"<td>{score:.4f}</td><td>{html.escape(str(row.get('before_file', row.get('first_file',''))))}</td>"
            f"<td>{html.escape(str(row.get('during_file', row.get('series_files',''))))}</td>"
            f"<td>{html.escape(str(row.get('after_file', row.get('last_file',''))))}</td>"
            f"<td>{html.escape(rel)}</td></tr>"
        )
    body = "\n".join(cards)
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(title)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif;background:#0f111a;color:#eee;margin:0;padding:16px;}}
table{{width:100%;border-collapse:collapse;font-size:13px;}}
th,td{{border-bottom:1px solid #2b2f3a;padding:8px;text-align:left;vertical-align:top;}}
th{{position:sticky;top:0;background:#161a24;}}
.summary{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px;}}
.stat{{background:#161a24;padding:12px 16px;border-radius:8px;min-width:140px;}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class='summary'>
<div class='stat'><div>Samples</div><strong>{summary.get('n_samples',0)}</strong></div>
<div class='stat'><div>Positives</div><strong>{summary.get('n_positive',0)}</strong></div>
<div class='stat'><div>Negatives</div><strong>{summary.get('n_negative',0)}</strong></div>
<div class='stat'><div>AUROC</div><strong>{summary.get('auroc','n/a')}</strong></div>
<div class='stat'><div>AP</div><strong>{summary.get('ap','n/a')}</strong></div>
<div class='stat'><div>Accuracy</div><strong>{summary.get('accuracy','n/a')}</strong></div>
</div>
<table><thead><tr><th>Location</th><th>Label</th><th>Score</th><th>Before</th><th>During</th><th>After</th><th>Thumb</th></tr></thead><tbody>{body}</tbody></table>
</body></html>"""


def _save_series_plots(output_dir: Path, rows: Sequence[dict[str, Any]], title_prefix: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        distances = row.get("distance_series", [])
        if not distances:
            continue
        dates = row.get("dates", [])
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(len(distances)), distances, marker="o")
        ax.set_title(f"{title_prefix} {row.get('location_key','')}")
        ax.set_ylabel("Embedding distance")
        ax.set_xlabel("Observation index")
        ax.grid(True, alpha=0.3)
        plot_path = plots_dir / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', str(row.get('location_key','series')))}.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        row["plot_path"] = str(plot_path)


def _triplet_mode(args: argparse.Namespace) -> int:
    device = _pick_device()
    print(f"Device: {device}")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME).eval().to(device)

    samples = load_golden_samples(args.manifest, args.positive_dir, args.negative_dir)
    samples = [s for s in samples if s.image_path.exists()]
    embeddings = embed_samples(samples, model, device)

    rows = build_triplet_rows(samples, embeddings, args.tide_tolerance)
    if not rows:
        raise RuntimeError("No triplets could be formed from the golden set")

    X_triplet = np.vstack([row["feature_vector"] for row in rows])
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    triplet_scores = _train_triplet_svm(X_triplet, y)

    center_files = [row["during_file"] for row in rows]
    image_embeddings = np.vstack([embeddings[name] for name in center_files])
    image_scores = _train_image_svm(image_embeddings, y)
    pair_scores = np.asarray([row["spawn_score"] for row in rows], dtype=float)

    triplet_metrics = _metrics(y, triplet_scores)
    pair_metrics = _metrics(y, pair_scores)
    image_metrics = _metrics(y, image_scores)

    summary = {
        "mode": "triplet",
        "n_samples": len(rows),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "triplet": {**triplet_metrics, **_threshold_sweep(y, triplet_scores)},
        "single_pair": pair_metrics,
        "single_image": image_metrics,
        "tide_matching": {
            "method": "utc_proxy",
            "tide_tolerance": args.tide_tolerance,
            "coverage": float(np.mean([row["tide_span"] <= args.tide_tolerance for row in rows])),
        },
    }

    output_dir = args.output_dir or DEFAULT_TRIPLET_OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    for row, ts, img in zip(rows, triplet_scores, image_scores):
        row["triplet_score"] = float(ts)
        row["image_score"] = float(img)
    _write_csv(output_dir / "scores.csv", rows, fieldnames=[
        "location_key", "label", "before_file", "during_file", "after_file", "before_date", "during_date", "after_date",
        "before_tide", "during_tide", "after_tide", "spawn_score", "temporal_score", "recovery_score",
        "tide_span", "tide_match_score", "reversal_ok", "combined_score", "triplet_score", "image_score",
    ])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "review.html").write_text(_build_review_html("Temporal Triplet Review", summary["triplet"], rows, output_dir), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Review page: file://{(output_dir / 'review.html').resolve()}")
    return 0


def _series_mode(args: argparse.Namespace) -> int:
    device = _pick_device()
    print(f"Device: {device}")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME).eval().to(device)

    samples = load_golden_samples(args.manifest, args.positive_dir, args.negative_dir)
    samples = [s for s in samples if s.image_path.exists()]
    embeddings = embed_samples(samples, model, device)

    rows = build_series_rows(samples, embeddings, args.tide_tolerance)
    if not rows:
        raise RuntimeError("No time series could be formed from the golden set")

    X = np.vstack([
        np.array([row["peak_distance"], row["rise_rate"], row["fall_rate"], row["bump_score"], row["seasonal_contrast"], row["tide_span"]], dtype=float)
        for row in rows
    ])
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    scores = _fit_predict_scores(X, y)

    summary = {
        "mode": "timeseries",
        "n_samples": len(rows),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "classifier": {**_metrics(y, scores), **_threshold_sweep(y, scores)},
        "rise_fall_positive_count": int(sum(1 for row in rows if row["label"] == 1 and row["rise_fall"])),
        "tide_matching": {"method": "utc_proxy", "tide_tolerance": args.tide_tolerance},
    }

    output_dir = args.output_dir or DEFAULT_SERIES_OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    for row, score in zip(rows, scores):
        row["score"] = float(score)
        row["dates"] = [s.observation_date.isoformat() for s in _group_samples(samples)[row["location_key"]]]
    _save_series_plots(output_dir, rows, "Temporal trajectory")
    _write_csv(output_dir / "scores.csv", rows, fieldnames=[
        "location_key", "label", "n_points", "first_date", "last_date", "peak_distance", "rise_rate", "fall_rate",
        "bump_score", "seasonal_contrast", "peak_index", "tide_span", "rise_fall", "score", "plot_path",
    ])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _scan_mode(args: argparse.Namespace) -> int:
    # Train on the golden set first, then score candidate locations.
    device = _pick_device()
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME).eval().to(device)
    samples = load_golden_samples(args.manifest, args.positive_dir, args.negative_dir)
    samples = [s for s in samples if s.image_path.exists()]
    embeddings = embed_samples(samples, model, device)
    train_rows = build_triplet_rows(samples, embeddings, args.tide_tolerance)
    if not train_rows:
        raise RuntimeError("No training triplets could be formed")

    X_train = np.vstack([row["feature_vector"] for row in train_rows])
    y_train = np.array([int(row["label"]) for row in train_rows], dtype=int)
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", gamma="scale"))
    clf.fit(X_train, y_train)
    image_train_embeddings = np.vstack([embeddings[row["during_file"]] for row in train_rows])
    image_clf: Any | None = None
    if len(np.unique(y_train)) > 1:
        image_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", gamma="scale"))
        image_clf.fit(image_train_embeddings, y_train)

    scan_samples = load_scan_samples(args.scan_dir)
    scan_samples = [s for s in scan_samples if s.image_path.exists()]
    scan_embeddings = embed_samples(scan_samples, model, device)
    scan_rows = build_triplet_rows(scan_samples, scan_embeddings, args.tide_tolerance)
    if not scan_rows:
        # Fall back to single-image scoring when temporal context is missing.
        scan_rows = []
        for sample in scan_samples:
            scan_emb = scan_embeddings[sample.filename].reshape(1, -1)
            score = float(image_clf.decision_function(scan_emb)[0]) if image_clf is not None else 0.0
            scan_rows.append(
                {
                    "location_key": sample.location_key,
                    "label": sample.label,
                    "during_file": sample.filename,
                    "during_date": sample.observation_date.isoformat(),
                    "triplet_score": score,
                    "image_path": str(sample.image_path),
                }
            )
    else:
        for row in scan_rows:
            row["triplet_score"] = float(clf.decision_function(row["feature_vector"].reshape(1, -1))[0])

    scan_rows.sort(key=lambda r: r.get("triplet_score", 0.0), reverse=True)
    summary = {
        "mode": "scan",
        "n_candidates": len(scan_rows),
        "train_triplets": len(train_rows),
        "tide_matching": {"method": "utc_proxy", "tide_tolerance": args.tide_tolerance},
    }
    output_dir = args.output_dir or DEFAULT_TRIPLET_OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "candidate_scores.csv", scan_rows)
    (output_dir / "scan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "review.html").write_text(_build_review_html("Temporal Scan Review", summary, scan_rows, output_dir), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal herring spawn detection with tide-aware matching")
    parser.add_argument("--mode", required=True, choices=["triplet", "timeseries", "scan"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--positive-dir", type=Path, default=DEFAULT_POSITIVE_DIR)
    parser.add_argument("--negative-dir", type=Path, default=DEFAULT_NEGATIVE_DIR)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tide-tolerance", type=float, default=DEFAULT_TIDE_TOLERANCE)
    parser.add_argument("--reversal-threshold", type=float, default=DEFAULT_REVERSAL_THRESHOLD)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "triplet":
        return _triplet_mode(args)
    if args.mode == "timeseries":
        return _series_mode(args)
    if args.mode == "scan":
        return _scan_mode(args)
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
