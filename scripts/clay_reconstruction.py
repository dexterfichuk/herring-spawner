#!/usr/bin/env python3
"""Clay MAE reconstruction-error analysis for herring spawn detection.

This script compares spawn vs. nospawn labels using reconstruction error from
Clay v1.5 in two settings:

  - mask_ratio=0.75 (MAE-style reconstruction)
  - mask_ratio=0.0  (pure autoencoder comparison)

It matches the saved labels in /Users/dexterfichuk/Downloads/herring-labels-v2.json
to the corresponding data/candidates_v2 thumbnails, downloads the matching
Sentinel-2 chip from GEE, runs Clay, and reports mean reconstruction error.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable
from functools import lru_cache


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS_PATH = Path("/Users/dexterfichuk/Downloads/herring-labels-v2.json")
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "candidates_v2" / "manifest.json"
DEFAULT_THUMB_DIR = REPO_ROOT / "data" / "candidates_v2"
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "review" / "clay_reconstruction_report.html"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "review" / "clay_reconstruction_cache"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "v1.5" / "clay-v1.5.ckpt"
METADATA_PATH = REPO_ROOT / "configs" / "metadata.yaml"

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BAND_NAMES = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
PLATFORM = "sentinel-2-l2a"
SIZE = 256
GSD = 10


@dataclass(frozen=True)
class LabelRecord:
    id: str
    region: str
    lat: float
    lon: float
    date: str
    label: str
    title: str = ""
    subtitle: str = ""
    scene_id: str | None = None
    thumbnail_path: Path | None = None
    score: float | None = None


def parse_label_id(label_id: str) -> LabelRecord:
    parts = label_id.split(":", 4)
    if len(parts) != 5 or parts[0] != "cand":
        raise ValueError(f"Unsupported label id: {label_id}")

    _, region, lat_str, lon_str, date_str = parts
    return LabelRecord(
        id=label_id,
        region=region,
        lat=float(lat_str),
        lon=float(lon_str),
        date=date_str,
        label="",
    )


def _manifest_key(region: str, lat: float, lon: float, date_str: str) -> tuple[str, str, str, str]:
    return (region, f"{lat:.6f}", f"{lon:.6f}", date_str)


def load_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    return json.loads(manifest_path.read_text())


def build_manifest_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        index[_manifest_key(str(row["region"]), float(row["lat"]), float(row["lon"]), str(row["date"]))] = row
    return index


def resolve_thumbnail_path(label_id: str, manifest_rows: list[dict[str, Any]], thumb_dir: Path) -> Path:
    record = parse_label_id(label_id)
    index = build_manifest_index(manifest_rows)
    row = index.get(_manifest_key(record.region, record.lat, record.lon, record.date))
    if row and row.get("thumbnail_path"):
        return thumb_dir / str(row["thumbnail_path"])

    # Fallback: scan files for a close filename match.
    lat_str = f"{record.lat:.6f}"
    lon_str = f"{record.lon:.6f}"
    date_compact = record.date.replace("-", "")
    matches = []
    for path in sorted(thumb_dir.glob("*.png")):
        name = path.name
        if (
            record.region in name
            and record.date in name
            and lat_str in name
            and lon_str in name
            and date_compact in name
        ):
            matches.append(path)
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not resolve thumbnail for {label_id}")


def load_label_records(
    labels_path: Path,
    manifest_path: Path,
    thumb_dir: Path,
) -> list[LabelRecord]:
    payload = json.loads(labels_path.read_text())
    manifest_rows = load_manifest_rows(manifest_path)
    manifest_index = build_manifest_index(manifest_rows)

    records: list[LabelRecord] = []
    for item in payload.get("items", []):
        label = item.get("label")
        if label not in {"spawn", "nospawn"}:
            continue

        record = parse_label_id(str(item["id"]))
        row = manifest_index.get(_manifest_key(record.region, record.lat, record.lon, record.date))
        try:
            thumbnail_path = resolve_thumbnail_path(str(item["id"]), manifest_rows, thumb_dir)
        except FileNotFoundError:
            thumbnail_path = None

        records.append(
            LabelRecord(
                id=str(item["id"]),
                region=record.region,
                lat=record.lat,
                lon=record.lon,
                date=record.date,
                label=str(label),
                title=str(item.get("title") or record.region),
                subtitle=str(item.get("subtitle") or record.date),
                scene_id=str(row.get("scene_id")) if row and row.get("scene_id") else None,
                thumbnail_path=thumbnail_path,
                score=float(row["score"]) if row and row.get("score") is not None else None,
            )
        )

    return records


def summarize_results(spawn_errors: Iterable[float], nospawn_errors: Iterable[float]) -> dict[str, float]:
    import numpy as np

    spawn = np.asarray([float(x) for x in spawn_errors], dtype=np.float64)
    nospawn = np.asarray([float(x) for x in nospawn_errors], dtype=np.float64)
    if len(spawn) == 0 or len(nospawn) == 0:
        raise ValueError("Need both spawn and nospawn errors to summarize")

    spawn_mean = float(spawn.mean())
    nospawn_mean = float(nospawn.mean())
    spawn_std = float(spawn.std(ddof=0)) if len(spawn) > 1 else 0.0
    nospawn_std = float(nospawn.std(ddof=0)) if len(nospawn) > 1 else 0.0
    pooled_std = math.sqrt((spawn_std**2 + nospawn_std**2) / 2) if (spawn_std or nospawn_std) else 0.0
    separation = spawn_mean - nospawn_mean
    midpoint_threshold = (spawn_mean + nospawn_mean) / 2
    correct = sum(1 for x in spawn if x >= midpoint_threshold) + sum(1 for x in nospawn if x < midpoint_threshold)
    accuracy = correct / (len(spawn) + len(nospawn))

    return {
        "spawn_mean": spawn_mean,
        "nospawn_mean": nospawn_mean,
        "spawn_std": spawn_std,
        "nospawn_std": nospawn_std,
        "pooled_std": pooled_std,
        "mean_difference": separation,
        "separation": separation,
        "effect_size": separation / pooled_std if pooled_std else 0.0,
        "midpoint_threshold": midpoint_threshold,
        "midpoint_accuracy": accuracy,
    }


@lru_cache(maxsize=1)
def load_rgb_stats():
    from box import Box
    import yaml

    metadata = Box(yaml.safe_load(METADATA_PATH.read_text()))
    mean = [metadata[PLATFORM].bands.mean[b] for b in BAND_NAMES]
    std = [metadata[PLATFORM].bands.std[b] for b in BAND_NAMES]
    waves = [metadata[PLATFORM].bands.wavelength[b] for b in BAND_NAMES]
    return mean, std, waves


def normalize_ts(d: date):
    week = d.isocalendar().week * 2 * math.pi / 52
    hour = 12 * 2 * math.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def normalize_ll(lat: float, lon: float):
    return (
        (math.sin(lat * math.pi / 180), math.cos(lat * math.pi / 180)),
        (math.sin(lon * math.pi / 180), math.cos(lon * math.pi / 180)),
    )


def load_clay_model(mask_ratio: float):
    from claymodel.module import ClayMAEModule

    kwargs = dict(
        model_size="large",
        metadata_path=str(METADATA_PATH),
        mask_ratio=mask_ratio,
        shuffle=False,
    )
    try:
        return ClayMAEModule.load_from_checkpoint(str(CHECKPOINT_PATH), **kwargs)
    except TypeError:
        kwargs.update(dolls=[16, 32, 64, 128, 256, 768, 1024], doll_weights=[1, 1, 1, 1, 1, 1, 1])
        return ClayMAEModule.load_from_checkpoint(str(CHECKPOINT_PATH), **kwargs)


def load_chip_from_gee(record: LabelRecord, cache_dir: Path) -> np.ndarray:
    import numpy as np
    import ee
    import requests
    import tifffile as tiff

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{record.scene_id or 'scene'}_{record.lat:.6f}_{record.lon:.6f}.tif".replace("/", "_")
    cache_path = cache_dir / cache_key

    expected_shape = (len(BANDS), SIZE, SIZE)

    if cache_path.exists():
        chip = tiff.imread(str(cache_path)).astype(np.float32)
        if chip.ndim == 3:
            chip = np.transpose(chip, (2, 0, 1))
        if chip.shape != expected_shape:
            cache_path.unlink(missing_ok=True)
            chip = None
    else:
        chip = None

    if chip is None:
        scene_id = record.scene_id
        if not scene_id:
            start = (date.fromisoformat(record.date) - timedelta(days=7)).isoformat()
            end = (date.fromisoformat(record.date) + timedelta(days=14)).isoformat()
            scenes = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(ee.Geometry.Point(record.lon, record.lat))
                .filterDate(start, end)
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60))
                .sort("CLOUDY_PIXEL_PERCENTAGE")
            )
            ids = scenes.aggregate_array("system:index").getInfo()
            if not ids:
                raise RuntimeError(f"No Sentinel-2 scenes found for {record.id}")
            scene_id = ids[0]

        img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{scene_id}")
        region = ee.Geometry.Point(record.lon, record.lat).buffer(GSD * SIZE / 2).bounds()
        url = img.select(BANDS).getDownloadURL(
            {"region": region, "dimensions": [SIZE, SIZE], "format": "GEO_TIFF"}
        )
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        chip = tiff.imread(str(cache_path)).astype(np.float32)

        if chip.ndim == 3:
            chip = np.transpose(chip, (2, 0, 1))
    if chip.shape != expected_shape:
        from skimage.transform import resize

        chip = np.stack([resize(chip[i], (SIZE, SIZE), preserve_range=True) for i in range(chip.shape[0])])
    return chip.astype(np.float32)


def build_datacube(record: LabelRecord, chip: np.ndarray, device):
    import numpy as np
    import torch
    from torchvision.transforms import v2

    mean, std, waves = load_rgb_stats()
    transform = v2.Compose([v2.Normalize(mean=mean, std=std)])
    pixel_tensor = transform(torch.from_numpy(chip)).unsqueeze(0).to(device)
    scenedate = date.fromisoformat(record.date)
    wn, hn = normalize_ts(scenedate)
    ln, lo = normalize_ll(record.lat, record.lon)
    return {
        "platform": [PLATFORM],
        "time": torch.tensor(np.hstack([wn, hn]), dtype=torch.float32, device=device).unsqueeze(0),
        "latlon": torch.tensor(np.hstack([ln, lo]), dtype=torch.float32, device=device).unsqueeze(0),
        "pixels": pixel_tensor,
        "gsd": torch.tensor(GSD, device=device),
        "waves": torch.tensor(waves, device=device),
    }


def _unpatchify(pred, input_shape, patch_size: int = 8):
    import torch

    if not isinstance(pred, torch.Tensor) or pred.ndim != 3:
        return None
    batch, num_patches, dim = pred.shape
    _, channels, height, width = input_shape
    if height % patch_size or width % patch_size:
        return None
    patches_per_side = height // patch_size
    if num_patches != patches_per_side * patches_per_side:
        return None
    if dim != channels * patch_size * patch_size:
        return None
    image = pred.reshape(batch, patches_per_side, patches_per_side, channels, patch_size, patch_size)
    return image.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)


def _find_reconstruction(obj, input_tensor):
    import torch

    preferred_keys = (
        "reconstruction",
        "recon",
        "recons",
        "pred_pixels",
        "pixels_pred",
        "decoded",
        "prediction",
        "pred",
        "output",
    )

    if isinstance(obj, torch.Tensor):
        if obj.shape == input_tensor.shape:
            return obj
        if obj.ndim == 4 and obj.shape[-1] == input_tensor.shape[1] and obj.shape[1:3] == input_tensor.shape[2:4]:
            return obj.permute(0, 3, 1, 2)
        if obj.ndim == 3 and obj.shape[1] == input_tensor.shape[2] * input_tensor.shape[3] and obj.shape[2] == input_tensor.shape[1]:
            return obj.reshape(input_tensor.shape[0], input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[1]).permute(0, 3, 1, 2)
        return _unpatchify(obj, input_tensor.shape)

    if isinstance(obj, dict):
        for key in preferred_keys:
            if key in obj:
                found = _find_reconstruction(obj[key], input_tensor)
                if found is not None:
                    return found
        for value in obj.values():
            found = _find_reconstruction(value, input_tensor)
            if found is not None:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_reconstruction(value, input_tensor)
            if found is not None:
                return found
        return None

    for key in preferred_keys:
        if hasattr(obj, key):
            found = _find_reconstruction(getattr(obj, key), input_tensor)
            if found is not None:
                return found
    return None


def _call_model(model, datacube):
    attempts = []
    if hasattr(model, "model"):
        attempts.extend([lambda: model.model(datacube), lambda: model.model.forward(datacube)])
    attempts.extend([lambda: model(datacube), lambda: model.forward(datacube)])

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:  # pragma: no cover - runtime dependent
            last_error = exc
    raise RuntimeError("Clay forward pass failed") from last_error


def reconstruct_error(model, datacube) -> float:
    import torch

    with torch.no_grad():
        output = _call_model(model, datacube)
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        reconstruction_loss = output[1]
        return float(reconstruction_loss.detach().item() if isinstance(reconstruction_loss, torch.Tensor) else reconstruction_loss)
    reconstructed = _find_reconstruction(output, datacube["pixels"])
    if reconstructed is None:
        raise RuntimeError(f"Could not find reconstruction tensor in model output: {type(output)!r}")
    return float(torch.nn.functional.mse_loss(reconstructed, datacube["pixels"]).item())


def evaluate_mask_ratio(
    records: list[LabelRecord],
    mask_ratio: float,
    cache_dir: Path,
    device,
) -> dict[str, Any]:
    import torch

    model = load_clay_model(mask_ratio)
    model.eval().to(device)

    rows: list[dict[str, Any]] = []
    for record in records:
        chip = load_chip_from_gee(record, cache_dir)
        datacube = build_datacube(record, chip, device)
        error = reconstruct_error(model, datacube)
        rows.append(
            {
                "id": record.id,
                "label": record.label,
                "region": record.region,
                "lat": record.lat,
                "lon": record.lon,
                "date": record.date,
                "title": record.title,
                "subtitle": record.subtitle,
                "scene_id": record.scene_id,
                "thumbnail_path": str(record.thumbnail_path) if record.thumbnail_path else None,
                "score": record.score,
                "error": error,
            }
        )

    spawn_errors = [row["error"] for row in rows if row["label"] == "spawn"]
    nospawn_errors = [row["error"] for row in rows if row["label"] == "nospawn"]
    summary = summarize_results(spawn_errors, nospawn_errors)
    summary.update(
        {
            "mask_ratio": mask_ratio,
            "count": len(rows),
            "spawn_count": len(spawn_errors),
            "nospawn_count": len(nospawn_errors),
        }
    )
    return {"rows": rows, "summary": summary}


def render_report(results_by_ratio: dict[float, dict[str, Any]], output_path: Path, unmatched: list[str]) -> str:
    best_ratio = max(results_by_ratio, key=lambda r: results_by_ratio[r]["summary"]["separation"])
    best = results_by_ratio[best_ratio]

    def fmt(x: float | None) -> str:
        return "—" if x is None else f"{x:.6f}"

    sections = []
    for ratio in sorted(results_by_ratio.keys(), reverse=True):
        result = results_by_ratio[ratio]
        summary = result["summary"]
        rows = sorted(result["rows"], key=lambda r: r["error"], reverse=True)
        max_error = max(r["error"] for r in rows) if rows else 1.0

        table_rows = []
        for row in rows:
            thumb_rel = None
            if row["thumbnail_path"]:
                thumb_rel = Path(row["thumbnail_path"]).name
                thumb_rel = f"../candidates_v2/{thumb_rel}"
            bar_width = (row["error"] / max_error * 100) if max_error else 0
            table_rows.append(
                f"""
                <tr class=\"{row['label']}\">
                  <td>{html.escape(row['label'])}</td>
                  <td>{html.escape(row['region'])}</td>
                  <td>{html.escape(row['date'])}</td>
                  <td>{html.escape(row['scene_id'] or '')}</td>
                  <td>{fmt(row['error'])}</td>
                  <td><div class=\"bar\"><span style=\"width:{bar_width:.1f}%\"></span></div></td>
                  <td>{html.escape(row['thumbnail_path'] or '')}</td>
                  <td>{f'<img src="{thumb_rel}" alt="thumb">' if thumb_rel else '—'}</td>
                </tr>
                """
            )

        sections.append(
            f"""
            <section class=\"panel\">
              <h2>mask_ratio = {ratio:.2f}</h2>
              <div class=\"stats\">
                <div class=\"card\"><div class=\"k\">spawn mean</div><div class=\"v\">{summary['spawn_mean']:.6f}</div></div>
                <div class=\"card\"><div class=\"k\">nospawn mean</div><div class=\"v\">{summary['nospawn_mean']:.6f}</div></div>
                <div class=\"card\"><div class=\"k\">separation</div><div class=\"v\">{summary['separation']:.6f}</div></div>
                <div class=\"card\"><div class=\"k\">effect size</div><div class=\"v\">{summary['effect_size']:.3f}</div></div>
                <div class=\"card\"><div class=\"k\">midpoint accuracy</div><div class=\"v\">{summary['midpoint_accuracy']:.1%}</div></div>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>label</th><th>region</th><th>date</th><th>scene id</th><th>error</th><th>relative</th><th>thumb file</th><th>preview</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(table_rows)}
                </tbody>
              </table>
            </section>
            """
        )

    unmatched_html = "<li>" + "</li><li>".join(html.escape(x) for x in unmatched) + "</li>" if unmatched else "<li>none</li>"

    report = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Clay Reconstruction Report</title>
  <style>
    body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f5f7fb; color: #1b1f2a; }}
    header {{ background: linear-gradient(135deg, #172033, #0f172a); color: white; padding: 24px 28px; }}
    h1 {{ margin: 0; font-size: 28px; }}
    .sub {{ opacity: .85; margin-top: 8px; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 20px 24px 40px; }}
    .panel {{ background: white; border-radius: 14px; box-shadow: 0 2px 10px rgba(0,0,0,.06); padding: 18px; margin-bottom: 20px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 12px 0 16px; }}
    .card {{ background: #f7f9fc; border: 1px solid #e4e9f2; border-radius: 12px; padding: 12px 14px; }}
    .k {{ font-size: 11px; text-transform: uppercase; color: #687387; letter-spacing: .05em; }}
    .v {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf1f6; vertical-align: top; text-align: left; }}
    th {{ position: sticky; top: 0; background: #f7f9fc; z-index: 1; }}
    tr.spawn td:first-child {{ color: #0f7a33; font-weight: 700; }}
    tr.nospawn td:first-child {{ color: #9a3412; font-weight: 700; }}
    img {{ width: 110px; height: 110px; object-fit: cover; border-radius: 8px; border: 1px solid #dfe6f0; }}
    .bar {{ width: 180px; height: 10px; background: #edf2f7; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: linear-gradient(90deg, #60a5fa, #ef4444); }}
    code {{ background: #eef2ff; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>Clay Reconstruction Error</h1>
    <div class=\"sub\">Zero-shot spawn detection using MAE-style reconstruction error on labeled candidate scenes.</div>
  </header>
  <div class=\"wrap\">
    <section class=\"panel\">
      <h2>Summary</h2>
      <div>Best separation: <code>mask_ratio = {best_ratio:.2f}</code></div>
      <div>Spawn mean: <code>{best['summary']['spawn_mean']:.6f}</code></div>
      <div>Nospawn mean: <code>{best['summary']['nospawn_mean']:.6f}</code></div>
      <div>Mean difference: <code>{best['summary']['mean_difference']:.6f}</code></div>
      <div>Midpoint accuracy: <code>{best['summary']['midpoint_accuracy']:.1%}</code></div>
    </section>
    <section class=\"panel\">
      <h2>Unmatched labels</h2>
      <ul>{unmatched_html}</ul>
    </section>
    {''.join(sections)}
  </div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clay reconstruction-error analysis")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--thumb-dir", type=Path, default=DEFAULT_THUMB_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-items", type=int, default=0, help="Limit labels for a quick smoke test")
    args = parser.parse_args()

    import torch
    import ee

    ee.Initialize(project="redd-fish")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    records = load_label_records(args.labels, args.manifest, args.thumb_dir)
    if args.max_items > 0:
        records = records[: args.max_items]

    unmatched: list[str] = []
    for record in records:
        if not record.thumbnail_path or not record.thumbnail_path.exists():
            unmatched.append(record.id)

    results_by_ratio: dict[float, dict[str, Any]] = {}
    for mask_ratio in (0.75, 0.0):
        print(f"Running Clay reconstruction with mask_ratio={mask_ratio:.2f} on {len(records)} labels...")
        results_by_ratio[mask_ratio] = evaluate_mask_ratio(records, mask_ratio, args.cache_dir, device)
        summary = results_by_ratio[mask_ratio]["summary"]
        print(
            f"  spawn={summary['spawn_mean']:.6f} nospawn={summary['nospawn_mean']:.6f} "
            f"separation={summary['separation']:.6f} accuracy={summary['midpoint_accuracy']:.1%}"
        )

    report_path = render_report(results_by_ratio, args.report, unmatched)
    print(f"Report written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
