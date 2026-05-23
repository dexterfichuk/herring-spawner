#!/usr/bin/env python3
"""Augmentation-consensus scoring for DINOv2 spawn detection.

This evaluates each labeled candidate with a single-crop baseline and with a
10-augmentation consensus score. The report compares separation for spawn vs
nospawn images before and after consensus penalization.

Default inputs:
  - labels: /Users/dexterfichuk/Downloads/herring-labels-v2.json
  - candidates: data/candidates_v2
  - output: data/review/dinov2_consensus_report.html
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_LABELS = Path("/Users/dexterfichuk/Downloads/herring-labels-v2.json")
DEFAULT_CANDIDATE_DIR = Path("data/candidates_v2")
DEFAULT_OUTPUT = Path("data/review/dinov2_consensus_report.html")
DEFAULT_AUGMENTATIONS = 10
MODEL_NAME = "dinov2_vits14"


def parse_candidate_id(item_id: str) -> dict[str, Any]:
    """Parse a candidate id like `cand:tofino:49.1:-125.9:2023-04-28`."""
    parts = item_id.split(":")
    if len(parts) != 5 or parts[0] != "cand":
        raise ValueError(f"Unsupported candidate id: {item_id}")
    _, site, lat_text, lon_text, date_text = parts
    return {
        "id": item_id,
        "site": site,
        "lat": float(lat_text),
        "lon": float(lon_text),
        "date": date_text,
        "date_compact": date_text.replace("-", ""),
    }


def _parse_candidate_filename(path: Path) -> dict[str, Any] | None:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 6:
        return None
    site = parts[0]
    date_text = parts[1]
    try:
        lat = float(parts[3])
        lon = float(parts[4])
    except ValueError:
        return None
    return {
        "site": site,
        "date": date_text,
        "date_compact": parts[5],
        "lat": lat,
        "lon": lon,
    }


def match_candidate_image(
    candidate_dir: Path,
    item_id: str,
    files: Sequence[Path] | None = None,
) -> Path | None:
    """Find the thumbnail matching a labeled candidate id."""
    if not item_id.startswith("cand:"):
        return None
    parsed = parse_candidate_id(item_id)
    search_files = list(files) if files is not None else sorted(candidate_dir.glob("*.png"))
    exact_matches: list[Path] = []

    for path in search_files:
        meta = _parse_candidate_filename(path)
        if meta is None:
            continue
        if meta["site"] != parsed["site"]:
            continue
        if meta["date"] != parsed["date"] and meta["date_compact"] != parsed["date_compact"]:
            continue
        if abs(meta["lat"] - parsed["lat"]) > 1e-6:
            continue
        if abs(meta["lon"] - parsed["lon"]) > 1e-6:
            continue
        exact_matches.append(path)

    if not exact_matches:
        return None
    exact_matches.sort(key=lambda p: p.name)
    return exact_matches[0]


def _candidate_lookup_key(parsed: dict[str, Any]) -> tuple[str, str, float, float]:
    return (
        parsed["site"],
        parsed["date_compact"],
        round(float(parsed["lat"]), 6),
        round(float(parsed["lon"]), 6),
    )


def _build_candidate_lookup(files: Sequence[Path]) -> dict[tuple[str, str, float, float], Path]:
    lookup: dict[tuple[str, str, float, float], Path] = {}
    for path in files:
        meta = _parse_candidate_filename(path)
        if meta is None:
            continue
        key = _candidate_lookup_key(meta)
        lookup.setdefault(key, path)
    return lookup


def _lookup_candidate_image(item_id: str, lookup: dict[tuple[str, str, float, float], Path]) -> Path | None:
    if not item_id.startswith("cand:"):
        return None
    parsed = parse_candidate_id(item_id)
    return lookup.get(_candidate_lookup_key(parsed))


def compute_consensus_metrics(scores: Sequence[float]) -> dict[str, float]:
    """Return mean, variance, and the consensus score (mean - variance)."""
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute consensus metrics from an empty score list")
    mean = float(arr.mean())
    variance = float(arr.var())
    return {
        "mean": mean,
        "variance": variance,
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "consensus": mean - variance,
    }


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) & 0x7FFFFFFF


def _load_labels(labels_path: Path) -> list[dict[str, Any]]:
    data = json.loads(labels_path.read_text())
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError("labels file is missing an items list")
    return items


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        return embedding
    return embedding / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize_embedding(a), _normalize_embedding(b)))


def _load_torch_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - import error path is runtime only
        raise RuntimeError(
            "DINOv2 consensus scoring requires torch, torchvision, and Pillow"
        ) from exc
    return torch, Image, transforms, F


def _pick_device(torch: Any) -> Any:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_transforms(transforms: Any) -> tuple[Any, Any]:
    baseline = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    augmentation = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.5, 0.9)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return baseline, augmentation


def _embed_image(model: Any, tensor: Any, torch: Any, device: Any) -> np.ndarray:
    return _embed_batch(model, [tensor], torch, device)[0]


def _embed_batch(model: Any, tensors: Sequence[Any], torch: Any, device: Any) -> np.ndarray:
    if not tensors:
        return np.empty((0, 0), dtype=float)
    batch = torch.stack(list(tensors), dim=0).to(device)
    with torch.no_grad():
        embs = model(batch)
    embs = embs.detach().cpu().numpy().astype(float)
    return np.asarray([_normalize_embedding(row) for row in embs])


def _compute_reference_embedding(
    model: Any,
    torch: Any,
    device: Any,
    image_cls: Any,
    baseline_transform: Any,
    items: Sequence[dict[str, Any]],
    candidate_dir: Path,
) -> tuple[np.ndarray, int]:
    positive_tensors: list[Any] = []
    for item in items:
        if not item.get("id", "").startswith("cand:"):
            continue
        if item.get("label") != "spawn":
            continue
        path = match_candidate_image(candidate_dir, item["id"])
        if path is None:
            continue
        with image_cls.open(path) as image_file:
            image = image_file.convert("RGB")
        positive_tensors.append(baseline_transform(image))

    if not positive_tensors:
        raise RuntimeError("No labeled spawn thumbnails were matched to candidate images")

    positive_embeddings = _embed_batch(model, positive_tensors, torch, device)
    mean_pos = np.mean(positive_embeddings, axis=0)
    return _normalize_embedding(mean_pos), len(positive_embeddings)


def _score_item(
    model: Any,
    torch: Any,
    device: Any,
    image_cls: Any,
    baseline_transform: Any,
    augmentation_transform: Any,
    item: dict[str, Any],
    image_path: Path,
    reference_embedding: np.ndarray,
    augmentations: int,
    seed: int,
) -> dict[str, Any]:
    with image_cls.open(image_path) as image_file:
        image = image_file.convert("RGB")

    baseline_embedding = _embed_image(model, baseline_transform(image), torch, device)
    baseline_score = _cosine(baseline_embedding, reference_embedding)

    aug_tensors: list[Any] = []
    for idx in range(augmentations):
        aug_seed = _stable_seed(item["id"], str(seed), str(idx))
        random.seed(aug_seed)
        np.random.seed(aug_seed % (2**32 - 1))
        torch.manual_seed(aug_seed)
        aug_tensors.append(augmentation_transform(image))

    aug_embeddings = _embed_batch(model, aug_tensors, torch, device)
    scores = [_cosine(embedding, reference_embedding) for embedding in aug_embeddings]

    metrics = compute_consensus_metrics(scores)
    return {
        "id": item["id"],
        "label": item.get("label", "unknown"),
        "image": image_path.name,
        "baseline": baseline_score,
        "mean": metrics["mean"],
        "variance": metrics["variance"],
        "consensus": metrics["consensus"],
        "std": metrics["std"],
        "min": metrics["min"],
        "max": metrics["max"],
        "scores": scores,
    }


def _class_stats(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, float]:
    spawn = [float(row[metric]) for row in rows if row["label"] == "spawn"]
    nospawn = [float(row[metric]) for row in rows if row["label"] == "nospawn"]
    if not spawn or not nospawn:
        raise ValueError(f"Need both spawn and nospawn rows to compute {metric} separation")
    return {
        "spawn": float(np.mean(spawn)),
        "nospawn": float(np.mean(nospawn)),
        "separation": float(np.mean(spawn) - np.mean(nospawn)),
        "accuracy": float(
            np.mean([(row[metric] >= 0) == (row["label"] == "spawn") for row in rows])
        ),
    }


def build_report_html(summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    """Render the comparison report as a standalone HTML document."""
    baseline = summary["baseline"]
    consensus = summary["consensus"]
    label_count = summary.get("label_count", len(rows))
    positive_count = summary.get("positive_reference_count", "n/a")
    baseline_accuracy = baseline.get("accuracy", 0.0)
    consensus_accuracy = consensus.get("accuracy", 0.0)

    def fmt(value: float) -> str:
        return f"{value:.4f}"

    row_html = []
    for row in rows:
        image_name = row.get("image") or row.get("id", "")
        row_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('label', '')))}</td>"
            f"<td>{html.escape(str(image_name))}</td>"
            f"<td>{fmt(row['baseline'])}</td>"
            f"<td>{fmt(row['mean'])}</td>"
            f"<td>{fmt(row['variance'])}</td>"
            f"<td>{fmt(row['consensus'])}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DINOv2 Augmentation Consensus Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f5f6fa; color: #1f2937; }}
    header {{ background: linear-gradient(135deg, #1f2937, #0f172a); color: white; padding: 24px; }}
    main {{ padding: 24px; max-width: 1200px; margin: 0 auto; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .kicker {{ text-transform: uppercase; font-size: 11px; color: #6b7280; letter-spacing: .08em; }}
    .value {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
    .muted {{ color: #6b7280; }}
    table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
    th {{ background: #f9fafb; position: sticky; top: 0; }}
    .good {{ color: #15803d; }}
    .bad {{ color: #b91c1c; }}
  </style>
</head>
<body>
  <header>
    <h1>Augmentation Consensus</h1>
    <p class="muted" style="color:#cbd5e1">DINOv2 spawn scoring with 10 random crops/augmentations per image.</p>
  </header>
  <main>
    <p class="muted">before/after comparison for {label_count} labeled candidates using {positive_count} spawn references.</p>
    <div class="cards">
      <div class="card"><div class="kicker">Baseline separation</div><div class="value">{fmt(baseline['separation'])}</div><div class="muted">spawn {fmt(baseline['spawn'])} / nospawn {fmt(baseline['nospawn'])}</div></div>
      <div class="card"><div class="kicker">Consensus separation</div><div class="value">{fmt(consensus['separation'])}</div><div class="muted">spawn {fmt(consensus['spawn'])} / nospawn {fmt(consensus['nospawn'])}</div></div>
      <div class="card"><div class="kicker">Separation delta</div><div class="value { 'good' if consensus['separation'] >= baseline['separation'] else 'bad' }">{fmt(consensus['separation'] - baseline['separation'])}</div><div class="muted">consensus - baseline</div></div>
      <div class="card"><div class="kicker">Baseline accuracy</div><div class="value">{baseline_accuracy:.1%}</div><div class="muted">threshold at zero</div></div>
      <div class="card"><div class="kicker">Consensus accuracy</div><div class="value">{consensus_accuracy:.1%}</div><div class="muted">threshold at zero</div></div>
    </div>
    <table>
      <thead>
        <tr><th>Label</th><th>Image</th><th>Baseline</th><th>Mean</th><th>Variance</th><th>Consensus</th></tr>
      </thead>
      <tbody>
        {''.join(row_html)}
      </tbody>
    </table>
  </main>
</body>
</html>"""


def run(labels_path: Path, candidate_dir: Path, output_path: Path, augmentations: int, seed: int) -> dict[str, Any]:
    torch, image_cls, transforms, _ = _load_torch_stack()
    baseline_transform, augmentation_transform = _build_transforms(transforms)
    candidate_files = sorted(candidate_dir.glob("*.png"))
    candidate_lookup = _build_candidate_lookup(candidate_files)
    device = _pick_device(torch)

    try:
        model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError(f"Failed to load DINOv2 model {MODEL_NAME}: {exc}") from exc

    model.eval().to(device)

    items = _load_labels(labels_path)
    reference_embedding, positive_count = _compute_reference_embedding(
        model, torch, device, image_cls, baseline_transform, items, candidate_dir
    )

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for item in items:
        if not item.get("id", "").startswith("cand:"):
            skipped.append(item.get("id", "<missing-id>"))
            continue
        if item.get("label") not in {"spawn", "nospawn"}:
            continue
        path = _lookup_candidate_image(item["id"], candidate_lookup)
        if path is None:
            skipped.append(item["id"])
            continue
        rows.append(
            _score_item(
                model,
                torch,
                device,
                image_cls,
                baseline_transform,
                augmentation_transform,
                item,
                path,
                reference_embedding,
                augmentations,
                seed,
            )
        )

    if not rows:
        raise RuntimeError("No labeled candidates could be scored")

    baseline_stats = _class_stats(rows, "baseline")
    consensus_stats = _class_stats(rows, "consensus")

    summary: dict[str, Any] = {
        "baseline": baseline_stats,
        "consensus": consensus_stats,
        "label_count": len(rows),
        "positive_reference_count": positive_count,
        "skipped_ids": skipped,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report_html(summary, rows))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--augmentations", type=int, default=DEFAULT_AUGMENTATIONS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    summary = run(args.labels, args.candidate_dir, args.output, args.augmentations, args.seed)
    print(
        "Baseline separation: {0:.4f} | Consensus separation: {1:.4f} | Delta: {2:.4f}".format(
            summary["baseline"]["separation"],
            summary["consensus"]["separation"],
            summary["consensus"]["separation"] - summary["baseline"]["separation"],
        )
    )
    print(f"Report written to: {args.output.resolve()}")
    if summary["skipped_ids"]:
        print(f"Skipped {len(summary['skipped_ids'])} unmatched labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
