#!/usr/bin/env python3
"""KNN voting classifier for herring spawn detection.

This evaluates labeled candidate thumbnails with leave-one-out KNN voting in
DINOv2 embedding space and compares it to a cosine-similarity baseline.

Default inputs:
  - labels: /Users/dexterfichuk/Downloads/herring-labels-v2.json
  - candidates: data/candidates_v2
  - output: data/review/knn_report.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision import transforms


DEFAULT_LABELS = Path("/Users/dexterfichuk/Downloads/herring-labels-v2.json")
DEFAULT_CANDIDATE_DIR = Path("data/candidates_v2")
DEFAULT_OUTPUT = Path("data/review/knn_report.html")
DEFAULT_KS = (3, 5, 7, 10, 15)
MODEL_NAME = "dinov2_vits14"

DINO_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def parse_candidate_id(item_id: str) -> dict[str, Any]:
    """Parse a labeled candidate id like `cand:tofino:49.1:-125.9:2023-04-28`."""
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
        return path

    return None


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        return embedding
    return embedding / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize_embedding(a), _normalize_embedding(b)))


def _load_labels(labels_path: Path) -> list[dict[str, Any]]:
    data = json.loads(labels_path.read_text())
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError("labels file is missing an items list")
    return items


def _pick_device() -> torch.device:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_image_embedding(model: torch.nn.Module, image_path: Path, device: torch.device) -> np.ndarray:
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
    tensor = DINO_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    return F.normalize(emb, dim=1).cpu().numpy().flatten().astype(float)


def _build_dataset(
    labels: Sequence[dict[str, Any]],
    candidate_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> list[dict[str, Any]]:
    candidate_files = sorted(candidate_dir.glob("*.png"))
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for item in labels:
        item_id = str(item.get("id", ""))
        label = str(item.get("label", "")).strip().lower()
        if not item_id.startswith("cand:") or label not in {"spawn", "nospawn"}:
            continue
        image_path = match_candidate_image(candidate_dir, item_id, candidate_files)
        if image_path is None:
            skipped.append(item_id)
            continue
        rows.append(
            {
                "id": item_id,
                "label": 1 if label == "spawn" else 0,
                "label_name": label,
                "image_path": image_path,
                "embedding": _load_image_embedding(model, image_path, device),
            }
        )

    if not rows:
        raise RuntimeError("No labeled candidate images could be matched")
    if skipped:
        print(f"  WARNING: skipped {len(skipped)} unmatched labels")

    return rows


def compute_baseline_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Score each sample against mean positive minus mean negative cosine similarity."""
    pos = embeddings[labels == 1]
    neg = embeddings[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Need both spawn and nospawn samples for baseline scoring")

    mean_pos = _normalize_embedding(np.mean(pos, axis=0))
    mean_neg = _normalize_embedding(np.mean(neg, axis=0))
    scores = np.asarray([_cosine(row, mean_pos) - _cosine(row, mean_neg) for row in embeddings], dtype=float)
    preds = (scores >= 0).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "scores": scores,
        "predictions": preds,
        "accuracy": float(accuracy_score(labels, preds)),
        "confusion_matrix": cm.tolist(),
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
    }


def evaluate_leave_one_out_knn(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[int, dict[str, Any]]:
    """Evaluate leave-one-out KNN voting for each K."""
    embeddings = np.asarray([_normalize_embedding(row) for row in embeddings], dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_samples = len(labels)
    if n_samples < 2:
        raise ValueError("Need at least two samples for leave-one-out KNN")

    pairwise = embeddings @ embeddings.T
    np.fill_diagonal(pairwise, -np.inf)

    results: dict[int, dict[str, Any]] = {}
    for k in ks:
        effective_k = min(int(k), n_samples - 1)
        if effective_k < 1:
            raise ValueError("K must be at least 1")

        preds: list[int] = []
        vote_details: list[dict[str, Any]] = []
        for idx in range(n_samples):
            neighbor_idx = np.argpartition(pairwise[idx], -effective_k)[-effective_k:]
            neighbor_idx = neighbor_idx[np.argsort(pairwise[idx][neighbor_idx])[::-1]]
            neighbor_labels = labels[neighbor_idx]
            spawn_votes = int(np.sum(neighbor_labels == 1))
            pred = int(spawn_votes > (effective_k / 2))
            preds.append(pred)
            vote_details.append(
                {
                    "index": idx,
                    "neighbor_indices": neighbor_idx.tolist(),
                    "spawn_votes": spawn_votes,
                    "neighbor_labels": neighbor_labels.tolist(),
                    "prediction": pred,
                }
            )

        pred_arr = np.asarray(preds, dtype=int)
        cm = confusion_matrix(labels, pred_arr, labels=[0, 1])
        results[int(k)] = {
            "effective_k": effective_k,
            "predictions": pred_arr,
            "accuracy": float(accuracy_score(labels, pred_arr)),
            "confusion_matrix": cm.tolist(),
            "votes": vote_details,
        }

    return results


def _format_cm(cm: Sequence[Sequence[int]]) -> str:
    return (
        "<table class='cm'><tr><th></th><th>Pred No</th><th>Pred Yes</th></tr>"
        f"<tr><th>Actual No</th><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr>"
        f"<tr><th>Actual Yes</th><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr></table>"
    )


def _as_label_int(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() == "spawn" else 0
    return int(value)


def build_report_html(summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    """Render the comparison report as a standalone HTML document."""
    baseline = summary["baseline"]
    knn = summary["knn"]
    k_results = summary.get("k_results", {})

    def fmt(value: float) -> str:
        return f"{value:.4f}"

    best_k = knn["best_k"]
    row_html = []
    for row in rows:
        actual = _as_label_int(row.get("actual", row.get("label", 0)))
        row_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('label_name', row.get('label', ''))))}</td>"
            f"<td>{html.escape(str(row.get('image_path', row.get('id', ''))))}</td>"
            f"<td>{fmt(float(row.get('baseline_score', 0.0)))}</td>"
            f"<td>{int(row.get('best_k_prediction', 0))}</td>"
            f"<td>{actual}</td>"
            "</tr>"
        )

    k_rows = []
    for k in sorted(k_results):
        result = k_results[k]
        effective_k = result.get("effective_k", k)
        k_rows.append(
            f"<tr><td>{k}</td><td>{effective_k}</td><td>{result['accuracy']:.1%}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KNN Voting Classifier Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f5f6fa; color: #1f2937; }}
    header {{ background: linear-gradient(135deg, #111827, #0f172a); color: white; padding: 24px; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .kicker {{ text-transform: uppercase; font-size: 11px; color: #6b7280; letter-spacing: .08em; }}
    .value {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
    .muted {{ color: #6b7280; }}
    table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-top: 12px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
    th {{ background: #f9fafb; position: sticky; top: 0; }}
    .cm td, .cm th {{ text-align: center; }}
  </style>
</head>
<body>
  <header>
    <h1>KNN Voting Classifier</h1>
    <p class="muted" style="color:#cbd5e1">Leave-one-out evaluation on DINOv2 embeddings.</p>
  </header>
  <main>
    <div class="cards">
      <div class="card"><div class="kicker">Baseline accuracy</div><div class="value">{baseline['accuracy']:.1%}</div><div class="muted">cos(pos mean) - cos(neg mean)</div></div>
      <div class="card"><div class="kicker">Best K</div><div class="value">{best_k}</div><div class="muted">leave-one-out majority vote</div></div>
      <div class="card"><div class="kicker">KNN accuracy</div><div class="value">{knn['accuracy']:.1%}</div><div class="muted">effective K capped at n-1</div></div>
      <div class="card"><div class="kicker">Accuracy gain</div><div class="value">{(knn['accuracy'] - baseline['accuracy']):+.1%}</div><div class="muted">best K vs baseline</div></div>
    </div>

    <h2>Confusion Matrix</h2>
    <div class="cards">
      <div class="card"><div class="kicker">Baseline</div>{_format_cm(baseline['confusion_matrix'])}</div>
      <div class="card"><div class="kicker">KNN (best K={best_k})</div>{_format_cm(knn['confusion_matrix'])}</div>
    </div>

    <h2>K Comparison</h2>
    <table>
      <thead><tr><th>K</th><th>Effective K</th><th>Accuracy</th></tr></thead>
      <tbody>{''.join(k_rows)}</tbody>
    </table>

    <h2>Per-sample summary</h2>
    <table>
      <thead><tr><th>Label</th><th>Image</th><th>Baseline Score</th><th>Best K Pred</th><th>Actual</th></tr></thead>
      <tbody>{''.join(row_html)}</tbody>
    </table>
  </main>
</body>
</html>"""


def run(labels_path: Path, candidate_dir: Path, output_path: Path, ks: Sequence[int]) -> dict[str, Any]:
    device = _pick_device()
    try:
        model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError(f"Failed to load DINOv2 model {MODEL_NAME}: {exc}") from exc

    model.eval().to(device)
    labels = _load_labels(labels_path)
    rows = _build_dataset(labels, candidate_dir, model, device)

    embeddings = np.asarray([row["embedding"] for row in rows], dtype=float)
    y = np.asarray([row["label"] for row in rows], dtype=int)

    baseline = compute_baseline_metrics(embeddings, y)
    k_results = evaluate_leave_one_out_knn(embeddings, y, ks=ks)
    best_k = max(sorted(k_results), key=lambda k: (k_results[k]["accuracy"], -int(k)))

    for idx, row in enumerate(rows):
        row["baseline_score"] = float(baseline["scores"][idx])
        row["actual"] = int(y[idx])
        row["best_k_prediction"] = int(k_results[best_k]["predictions"][idx])

    summary = {
        "baseline": {
            "accuracy": baseline["accuracy"],
            "confusion_matrix": baseline["confusion_matrix"],
        },
        "knn": {
            "best_k": best_k,
            "accuracy": k_results[best_k]["accuracy"],
            "confusion_matrix": k_results[best_k]["confusion_matrix"],
        },
        "k_results": {k: {"accuracy": v["accuracy"], "effective_k": v["effective_k"]} for k, v in k_results.items()},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report_html(summary, rows), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    args = parser.parse_args(argv)

    summary = run(args.labels, args.candidate_dir, args.output, args.ks)
    print(
        "Baseline accuracy: {0:.1%} | Best K: {1} | KNN accuracy: {2:.1%}".format(
            summary["baseline"]["accuracy"],
            summary["knn"]["best_k"],
            summary["knn"]["accuracy"],
        )
    )
    print(f"Report written to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
