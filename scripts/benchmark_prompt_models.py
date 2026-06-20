#!/usr/bin/env python3
"""Benchmark prompt-based herring spawn models on the canonical golden set.

Runs full-image and multi-crop contrastive scoring against the 15-positive,
164-negative canonical training manifest, then writes JSON/CSV/HTML outputs
under data/prompt_benchmarks/.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover - optional dependency
    average_precision_score = None
    roc_auc_score = None

from scripts.prompt_detector import (
    DEFAULT_PROMPT_BANK,
    IMAGE_SUFFIXES,
    _prompt_bank,
    load_backend,
    prepare_prompt_embeddings,
    score_image,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "data" / "samples" / "training_manifest.json"
POSITIVE_DIR = ROOT_DIR / "data" / "samples" / "positive"
NEGATIVE_DIR = ROOT_DIR / "data" / "samples" / "negative"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "prompt_benchmarks"


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _golden_items(limit: int | None = None) -> list[dict]:
    manifest = _load_manifest()
    items: list[dict] = []

    for filename in manifest.get("positives", []):
        path = POSITIVE_DIR / filename
        if path.suffix.lower() in IMAGE_SUFFIXES and path.exists():
            items.append({"image_path": str(path), "label": 1})

    negative_paths = [
        p
        for p in sorted(NEGATIVE_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    for path in negative_paths:
        items.append({"image_path": str(path), "label": 0})

    if limit is not None:
        items = items[: max(0, int(limit))]
    return items


def _safe_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.0) -> dict:
    if labels.size == 0:
        return {
            "threshold": threshold,
            "n": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auroc": None,
            "ap": None,
        }

    preds = (scores > threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    accuracy = float((preds == labels).mean())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0

    auroc = None
    ap = None
    if roc_auc_score is not None and len(np.unique(labels)) > 1:
        auroc = float(roc_auc_score(labels, scores))
    if average_precision_score is not None and len(np.unique(labels)) > 1:
        ap = float(average_precision_score(labels, scores))

    return {
        "threshold": threshold,
        "n": int(labels.size),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "auroc": auroc,
        "ap": ap,
    }


def _threshold_candidates(scores: np.ndarray) -> list[float]:
    unique_scores = np.unique(np.asarray(scores, dtype=float))
    if unique_scores.size == 0:
        return [0.0]
    if unique_scores.size == 1:
        value = float(unique_scores[0])
        return [value - 1e-9, value + 1e-9]
    thresholds: list[float] = [float(unique_scores[0] - 1e-9)]
    thresholds.extend(
        float((left + right) / 2.0)
        for left, right in zip(unique_scores[:-1], unique_scores[1:], strict=False)
    )
    thresholds.append(float(unique_scores[-1] + 1e-9))
    return thresholds


def _pick_best_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: list[float],
    *,
    metric_name: str,
) -> dict:
    best: dict | None = None
    for threshold in thresholds:
        metrics = _safe_metrics(labels, scores, threshold=threshold)
        if metric_name == "balanced_accuracy":
            tp = metrics["tp"]
            tn = metrics["tn"]
            fp = metrics["fp"]
            fn = metrics["fn"]
            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            primary = (sensitivity + specificity) / 2.0
        else:
            primary = metrics[metric_name]

        candidate = {**metrics, "balanced_accuracy": None}
        tp = metrics["tp"]
        tn = metrics["tn"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        candidate["balanced_accuracy"] = float((sensitivity + specificity) / 2.0)

        if best is None:
            best = candidate
            best["_primary"] = float(primary)
            continue

        best_primary = best["_primary"]
        better = primary > best_primary
        tie = np.isclose(primary, best_primary)
        if better or (tie and threshold < best["threshold"]):
            best = candidate
            best["_primary"] = float(primary)

    assert best is not None
    best.pop("_primary", None)
    return best


def sweep_thresholds(rows: list[dict]) -> dict:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
    if labels.size == 0:
        empty = _safe_metrics(labels, scores, threshold=0.0)
        return {
            "baseline": empty,
            "best_accuracy": empty,
            "best_f1": empty,
            "best_balanced_accuracy": empty,
            "threshold_count": 0,
            "thresholds": [],
        }
    thresholds = _threshold_candidates(scores)
    baseline = _safe_metrics(labels, scores, threshold=0.0)
    best_accuracy = _pick_best_threshold(labels, scores, thresholds, metric_name="accuracy")
    best_f1 = _pick_best_threshold(labels, scores, thresholds, metric_name="f1")
    best_balanced = _pick_best_threshold(
        labels, scores, thresholds, metric_name="balanced_accuracy"
    )
    return {
        "baseline": baseline,
        "best_accuracy": best_accuracy,
        "best_f1": best_f1,
        "best_balanced_accuracy": best_balanced,
        "threshold_count": len(thresholds),
        "thresholds": thresholds,
    }


def build_benchmark_summary(
    rows: list[dict],
    *,
    model_name: str | None = None,
    mode_name: str | None = None,
) -> dict:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
    ranked_rows = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    return {
        "model": model_name,
        "mode": mode_name,
        "row_count": len(rows),
        "metrics": _safe_metrics(labels, scores),
        "calibration": sweep_thresholds(rows),
        "rows": rows,
        "ranked_rows": ranked_rows,
    }


def _write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "mode",
        "label",
        "score",
        "prediction",
        "image_path",
        "backend",
        "prompt_bank",
        "crop_aggregation",
        "n_crops",
        "full_image_score",
        "best_crop_score",
        "raw_spawn_score",
        "spawn_mean",
        "max_confounder_group",
        "max_confounder_mean",
        "max_confounder_score",
        "margin",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _build_review_html(summary: dict, rows: list[dict]) -> str:
    summary_rows = []
    for bench in summary["benchmarks"]:
        metrics = bench["metrics"]
        calibration = bench.get("calibration", {})
        best_accuracy = calibration.get("best_accuracy", {})
        best_f1 = calibration.get("best_f1", {})
        best_balanced = calibration.get("best_balanced_accuracy", {})
        summary_rows.append(
            "<tr>"
            f"<td>{escape(str(bench['model']))}</td>"
            f"<td>{escape(str(bench['mode']))}</td>"
            f"<td>{bench['row_count']}</td>"
            f"<td>{metrics['accuracy']}</td>"
            f"<td>{metrics['auroc']}</td>"
            f"<td>{metrics['ap']}</td>"
            f"<td>{best_accuracy.get('threshold')}</td>"
            f"<td>{best_f1.get('threshold')}</td>"
            f"<td>{best_balanced.get('threshold')}</td>"
            "</tr>"
        )

    row_rows = []
    for row in rows:
        row_rows.append(
            "<tr>"
            f"<td>{escape(str(row['model']))}</td>"
            f"<td>{escape(str(row['mode']))}</td>"
            f"<td>{row['label']}</td>"
            f"<td>{row['score']:.4f}</td>"
            f"<td>{row['prediction']}</td>"
            f"<td>{escape(str(Path(row['image_path']).name))}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Prompt Benchmark Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f6f6f6; }}
  </style>
</head>
<body>
  <h1>Prompt Benchmark Review</h1>
  <p>Generated: {escape(summary['created_at'])}</p>
  <h2>Configuration Summary</h2>
  <table>
    <thead><tr><th>Model</th><th>Mode</th><th>Rows</th><th>Accuracy</th><th>AUROC</th>
      <th>AP</th><th>Best Acc T</th><th>Best F1 T</th><th>Best Bal T</th></tr></thead>
    <tbody>
      {''.join(summary_rows)}
    </tbody>
  </table>
  <h2>All Scored Rows</h2>
  <table>
    <thead><tr><th>Model</th><th>Mode</th><th>Label</th><th>Score</th><th>Prediction</th><th>File</th></tr></thead>
    <tbody>
      {''.join(row_rows)}
    </tbody>
  </table>
</body>
</html>
"""


def _run_configuration(
    backend_name: str,
    items: list[dict],
    *,
    prompt_bank_name: str,
    device: str,
    mode: str,
    crop_aggregation: str,
    top_k: int,
    crop_size: int,
    stride: int,
    max_crops: int,
) -> dict:
    backend = load_backend(backend_name, device=device)
    prompt_bank = _prompt_bank(prompt_bank_name)
    prompt_embeddings = prepare_prompt_embeddings(backend, prompt_bank)

    rows: list[dict] = []
    errors: list[str] = []
    for item in items:
        row = score_image(
            item["image_path"],
            backend,
            prompt_bank,
            mode=mode,
            crop_aggregation=crop_aggregation,
            top_k=top_k,
            crop_size=crop_size,
            stride=stride,
            max_crops=max_crops,
            prompt_embeddings=prompt_embeddings,
        )
        if row is None:
            errors.append(item["image_path"])
            continue
        row.update(
            {
                "model": backend_name,
                "mode": mode,
                "label": int(item["label"]),
            }
        )
        rows.append(row)

    del prompt_embeddings
    del backend
    gc.collect()

    summary = build_benchmark_summary(rows, model_name=backend_name, mode_name=mode)
    summary["backend_status"] = "ok"
    summary["errors"] = errors
    summary["error_count"] = len(errors)
    return summary


def _resolve_backends(requested: list[str]) -> list[str]:
    expanded: list[str] = []
    for backend in requested:
        if backend == "auto":
            expanded.extend(["senclip", "remoteclip"])
        else:
            expanded.append(backend)
    deduped: list[str] = []
    for backend in expanded:
        if backend not in deduped:
            deduped.append(backend)
    return deduped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt-bank", default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--backends", nargs="+", default=["auto"])
    parser.add_argument("--modes", nargs="+", default=["full", "multicrop"])
    parser.add_argument(
        "--crop-aggregation",
        default="topk_mean",
        choices=["full", "max", "mean", "topk_mean"],
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--max-crops", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = _golden_items(limit=args.limit)
    resolved_backends = _resolve_backends(args.backends)

    benchmarks: list[dict] = []
    all_rows: list[dict] = []
    skipped: list[dict] = []

    for backend_name in resolved_backends:
        try:
            for mode in args.modes:
                summary = _run_configuration(
                    backend_name,
                    items,
                    prompt_bank_name=args.prompt_bank,
                    device=args.device,
                    mode=mode,
                    crop_aggregation=args.crop_aggregation,
                    top_k=args.top_k,
                    crop_size=args.crop_size,
                    stride=args.stride,
                    max_crops=args.max_crops,
                )
                benchmarks.append(summary)
                all_rows.extend(summary["rows"])
        except Exception as exc:
            skipped.append({"backend": backend_name, "error": str(exc)})

    all_rows = sorted(all_rows, key=lambda row: float(row["score"]), reverse=True)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "manifest_path": str(MANIFEST_PATH),
        "positive_count": len([item for item in items if item["label"] == 1]),
        "negative_count": len([item for item in items if item["label"] == 0]),
        "row_count": len(all_rows),
        "benchmarks": benchmarks,
        "skipped": skipped,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(all_rows, output_dir / "scores.csv")
    (output_dir / "review.html").write_text(_build_review_html(summary, all_rows), encoding="utf-8")

    print(f"Wrote benchmark outputs to {output_dir}")
    print(f"Rows: {len(all_rows)}")
    if skipped:
        print(f"Skipped {len(skipped)} backend configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
