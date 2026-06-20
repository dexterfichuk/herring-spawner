#!/usr/bin/env python3
"""Frozen-embedding linear probe for remote-sensing foundation backbones.

This is a low-RAM additive benchmark: one image at a time, frozen embeddings
only, then a small sklearn linear classifier over the cached feature matrix.
It defaults to DINOv2 as the guaranteed fallback and can optionally try
Prithvi/SatMAE/SkySense-style Hugging Face backbones when the dependencies and
model ids are available.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

try:  # Optional at import time for lightweight environments.
    import torch
except Exception:  # pragma: no cover - import guard
    torch = None

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scripts.benchmark_prompt_models import sweep_thresholds

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "data" / "samples" / "training_manifest.json"
POSITIVE_DIR = ROOT_DIR / "data" / "samples" / "positive"
NEGATIVE_DIR = ROOT_DIR / "data" / "samples" / "negative"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "rs_probe_benchmarks"
DEFAULT_DINOV2_MODEL = "dinov2_vits14"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(slots=True)
class RSBackbone:
    name: str
    device: str
    encode_image: Callable[[Image.Image], np.ndarray]
    model_id: str | None = None


def _require_torch(action: str) -> None:
    if torch is None:
        raise RuntimeError(f"{action} requires torch to be installed")


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


def _normalize_numpy(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr if norm == 0.0 else arr / norm
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def _pil_to_tensor(image: Image.Image, *, size: int = 224):
    _require_torch("DINOv2 preprocessing")
    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def _to_numpy(tensor) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32, copy=False)
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        return np.asarray(tensor.numpy(), dtype=np.float32)
    return np.asarray(tensor, dtype=np.float32)


def _load_dinov2_backbone(device: str) -> RSBackbone:
    _require_torch("DINOv2 loading")
    try:
        model = torch.hub.load("facebookresearch/dinov2", DEFAULT_DINOV2_MODEL)
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "DINOv2 fallback failed to load. Make sure torch can access the "
            "facebookresearch/dinov2 hub repo, or provide a cached checkpoint."
        ) from exc

    model.eval().to(device)

    def encode_image(image: Image.Image) -> np.ndarray:
        tensor = _pil_to_tensor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, "forward_features"):
                outputs = model.forward_features(tensor)
                if isinstance(outputs, dict):
                    emb = outputs.get("x_norm_clstoken")
                    if emb is None:
                        emb = next(iter(outputs.values()))
                else:
                    emb = outputs
            elif hasattr(model, "get_intermediate_layers"):
                patch_tokens, cls_tokens = model.get_intermediate_layers(
                    tensor, n=1, reshape=True, return_class_token=True
                )[0]
                emb = cls_tokens
            else:
                emb = model(tensor)
        return _normalize_numpy(_to_numpy(emb).reshape(-1))

    return RSBackbone(name="dinov2", device=device, encode_image=encode_image)


def _pool_transformers_output(outputs):
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state.mean(dim=1)
    if isinstance(outputs, tuple) or isinstance(outputs, list):
        for item in outputs:
            if hasattr(item, "shape"):
                if item.ndim == 2:
                    return item
                if item.ndim >= 3:
                    return item.mean(dim=1)
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


def _load_transformers_backbone(
    backend_name: str,
    device: str,
    model_id: str | None,
) -> RSBackbone:
    _require_torch(f"{backend_name} loading")
    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            f"{backend_name} requires transformers and huggingface_hub. "
            "Install those packages and retry, or use --backbone dinov2."
        ) from exc

    if not model_id:
        default_ids = {
            "prithvi": "ibm-nasa-geospatial/Prithvi-100M",
            "satmae": "facebook/satmae-base",
            "skysense": "google/siglip-base-patch16-224",
        }
        model_id = default_ids.get(backend_name)

    if not model_id:
        raise RuntimeError(
            f"No default model id configured for {backend_name}. Provide --model-id."
        )

    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            f"Failed to load {backend_name} checkpoint '{model_id}'. "
            "Check network access, HF auth/cache, and model availability."
        ) from exc

    model.eval().to(device)

    def encode_image(image: Image.Image) -> np.ndarray:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                emb = model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                emb = _pool_transformers_output(outputs)
        return _normalize_numpy(_to_numpy(emb).reshape(-1))

    return RSBackbone(
        name=backend_name,
        device=device,
        encode_image=encode_image,
        model_id=model_id,
    )


def load_backbone(
    name: str,
    *,
    device: str = "cpu",
    model_id: str | None = None,
) -> RSBackbone:
    resolved = _resolve_device(device)
    if name == "dinov2":
        return _load_dinov2_backbone(resolved)
    if name in {"prithvi", "satmae", "skysense"}:
        return _load_transformers_backbone(name, resolved, model_id)
    raise ValueError("backbone must be one of: dinov2, prithvi, satmae, skysense")


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _golden_items(limit: int | None = None) -> list[dict]:
    manifest = _load_manifest()
    items: list[dict] = []

    for filename in manifest.get("positives", []):
        path = POSITIVE_DIR / filename
        if path.exists() and path.suffix.lower() in IMAGE_SUFFIXES:
            items.append({"image_path": str(path), "label": 1})

    for path in sorted(NEGATIVE_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            items.append({"image_path": str(path), "label": 0})

    if limit is not None:
        items = items[: max(0, int(limit))]
    return items


def _read_image_embedding(path: str | Path, backbone: RSBackbone) -> np.ndarray | None:
    try:
        with Image.open(path) as src:
            image = src.convert("RGB")
    except Exception:
        return None

    try:
        emb = backbone.encode_image(image)
        return _normalize_numpy(emb)
    finally:
        try:
            image.close()
        except Exception:
            pass
        gc.collect()


def _resolve_cv(labels: np.ndarray, cv: str, random_state: int = 42):
    if cv == "loo":
        return LeaveOneOut()
    if cv != "auto":
        raise ValueError("cv must be 'auto' or 'loo'")

    class_counts = np.bincount(labels)
    class_counts = class_counts[class_counts > 0]
    min_class_count = int(class_counts.min()) if class_counts.size else 0
    if min_class_count < 2:
        return LeaveOneOut()
    return StratifiedKFold(
        n_splits=min(5, min_class_count),
        shuffle=True,
        random_state=random_state,
    )


def _build_classifier(name: str, random_state: int = 42):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=random_state,
            ),
        )
    if name == "linearsvc":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", random_state=random_state),
        )
    raise ValueError("classifier must be 'logreg' or 'linearsvc'")


def _predict_scores(
    X: np.ndarray,
    y: np.ndarray,
    *,
    classifier: str,
    cv: str,
    random_state: int = 42,
) -> np.ndarray:
    estimator = _build_classifier(classifier, random_state=random_state)
    splitter = _resolve_cv(y, cv, random_state=random_state)
    return cross_val_predict(estimator, X, y, cv=splitter, method="decision_function")


def _write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "classifier",
        "label",
        "score",
        "prediction",
        "image_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _build_review_html(summary: dict, rows: list[dict]) -> str:
    rows_html = []
    for row in rows:
        rows_html.append(
            "<tr>"
            f"<td>{escape(str(row['backend']))}</td>"
            f"<td>{escape(str(row['classifier']))}</td>"
            f"<td>{row['label']}</td>"
            f"<td>{row['score']:.4f}</td>"
            f"<td>{row['prediction']}</td>"
            f"<td>{escape(str(Path(row['image_path']).name))}</td>"
            "</tr>"
        )

    calibration = summary.get("calibration", {})
    best_accuracy = calibration.get("best_accuracy", {})
    best_f1 = calibration.get("best_f1", {})
    best_balanced = calibration.get("best_balanced_accuracy", {})

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RS Foundation Linear Probe Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f6f6f6; }}
  </style>
</head>
<body>
  <h1>RS Foundation Linear Probe Review</h1>
  <p>Generated: {escape(summary['created_at'])}</p>
  <p>Backbone: {escape(str(summary['backbone']))} {escape(str(summary.get('model_id') or ''))}</p>
  <p>Classifier: {escape(str(summary['classifier']))} &middot; CV: {escape(str(summary['cv']))}</p>
  <h2>Metrics</h2>
  <table>
    <thead><tr><th>Rows</th><th>Accuracy</th><th>AUROC</th><th>AP</th>
      <th>Best Acc T</th><th>Best F1 T</th><th>Best Bal T</th></tr></thead>
    <tbody>
      <tr>
        <td>{summary['row_count']}</td>
        <td>{summary['metrics']['accuracy']}</td>
        <td>{summary['metrics']['auroc']}</td>
        <td>{summary['metrics']['ap']}</td>
        <td>{best_accuracy.get('threshold')}</td>
        <td>{best_f1.get('threshold')}</td>
        <td>{best_balanced.get('threshold')}</td>
      </tr>
    </tbody>
  </table>
  <h2>Scored Rows</h2>
  <table>
    <thead><tr><th>Backbone</th><th>Classifier</th><th>Label</th><th>Score</th><th>Prediction</th><th>File</th></tr></thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""


def evaluate_linear_probe(
    items: list[dict],
    *,
    backbone: RSBackbone,
    classifier: str = "logreg",
    cv: str = "auto",
    random_state: int = 42,
) -> dict:
    embeddings: list[np.ndarray] = []
    rows: list[dict] = []

    for item in items:
        emb = _read_image_embedding(item["image_path"], backbone)
        if emb is None:
            continue
        embeddings.append(emb)
        rows.append({"image_path": item["image_path"], "label": int(item["label"])})

    if not embeddings:
        return {
            "backend": backbone.name,
            "backbone": backbone.name,
            "model_id": getattr(backbone, "model_id", None),
            "classifier": classifier,
            "cv": cv,
            "row_count": 0,
            "metrics": {"accuracy": None, "auroc": None, "ap": None},
            "calibration": {},
            "rows": [],
            "ranked_rows": [],
        }

    X = np.vstack([np.asarray(emb, dtype=np.float32).reshape(1, -1) for emb in embeddings])
    y = np.asarray([row["label"] for row in rows], dtype=int)
    scores = _predict_scores(X, y, classifier=classifier, cv=cv, random_state=random_state)
    predictions = (scores > 0.0).astype(int)

    scored_rows: list[dict] = []
    for item, score, prediction in zip(rows, scores, predictions, strict=False):
        scored_rows.append(
            {
                "backend": backbone.name,
                "model_id": getattr(backbone, "model_id", None),
                "classifier": classifier,
                "image_path": item["image_path"],
                "label": int(item["label"]),
                "score": float(score),
                "prediction": int(prediction),
            }
        )

    metrics = sweep_thresholds(scored_rows)
    baseline_metrics = metrics["baseline"]
    ranked_rows = sorted(scored_rows, key=lambda row: float(row["score"]), reverse=True)
    return {
        "backend": backbone.name,
        "backbone": backbone.name,
        "model_id": getattr(backbone, "model_id", None),
        "classifier": classifier,
        "cv": cv,
        "row_count": len(scored_rows),
        "metrics": baseline_metrics,
        "calibration": metrics,
        "rows": scored_rows,
        "ranked_rows": ranked_rows,
    }


def save_probe_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(summary.get("rows", []), output_dir / "scores.csv")
    (output_dir / "review.html").write_text(
        _build_review_html(summary, summary.get("rows", [])), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--backbone",
        default="dinov2",
        choices=["dinov2", "prithvi", "satmae", "skysense"],
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--classifier", default="logreg", choices=["logreg", "linearsvc"])
    parser.add_argument("--cv", default="auto", choices=["auto", "loo"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    items = _golden_items(limit=args.limit)
    backbone = load_backbone(args.backbone, device=args.device, model_id=args.model_id)
    summary = evaluate_linear_probe(
        items,
        backbone=backbone,
        classifier=args.classifier,
        cv=args.cv,
    )
    summary.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "manifest_path": str(MANIFEST_PATH),
            "positive_count": len([item for item in items if item["label"] == 1]),
            "negative_count": len([item for item in items if item["label"] == 0]),
        }
    )
    save_probe_outputs(summary, Path(args.output_dir))
    print(f"Wrote probe outputs to {args.output_dir}")
    print(f"Rows: {summary['row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
