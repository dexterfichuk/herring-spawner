#!/usr/bin/env python3
"""Prompt-based herring spawn scoring with contrastive prompts and crops.

This module is additive to the existing RemoteCLIP pipeline. It keeps memory
usage conservative by loading one backend at a time, scoring one image at a
time, and evaluating crops sequentially.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:  # Optional, but expected in normal model-running environments.
    import torch
except Exception:  # pragma: no cover - import guard for lightweight installs
    torch = None

ROOT_DIR = Path(__file__).resolve().parent.parent
SENCLIP_REPO_ID = "pallavijainpj/SenCLIP"
DEFAULT_PROMPT_BANK = "default"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class PromptBank:
    name: str
    spawn: tuple[str, ...]
    confounders: dict[str, tuple[str, ...]]


@dataclass(slots=True)
class PromptBackend:
    name: str
    device: str
    encode_texts: Callable[[Sequence[str]], np.ndarray]
    encode_image: Callable[[Image.Image], np.ndarray]


@dataclass(slots=True)
class PromptEmbeddings:
    spawn: np.ndarray
    confounders: dict[str, np.ndarray]


PROMPT_BANKS: dict[str, PromptBank] = {
    DEFAULT_PROMPT_BANK: PromptBank(
        name=DEFAULT_PROMPT_BANK,
        spawn=(
            "a satellite image of milky white herring spawn in shallow coastal water",
            "an aerial view of chalky turquoise milt clouds near a rocky shoreline",
            "opaque white water plumes from herring spawn in a sheltered bay",
            "vibrant cyan white swirls of herring milt along the coast",
            "bright turquoise spawn plume in shallow sheltered coastal water",
            "milky turquoise coastal water from herring spawning",
        ),
        confounders={
            "foam_waves": (
                "sea foam and breaking waves on a rocky shore",
                "white surf lines from wave action along the coastline",
                "wind-driven foam streaks over open water near rocks",
            ),
            "glint": (
                "sun glint and specular reflection on the sea surface",
                "bright reflected sunlight sparkling on ocean water",
                "sunglint on calm coastal water with a mirror-like sheen",
            ),
            "sediment": (
                "sediment plume showing brown and grey turbid water from river outflow",
                "river plume of muddy water spreading into the ocean",
                "tan sediment mixing into blue coastal water near an estuary",
            ),
            "clouds": (
                "clouds and haze obscuring the ocean in a satellite image",
                "bright overexposed cloud cover over coastal water",
                "thin haze and white cloud patches over the sea surface",
            ),
            "surf_beach": (
                "surf breaking over rocks on a bright beach",
                "white breaking waves on a sandy shoreline",
                "bright exposed beach sand and surf near the coast",
            ),
        },
    )
}


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
        denom = float(np.linalg.norm(arr))
        return arr if denom == 0.0 else arr / denom
    denom = np.linalg.norm(arr, axis=-1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return arr / denom


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


def _move_batch_to_device(batch, device: str):
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [_move_batch_to_device(v, device) for v in batch]
        return type(batch)(moved)
    return batch


def _prompt_bank(name: str) -> PromptBank:
    try:
        return PROMPT_BANKS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_BANKS))
        raise KeyError(f"Unknown prompt bank '{name}'. Available: {available}") from exc


def _stack_text_embeddings(backend: PromptBackend, texts: Sequence[str]) -> np.ndarray:
    embeddings = backend.encode_texts(list(texts))
    return _normalize_numpy(embeddings)


def prepare_prompt_embeddings(backend: PromptBackend, prompt_bank: PromptBank) -> PromptEmbeddings:
    spawn = _stack_text_embeddings(backend, prompt_bank.spawn)
    confounders: dict[str, np.ndarray] = {}
    for group_name, prompts in prompt_bank.confounders.items():
        confounders[group_name] = _stack_text_embeddings(backend, prompts)
    return PromptEmbeddings(spawn=spawn, confounders=confounders)


def contrastive_score(
    spawn_scores: Sequence[float],
    confounder_scores_by_group: dict[str, Sequence[float]],
) -> float:
    spawn_scores = list(spawn_scores)
    if not spawn_scores:
        raise ValueError("spawn_scores must not be empty")
    spawn_mean = float(np.mean(spawn_scores))
    if not confounder_scores_by_group:
        return spawn_mean
    confounder_means: list[float] = []
    for scores in confounder_scores_by_group.values():
        score_list = list(scores)
        if score_list:
            confounder_means.append(float(np.mean(score_list)))
    if not confounder_means:
        return spawn_mean
    return float(spawn_mean - max(confounder_means))


def aggregate_crop_scores(
    scores: Sequence[float],
    mode: str = "topk_mean",
    top_k: int = 2,
) -> float:
    values = [float(v) for v in scores]
    if not values:
        raise ValueError("scores must not be empty")
    if mode == "full":
        return float(values[0])
    if mode == "max":
        return float(max(values))
    if mode == "mean":
        return float(np.mean(values))
    if mode == "topk_mean":
        k = max(1, min(int(top_k), len(values)))
        topk = sorted(values, reverse=True)[:k]
        return float(np.mean(topk))
    raise ValueError(f"Unknown crop aggregation mode '{mode}'")


def generate_sliding_crops(
    image_size: tuple[int, int],
    *,
    crop_size: int = 224,
    stride: int = 112,
    max_crops: int = 8,
) -> list[tuple[int, int, int, int]]:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    if crop_size <= 0 or stride <= 0:
        raise ValueError("crop_size and stride must be positive")
    if max_crops <= 0:
        raise ValueError("max_crops must be positive")

    full_box = (0, 0, width, height)
    if width <= crop_size and height <= crop_size:
        return [full_box][:max_crops]

    boxes: list[tuple[int, int, int, int]] = [full_box]

    def _positions(length: int) -> list[int]:
        if length <= crop_size:
            return [0]
        positions = list(range(0, max(1, length - crop_size + 1), stride))
        last = max(0, length - crop_size)
        if positions[-1] != last:
            positions.append(last)
        return sorted(set(positions))

    x_positions = _positions(width)
    y_positions = _positions(height)
    for top in y_positions:
        for left in x_positions:
            if len(boxes) >= max_crops:
                return boxes[:max_crops]
            right = min(left + crop_size, width)
            bottom = min(top + crop_size, height)
            box = (left, top, right, bottom)
            if box not in boxes:
                boxes.append(box)

    return boxes[:max_crops]


def score_prompt_groups(
    prompt_embeddings: PromptEmbeddings,
    image_embedding: np.ndarray,
) -> dict:
    image_embedding = _normalize_numpy(image_embedding)
    spawn_scores = (prompt_embeddings.spawn @ image_embedding).astype(np.float32)
    confounder_scores: dict[str, np.ndarray] = {}
    for group_name, embeddings in prompt_embeddings.confounders.items():
        confounder_scores[group_name] = (embeddings @ image_embedding).astype(np.float32)
    confounder_group_means = {
        k: float(np.mean(v)) for k, v in confounder_scores.items() if len(v)
    }
    max_confounder_group = None
    max_confounder_mean = None
    if confounder_group_means:
        max_confounder_group = max(confounder_group_means, key=confounder_group_means.get)
        max_confounder_mean = float(confounder_group_means[max_confounder_group])
    margin = contrastive_score(
        spawn_scores.tolist(),
        {k: v.tolist() for k, v in confounder_scores.items()},
    )
    return {
        "spawn_scores": spawn_scores.tolist(),
        "confounder_scores": {k: v.tolist() for k, v in confounder_scores.items()},
        "spawn_mean": float(np.mean(spawn_scores)),
        "raw_spawn_score": float(np.mean(spawn_scores)),
        "confounder_group_means": confounder_group_means,
        "max_confounder_group": max_confounder_group,
        "max_confounder_mean": max_confounder_mean,
        "max_confounder_score": max_confounder_mean,
        "margin": margin,
        "score": margin,
    }


def _load_remoteclip_backend(device: str) -> PromptBackend:
    _require_torch("RemoteCLIP loading")
    try:
        from scripts.remoteclip_zero_shot import load_model as load_remoteclip_model
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "RemoteCLIP backend is unavailable. Install open-clip-torch, "
            "huggingface-hub, torch, and torchvision."
        ) from exc

    model, preprocess, tokenize = load_remoteclip_model(device=device)

    def encode_texts(texts: Sequence[str]) -> np.ndarray:
        from scripts.remoteclip_zero_shot import get_text_embedding

        emb = get_text_embedding(model, tokenize, list(texts), device)
        return _to_numpy(emb)

    def encode_image(image: Image.Image) -> np.ndarray:
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensor)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return _to_numpy(emb).reshape(-1)

    return PromptBackend(
        name="remoteclip",
        device=device,
        encode_texts=encode_texts,
        encode_image=encode_image,
    )


def _load_senclip_backend(device: str) -> PromptBackend:
    _require_torch("SenCLIP loading")
    try:
        from transformers import AutoModel, AutoProcessor
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "SenCLIP backend requires transformers and huggingface_hub. "
            "Install them, then retry with --backend senclip."
        ) from exc

    try:
        processor = AutoProcessor.from_pretrained(SENCLIP_REPO_ID)
        model = AutoModel.from_pretrained(SENCLIP_REPO_ID)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load SenCLIP checkpoint '{SENCLIP_REPO_ID}'. "
            "Check network access, HF auth, and install transformers/"
            "huggingface_hub/torch."
        ) from exc

    model.eval().to(device)

    def _encode_batch(inputs) -> np.ndarray:
        inputs = _move_batch_to_device(inputs, device)
        with torch.no_grad():
            if hasattr(model, "get_text_features") and "input_ids" in inputs:
                emb = model.get_text_features(**inputs)
            elif hasattr(model, "get_image_features") and "pixel_values" in inputs:
                emb = model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                if hasattr(outputs, "text_embeds"):
                    emb = outputs.text_embeds
                elif hasattr(outputs, "image_embeds"):
                    emb = outputs.image_embeds
                else:
                    emb = outputs[0]
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return _to_numpy(emb)

    def encode_texts(texts: Sequence[str]) -> np.ndarray:
        inputs = processor(text=list(texts), padding=True, truncation=True, return_tensors="pt")
        return _encode_batch(inputs)

    def encode_image(image: Image.Image) -> np.ndarray:
        inputs = processor(images=image, return_tensors="pt")
        return _encode_batch(inputs).reshape(-1)

    return PromptBackend(
        name="senclip",
        device=device,
        encode_texts=encode_texts,
        encode_image=encode_image,
    )


def load_backend(name: str, device: str = "cpu") -> PromptBackend:
    resolved = _resolve_device(device)
    if name == "remoteclip":
        return _load_remoteclip_backend(resolved)
    if name == "senclip":
        return _load_senclip_backend(resolved)
    raise ValueError("backend must be 'remoteclip' or 'senclip'")


def _score_crop(
    crop: Image.Image,
    backend: PromptBackend,
    prompt_embeddings: PromptEmbeddings,
) -> dict:
    image_embedding = backend.encode_image(crop)
    return score_prompt_groups(prompt_embeddings, image_embedding)


def score_image(
    image_path: str | Path,
    backend: PromptBackend,
    prompt_bank: PromptBank,
    *,
    mode: str = "multicrop",
    crop_aggregation: str = "topk_mean",
    top_k: int = 2,
    crop_size: int = 224,
    stride: int = 112,
    max_crops: int = 8,
    prompt_embeddings: PromptEmbeddings | None = None,
) -> dict | None:
    path = Path(image_path)
    try:
        with Image.open(path) as src:
            image = src.convert("RGB")
    except Exception:
        return None

    try:
        if prompt_embeddings is None:
            prompt_embeddings = prepare_prompt_embeddings(backend, prompt_bank)

        if mode == "full":
            crop_boxes = [(0, 0, image.width, image.height)]
        elif mode == "multicrop":
            crop_boxes = generate_sliding_crops(
                (image.width, image.height),
                crop_size=crop_size,
                stride=stride,
                max_crops=max_crops,
            )
        else:
            raise ValueError("mode must be 'full' or 'multicrop'")

        per_crop: list[dict] = []
        crop_scores: list[float] = []
        for box in crop_boxes:
            crop = image.crop(box)
            result = _score_crop(crop, backend, prompt_embeddings)
            crop_score = float(result["score"])
            crop_scores.append(crop_score)
            per_crop.append({"box": box, **result})
            del crop, result

        full_image_score = float(crop_scores[0])
        aggregated_score = aggregate_crop_scores(crop_scores, mode=crop_aggregation, top_k=top_k)
        best_idx = int(np.argmax(crop_scores))
        best_crop = per_crop[best_idx]

        return {
            "image_path": str(path),
            "backend": backend.name,
            "prompt_bank": prompt_bank.name,
            "mode": mode,
            "crop_aggregation": crop_aggregation,
            "top_k": int(top_k),
            "score": float(aggregated_score),
            "full_image_score": full_image_score,
            "best_crop_score": float(crop_scores[best_idx]),
            "best_crop_box": list(best_crop["box"]),
            "n_crops": len(crop_boxes),
            "crop_scores": crop_scores,
            "crop_boxes": [list(box) for box in crop_boxes],
            "raw_spawn_score": float(best_crop["spawn_mean"]),
            "spawn_mean": float(best_crop["spawn_mean"]),
            "confounder_group_means": best_crop["confounder_group_means"],
            "max_confounder_group": best_crop.get("max_confounder_group"),
            "max_confounder_mean": best_crop.get("max_confounder_mean"),
            "max_confounder_score": best_crop.get("max_confounder_score"),
            "margin": float(best_crop["margin"]),
            "prediction": int(aggregated_score > 0.0),
        }
    except Exception:
        return None
    finally:
        try:
            image.close()
        except Exception:
            pass
        gc.collect()


def _iter_image_paths(image_dir: str | Path) -> list[Path]:
    paths = []
    for path in sorted(Path(image_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)
    return paths


def score_directory(
    image_dir: str | Path,
    backend: PromptBackend,
    prompt_bank: PromptBank,
    *,
    mode: str = "multicrop",
    crop_aggregation: str = "topk_mean",
    top_k: int = 2,
    crop_size: int = 224,
    stride: int = 112,
    max_crops: int = 8,
    limit: int | None = None,
) -> tuple[list[dict], list[str]]:
    prompt_embeddings = prepare_prompt_embeddings(backend, prompt_bank)
    paths = _iter_image_paths(image_dir)
    if limit is not None:
        paths = paths[: max(0, int(limit))]

    rows: list[dict] = []
    errors: list[str] = []
    for path in paths:
        row = score_image(
            path,
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
            errors.append(str(path))
            continue
        rows.append(row)
    return rows, errors


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-json", default="data/prompt_detector_results.json")
    parser.add_argument("--backend", default="senclip", choices=["senclip", "remoteclip"])
    parser.add_argument("--prompt-bank", default=DEFAULT_PROMPT_BANK)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default="multicrop", choices=["full", "multicrop"])
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

    backend = load_backend(args.backend, device=args.device)
    prompt_bank = _prompt_bank(args.prompt_bank)
    rows, errors = score_directory(
        args.image_dir,
        backend,
        prompt_bank,
        mode=args.mode,
        crop_aggregation=args.crop_aggregation,
        top_k=args.top_k,
        crop_size=args.crop_size,
        stride=args.stride,
        max_crops=args.max_crops,
        limit=args.limit,
    )
    payload = {
        "backend": backend.name,
        "device": backend.device,
        "prompt_bank": prompt_bank.name,
        "mode": args.mode,
        "crop_aggregation": args.crop_aggregation,
        "row_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "rows": rows,
    }
    _write_json(Path(args.output_json), payload)
    print(f"Wrote {len(rows)} rows to {args.output_json}")
    if errors:
        print(f"Skipped {len(errors)} unreadable images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
