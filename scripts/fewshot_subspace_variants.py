#!/usr/bin/env python3
"""Benchmark few-shot DINOv2 SubspaceAD variants for herring spawn detection.

This script keeps the evaluation additive and RAM-conservative:
- DINOv2 ViT-S/14 is the backbone.
- Images are embedded one at a time.
- Global-image variants use leave-one-positive-out scoring where practical.
- Patch/MIL variants are transductive and explicitly marked as such.

Outputs:
- data/fewshot_subspace_variants/summary.json
- data/fewshot_subspace_variants/scores.csv
- data/fewshot_subspace_variants/review.html
- optional feature cache: data/fewshot_subspace_variants/features_cache.npz
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.benchmark_prompt_models import sweep_thresholds
from scripts.train_classifier import DINO_TRANSFORM, EMBED_DIM, MODEL_NAME

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "data" / "samples" / "training_manifest.json"
POSITIVE_DIR = ROOT_DIR / "data" / "samples" / "positive"
NEGATIVE_DIR = ROOT_DIR / "data" / "samples" / "negative"
REJECTED_DIR = ROOT_DIR / "data" / "samples" / "rejected"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "fewshot_subspace_variants"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_PATCH_SAMPLE_FRAC = 0.15
DEFAULT_PATCH_TOP_K = 8
DEFAULT_PATCH_PERCENTILE = 95.0
SAFE_POSITIVE_PCA_CAP = 8


@dataclass(slots=True)
class SampleRecord:
    image_path: Path
    label: int
    group: str


@dataclass(slots=True)
class FeatureBank:
    canonical: list[SampleRecord]
    confounders: list[SampleRecord]
    cls_embeddings: np.ndarray
    patch_tokens: np.ndarray
    confounder_embeddings: np.ndarray


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _image_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]


def _normalise_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _safe_float(value: float | np.floating | np.ndarray) -> float:
    return float(np.asarray(value).reshape(()))


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _safe_std(values: np.ndarray) -> float:
    std = float(np.std(np.asarray(values, dtype=np.float32)))
    return std if std > 1e-12 else 1.0


def safe_pca_components(n_samples: int, n_features: int, *, positive_side: bool = False) -> int:
    """Select a safe PCA component count for tiny few-shot sets.

    Positive-side PCA is capped more aggressively because the canonical positive
    set is small and we want to avoid memorising it.
    """

    if n_samples <= 1 or n_features <= 0:
        return 1
    base_cap = SAFE_POSITIVE_PCA_CAP if positive_side else 32
    max_possible = min(n_samples - 1, n_features, max(1, n_samples // 2))
    return max(1, min(base_cap, max_possible))


def zscore(values: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    ref = arr if reference is None else np.asarray(reference, dtype=np.float32)
    mean = _safe_mean(ref)
    std = _safe_std(ref)
    return (arr - mean) / std


def combine_positive_aware_scores(
    residuals: np.ndarray,
    similarities: np.ndarray,
    mode: str,
    *,
    residual_reference: np.ndarray | None = None,
    similarity_reference: np.ndarray | None = None,
) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float32)
    similarities = np.asarray(similarities, dtype=np.float32)
    if mode == "residual_x_pos_sim":
        return residuals * np.clip(similarities, 0.0, None)
    if mode == "zsum":
        return zscore(residuals, residual_reference) + zscore(similarities, similarity_reference)
    raise ValueError("mode must be 'residual_x_pos_sim' or 'zsum'")


def aggregate_patch_scores(
    patch_scores: np.ndarray,
    *,
    method: str,
    top_k: int = DEFAULT_PATCH_TOP_K,
    percentile: float = DEFAULT_PATCH_PERCENTILE,
) -> float:
    scores = np.asarray(patch_scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return 0.0
    if method == "max":
        return _safe_float(np.max(scores))
    if method == "topk_mean":
        k = max(1, min(int(top_k), scores.size))
        return _safe_float(np.mean(np.sort(scores)[-k:]))
    if method == "percentile":
        return _safe_float(np.percentile(scores, percentile))
    if method == "mean":
        return _safe_float(np.mean(scores))
    raise ValueError("method must be one of: max, topk_mean, percentile, mean")


def _load_dinov2(device: str) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval().to(device)
    return model


def _extract_image_features(
    path: Path,
    model: torch.nn.Module,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as src:
        image = src.convert("RGB")
    tensor = DINO_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        patch_tokens, cls_token = model.get_intermediate_layers(
            tensor, n=1, reshape=True, return_class_token=True
        )[0]
    cls_embedding = F.normalize(cls_token, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    patch_matrix = (
        patch_tokens.flatten(2).transpose(1, 2).squeeze(0).cpu().numpy().astype(np.float16)
    )
    return cls_embedding, patch_matrix


def _extract_fewshot_bank(
    canonical: list[SampleRecord],
    confounders: list[SampleRecord],
    *,
    device: str,
    cache_path: Path,
    use_cache: bool = True,
) -> FeatureBank:
    if use_cache and cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        cached_canonical_paths = loaded["canonical_paths"].tolist()
        requested_paths = [str(record.image_path) for record in canonical]
        cached_confounder_paths = loaded["confounder_paths"].tolist()
        requested_confounder_paths = [str(record.image_path) for record in confounders]
        if (
            cached_canonical_paths == requested_paths
            and cached_confounder_paths == requested_confounder_paths
        ):
            return FeatureBank(
                canonical=canonical,
                confounders=confounders,
                cls_embeddings=loaded["cls_embeddings"].astype(np.float32, copy=False),
                patch_tokens=loaded["patch_tokens"],
                confounder_embeddings=loaded["confounder_embeddings"].astype(
                    np.float32, copy=False
                ),
            )

    model = _load_dinov2(device)
    cls_embeddings: list[np.ndarray] = []
    patch_tokens: list[np.ndarray] = []
    confounder_embeddings: list[np.ndarray] = []

    for record in canonical:
        cls_embedding, patch_matrix = _extract_image_features(record.image_path, model, device)
        cls_embeddings.append(cls_embedding)
        patch_tokens.append(patch_matrix)

    for record in confounders:
        cls_embedding, _ = _extract_image_features(record.image_path, model, device)
        confounder_embeddings.append(cls_embedding)

    del model
    gc.collect()

    cls_array = np.stack(cls_embeddings).astype(np.float32)
    patch_array = np.stack(patch_tokens).astype(np.float16)
    confounder_array = (
        np.stack(confounder_embeddings).astype(np.float32)
        if confounder_embeddings
        else np.empty((0, cls_array.shape[1]), dtype=np.float32)
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        canonical_paths=np.array([str(record.image_path) for record in canonical], dtype=object),
        confounder_paths=np.array([str(record.image_path) for record in confounders], dtype=object),
        cls_embeddings=cls_array,
        patch_tokens=patch_array,
        confounder_embeddings=confounder_array,
    )

    return FeatureBank(
        canonical=canonical,
        confounders=confounders,
        cls_embeddings=cls_array,
        patch_tokens=patch_array,
        confounder_embeddings=confounder_array,
    )


def _load_records() -> tuple[list[SampleRecord], list[SampleRecord]]:
    manifest = _load_manifest()
    positives = [
        SampleRecord(image_path=POSITIVE_DIR / name, label=1, group="positive")
        for name in manifest.get("positives", [])
        if (POSITIVE_DIR / name).exists()
    ]

    rejected_names = set(manifest.get("rejected", []))
    rejected_names.update(p.name for p in _image_paths(REJECTED_DIR))
    confounders = []
    for name in sorted(rejected_names):
        path = REJECTED_DIR / name
        if path.exists():
            confounders.append(SampleRecord(image_path=path, label=0, group="confounder"))

    negative_paths = [
        path
        for path in _image_paths(NEGATIVE_DIR)
        if path.name not in {record.image_path.name for record in positives}
    ]
    negatives = [
        SampleRecord(image_path=path, label=0, group="negative")
        for path in negative_paths
    ]
    canonical = positives + negatives
    return canonical, confounders


def _normalise_images(embeddings: np.ndarray) -> np.ndarray:
    return _normalise_rows(np.asarray(embeddings, dtype=np.float32))


def _fit_pca(matrix: np.ndarray, *, positive_side: bool = False) -> PCA:
    matrix = np.asarray(matrix, dtype=np.float32)
    n_components = safe_pca_components(
        matrix.shape[0], matrix.shape[1], positive_side=positive_side
    )
    pca = PCA(n_components=n_components, whiten=False, random_state=42)
    pca.fit(matrix)
    return pca


def _pca_residuals(pca: PCA, matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    projected = pca.transform(matrix)
    reconstructed = pca.inverse_transform(projected)
    return np.mean((matrix - reconstructed) ** 2, axis=1).astype(np.float32)


def _patch_residuals(pca: PCA, patch_tokens: np.ndarray) -> np.ndarray:
    patches = np.asarray(patch_tokens, dtype=np.float32).reshape(-1, patch_tokens.shape[-1])
    projected = pca.transform(patches)
    reconstructed = pca.inverse_transform(projected)
    residuals = np.mean((patches - reconstructed) ** 2, axis=1)
    return residuals.reshape(patch_tokens.shape[0], patch_tokens.shape[1]).astype(np.float32)


def _image_patch_scores(
    bank: FeatureBank,
    pca: PCA,
    positive_proto: np.ndarray,
    *,
    sample_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patch_tokens = bank.patch_tokens.astype(np.float32, copy=False)
    n_images, n_patches, _ = patch_tokens.shape
    residual_matrix = _patch_residuals(pca, patch_tokens)
    patch_vectors = patch_tokens.reshape(n_images * n_patches, -1)
    proto = _normalise_vector(positive_proto)
    similarity_vector = _normalise_rows(patch_vectors) @ proto
    similarity_matrix = similarity_vector.reshape(n_images, n_patches).astype(np.float32)
    return residual_matrix, similarity_matrix, patch_vectors


def _positive_loo_scores(
    labels: np.ndarray,
    scorer,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    if pos_indices.size == 0:
        raise ValueError("Need at least one positive example for leave-one-positive-out scoring")

    final_scores = np.full(labels.shape[0], np.nan, dtype=np.float32)
    negative_fold_scores: list[np.ndarray] = []

    for holdout_idx in pos_indices:
        train_pos_mask = labels == 1
        train_pos_mask[holdout_idx] = False
        fold_scores = np.asarray(scorer(train_pos_mask), dtype=np.float32)
        final_scores[holdout_idx] = fold_scores[holdout_idx]
        negative_fold_scores.append(fold_scores[neg_indices])

    final_scores[neg_indices] = np.mean(np.stack(negative_fold_scores), axis=0).astype(np.float32)
    return final_scores


def _metrics_from_rows(rows: list[dict]) -> dict:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float32)
    baseline = sweep_thresholds(rows)
    metrics = baseline["baseline"]
    if labels.size > 1 and len(np.unique(labels)) > 1:
        auroc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
    else:
        auroc = None
        ap = None
    metrics = {**metrics, "auroc": auroc, "ap": ap}
    return {
        "metrics": metrics,
        "calibration": baseline,
    }


def _scored_rows(
    records: list[SampleRecord],
    scores: np.ndarray,
    *,
    benchmark: str,
    mode: str,
) -> list[dict]:
    rows: list[dict] = []
    for record, score in zip(records, scores, strict=False):
        rows.append(
            {
                "benchmark": benchmark,
                "mode": mode,
                "label": int(record.label),
                "score": float(score),
                "prediction": int(score > 0.0),
                "image_path": str(record.image_path),
                "filename": record.image_path.name,
                "group": record.group,
            }
        )
    return rows


def _global_positive_prototype(embeddings: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return _normalise_vector(np.mean(embeddings[mask], axis=0))


def _global_fold_stats(values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_values = values[train_mask]
    return zscore(values, train_values), train_values


def benchmark_global_variants(bank: FeatureBank) -> list[dict]:
    labels = np.asarray([record.label for record in bank.canonical], dtype=int)
    embeddings = _normalise_images(bank.cls_embeddings)
    negative_mask = labels == 0
    negative_pca = _fit_pca(embeddings[negative_mask], positive_side=False)
    negative_residuals = _pca_residuals(negative_pca, embeddings)
    confounder_proto = (
        _normalise_vector(np.mean(bank.confounder_embeddings, axis=0))
        if bank.confounder_embeddings.size
        else np.zeros(embeddings.shape[1], dtype=np.float32)
    )
    confounder_sim = (
        embeddings @ confounder_proto
        if bank.confounder_embeddings.size
        else np.zeros(labels.size, dtype=np.float32)
    )

    benchmarks: list[dict] = []

    def fold_pos_aware_mul(train_pos_mask: np.ndarray) -> np.ndarray:
        positive_proto = _global_positive_prototype(embeddings, train_pos_mask)
        positive_sim = embeddings @ positive_proto
        scores = combine_positive_aware_scores(
            negative_residuals,
            positive_sim,
            mode="residual_x_pos_sim",
        )
        return scores

    def fold_pos_aware_zsum(train_pos_mask: np.ndarray) -> np.ndarray:
        positive_proto = _global_positive_prototype(embeddings, train_pos_mask)
        positive_sim = embeddings @ positive_proto
        train_mask = train_pos_mask | negative_mask
        return combine_positive_aware_scores(
            negative_residuals,
            positive_sim,
            mode="zsum",
            residual_reference=negative_residuals[train_mask],
            similarity_reference=positive_sim[train_mask],
        )

    def fold_dual_contrast(train_pos_mask: np.ndarray) -> np.ndarray:
        positive_pca = _fit_pca(embeddings[train_pos_mask], positive_side=True)
        positive_residuals = _pca_residuals(positive_pca, embeddings)
        train_mask = train_pos_mask | negative_mask
        return zscore(negative_residuals, negative_residuals[train_mask]) - zscore(
            positive_residuals, positive_residuals[train_mask]
        )

    def fold_confounder_margin(train_pos_mask: np.ndarray) -> np.ndarray:
        positive_proto = _global_positive_prototype(embeddings, train_pos_mask)
        positive_sim = embeddings @ positive_proto
        train_mask = train_pos_mask | negative_mask
        return (
            zscore(negative_residuals, negative_residuals[train_mask])
            + zscore(positive_sim, positive_sim[train_mask])
            - zscore(confounder_sim, confounder_sim[train_mask])
        )

    def fold_gaussian_llr(train_pos_mask: np.ndarray) -> np.ndarray:
        pos_train = embeddings[train_pos_mask]
        neg_train = embeddings[negative_mask]
        pos_mean = np.mean(pos_train, axis=0)
        neg_mean = np.mean(neg_train, axis=0)
        pos_var = np.maximum(np.var(pos_train, axis=0), 1e-6)
        neg_var = np.maximum(np.var(neg_train, axis=0), 1e-6)
        pos_ll = _diag_gaussian_logpdf(embeddings, pos_mean, pos_var)
        neg_ll = _diag_gaussian_logpdf(embeddings, neg_mean, neg_var)
        return pos_ll - neg_ll

    def fold_mahalanobis(train_pos_mask: np.ndarray) -> np.ndarray:
        pos_train = embeddings[train_pos_mask]
        neg_train = embeddings[negative_mask]
        pos_mean = np.mean(pos_train, axis=0)
        neg_mean = np.mean(neg_train, axis=0)
        pos_var = np.maximum(np.var(pos_train, axis=0), 1e-6)
        neg_var = np.maximum(np.var(neg_train, axis=0), 1e-6)
        pos_dist = _diag_mahalanobis(embeddings, pos_mean, pos_var)
        neg_dist = _diag_mahalanobis(embeddings, neg_mean, neg_var)
        return neg_dist - pos_dist

    variant_specs = [
        {
            "name": "positive_aware_residual_mul",
            "family": "positive_aware_residual",
            "mode": "global_loo",
            "evaluation": "leave-one-positive-out",
            "leakage_caveat": (
                "Positive prototype is re-fit per held-out positive; negatives are "
                "averaged across folds."
            ),
            "scorer": fold_pos_aware_mul,
        },
        {
            "name": "positive_aware_residual_zsum",
            "family": "positive_aware_residual",
            "mode": "global_loo",
            "evaluation": "leave-one-positive-out",
            "leakage_caveat": (
                "Standardisation is learned from each training fold; negatives are "
                "averaged across folds."
            ),
            "scorer": fold_pos_aware_zsum,
        },
        {
            "name": "dual_subspace_contrast",
            "family": "dual_subspace",
            "mode": "global_loo",
            "evaluation": "leave-one-positive-out",
            "leakage_caveat": (
                "Positive PCA is re-fit per held-out positive; negatives are "
                "averaged across folds."
            ),
            "scorer": fold_dual_contrast,
        },
        {
            "name": "confounder_aware_margin",
            "family": "confounder_aware",
            "mode": "global_loo",
            "evaluation": "leave-one-positive-out",
            "leakage_caveat": (
                "Explicit rejects are used only as confounder prototypes, not as "
                "evaluation labels."
            ),
            "scorer": fold_confounder_margin,
        },
        {
            "name": "gaussian_llr",
            "family": "fewshot_gaussian",
            "mode": "global_loo",
            "evaluation": "leave-one-positive-out",
            "leakage_caveat": (
                "Positive Gaussian parameters are re-fit per held-out positive; "
                "negatives are averaged across folds."
            ),
            "scorer": fold_gaussian_llr,
        },
        {
            "name": "mahalanobis_contrast",
            "family": "fewshot_gaussian",
            "mode": "global_loo",
            "evaluation": "leave-one-positive-out",
            "leakage_caveat": (
                "Positive Gaussian parameters are re-fit per held-out positive; "
                "negatives are averaged across folds."
            ),
            "scorer": fold_mahalanobis,
        },
    ]

    for spec in variant_specs:
        scores = _positive_loo_scores(labels, spec["scorer"])
        rows = _scored_rows(bank.canonical, scores, benchmark=spec["name"], mode=spec["mode"])
        metrics_bundle = _metrics_from_rows(rows)
        benchmarks.append(
            {
                **spec,
                "row_count": len(rows),
                "metrics": metrics_bundle["metrics"],
                "calibration": metrics_bundle["calibration"],
                "rows": rows,
            }
        )

    return benchmarks


def _diag_gaussian_logpdf(samples: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    var = np.maximum(np.asarray(var, dtype=np.float32), 1e-6)
    diff = samples - mean
    return -0.5 * np.sum(np.log(2.0 * np.pi * var) + (diff * diff) / var, axis=1)


def _diag_mahalanobis(samples: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    var = np.maximum(np.asarray(var, dtype=np.float32), 1e-6)
    diff = samples - mean
    return np.sum((diff * diff) / var, axis=1)


def _sample_patch_tokens(
    patch_tokens: np.ndarray,
    sample_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    patch_tokens = np.asarray(patch_tokens, dtype=np.float32)
    n_patches = patch_tokens.shape[0]
    sample_count = max(1, min(n_patches, int(round(n_patches * sample_frac))))
    if sample_count >= n_patches:
        return patch_tokens
    indices = rng.choice(n_patches, size=sample_count, replace=False)
    return patch_tokens[indices]


def benchmark_patch_variants(
    bank: FeatureBank,
    *,
    patch_sample_frac: float,
    patch_top_k: int,
    patch_percentile: float,
) -> list[dict]:
    rng = np.random.default_rng(42)
    patch_tokens = bank.patch_tokens.astype(np.float32, copy=False)
    labels = np.asarray([record.label for record in bank.canonical], dtype=int)
    negative_indices = np.where(labels == 0)[0]
    positive_indices = np.where(labels == 1)[0]

    negative_patch_samples = []
    positive_patch_samples = []
    for index in negative_indices:
        negative_patch_samples.append(
            _sample_patch_tokens(patch_tokens[index], patch_sample_frac, rng)
        )
    for index in positive_indices:
        positive_patch_samples.append(patch_tokens[index])

    negative_patch_matrix = np.vstack(negative_patch_samples).astype(np.float32)
    positive_patch_matrix = np.vstack(positive_patch_samples).astype(np.float32)
    negative_pca = _fit_pca(negative_patch_matrix, positive_side=False)
    positive_proto = _normalise_vector(np.mean(positive_patch_matrix, axis=0))

    residual_cube = _patch_residuals(negative_pca, patch_tokens)
    patch_vectors = patch_tokens.reshape(-1, patch_tokens.shape[-1])
    similarity_cube = (_normalise_rows(patch_vectors) @ positive_proto).reshape(residual_cube.shape)

    residual_z = zscore(residual_cube.reshape(-1), residual_cube.reshape(-1)).reshape(
        residual_cube.shape
    )
    similarity_z = zscore(
        similarity_cube.reshape(-1), similarity_cube.reshape(-1)
    ).reshape(similarity_cube.shape)
    patch_proto_scores = residual_z + similarity_z
    mil_scores = residual_z

    def aggregate_matrix(matrix: np.ndarray, method: str) -> np.ndarray:
        return np.asarray(
            [
                aggregate_patch_scores(
                    row,
                    method=method,
                    top_k=patch_top_k,
                    percentile=patch_percentile,
                )
                for row in matrix
            ],
            dtype=np.float32,
        )

    benchmarks: list[dict] = []
    patch_variants = [
        ("patch_positive_proto_max", patch_proto_scores, "max"),
        ("patch_positive_proto_topk_mean", patch_proto_scores, "topk_mean"),
        ("patch_positive_proto_p95", patch_proto_scores, "percentile"),
        ("mil_subspace_max", mil_scores, "max"),
        ("mil_subspace_topk_mean", mil_scores, "topk_mean"),
        ("mil_subspace_p95", mil_scores, "percentile"),
    ]

    for name, matrix, agg_method in patch_variants:
        aggregated = aggregate_matrix(matrix, agg_method)
        rows = _scored_rows(
            bank.canonical,
            aggregated,
            benchmark=name,
            mode="transductive_patch_mil",
        )
        metrics_bundle = _metrics_from_rows(rows)
        benchmarks.append(
            {
                "name": name,
                "family": "patch_positive_proto" if name.startswith("patch_") else "mil_subspace",
                "mode": "transductive_patch_mil",
                "evaluation": "transductive",
                "leakage_caveat": (
                    "Patch tokens, prototypes, and z-score normalisation are computed "
                    "transductively over the benchmark set."
                ),
                "aggregation": agg_method,
                "top_k": patch_top_k,
                "percentile": patch_percentile,
                "row_count": len(rows),
                "metrics": metrics_bundle["metrics"],
                "calibration": metrics_bundle["calibration"],
                "rows": rows,
            }
        )

    return benchmarks


def _relative_path(path: str | Path, base_dir: Path) -> str:
    return os.path.relpath(Path(path), base_dir)


def _build_review_html(
    summary: dict,
    benchmarks: list[dict],
    best_rows: list[dict],
    output_dir: Path,
) -> str:
    benchmark_rows = []
    for bench in benchmarks:
        metrics = bench["metrics"]
        calibration = bench["calibration"]
        best_accuracy = calibration["best_accuracy"]
        best_f1 = calibration["best_f1"]
        best_balanced = calibration["best_balanced_accuracy"]
        benchmark_rows.append(
            "<tr>"
            f"<td>{escape(str(bench['name']))}</td>"
            f"<td>{escape(str(bench['family']))}</td>"
            f"<td>{escape(str(bench['evaluation']))}</td>"
            f"<td>{bench['row_count']}</td>"
            f"<td>{metrics['accuracy']}</td>"
            f"<td>{metrics['auroc']}</td>"
            f"<td>{metrics['ap']}</td>"
            f"<td>{best_accuracy.get('threshold')}</td>"
            f"<td>{best_f1.get('threshold')}</td>"
            f"<td>{best_balanced.get('threshold')}</td>"
            f"<td>{escape(str(bench['leakage_caveat']))}</td>"
            "</tr>"
        )

    row_rows = []
    for row in best_rows[:100]:
        rel_path = _relative_path(row["image_path"], output_dir)
        row_rows.append(
            "<tr>"
            f"<td>{escape(str(row['filename']))}</td>"
            f"<td>{row['label']}</td>"
            f"<td>{row['score']:.6f}</td>"
            f"<td>{row['prediction']}</td>"
            f"<td>{escape(str(row['benchmark']))}</td>"
            f"<td>{escape(str(row['group']))}</td>"
            f"<td><img src='{escape(rel_path)}' alt='{escape(str(row['filename']))}'></td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Few-shot SubspaceAD Variants Review</title>
    <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    th, td {{
      border: 1px solid #ddd;
      padding: 0.35rem 0.5rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f6f6f6; }}
    img {{ width: 140px; max-width: 140px; display: block; }}
    .note {{ color: #444; font-size: 0.95rem; }}
  </style>
</head>
<body>
  <h1>Few-shot DINOv2 SubspaceAD Variants</h1>
  <p>Generated: {escape(summary['created_at'])}</p>
  <p class="note">
    Global-image variants use leave-one-positive-out scoring where practical.
    Patch/MIL variants are transductive and use the full benchmark set for
    prototypes, PCA sampling, and normalisation.
  </p>

  <h2>Benchmark Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Name</th><th>Family</th><th>Eval</th><th>Rows</th><th>Accuracy</th>
        <th>AUROC</th><th>AP</th><th>Best Acc T</th><th>Best F1 T</th>
        <th>Best Bal T</th><th>Leakage caveat</th>
      </tr>
    </thead>
    <tbody>
      {''.join(benchmark_rows)}
    </tbody>
  </table>

  <h2>Top-scoring rows for the best benchmark</h2>
  <p class="note">
    Sorted by score descending for
    <strong>{escape(summary['best_benchmark'])}</strong>.
  </p>
  <table>
    <thead>
      <tr><th>File</th><th>Label</th><th>Score</th><th>Pred</th><th>Benchmark</th><th>Group</th><th>Thumbnail</th></tr>
    </thead>
    <tbody>
      {''.join(row_rows)}
    </tbody>
  </table>
</body>
</html>
"""


def _write_scores_csv(all_rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "mode",
        "group",
        "label",
        "score",
        "prediction",
        "image_path",
        "filename",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _best_benchmark(benchmarks: list[dict]) -> dict:
    def key(bench: dict) -> tuple[float, float]:
        metrics = bench["metrics"]
        auroc = metrics.get("auroc") if metrics.get("auroc") is not None else -1.0
        ap = metrics.get("ap") if metrics.get("ap") is not None else -1.0
        return float(auroc), float(ap)

    return max(benchmarks, key=key)


def build_summary(
    bank: FeatureBank,
    benchmarks: list[dict],
    *,
    device: str,
    patch_sample_frac: float,
    patch_top_k: int,
    patch_percentile: float,
    cache_path: Path,
) -> tuple[dict, list[dict], dict]:
    best = _best_benchmark(benchmarks)
    all_rows: list[dict] = []
    for bench in benchmarks:
        all_rows.extend(bench["rows"])
    all_rows = sorted(all_rows, key=lambda row: float(row["score"]), reverse=True)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "manifest_path": str(MANIFEST_PATH),
        "model_name": MODEL_NAME,
        "embed_dim": EMBED_DIM,
        "device": device,
        "canonical_count": len(bank.canonical),
        "positive_count": len([record for record in bank.canonical if record.label == 1]),
        "negative_count": len([record for record in bank.canonical if record.label == 0]),
        "confounder_count": len(bank.confounders),
        "patch_sample_frac": patch_sample_frac,
        "patch_top_k": patch_top_k,
        "patch_percentile": patch_percentile,
        "cache_path": str(cache_path),
        "best_benchmark": best["name"],
        "benchmarks": [
            {
                "name": bench["name"],
                "family": bench["family"],
                "mode": bench["mode"],
                "evaluation": bench["evaluation"],
                "leakage_caveat": bench["leakage_caveat"],
                "aggregation": bench.get("aggregation"),
                "top_k": bench.get("top_k"),
                "percentile": bench.get("percentile"),
                "row_count": bench["row_count"],
                "metrics": bench["metrics"],
                "calibration": bench["calibration"],
            }
            for bench in benchmarks
        ],
        "notes": [
            (
                "Global-image variants use leave-one-positive-out folds so each "
                "positive is scored without seeing itself in the positive prototype / "
                "Gaussian fit."
            ),
            (
                "Negative scores are averaged across positive holdout folds for the "
                "global-image variants."
            ),
            (
                "Patch/MIL variants are transductive and explicitly use "
                "benchmark-wide patch statistics; interpret them as screening scores, "
                "not final truth."
            ),
            (
                "Explicit rejects are used only as confounder prototypes for the "
                "confounder-aware variant."
            ),
        ],
    }
    return summary, all_rows, best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patch-sample-frac", type=float, default=DEFAULT_PATCH_SAMPLE_FRAC)
    parser.add_argument("--patch-top-k", type=int, default=DEFAULT_PATCH_TOP_K)
    parser.add_argument("--patch-percentile", type=float, default=DEFAULT_PATCH_PERCENTILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "features_cache.npz"

    canonical, confounders = _load_records()
    if args.limit is not None:
        canonical = canonical[: max(0, int(args.limit))]

    device = _resolve_device(args.device)
    bank = _extract_fewshot_bank(
        canonical,
        confounders,
        device=device,
        cache_path=cache_path,
        use_cache=not args.no_cache,
    )

    global_benchmarks = benchmark_global_variants(bank)
    patch_benchmarks = benchmark_patch_variants(
        bank,
        patch_sample_frac=args.patch_sample_frac,
        patch_top_k=args.patch_top_k,
        patch_percentile=args.patch_percentile,
    )
    benchmarks = global_benchmarks + patch_benchmarks

    summary, all_rows, best = build_summary(
        bank,
        benchmarks,
        device=device,
        patch_sample_frac=args.patch_sample_frac,
        patch_top_k=args.patch_top_k,
        patch_percentile=args.patch_percentile,
        cache_path=cache_path,
    )

    _write_scores_csv(all_rows, output_dir / "scores.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    best_rows = sorted(best["rows"], key=lambda row: float(row["score"]), reverse=True)
    (output_dir / "review.html").write_text(
        _build_review_html(summary, benchmarks, best_rows, output_dir), encoding="utf-8"
    )

    print(f"Wrote few-shot SubspaceAD variant outputs to {output_dir}")
    print(f"Benchmarks: {len(benchmarks)}")
    print(f"Best benchmark: {best['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
