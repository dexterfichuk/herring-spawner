#!/usr/bin/env python3
"""Scan the Issue #1 expanded BC herring habitat regions.

Scans ONLY the new regions added for Issue #1 coverage (Barkley Sound,
Hesquiat Harbour, Howe Sound, Kyuquot, Port McNeill, Nanoose) using the
same KNN/DINOv2 pipeline from scan_bc_coast_knn.py.

Output is saved to data/candidates_knn_expanded/ so it doesn't disturb
the existing 725-candidate set at data/candidates_knn/.

Usage:
    source .venv/bin/activate
    python scripts/scan_issue1_expansion.py --workers 6
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scan_bc_coast import (
    REGIONS,
    download_thumbnail,
    find_best_scene,
    generate_grid_points,
    print_progress,
    save_candidate,
    update_manifest,
)
from scripts.knn_detector import (
    DINO_TRANSFORM,
    _load_image_embedding,
    _pick_device,
)
from scripts.scan_bc_coast_knn import (
    DEFAULT_END,
    DEFAULT_GRID_SPACING,
    DEFAULT_INGRESSED_DIR,
    DEFAULT_K,
    DEFAULT_LABELS,
    DEFAULT_MAX_CLOUD,
    DEFAULT_NEGATIVE_DIR,
    DEFAULT_ROSE_REVIEW,
    DEFAULT_SOG_FILES,
    DEFAULT_SOG_OUTPUT,
    DEFAULT_START,
    DEFAULT_WORKERS,
    MANIFEST_LOCK as KNN_MANIFEST_LOCK,
    MODEL_LOCK as KNN_MODEL_LOCK,
    MODEL_NAME,
    PRINT_LOCK as KNN_PRINT_LOCK,
    DEFAULT_PROJECT,
    KnnIndex,
    _embedding_from_png_bytes,
    _load_or_compute_embeddings,
    _write_json,
    build_training_records,
    build_review_html,
    html_escape,
    ingest_sog_records,
)

# ---------------------------------------------------------------------------
# New Issue #1 regions only (subset of the expanded REGIONS)
# ---------------------------------------------------------------------------
ISSUE1_REGION_NAMES = {
    "barkley-sound",
    "hesquiat-harbour",
    "howe-sound",
    "kyuquot",
    "port-mcneill",
    "nanoose",
}

ISSUE1_REGIONS = [r for r in REGIONS if r["name"] in ISSUE1_REGION_NAMES]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "candidates_knn_expanded"


def _process_point(
    point: dict,
    knn_index: KnnIndex,
    model: torch.nn.Module,
    device: torch.device,
    ee_module: Any,
    idx: int,
    total: int,
    start_date: str,
    end_date: str,
    max_cloud: float,
    output_dir: Path,
) -> dict:
    """Process a single grid point: find scene, download, embed, classify."""
    result = {"processed": 1, "candidates": 0, "no_scene": 0, "download_errors": 0, "low_score": 0}

    scene_info = find_best_scene(ee_module, point["lat"], point["lon"], start_date, end_date, max_cloud)
    if scene_info is None:
        with KNN_PRINT_LOCK:
            print_progress(idx, total, point["region"], point["lat"], point["lon"], "no scene", 0)
        result["no_scene"] = 1
        return result

    thumb_bytes = download_thumbnail(ee_module, point["lat"], point["lon"], scene_info["scene_id"])
    if thumb_bytes is None:
        with KNN_PRINT_LOCK:
            print_progress(idx, total, point["region"], point["lat"], point["lon"], "download error", 0)
        result["download_errors"] = 1
        return result

    try:
        embedding = _embedding_from_png_bytes(model, device, thumb_bytes)
    except Exception:
        with KNN_PRINT_LOCK:
            print_progress(idx, total, point["region"], point["lat"], point["lon"], "embedding error", 0)
        result["download_errors"] = 1
        return result

    vote = knn_index.predict(embedding)
    if vote["prediction"] != 1:
        with KNN_PRINT_LOCK:
            print_progress(idx, total, point["region"], point["lat"], point["lon"],
                           f"below threshold ({vote['vote_fraction']:.2f})", 0)
        result["low_score"] = 1
        return result

    info = {
        "region": point["region"],
        "lat": point["lat"],
        "lon": point["lon"],
        "date": scene_info["date"],
        "scene_id": scene_info["scene_id"],
        "cloud": scene_info["cloud"],
        "score": round(float(vote["vote_fraction"]), 4),
        "spawn_votes": int(vote["spawn_votes"]),
        "k": int(knn_index.k),
    }
    fname = save_candidate(output_dir, thumb_bytes, info, float(vote["vote_fraction"]))
    entry = {
        **info,
        "thumbnail_path": fname,
        "neighbors": vote["neighbors"],
    }
    with KNN_MANIFEST_LOCK:
        update_manifest(output_dir, entry)
    with KNN_PRINT_LOCK:
        print_progress(idx, total, point["region"], point["lat"], point["lon"],
                       f"CANDIDATE votes={vote['spawn_votes']}/{knn_index.k} {fname}", 0)
    result["candidates"] = 1
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD)
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--rose-review", type=Path, default=DEFAULT_ROSE_REVIEW)
    parser.add_argument("--sog-files", type=Path, nargs="+", default=DEFAULT_SOG_FILES)
    parser.add_argument("--negative-dir", type=Path, default=DEFAULT_NEGATIVE_DIR)
    parser.add_argument("--sog-output", type=Path, default=Path("data/ingressed/sog_events.json"))
    parser.add_argument("--ingressed-dir", type=Path, default=DEFAULT_INGRESSED_DIR)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    args = parser.parse_args()

    t0 = time.time()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    search_dirs = [PROJECT_ROOT / "data" / "candidates_v2", args.ingressed_dir]

    # ---- Load training data ----
    sog_records, sog_year_counts, sog_raw_count = ingest_sog_records(
        args.sog_files, args.sog_output, search_dirs
    )
    print(f"Ingested {len(sog_records)} SoG records from {sog_raw_count} features")
    print(f"SoG by year: {sog_year_counts}")

    training_records, build_summary = build_training_records(
        args.labels, args.rose_review, sog_records, args.negative_dir, search_dirs,
    )
    if not training_records:
        raise RuntimeError("No training records could be matched to thumbnails")

    labels = np.asarray([r["label_int"] for r in training_records], dtype=int)
    print(f"Training set: {len(training_records)}  spawn={int(np.sum(labels==1))}  "
          f"nospawn={int(np.sum(labels==0))}")
    print(f"Match summary: {dict(build_summary)}")

    # ---- Load model ----
    device = _pick_device()
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval().to(device)

    cache_path = PROJECT_ROOT / "data" / "embeddings" / "knn_training_embeddings.npz"
    embeddings = _load_or_compute_embeddings(training_records, model, device, cache_path)
    knn_index = KnnIndex(embeddings, labels, training_records, k=args.k)

    # ---- Earth Engine ----
    import ee  # type: ignore[import-untyped]
    ee.Initialize(project=args.project)

    # ---- Generate grid points for Issue #1 regions only ----
    points = generate_grid_points(ISSUE1_REGIONS, args.grid_spacing)
    print(f"Generated {len(points)} grid points across {len(ISSUE1_REGIONS)} Issue #1 regions")
    for r in ISSUE1_REGIONS:
        region_points = [p for p in points if p["region"] == r["name"]]
        print(f"  {r['name']}: {len(region_points)} points")

    # ---- Scan ----
    stats: Counter[str] = Counter()
    candidate_regions: Counter[str] = Counter()

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [
            executor.submit(
                _process_point, point, knn_index, model, device, ee,
                idx, len(points), args.start, args.end, args.max_cloud, output_dir,
            )
            for idx, point in enumerate(points)
        ]
        for future in as_completed(futures):
            stats.update(future.result())

    # ---- Load manifest ----
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        import json
        candidate_entries = json.loads(manifest_path.read_text())
    else:
        candidate_entries = []

    for row in candidate_entries:
        candidate_regions[str(row.get("region", "unknown"))] += 1

    elapsed = time.time() - t0
    summary = {
        "training_size": len(training_records),
        "regions": [r["name"] for r in ISSUE1_REGIONS],
        "points_scanned": len(points),
        "candidates": int(stats["candidates"]),
        "candidate_count": len(candidate_entries),
        "candidate_regions": dict(candidate_regions),
        "processed": int(stats["processed"]),
        "no_scene": int(stats["no_scene"]),
        "download_errors": int(stats["download_errors"]),
        "low_score": int(stats["low_score"]),
        "elapsed_seconds": elapsed,
    }
    _write_json(output_dir / "summary.json", summary)

    # Generate review page
    review_html = build_review_html(candidate_entries, summary)
    (output_dir / "review.html").write_text(review_html, encoding="utf-8")

    print(f"\nDone. {len(candidate_entries)} candidates from {len(ISSUE1_REGIONS)} new regions")
    print(f"Top regions: {dict(candidate_regions.most_common(5))}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Review: file://{(output_dir / 'review.html').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
