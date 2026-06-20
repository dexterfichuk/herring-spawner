#!/usr/bin/env python3
"""Consolidate all herring spawn candidate images into one unified labeled set.

Deduplicates by MD5 hash, preserves existing labels from known sources,
symlinks into `data/unified/thumbs/`, and writes a single manifest.

Usage:
    python scripts/consolidate_candidates.py --output data/unified
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Source directories to scan (priority order — later dirs won't override
# labels from earlier dirs)
# ---------------------------------------------------------------------------
SOURCE_DIRS = [
    "data/samples/positive",
    "data/samples/negative",
    "data/candidates_knn",
    "data/candidates_shsi",
    "data/candidates_final",
    "data/candidates_v2",
    "data/candidates",
    "data/sog_candidates/thumbnails",
    "data/ingressed/thumbnails",
    "data/candidates_multiyear/thumbs",
    "data/candidates_salmon_coast",
    "data/candidates_salmon_coast_2023",
    "data/candidates_salmon_coast_2025",
    "data/candidates_salmon_coast_2026",
    "data/candidates_svm_2023",
    "data/candidates_svm_2024",
    "data/candidates_svm_2025",
]

# ---------------------------------------------------------------------------
# Known label files (label_source_name -> {filename: "spawn"|"no-spawn"|"skip"})
# ---------------------------------------------------------------------------
LABEL_SOURCES: list[tuple[str, str]] = [
    # (label, source_path)
    ("golden_pos", "data/samples/training_manifest.json"),
    ("golden_neg", "data/samples/training_manifest.json"),
    ("knn_silo", "data/candidates_knn/silo_labels.json"),
    ("sog_silo", "data/sog_candidates/silo_labels.json"),
    ("dual_subspace", "data/dual_subspace_review/labels.json"),
    ("shsi", "data/candidates_shsi/labels.json"),
    ("rose", "data/candidates_v2/rose_labels.json"),
    ("rose_super", "data/candidates_v2/rose_super_review.json"),
    ("final_golden", "data/final_review/golden_set.json"),
    ("env_match", "data/environmental_matching/s2_landsat/labels.json"),
]


def md5_file(path: Path) -> str:
    """Return hex digest of file contents."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def parse_region_from_filename(fname: str) -> str:
    """Extract region name from filename patterns."""
    # Pattern: region_YYYY-MM-DD_...
    m = re.match(r"^([a-z][-a-z0-9]+(?:_[a-z][-a-z0-9]+)*)_\d{4}-\d{2}", fname)
    if m:
        return m.group(1).replace("_", "-")
    # Pattern: region-YYYY-MM-DD...
    m = re.match(r"^([a-z][-a-z0-9]+)-\d{4}", fname)
    if m:
        return m.group(1)
    # Pattern: SoG_YYYY-...
    if fname.startswith("SoG_"):
        return "strait-of-georgia"
    # Pattern: dfo-verified-REGION_...
    m = re.match(r"^dfo-verified-(.+?)_\d{4}", fname)
    if m:
        return m.group(1)
    # Pattern: manual-YYYY-...
    m = re.match(r"^manual-\d{4}", fname)
    if m:
        return "manual"
    return "unknown"


def load_training_manifest(path: Path) -> dict[str, str]:
    """Load training_manifest.json format: positives list + negatives from file list."""
    labels: dict[str, str] = {}
    if not path.exists():
        return labels
    data = json.loads(path.read_text())
    for fname in data.get("positives", []):
        labels[fname] = "spawn"
    # Negatives are inferred — the file lists are in the manifest but actual
    # negative dir contents matter. We'll get those from the directory scan.
    # But training_manifest has a negative list pattern — try to find it.
    for fname in data.get("rejected", []):
        labels[fname] = "no-spawn"
    # Also check for "negatives" key
    for fname in data.get("negatives", []):
        labels[fname] = "no-spawn"
    return labels


def load_simple_labels(path: Path) -> dict[str, str]:
    """Load a flat {filename: label} JSON dict."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        # Check if it has nested structure
        if "labels" in data and isinstance(data["labels"], dict):
            return {k: v for k, v in data["labels"].items()
                    if v in ("spawn", "no-spawn", "skip")}
        # Flat dict — check values
        return {k: v for k, v in data.items()
                if isinstance(v, str) and v in ("spawn", "no-spawn", "skip")}
    return {}


def load_golden_set(path: Path) -> dict[str, str]:
    """Load final_review/golden_set.json format."""
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    data = json.loads(path.read_text())
    for entry in data.get("positives", []):
        fname = entry if isinstance(entry, str) else entry.get("filename", "")
        if fname:
            labels[fname] = "spawn"
    for entry in data.get("negatives", []):
        fname = entry if isinstance(entry, str) else entry.get("filename", "")
        if fname:
            labels[fname] = "no-spawn"
    return labels


def load_rose_labels(path: Path) -> dict[str, str]:
    """Load rose_labels.json or rose_super_review.json format."""
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        for entry in data:
            fname = entry.get("filename", entry.get("image", ""))
            label_val = entry.get("label", entry.get("class", ""))
            if fname and label_val:
                if label_val in (1, "1", "spawn", "positive", True):
                    labels[fname] = "spawn"
                elif label_val in (0, "0", "no-spawn", "negative", "no_spawn", False):
                    labels[fname] = "no-spawn"
                elif label_val in ("skip", "uncertain", "maybe"):
                    labels[fname] = "skip"
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (int, bool)):
                labels[k] = "spawn" if v else "no-spawn"
            elif isinstance(v, str) and v in ("spawn", "no-spawn", "skip"):
                labels[k] = v
    return labels


LABEL_LOADERS = {
    "golden_pos": load_training_manifest,
    "golden_neg": load_training_manifest,
    "knn_silo": load_simple_labels,
    "sog_silo": load_simple_labels,
    "dual_subspace": load_simple_labels,
    "shsi": load_simple_labels,
    "rose": load_rose_labels,
    "rose_super": load_rose_labels,
    "final_golden": load_golden_set,
    "env_match": load_simple_labels,
}


def consolidate(
    repo_root: Path,
    output_dir: Path,
) -> dict:
    """Main consolidation logic."""
    thumbs_dir = output_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: collect all images, deduplicate by hash ----
    seen_hashes: dict[str, Path] = {}  # md5 -> first seen path
    hash_to_fname: dict[str, str] = {}  # md5 -> filename

    for rel_dir in SOURCE_DIRS:
        src = repo_root / rel_dir
        if not src.is_dir():
            continue
        for png in sorted(src.rglob("*.png")):
            # Skip deep temporal_cache subdirs
            if "temporal_cache" in str(png):
                continue
            # Skip overlay_cache
            if ".overlay_cache" in str(png):
                continue
            try:
                h = md5_file(png)
            except Exception:
                continue
            if h not in seen_hashes:
                seen_hashes[h] = png
                hash_to_fname[h] = png.name

    print(f"  Unique images (by MD5): {len(seen_hashes)}")

    # ---- Phase 2: load all existing labels ----
    existing_labels: dict[str, str] = {}  # filename -> label
    label_source: dict[str, list[str]] = {}  # filename -> [sources]

    for source_name, rel_path in LABEL_SOURCES:
        full_path = repo_root / rel_path
        loader = LABEL_LOADERS.get(source_name, load_simple_labels)
        source_labels = loader(full_path)
        for fname, label in source_labels.items():
            if fname not in existing_labels:
                existing_labels[fname] = label
                label_source[fname] = [source_name]
            else:
                if source_name not in label_source.get(fname, []):
                    label_source[fname].append(source_name)

    # ---- Phase 3: also label negatives from samples/negative dir ----
    neg_dir = repo_root / "data/samples/negative"
    if neg_dir.is_dir():
        for png in neg_dir.glob("*.png"):
            if png.name not in existing_labels:
                existing_labels[png.name] = "no-spawn"
                label_source[png.name] = ["negative_dir"]

    # Mark known positives from samples/positive dir
    pos_dir = repo_root / "data/samples/positive"
    if pos_dir.is_dir():
        for png in pos_dir.glob("*.png"):
            if png.name not in existing_labels:
                existing_labels[png.name] = "spawn"
                label_source[png.name] = ["positive_dir"]

    # Also mark rejected dir
    rej_dir = repo_root / "data/samples/rejected"
    if rej_dir.is_dir():
        for png in rej_dir.glob("*.png"):
            if png.name not in existing_labels:
                existing_labels[png.name] = "no-spawn"
                label_source[png.name] = ["rejected_dir"]

    # ---- Phase 4: symlink unique images ----
    manifest: list[dict] = []
    symlinked = 0

    for md5_hash, src_path in sorted(seen_hashes.items(), key=lambda x: x[1].name):
        fname = src_path.name
        dest = thumbs_dir / fname

        # Handle name collisions (different hashes, same filename)
        counter = 1
        while dest.exists() and md5_file(dest) != md5_hash:
            stem, ext = fname.rsplit(".", 1) if "." in fname else (fname, "png")
            dest = thumbs_dir / f"{stem}_{counter}.{ext}"
            counter += 1

        if not dest.exists():
            dest.symlink_to(src_path.resolve())
            symlinked += 1

        region = parse_region_from_filename(fname)
        label = existing_labels.get(fname)
        sources = label_source.get(fname, [])

        manifest.append({
            "filename": dest.name,
            "source_path": str(src_path.relative_to(repo_root)),
            "region": region,
            "label": label,
            "label_sources": sources,
        })

    print(f"  Symlinked: {symlinked} images")

    # ---- Phase 5: stats ----
    n_spawn = sum(1 for m in manifest if m["label"] == "spawn")
    n_nospawn = sum(1 for m in manifest if m["label"] == "no-spawn")
    n_skip = sum(1 for m in manifest if m["label"] == "skip")
    n_unlabeled = sum(1 for m in manifest if m["label"] is None)

    print(f"  Labeled: {n_spawn} spawn, {n_nospawn} no-spawn, {n_skip} skip")
    print(f"  Unlabeled: {n_unlabeled}")

    # ---- Phase 6: save ----
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    labels_out = {m["filename"]: m["label"] for m in manifest if m["label"]}
    labels_path = output_dir / "labels.json"
    labels_path.write_text(json.dumps(labels_out, indent=2, sort_keys=True))

    summary = {
        "total_images": len(manifest),
        "spawn": n_spawn,
        "no_spawn": n_nospawn,
        "skip": n_skip,
        "unlabeled": n_unlabeled,
        "regions": sorted(set(m["region"] for m in manifest)),
        "label_sources_used": sorted(set(
            s for m in manifest for s in m["label_sources"]
        )),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  Manifest: {manifest_path}")
    print(f"  Labels:   {labels_path}")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate all herring spawn candidate images into one unified set",
    )
    parser.add_argument(
        "--output", default="data/unified",
        help="Output directory (default: data/unified)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / args.output

    print("=== Consolidating candidates ===")
    summary = consolidate(repo_root, output_dir)

    print("\n=== Ready to label ===")
    print(f"  python scripts/label_gradio.py \\")
    print(f"      --manifest data/unified/manifest.json \\")
    print(f"      --image-dir data/unified/thumbs \\")
    print(f"      --labels data/unified/labels.json \\")
    print(f"      --port 7888")

    return 0


if __name__ == "__main__":
    sys.exit(main())
