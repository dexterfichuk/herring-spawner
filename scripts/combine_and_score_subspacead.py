#!/usr/bin/env python3
"""Combine original + Issue #1 expansion candidates and run SubspaceAD scoring.

Creates a combined candidate directory with symlinks to avoid copying PNGs,
then runs the SubspaceAD batch scorer to generate overlays and review page.

Usage:
    source .venv/bin/activate
    python scripts/combine_and_score_subspacead.py --port 8772
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ORIGINAL_DIR = PROJECT_ROOT / "data" / "candidates_knn"
EXPANDED_DIR = PROJECT_ROOT / "data" / "candidates_knn_expanded"
COMBINED_DIR = PROJECT_ROOT / "data" / "candidates_knn_combined"
OUTPUT_DIR = PROJECT_ROOT / "data" / "subspace_ad_review_expanded"

DEFAULT_NEGATIVE_DIR = PROJECT_ROOT / "data" / "samples" / "negative"
DEFAULT_PORT = 8772


def build_combined_candidate_dir() -> int:
    """Symlink all original + new candidates into a single directory."""
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    seen = set()

    for src_dir, label in [(ORIGINAL_DIR, "original"), (EXPANDED_DIR, "expansion")]:
        if not src_dir.exists():
            print(f"  WARNING: {label} dir not found: {src_dir}")
            continue
        for png in sorted(src_dir.glob("*.png")):
            if png.name in seen:
                continue
            seen.add(png.name)
            link = COMBINED_DIR / png.name
            if not link.exists():
                link.symlink_to(png.resolve())
            count += 1

    print(f"  Combined {count} candidates into {COMBINED_DIR}")
    manifest_src = ORIGINAL_DIR / "manifest.json"
    if manifest_src.exists():
        manifest = json.loads(manifest_src.read_text())
        exp_manifest_src = EXPANDED_DIR / "manifest.json"
        if exp_manifest_src.exists():
            manifest.extend(json.loads(exp_manifest_src.read_text()))
        (COMBINED_DIR / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        print(f"  Combined manifest has {len(manifest)} entries")

    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--sample-frac", type=float, default=0.15)
    parser.add_argument("--serve-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.serve_only:
        print("=" * 60)
        print("  Step 1: Building combined candidate directory")
        print("=" * 60)
        total = build_combined_candidate_dir()
        print()

        # Import and run SubspaceAD
        from scripts.run_subspace_ad_review import (
            OUTPUT_DIR as _IGNORED,
            DEFAULT_NEGATIVE_DIR as _UNUSED,
            DEFAULT_CANDIDATE_DIR as _UNUSED2,
            DEFAULT_PORT as _UNUSED3,
            _resolve_device,
            batch_process,
            generate_review_page,
            save_manifest,
            start_server,
            train_pca_on_negatives,
        )
        import webbrowser

        device = _resolve_device(args.device)
        negative_dir = str(DEFAULT_NEGATIVE_DIR)
        candidate_dir = str(COMBINED_DIR)
        output_dir = str(OUTPUT_DIR)

        print("=" * 60)
        print("  Step 2: Training PCA on negatives")
        print("=" * 60)
        train_result = train_pca_on_negatives(
            negative_dir,
            n_components=args.n_components,
            device=device,
            sample_frac=args.sample_frac,
        )
        if "error" in train_result:
            print(f"ERROR: PCA training failed: {train_result['error']}")
            return 1
        pca_model = train_result["pca_model"]
        print()

        print("=" * 60)
        print("  Step 3: Scoring all 931 candidates & generating overlays")
        print("=" * 60)
        manifest = batch_process(pca_model, candidate_dir, output_dir, device=device)
        if not manifest:
            print("ERROR: No candidates were successfully scored.")
            return 1
        print(f"  Scored {len(manifest)} candidates")
        print()

        print("=" * 60)
        print("  Step 4: Generating review page")
        print("=" * 60)
        generate_review_page(manifest, output_dir, train_meta=train_result)
        print()

        print("=" * 60)
        print("  Step 5: Saving manifest")
        print("=" * 60)
        save_manifest(manifest, output_dir, train_meta=train_result)
        print()

        # Summary
        top = manifest[:10]
        print("=" * 60)
        print("  Top 10 Candidates by SubspaceAD Score")
        print("=" * 60)
        for i, c in enumerate(top, 1):
            region = c["filename"].split("_")[0] if "_" in c["filename"] else "?"
            print(f"  {i}. [{region}] score={c['score_top10p']:.6f}  "
                  f"area={c['spawn_area_frac']*100:.1f}%  "
                  f"patches={c['n_spawn_patches']}/{c['n_patches']}")
        print()

        # Score distribution
        area_buckets = {">5%": 0, "1-5%": 0, "<1%": 0}
        for c in manifest:
            af = c.get("spawn_area_frac", 0)
            if af > 0.05:
                area_buckets[">5%"] += 1
            elif af > 0.01:
                area_buckets["1-5%"] += 1
            else:
                area_buckets["<1%"] += 1
        print("  Spawn area distribution:")
        for k, v in area_buckets.items():
            print(f"    {k}: {v} candidates")
        print()

    # Start server
    from scripts.run_subspace_ad_review import start_server

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not (OUTPUT_DIR / "review.html").exists():
        print(f"ERROR: No review page found in {OUTPUT_DIR}")
        print("  Run without --serve-only to generate the review first.")
        return 1

    print(f"  Serving from {OUTPUT_DIR}")
    print(f"  URL: http://localhost:{args.port}/review.html")
    print(f"  Press Ctrl+C to stop.\n")

    server = start_server(str(OUTPUT_DIR), args.port)

    if not args.no_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:{args.port}/review.html")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
