#!/usr/bin/env python3
"""
Export labeled spawn dataset: locations, scenes, and commands to reproduce.

Outputs:
  /Volumes/Z Slim/herring-spawn-data/candidates_fresh/labeled_dataset.csv
    — One row per labeled image with DFO record + Sentinel-2 scene metadata

Usage:
  .venv/bin/python3 scripts/export_labeled_dataset.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

IMAGE_DIR = Path("/Volumes/Z Slim/herring-spawn-data/candidates_fresh")
MANIFEST_PATH = IMAGE_DIR / "manifest.json"
LABELS_PATH = IMAGE_DIR / "labels.json"
OUTPUT_CSV = IMAGE_DIR / "labeled_dataset.csv"

# Map DFO region codes to full names
REGION_NAMES = {
    "A27": "Area 27",
    "A2W": "Area 2W",
    "CC": "Central Coast",
    "HG": "Haida Gwaii",
    "PRD": "Prince Rupert District",
    "SoG": "Strait of Georgia",
    "WCVI": "West Coast Vancouver Island",
}


def main():
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Build lookup
    lookup = {e["filename"]: e for e in manifest if e.get("filename")}

    # Build rows
    rows = []
    for fname, label in sorted(labels.items()):
        if label == "skip" or fname not in lookup:
            continue
        entry = lookup[fname]
        rows.append({
            "filename": fname,
            "label": label,
            "region": entry.get("region", ""),
            "region_name": REGION_NAMES.get(entry.get("region", ""), ""),
            "location_name": entry.get("location_name", ""),
            "dfo_date": entry.get("date", ""),
            "scene_date": entry.get("scene_date", ""),
            "scene_id": entry.get("scene_id", ""),
            "lat": entry.get("lat", ""),
            "lon": entry.get("lon", ""),
            "cloud_cover": entry.get("cloud_cover", ""),
            "days_from_spawn": entry.get("days_from_spawn", ""),
            "spawn_length_m": entry.get("spawn_length_m", ""),
            "spawn_width_m": entry.get("spawn_width_m", ""),
            "survey_method": entry.get("method", ""),
            "spawn_score": entry.get("spawn_score", ""),
        })

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Stats
    c = Counter(labels.values())
    spawns = len([r for r in rows if r["label"] == "spawn"])
    nospawns = len([r for r in rows if r["label"] == "no-spawn"])
    regions = Counter(r["region"] for r in rows)

    print(f"Exported {len(rows)} labeled records to {OUTPUT_CSV}")
    print(f"  Spawn:    {spawns}")
    print(f"  No-spawn: {nospawns}")
    print(f"  Regions:  {dict(regions)}")
    print()
    print(f"Next: push metadata + scripts to GitHub, then build temporal model.")


if __name__ == "__main__":
    main()
