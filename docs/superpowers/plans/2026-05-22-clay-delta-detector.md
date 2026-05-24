# Clay Embedding Delta Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a script that computes Clay v1.5 embedding deltas between spawn-season and off-season Sentinel-2 chips, comparing spawn vs non-spawn locations to determine if delta-based detection works better than DINOv2 single-image scoring.

**Architecture:** Single Python script `scripts/clay_delta_detector.py` that (1) parses spawn/non-spawn locations from `rose_200_labels.json`, (2) downloads paired GeoTIFF chips (spawn-season + off-season) for each location via GEE, (3) runs Clay v1.5 on both to compute delta vectors, (4) analyzes deltas statistically, and (5) generates an HTML report.

**Tech Stack:** Earth Engine Python API, Clay v1.5 (`claymodel`), NumPy, scikit-learn (PCA), Plotly (interactive HTML report)

---

### Task 1: Create `scripts/clay_delta_detector.py` — full implementation

**Files:**
- Create: `scripts/clay_delta_detector.py`
- Creates: `data/review/clay_delta_report.html`

- [ ] **Step 1: Write the complete script**

The script does the following:

1. Parse `rose_200_labels.json` — extract all entries with `spawn: true` (6 locations) and pick 10 `spawn: false` entries with spread-out coordinates as non-spawn control
2. For each location:
   - Download a spawn-season GeoTIFF (B2/B3/B4/B8) from GEE around March 2024 spawn window
   - Download an off-season GeoTIFF (same bands) from July 2024  
3. Load Clay v1.5 model (same as `run_clay_multispectral.py`)
4. For each chip, compute embedding via Clay encoder
5. Compute delta = spawn_emb - off_emb for each location
6. Compare spawn vs non-spawn: delta magnitudes, PCA, spectral band analysis
7. Generate `data/review/clay_delta_report.html` with Plotly visualizations:
   - Box plot of delta magnitudes (spawn vs non-spawn)
   - PCA scatter of delta vectors colored by spawn/non-spawn
   - Bar chart of top discriminating embedding dimensions
   - Summary statistics table

Key implementation details:
- Follow the exact same datacube format as `run_clay_multispectral.py` for Clay input
- Use `ee.Initialize(project="redd-fish")` for GEE
- Caching: reuse existing chips from `data/chips/` if they match, save new chips to `data/chips_delta/`
- Handle missing scenes gracefully (skip location, report in output)
- Filename extraction regex: `{name}_{date}_score{score}_{lat}_{lon}_{satdate}.png`

- [ ] **Step 2: Run the script**

```bash
python scripts/clay_delta_detector.py 2>&1 | tail -80
```

Expected: Script runs, downloads chips for ~16 locations, runs Clay, prints comparison stats, saves report to `data/review/clay_delta_report.html`

- [ ] **Step 3: Verify the report**

Check that `data/review/clay_delta_report.html` exists and contains the expected visualizations. Report the key finding: does Clay delta separate spawn from non-spawn better than DINOv2 single-image?
