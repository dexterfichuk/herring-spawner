# BC Coast Scanning Implementation Plan

> **For agentic workers:** Single-script implementation. No subagent dispatch needed.

**Goal:** Create `scripts/scan_bc_coast.py` that scans BC coastline regions for new herring spawn events using our trained DINOv2 model.

**Architecture:** Standalone CLI script that (1) generates grid points in defined herring habitat regions, (2) searches Sentinel-2 via GEE for each point, (3) downloads RGB thumbnails, (4) scores with DINOv2 reference vectors, (5) saves only candidates above threshold.

**Tech Stack:** Python 3.11+, earthengine-api, torch+torchvision, numpy, Pillow, requests, shapely

---

### Task 1: Create `scripts/scan_bc_coast.py`

**Files:**
- Create: `scripts/scan_bc_coast.py`

**Key design decisions:**
- Reference vectors computed once from labeled samples (`data/samples/positive/`, `data/samples/negative/`) on first run, cached as `data/embeddings/reference_vectors.npz`
- DINOv2 ViT-S/14 loaded once, runs on CPU (matches existing pipeline)
- Grid points generated at `--grid-spacing` intervals within each region's radius
- For each point, best scene selected: lowest cloud in the date range. If no scene with cloud < `--max-cloud`, skip point.
- Thumbnail downloaded via GEE `getThumbURL` (512x512, RGB bands B4/B3/B2)
- Score `= cosine_similarity(embedding, mean_positive) - cosine_similarity(embedding, mean_negative)`
- Only candidates with score > `--threshold` saved to disk
- Manifest updated atomically (read-modify-write per candidate)
- Rich progress output with timing estimates

**Dependencies (already in project):** earthengine-api, numpy, shapely, Pillow
**Runtime dependencies (not in pyproject but used by other scripts):** torch, torchvision, requests

**Scoring reference: from AGENTS.md:**
- DINOv2 ViT-S/14 for embeddings (384-dim, 84MB model, fast on CPU)
- Mean of positive embeddings as reference vector
- Cosine similarity to reference minus similarity to negative mean
- Threshold at 0 gives ~88.9% accuracy

**Implementation:**

```python
#!/usr/bin/env python3
"""Scan BC coastline for new herring spawn events using DINOv2 scoring."""
```

The script follows this flow:
1. Parse CLI args (output dir, threshold, date range, max cloud, grid spacing)
2. Initialize GEE with `redd-fish` project
3. Compute reference vectors from labeled samples (with cache)
4. Generate grid points from defined regions
5. Process each point:
   a. Search Sentinel-2 for scenes in date range with cloud < max
   b. Select best scene (lowest cloud)
   c. Download RGB thumbnail via getThumbURL
   d. Run DINOv2 embedding and score
   e. If score > threshold: save candidate PNG + update manifest row
   f. Otherwise: discard (never written to disk)
6. Print summary (processed, candidates, time, positive rate)

**Region definitions** (mapped to sheltered herring habitat bays/inlets):

| Region | Lat | Lon | Radius(km) | Notes |
|--------|-----|-----|-----------|-------|
| qualicum | 49.35 | -124.45 | 15 | Strait of Georgia hot spot |
| nanaimo | 49.15 | -123.85 | 15 | Strait of Georgia hot spot |
| comox | 49.68 | -124.88 | 15 | Strait of Georgia hot spot |
| denman-island | 49.55 | -124.80 | 10 | Between Denman/Vancouver Is |
| tofino | 49.15 | -125.90 | 15 | WCVI |
| ucluelet | 48.94 | -125.55 | 10 | WCVI |
| nootka-sound | 49.60 | -126.60 | 15 | WCVI |
| quatsino-sound | 50.50 | -128.00 | 15 | North WCVI |
| spiller-channel | 52.30 | -128.30 | 15 | Central Coast |
| milbanke-sound | 52.50 | -128.80 | 15 | Central Coast |
| prince-rupert | 54.30 | -130.40 | 20 | North Coast |
| haida-gwaii-south | 52.40 | -131.40 | 15 | Haida Gwaii |
| masset-inlet | 53.70 | -132.90 | 15 | Haida Gwaii |

**Estimated scale:**
- 13 regions, each ~300 grid points at 0.01° (~1km) spacing within 15km radius ≈ ~4,000 total points
- 2-3 clear scenes per point → would be crazy to check all scenes for all points
- Optimized: per point, ONLY the single best scene (lowest cloud)
- At ~4,000 points × 1 best scene = ~4,000 thumbnails max
- Realistically: many points share the same scenes, GEE handles this
- ~20% candidate rate → ~800 candidates
- DINOv2: ~0.1s per image on CPU → ~7 min total
- Thumbnail downloads: ~0.5-2s each → could be bottleneck; add concurrent downloads later if needed

**Edge cases handled:**
- Region with zero matching scenes → skipped with warning
- Point at sea with no nearby Sentinel-2 coverage → skipped
- GEE quota exceeded → graceful error with partial results saved
- Empty samples directories → clear error message before computation
- Corrupt PNG samples → skipped with warning
- File exists on thumbnail download → skip re-download
- Manifest file doesn't exist yet → create with header
- KeyboardInterrupt during long scan → saves progress so far

**Cache for reference vectors:**
Save as `data/embeddings/reference_vectors.npz` with keys:
- `mean_pos`: (384,) float32
- `mean_neg`: (384,) float32
- `n_pos`: int
- `n_neg`: int
- `separation`: float (difference between mean scores of training sets)

Invalidate when sample files change (check file count + modification times).

This keeps the script reusable without recomputing embeddings every run.
