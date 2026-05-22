# Herring Spawn Detection — Project Status & Plan

## Repository
https://github.com/dexterfichuk/herring-spawner

## What's Built

### Pipeline
- `scripts/run_gee_search.py` — Search Sentinel-2 for known events, download thumbnails
- `scripts/run_embeddings.py` — DINOv2 embedding ranking with positive/negative scoring
- `scripts/run_clay_multispectral.py` — Clay v1.5 encoder on multi-spectral GeoTIFF chips
- `scripts/download_and_review.py` — Batch thumbnail download + review page generator
- `scripts/label_images.py` — Terminal-based labeling tool
- `scripts/build_event_catalog.py` — Combines DFO + manual + track events into GeoJSON

### Data
- `data/samples/positive/` — 14 confirmed spawn images (user-validated)
- `data/samples/negative/` — 40 confirmed non-spawn images (user-validated)
- `data/review/thumbnails/` — 27 thumbnails from initial 11 DFO/manual events
- `data/review/thumbnails2/` — 47 thumbnails from 30 BC + WA events
- `data/chips/` — 20 multi-spectral GeoTIFF chips for Clay
- `data/embeddings/` — All embedding vectors saved as npz

### Model Performance
- **DINOv2 similarity**: 88.9% accuracy (48/54), 0.0607 separation, 14 spawn + 40 no spawn
- **DINOv2 + SVM**: 84.3% ± 8.3% 5-fold CV, 1.4811 separation, 98.6% full-dataset accuracy (20 pos + 50 neg)
- **Clay v1.5**: 0.0951 separation, needs more labeled multi-spectral data

### Review Pages
- `data/review/interactive_review.html` — Working interactive labeler (batch 1)
- `data/review/label.html` — Interactive labeler served via HTTP at :8766
- `data/review/review2_ref.html` — Zero-JS reference (numbers only)
- `data/review/combined_results.html` — Clay + DINOv2 side-by-side

## Current Approach

Given the small dataset (54 labeled, 14 spawn), DINOv2 on RGB thumbnails outperforms Clay (0.0607 vs 0.0465 separation). We use:
1. DINOv2 ViT-S/14 for embeddings (384-dim, 84MB model, fast on CPU)
2. Mean of 14 positive embeddings as reference vector
3. Cosine similarity to reference minus similarity to negative mean
4. Threshold at 0 (zero) gives 88.9% accuracy

## Next Phase: BC Coast Scanning

### Goal
Scan the entire BC coastline during herring spawn season (Feb-April) to find new spawn events.

### Method
1. Generate ~5,000 sampling points along the BC coastline
2. For each point, check Sentinel-2 scenes during spawn window
3. Download RGB thumbnail → run DINOv2 → score
4. Score above threshold? Save as candidate. Below? Discard immediately.
5. Present candidates for human review

### Coastline Sampling
- Use Natural Earth or DFO coastline data
- Generate points at 500m-1km intervals along coastline
- Only include points in known herring habitat (sheltered bays, inlets, ≤50m depth)
- Focus on March-April window (peak spawn)
- Points should be ~50-100m offshore (not on land)

### Candidate Criteria
- DINOv2 score > 0.0 (above zero threshold)
- Cloud < 50%
- Scene within ±14 days of expected spawn window
- Not already in our known events

### Storage
- No storage of non-candidate imagery
- Candidates stored as: `data/candidates/{event_id}_{date}_{score}.png`
- Candidate manifest: `data/candidates/manifest.json`
- Each candidate includes: lat, lon, date, score, scene_id, thumbnail path

### Estimated Scale
- 5,000 coastal points × 2-3 clear scenes each = ~12,500 thumbnails processed
- At ~20% candidate rate: ~2,500 candidates to review
- GEE getThumbURL calls: ~12,500 (free within quota)
- Processing time: ~1-2 hours (DINOv2 is fast on CPU)

### Running
```bash
# Scan with SVM classifier (better precision)
python scripts/scan_bc_coast.py \
  --output data/candidates_v2 \
  --classifier svm \
  --start 2024-03-01 \
  --end 2024-04-15 \
  --max-cloud 50 \
  --grid-spacing 0.02 \
  --workers 8

# Review candidates
python -m http.server 8766 --directory data/candidates_v2
# Then open http://localhost:8766/review.html
```

## Future Work
- More labels → train SVM on embeddings → higher accuracy
- Full Clay pipeline with proper GeoTIFF exports
- Multi-year scanning (2023, 2024, 2025)
- Web dashboard for candidate review
- Kelp forest detection adaptation
