# Herring Spawn Detection — Agent Context

## Repository
https://github.com/dexterfichuk/herring-spawner

For the detailed audit and current handoff, read `docs/agent_handoff.md` first.

## What's Built

### Pipeline
- `scripts/scan_bc_coast_knn.py` — Current KNN/DINOv2 scan pipeline for BC habitat regions
- `scripts/scan_bc_coast.py` — Original BC coast scan with DINOv2/SVM scoring
- `scripts/final_bc_sweep.py` — Rose-verified training sweep, model retrain, temporal review orchestration
- `scripts/ingress_dfo_gee_search.py` — DFO event Sentinel-2 thumbnail ingress and review page builder
- `scripts/knn_detector.py` — DINOv2 KNN voting evaluation and report builder
- `scripts/run_gee_search.py` — Search Sentinel-2 for known events, download thumbnails
- `scripts/run_embeddings.py` — DINOv2 embedding ranking with positive/negative scoring
- `scripts/run_clay_multispectral.py` — Clay v1.5 encoder on multispectral GeoTIFF chips
- `scripts/download_and_review.py` — Batch thumbnail download and review page generator
- `scripts/label_images.py` — Terminal-based labeling tool
- `scripts/build_event_catalog.py` — Combines DFO, manual, and track events into GeoJSON

### Data
- `data/samples/positive/` — Current positive training thumbnails; see `data/samples/training_manifest.json`
- `data/samples/negative/` — Current negative training thumbnails
- `data/candidates_v2/` — Earlier SVM candidate set, review pages, labels, and temporal artifacts
- `data/candidates_knn/` — Current KNN scan output: 725 candidates from 2,863 scanned points
- `data/candidates_final/` — Final SVM sweep metadata and generated review artifacts
- `data/sog_candidates/` — Strait of Georgia candidate thumbnails: 452 thumbnails from 333 filtered records
- `data/ingressed/` — DFO/external ingressed records, thumbnails, manifests, and review pages
- `data/models/` — DINOv2 SVM and improved feature model artifacts
- `data/chips/` and `data/embeddings/` — Clay/DINO intermediate artifacts
- Public generated-image dataset — `https://huggingface.co/datasets/dfichuk/herring-spawn-candidates`

### Model Performance
- **Current DINOv2 + SVM**: 94.5% mean CV, 1.8658 separation, trained on 5 rose-verified positives + 50 negatives
- **KNN DINOv2 scan**: 205 training labels, 2,863 points scanned, 725 candidates saved in `data/candidates_knn/`
- **Improved feature model**: 85.1% mean CV, 97.0% combined-feature accuracy on 67 labeled samples
- **Earlier DINOv2 similarity**: 88.9% accuracy on 54 labeled thumbnails, useful as historical baseline only
- **Clay/delta direction**: preferred research path because paired temporal/multispectral change reduces shoreline bias

### Human-Reviewed Positives
- Current final-sweep human-reviewed positives are exactly the 5 Rose-verified files in `data/samples/training_manifest.json`.
- The filenames are:
  - `qualicum_2024-03-18_score0.01_49.254865_-124.497442_20240318.png`
  - `tofino_2024-03-16_score0.00_49.114865_-125.806603_20240316.png`
  - `tofino_2024-03-16_score0.01_49.194865_-126.026603_20240316.png`
  - `nootka-sound_2024-03-16_score0.00_49.584865_-126.528503_20240316.png`
  - `nootka-sound_2024-02-12_score0.00_49.564865_-126.508503_20240212.png`
- Model-ranked candidates, high-confidence score buckets, temporal positives, and generated review pages are not human-reviewed positives unless they also appear in a human label file.
- Treat `rose_*`, `silo_labels.json`, and other explicit review-label files as provenance sources; preserve reviewer/source names when promoting labels into training manifests.

### Review Pages
- `data/candidates_knn/review.html` — Current KNN candidate review page
- `data/sog_candidates/review.html` and `data/sog_candidates/top.html` — Strait of Georgia candidate reviews
- `data/ingressed/review.html` and `data/ingressed/label.html` — DFO/external ingress review pages
- `data/candidates_v2/review.html`, `koko_review.html`, `rose_spawns.html`, `temporal_review.html` — Earlier and temporal review pages
- `data/review/` — Earlier review experiments, ignored by git by default
- `data/candidates_final/review.html` — Final sweep review page, uploaded to Hugging Face when generated

## Current Approach

Use DINOv2 thumbnail models to triage candidates, then require temporal support or human review before treating anything as a real spawn.

1. Build candidates from known event records or BC habitat grid points.
2. Download Sentinel-2 RGB thumbnails from Earth Engine project `redd-fish`.
3. Embed thumbnails with DINOv2 ViT-S/14.
4. Rank with KNN or SVM using current human-reviewed labels.
5. Store candidate thumbnails, manifests, summaries, and static review pages.
6. Confirm with temporal repeatability, paired deltas, or human labels.

Do not rely on single-image model score alone for final truth. The model can learn shoreline, surf, sediment, and bright beach patterns.

## Next Phase: BC Coast Scanning

### Goal
Scan the entire BC coastline during herring spawn season (Feb-April) to find new spawn events.

### Method
1. Generate sampling points across the 13 known BC herring habitat regions.
2. For each point, check Sentinel-2 scenes during the spawn window.
3. Download RGB thumbnail and run DINOv2 embedding.
4. Classify with KNN/SVM and save only candidate thumbnails.
5. Present candidates for human review and temporal validation.

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
- Image-heavy generated assets should be uploaded to Hugging Face with `scripts/upload_hf_dataset.py` instead of committed to GitHub.

### Estimated Scale
- 5,000 coastal points × 2-3 clear scenes each = ~12,500 thumbnails processed
- At ~20% candidate rate: ~2,500 candidates to review
- GEE getThumbURL calls: ~12,500 (free within quota)
- Processing time: ~1-2 hours (DINOv2 is fast on CPU)

### Running
```bash
source .venv/bin/activate
python scripts/scan_bc_coast_knn.py \
  --output data/candidates_knn \
  --start 2024-02-01 \
  --end 2024-05-31 \
  --max-cloud 50 \
  --grid-spacing 0.02 \
  --workers 6 \
  --k 3

python -m http.server 8766 --directory data/candidates_knn
# Then open http://localhost:8766/review.html
```

## Commit/Storage Notes

- Commit scripts, docs, tests, manifests, labels, model summaries, and reasonably sized review artifacts.
- Do not commit `.venv/`, caches, raw `checkpoints/`, generated candidate imagery, or files larger than GitHub's normal 100 MB limit unless Git LFS is configured.
- Upload image-heavy generated outputs to `dfichuk/herring-spawn-candidates`:

```bash
source .venv/bin/activate
python -m pip install huggingface_hub
huggingface-cli login
python scripts/upload_hf_dataset.py --repo-id dfichuk/herring-spawn-candidates
```

- Large generated temporal artifacts in `data/candidates_v2/` should remain local unless explicitly moved to LFS or external storage.

## Future Work
- **Delta-based approach** — Instead of scoring single images, compare pre-spawn baseline vs spawn-season imagery at each location. A spawn event = large embedding change; barren shoreline = minimal change.
- Full Clay pipeline with proper GeoTIFF exports
- Multi-year scanning and review consolidation across 2023-2026
- Web dashboard for candidate review
- Kelp forest detection adaptation
