# Herring Spawn Detection — Agent Context

## Repository
https://github.com/dexterfichuk/herring-spawner

For the detailed audit and current handoff, read `docs/agent_handoff.md` first.

## What's Built

### Pipeline
- `scripts/label_gradio.py` — **Current labeling app**: Gradio with Single Image + Grid Scan tabs, keyboard shortcuts, auto-save JSON. Replaces all Flask labeling apps.
- `scripts/consolidate_candidates.py` — MD5-deduplicate all candidate directories into `data/unified/`, preserving labels from 6+ sources
- `scripts/scan_shsi_candidates.py` — SHSI (Green²/Red) Earth Engine pre-screener: scans BC regions, downloads top-N candidate thumbnails, builds review page
- `scripts/scan_bc_coast_knn.py` — KNN/DINOv2 scan pipeline for BC habitat regions
- `scripts/scan_bc_coast.py` — Original BC coast scan with DINOv2/SVM scoring
- `scripts/train_from_manifest.py` — Train DINOv2 SVM from training_manifest.json
- `scripts/final_bc_sweep.py` — Rose-verified training sweep, model retrain, temporal review orchestration
- `scripts/ingress_dfo_gee_search.py` — DFO event Sentinel-2 thumbnail ingress and review page builder
- `scripts/knn_detector.py` — DINOv2 KNN voting evaluation and report builder
- `scripts/delta_detector.py` — DINOv2 paired-temporal delta detector (100% LOO CV on 37 locations)
- `scripts/scan_13regions_subspacead.py` — Multi-year SubspaceAD scan of all 13 BC habitat regions
- `scripts/subspace_ad.py` — DINOv2 SubspaceAD anomaly detector
- `scripts/patch_subspace_ad.py` — Patch-level SubspaceAD with spatial segmentation
- `scripts/fewshot_subspace_variants.py` — DINOv2 few-shot SubspaceAD variants
- `scripts/temporal_detector.py` — Before-during-after triplet delta and intra-season timeseries detector
- `scripts/temporal_v2.py` — Five temporal modes: outlier, YOY, spectral, trajectory, cloud-fusion
- `scripts/environmental_matcher.py` — Sun+tide-matched embedding outlier detection per location
- `scripts/multi_year_confirm.py` — Multi-year candidate confirmation with 0.01° coordinate grouping
- `scripts/prompt_detector.py` — SenCLIP/RemoteCLIP contrastive prompt scoring with multi-scale crops
- `scripts/benchmark_prompt_models.py` — Prompt model benchmark on golden set
- `scripts/rs_foundation_probe.py` — frozen-embedding linear probe benchmark for remote-sensing foundation backbones
- `scripts/segment_spawn.py` — DINOv3 + SAM interactive click-to-segment web app
- `scripts/dinov3_feature_extractor.py` — DINOv3 ViT-L/16 SAT-493M feature extraction with similarity heatmaps
- `scripts/build_event_catalog.py` — Combines DFO, manual, and track events into GeoJSON
- `scripts/download_and_review.py` — Batch thumbnail download and review page generator
- `scripts/run_gee_search.py` — Search Sentinel-2 for known events, download thumbnails
- `scripts/run_embeddings.py` — DINOv2 embedding ranking with positive/negative scoring
- `scripts/run_clay_multispectral.py` — Clay v1.5 encoder on multispectral GeoTIFF chips

### Data
- `data/unified/` — **Current unified dataset**: 4,605 unique images (MD5-deduplicated from all sources), symlinked from `thumbs/`, manifest + labels in root. Point Gradio at this.
- `data/candidates_shsi/` — SHSI pipeline output: 150 candidates from 2024 scan
- `data/olmoearth_chips/` — 12-band Sentinel-2 GeoTIFF chips (npy) for OlmoEarth embedding
- `data/samples/positive/` — Positive training thumbnails; see `data/samples/training_manifest.json`
- `data/samples/negative/` — Negative training thumbnails (164 files)
- `data/samples/rejected/` — Explicitly rejected images (11 files)
- `data/candidates_v2/` — Earlier SVM candidate set, review pages, labels, and temporal artifacts
- `data/candidates_knn/` — KNN scan output: 725 candidates from 2,863 scanned points
- `data/candidates_final/` — Final SVM sweep metadata and generated review artifacts
- `data/sog_candidates/` — Strait of Georgia candidate thumbnails: 452 thumbnails from 333 filtered records
- `data/ingressed/` — DFO/external ingressed records, thumbnails, manifests, and review pages
- `data/models/` — DINOv2 SVM and improved feature model artifacts
- `data/chips/` and `data/embeddings/` — Clay/DINO intermediate artifacts
- Public generated-image dataset — `https://huggingface.co/datasets/dfichuk/herring-spawn-candidates`

### Model Benchmarks (2026-06-20, on full unified dataset: 588 images, 28 spawn / 560 no-spawn)

All benchmarks use KNN (cosine metric) with LOO cross-validation unless noted.

| Rank | Model | Dim | AUROC | AP | Input | Notes |
|---|---|---|---|---|---|---|
| **1** | **GeoRSCLIP ViT-B-32** | 512 | **0.969** | 0.653 | RGB thumbnails (512px, 0-3000 stretch) | Trained on RS5M remote sensing dataset. Load via `open_clip` from `Zilun/GeoRSCLIP`. Also available: ViT-L-14 variant. |
| 2 | DINOv2 ViT-S/14 | 384 | 0.900 | 0.323 | RGB thumbnails | Previously 0.981 on 16/164 golden set. Drops on harder 588-image set. |
| 3 | DINOv3 ViT-L/16 SAT-493M | 1024 | 0.875 | 0.266 | RGB thumbnails | Satellite-pretrained, but tuned for segmentation not classification. |
| 4 | OlmoEarth v1.1 Nano | 128 | 0.842 | 0.744 | 12-band S2 GeoTIFF chips | Requires full multi-spectral download. 1.7M params. |
| 5 | OlmoEarth v1.1 Base | 768 | 0.831 | 0.715 | 12-band S2 GeoTIFF chips | 88.8M params but underperforms Nano — missing modalities confuse it. |
| — | SHSI (Green²/Red) | 1 | — | — | S2 bands in GEE | 44% recall at best threshold (0.02). Sediment confound. Screening only. |

**Imagery quality tests**: Higher resolution (1024px), different rendering stretches (0-500, 0-1000), percentile stretch, and raw GeoTIFF RGB all scored LOWER than current 512px/0-3000 thumbnails. Current approach is optimal.

**Key insight**: GeoRSCLIP's remote-sensing-specific pretraining on RS5M gives it the edge over general-purpose DINOv2 on this task. It also works on simple RGB thumbnails — no need for multi-band GeoTIFF downloads.

### Labeling Infrastructure

**Gradio app** (`scripts/label_gradio.py`) replaces all 6 old Flask labeling apps:
- **Single Image tab**: metadata sidebar, spawn/no-spawn/skip buttons, keyboard shortcuts (Y/N/S/arrows)
- **Grid Scan tab**: 48-image grid, click=spawn, right-click=no-spawn, paginated
- Auto-saves `labels.json` on every decision, resumes from where you left off
- Pre-loaded with existing labels from 6 sources (golden set, KNN, SoG, dual-subspace, SHSI, Rose)

```bash
source .venv/bin/activate
python scripts/label_gradio.py \
    --manifest data/unified/manifest.json \
    --image-dir data/unified/thumbs \
    --labels data/unified/labels.json \
    --port 7888
# Open http://localhost:7888
```

### SHSI Earth Engine Pipeline

`scripts/scan_shsi_candidates.py` implements the Spectral Herring Spawning Index (SHSI = Green²/Red) from UVic research:
- Scans BC habitat regions via GEE, computes raw SHSI at each grid point
- No pre-filters (NIR<0.025 kills spawn since milt reflects NIR)
- Threshold 0.02 for high recall, downloads top-N candidate thumbnails
- Generates `review.html` card grid for manual review
- 150 candidates from 2024 scan, 4 spawns found (all in Milbanke/Nootka)

### Human-Reviewed Positives (28 total, up from 16)

**Unified set** at `data/unified/`:
- 28 spawn positives, 340 no-spawn negatives, 1 skip (as of 2026-06-20)
- Labels consolidated from: golden set, KNN silo, SoG silo, dual-subspace review, SHSI labeling, Rose review
- Remaining: 4,245 unlabeled images to work through

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

7. Segmentation: use `segment_spawn.py` to produce pixel-level binary masks. Click suspected spawn regions → DINOv3 ViT-L/16 SAT-493M computes cosine-similarity heatmap → SAM refines to pixel mask. Output saved to `data/segmentation_masks/`.

Do not rely on single-image model score alone for final truth. The model can learn shoreline, surf, sediment, and bright beach patterns.

## Segmentation Tool

Run the interactive click-to-segment web app:

```bash
source .venv/bin/activate
python scripts/segment_spawn.py \
  --image-dir data/samples/positive \
  --output-dir data/segmentation_masks \
  --sam-checkpoint data/models/sam_vit_h_4b8939.pth \
  --device auto \
  --port 8777

# Then open http://localhost:8777
# Click on spawn turbidity → green overlay appears → Accept (A) or Reject (R)
# Arrow keys navigate images, C clears mask, toggle checkbox hides overlay
```

Masks saved as PNGs in `data/segmentation_masks/` with a `manifest.json` tracking accept/reject labels.

### Model Storage

Large model files are stored on external drive `/Volumes/Z Slim` to keep the main disk lean:
- `/Volumes/Z Slim/herring-models/` — SAM checkpoint (702MB), DINOv3 SAT model (1.1GB), HF/torch caches (18GB)
- Symlinks from `data/models/` and `~/.cache/` point to Z Slim
- All model downloads auto-cache to Z Slim via symlinked cache dirs

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

## Session Status (2026-06-20)

### Accomplished Today
- **Unified labeling infrastructure**: Built `consolidate_candidates.py` (MD5-deduplicates all candidate dirs into `data/unified/` — 4,605 unique images) and `label_gradio.py` (Gradio app with Single Image + Grid Scan tabs, keyboard shortcuts, auto-save JSON). Removed 6 old Flask labeling apps.
- **SHSI Earth Engine pipeline**: Ported and ran the UVic SHSI (Green²/Red) detector in GEE. Found best config: raw SHSI at threshold 0.02 with no pre-filters (NIR kills spawn). 150 candidates from 2024 scan, 4 spawns found.
- **Model benchmarks on full unified set** (588 images, 28 spawn / 560 no-spawn):
  - **GeoRSCLIP ViT-B-32**: **AUROC 0.969**, AP 0.653 — new champion, beats DINOv2 on hard set
  - DINOv2 ViT-S/14: AUROC 0.900, AP 0.323 — dropped from 0.981 on easier golden set
  - DINOv3 ViT-L/16 SAT: AUROC 0.875, AP 0.266 — tuned for segmentation, not classification
  - OlmoEarth v1.1 Nano: AUROC 0.842, AP 0.744 (on 77 S2 chips)
  - OlmoEarth v1.1 Base: AUROC 0.831, AP 0.715 (underperforms Nano — missing modalities hurt)
- **Imagery quality tests**: Current 512px thumbnails with 0-3000 rendering are optimal. Higher res, different stretches, and raw GeoTIFFs all scored lower.
- **GeoRSCLIP model integration**: Loaded via `open_clip` from `Zilun/GeoRSCLIP` (trained on RS5M dataset). Works on RGB thumbnails — no S2 chip downloads needed.
- **Extended golden set**: 28 spawn positives (up from 16), 340 no-spawn negatives, 1 skip. Consolidated from 6 label sources.
- **OlmoEarth evaluation**: Built custom encoder for both Nano and Base variants. Requires 12-band S2 GeoTIFF chips (77 chips downloaded). Base model needs full multi-modal input to perform well.
- **Commit and push**: New labeling infrastructure committed, model blobs gitignored.

### Key Findings
1. **GeoRSCLIP is the new champion** — remote-sensing-specific pretraining on RS5M gives it the edge. AUROC 0.969 vs DINOv2's 0.900 on the full 588-image set.
2. **Current imagery is optimal** — 512px RGB thumbnails with 0-3000 GEE rendering. Tested higher res, different stretches, raw GeoTIFFs — all worse.
3. **SHSI is a weak screener** — max 44% recall on known positives. Sediment/turbidity confound. Best used as coarse pre-filter before DINOv2/GeoRSCLIP.
4. **OlmoEarth needs data we don't have** — requires multi-modal (S1+S2+Landsat+SRTM+landcover), multi-temporal input. Single-sensor S2-only gives AUROC 0.831.
5. **DINOv3 SAT underperforms** — designed for dense segmentation tasks, not image-level classification.

### Recommended Next Steps
1. **Active learning scan**: Use GeoRSCLIP to scan BC coast, review top candidates, add to golden set, retrain
2. **Try GeoRSCLIP ViT-L-14**: Larger variant in `Zilun/GeoRSCLIP` (`ckpt/RS5M_ViT-L-14.pt`) — may push AUROC above 0.97
3. **Multi-scale crop ensembling**: Full + center + tight crops for small spawn plumes
4. **Landsat integration**: Add Landsat 8/9 thumbnails for locations where S2 is cloudy (2.4x more scenes)
5. **Label remaining 4,245 unlabeled**: Use Grid Scan tab for rapid visual review
6. **Update AGENTS.md with all model benchmark results**
