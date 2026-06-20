# Herring Spawn Detection — Agent Context

## Repository
https://github.com/dexterfichuk/herring-spawner

For the detailed audit and current handoff, read `docs/agent_handoff.md` first.

## What's Built

### Pipeline
- `scripts/scan_bc_coast_knn.py` — Current KNN/DINOv2 scan pipeline for BC habitat regions
- `scripts/scan_bc_coast.py` — Original BC coast scan with DINOv2/SVM scoring
- `scripts/train_from_manifest.py` — Train DINOv2 SVM from training_manifest.json
- `scripts/final_bc_sweep.py` — Rose-verified training sweep, model retrain, temporal review orchestration
- `scripts/ingress_dfo_gee_search.py` — DFO event Sentinel-2 thumbnail ingress and review page builder
- `scripts/knn_detector.py` — DINOv2 KNN voting evaluation and report builder
- `scripts/delta_detector.py` — DINOv2 paired-temporal delta detector (100% LOO CV on 37 locations)
- `scripts/scan_13regions_subspacead.py` — Multi-year SubspaceAD scan of all 13 BC habitat regions
- `scripts/subspace_ad.py` — DINOv2 SubspaceAD anomaly detector
- `scripts/patch_subspace_ad.py` — Patch-level SubspaceAD with spatial segmentation
- `scripts/fewshot_subspace_variants.py` — DINOv2 few-shot SubspaceAD variants (dual-subspace, positive-aware, MIL, confounder-aware, Mahalanobis)
- `scripts/temporal_detector.py` — Before-during-after triplet delta and intra-season timeseries detector
- `scripts/temporal_v2.py` — Five temporal modes: outlier, YOY, spectral, trajectory, cloud-fusion
- `scripts/environmental_matcher.py` — Sun+tide-matched embedding outlier detection per location
- `scripts/multi_year_confirm.py` — Multi-year candidate confirmation with 0.01° coordinate grouping
- `scripts/prompt_detector.py` — SenCLIP/RemoteCLIP contrastive prompt scoring with multi-scale crops
- `scripts/benchmark_prompt_models.py` — Prompt model benchmark on golden set
- `scripts/rs_foundation_probe.py` — frozen-embedding linear probe benchmark for remote-sensing foundation backbones
- `scripts/label_subspace_app.py` — Flask labeling app for SubspaceAD candidates
- `scripts/label_svm_app.py` — Flask labeling app for SVM candidates
- `scripts/final_review_app.py` — Flask final review app for golden set labeling
- `scripts/dual_subspace_review_app.py` — Flask labeling app for dual-subspace contrast candidates
- `scripts/run_gee_search.py` — Search Sentinel-2 for known events, download thumbnails
- `scripts/run_embeddings.py` — DINOv2 embedding ranking with positive/negative scoring
- `scripts/run_clay_multispectral.py` — Clay v1.5 encoder on multispectral GeoTIFF chips
- `scripts/download_and_review.py` — Batch thumbnail download and review page generator
- `scripts/label_images.py` — Terminal-based labeling tool
- `scripts/build_event_catalog.py` — Combines DFO, manual, and track events into GeoJSON
- `scripts/segment_spawn.py` — DINOv3 + SAM interactive click-to-segment web app for pixel-level spawn masks
- `scripts/dinov3_feature_extractor.py` — DINOv3 ViT-L/16 SAT-493M feature extraction with similarity heatmaps

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
- **Current DINOv2 + SVM**: 95.6% full accuracy, 1.8540 separation, trained on 16 golden positives + 164 negatives (model: `data/models/svm_16pos_164neg.pkl`)
- **DINOv2 Dual-Subspace Contrast (few-shot)**: AUROC 0.981, AP 0.901, best F1 0.846 — trains separate PCA on positives and negatives, score = negative_residual - positive_residual
- **DINOv2 SubspaceAD (zero-shot)**: AUROC 0.997, AP 0.965 — PCA reconstruction residual on DINOv2 patch tokens; detects generic shoreline anomalies, use as screening tool only
- **DINOv2 RS Foundation Linear Probe**: AUROC 0.972, AP 0.728, acc 0.939 — frozen DINOv2 + sklearn logistic regression; Prithvi/SatMAE/SkySense optional backbones fail clearly if deps unavailable
- **DINOv2 Delta detector**: 100% LOO CV on 37 locations — compares pre-spawn vs spawn-season DINOv2 embeddings; 2x better separation than single-image scoring
- **Environmental Matcher (S2 + Landsat)**: AUROC 0.745, AP 0.450 — sun+tide-matched embedding outliers across S2 + Landsat 8/9 scenes; 2.4x more scenes per location than S2-only
- **Prompt-based CLIP scoring**: try `pallavijainpj/SenCLIP` first for satellite prompt scoring; keep the existing RemoteCLIP pipeline as a separate additive baseline. Use contrastive scoring `spawn_mean - max(confounder_group_means)` and multi-scale crop scoring for small plumes. Process one image at a time, prefer CPU by default, and keep crop batches tiny to stay RAM-safe. BioCLIP/BiomedCLIP are not recommended for nadir Sentinel-2 imagery.
- **Prompt calibration (first-class)**: benchmark outputs now include raw spawn score, per-confounder group scores, max confounder group/mean, margin, and threshold sweeps (baseline 0.0, best accuracy, best balanced accuracy, best F1).
- **RS foundation linear probe**: `scripts/rs_foundation_probe.py` trains a frozen-embedding sklearn linear probe with DINOv2 as the guaranteed fallback; optional Prithvi/SatMAE/SkySense-style backbones are best-effort and fail clearly if deps/model ids are unavailable. Keep it low-RAM by embedding one image at a time.

### Human-Reviewed Positives
- **Canonical training data** is in `data/samples/training_manifest.json` (16 positives, 164 negatives).
- Positive images: `data/samples/positive/` (16 files)
- Negative images: `data/samples/negative/` (164 files)
- Rejected images from the final review cleanup: `data/samples/rejected/` (9 files)

- **14 final-review positives (all verified by dexterfichuk via dual_subspace_contrast review on 2026-05-25):**
  - 4 Strait of Georgia: `SoG_2018-03-10_score0.33_49.465556_-124.736111_20180310.png`, `SoG_2019-03-20_score0.00_49.481944_-124.731667_20190320.png`, `SoG_2019-03-20_score0.00_49.701389_-124.86_20190320.png`, `SoG_2019-03-20_score0.33_49.474722_-124.685_20190320.png`
  - 4 Strait of Georgia: `SoG_2021-03-11_score0.00_49.2449_-124.023_20210311.png`, `SoG_2021-03-11_score0.00_49.248611_-124.033611_20210311.png`, `SoG_2021-03-11_score0.00_49.5175_-124.577222_20210311.png`, `SoG_2021-03-11_score0.00_49.528056_-124.606667_20210311.png`
  - 1 breakwater-island: `dfo-verified-breakwater-island_2024-03-18_cloud0.png`
  - 3 milbanke-sound: `milbanke-sound_2024-03-24_score0.25_52.544865_-128.741984_20240324.png`, `milbanke-sound_2024-03-24_score0.26_52.544865_-128.721984_20240324.png`, `milbanke-sound_2024-03-24_score0.51_52.524865_-128.741984_20240324.png`
  - 2 nanaimo: `nanaimo_2024-03-18_score0.15_49.134865_-123.676603_20240318.png`, `nanaimo_2024-03-18_score0.21_49.134865_-123.696603_20240318.png`

- **1 legacy Rose-verified positive** (verified by dexterfichuk via `data/candidates_v2/rose_training_verify.json`):
  - `dfo-verified-qualicum-beach_2024-03-15_cloud16.png`

- **9 explicit rejects** (4 final-review rejects + 5 preexisting rejects in `data/samples/rejected/`):
  - `big-bay-prince-rupert_2023-03-28_cld11.png`, `big-bay-prince-rupert_2023-03-28_cld12.png`, `dfo-verified-anderson-point_2024-03-18_cloud24.png`, `dfo-verified-ucluelet_2024-03-18_cloud2.png`, `nootka-sound_2024-02-12_score0.00_49.564865_-126.508503_20240212.png`, `nootka-sound_2024-03-16_score0.00_49.584865_-126.528503_20240316.png`, `qualicum_2024-03-18_score0.01_49.254865_-124.497442_20240318.png`, `tofino_2024-03-16_score0.00_49.114865_-125.806603_20240316.png`, `tofino_2024-03-16_score0.01_49.194865_-126.026603_20240316.png`

- All labeling provenance is tracked in `data/final_review/golden_set.json` and `data/samples/training_manifest.json`.
- Do NOT use model-ranked candidates or high-confidence score buckets as training labels unless they also appear in a human label file.

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

## Session Status (2026-05-25)

### Accomplished Today
- **Golden set**: finalized 16 positives + 164 negatives through dual_subspace_contrast human review
- **Detection methods benchmarked**: dual-subspace contrast (AUROC 0.981), RS linear probe (0.972), SVM (95.6%), SubspaceAD (0.997)
- **Temporal approaches**: before-during-after triplet, YOY contrast, environmental sun+tide matching, trajectory classification, cloud-fusion
- **Landsat enrichment**: Added Landsat 8/9 embeddings alongside Sentinel-2, achieving 2.4x more scenes per location (+32% AUROC for environmental matcher)
- **Prompt-based**: SenCLIP/RemoteCLIP contrastive scoring with multi-scale crops, calibration with threshold sweeps
- **Multi-year scan**: SVM scan across 2023/2024/2025, 53 candidates, 3 multi-year confirmed locations (all nanaimo)
- **Few-shot SubspaceAD variants**: 6 variants benchmarked, dual_subspace_contrast is champion
- **Labeling infrastructure**: final_review_app, dual_subspace_review_app, label_svm_app, label_subspace_app, env_match_review
- **Documentation**: AGENTS.md and agent_handoff.md synced with current state

### Key Findings
1. Single-image DINOv2 methods remain practical champions: dual-subspace (AUROC 0.981), SVM (95.6%), RS probe (0.972)
2. Temporal approaches are data-limited: only 2 of 16 positives could form complete triplets, only 4 had multi-year coverage
3. Landsat 8/9 addition improves temporal coverage 2.4x (1.9 → 4.6 scenes/location) with near-zero disk cost
4. Environmental matching (sun+tide) is the best temporal method when imagery is available (AUROC 0.745 with S2+Landsat)
5. Prompt-based methods lag behind embedding methods (RemoteCLIP AUROC 0.729 vs DINOv2 0.981)

### Recommended Next Steps
1. **Expand Landsat coverage** to all 16 positive locations and scan negatives — more temporal coverage is the highest-leverage improvement
2. **Re-run multi-year scan** with SVM trained on all 16+164, ideally with 0.01° grid spacing for finer coverage
3. **Install SenCLIP dependencies** (`pip install transformers huggingface_hub`) and benchmark against RemoteCLIP
4. **Upload image-heavy outputs** to Hugging Face and clean local disk
5. **Active learning loop**: scan → review top candidates → add to golden set → retrain → repeat
