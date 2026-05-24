# Agent Handoff: Herring Spawn Flagging And Classification

This repository is a research prototype for finding Pacific herring spawn in satellite imagery. The goal is to flag candidate nearshore Sentinel-2 thumbnails and classify them as likely spawn or non-spawn for human review.

## Current Bottom Line

The strongest practical approach is not a single-image classifier by itself. Use known event records and image retrieval to build candidates, use DINOv2/KNN or SVM as cheap thumbnail ranking, then require temporal or human review before making detection claims.

Recent findings:

- Single-image DINOv2 can rank obvious turquoise spawn plumes, but it also learns shoreline and surf patterns.
- KNN over DINOv2 thumbnails is the current working large-scan pipeline and produced `data/candidates_knn/`.
- Clay or other multispectral delta methods are the preferred research direction because comparing pre-spawn and spawn-season imagery reduces shoreline bias.
- Human review labels from Rose/Silo/Koko-style review pages are treated as the highest-quality labels in this repo.

## Repository Map

Top-level code and docs:

- `README.md` - setup, Earth Engine auth, smoke workflow, and test commands.
- `AGENTS.md` - short status page for agent startup context.
- `herring_spawner/` - reusable package code for datasets, imagery, embeddings, chips, features, and review helpers.
- `scripts/` - research and batch-processing entry points.
- `tests/` - pytest coverage for reusable code and newer scripts.
- `docs/` - research notes, recommendations, plans, and this handoff.
- `data/` - local sample images, candidates, review pages, labels, manifests, model artifacts, and generated outputs.
- `notebooks/` - exploratory notebooks.

Important local-only or generated directories:

- `checkpoints/` - local model checkpoints. Do not commit raw checkpoint files unless Git LFS is configured.
- `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` - local environment/cache only.
- Image-heavy generated outputs are published to `https://huggingface.co/datasets/dfichuk/herring-spawn-candidates` instead of normal Git history.

## Core Package Modules

- `herring_spawner/datasets/dfo.py` - DFO spawn-index ingestion and normalization.
- `herring_spawner/datasets/washington.py` - Washington event/source helpers.
- `herring_spawner/datasets/alaska.py` - Alaska event/source helpers.
- `herring_spawner/datasets/manual.py` - manual event records.
- `herring_spawner/datasets/tracks.py` - local track parsing.
- `herring_spawner/imagery/gee.py` - Google Earth Engine search/export helpers.
- `herring_spawner/embeddings/search.py` - embedding-based scoring and search utilities.
- `herring_spawner/features/spectral.py` - spectral feature helpers.
- `herring_spawner/chips/catalog.py` - chip catalog support.
- `herring_spawner/review/static.py` - static review page helpers.
- `herring_spawner/models.py` - shared model/data structures.

## Main Scripts

Candidate discovery and scanning:

- `scripts/scan_bc_coast.py` - original BC coast Sentinel-2 thumbnail scan with DINOv2/SVM scoring.
- `scripts/scan_bc_coast_knn.py` - current KNN-based BC scan. It builds a DINOv2 embedding-space KNN classifier, scans 13 BC habitat regions, and stores majority-spawn candidates.
- `scripts/final_bc_sweep.py` - orchestrates a rose-verified training sweep, SVM retrain, temporal validation, and review generation.
- `scripts/scan_salmon_coast.py` - related coast scanning adaptation for salmon-coast-style locations.

Training and scoring:

- `scripts/run_embeddings.py` - DINOv2 embedding ranking from labeled positives/negatives.
- `scripts/train_classifier.py` - trains the DINOv2 SVM model saved under `data/models/`.
- `scripts/knn_detector.py` - evaluates DINOv2 KNN voting against labels and builds a report.
- `scripts/dinov2_consensus.py` - consensus experiments over DINOv2 candidate sets.
- `scripts/improved_detector.py` - HSV/texture/delta-inspired feature classifier with summary in `data/models/improved_model.summary.json`.
- `scripts/color_classify.py` - color-feature classification experiments for turquoise/foam/sediment cues.

Clay and temporal experiments:

- `scripts/run_clay_multispectral.py` - Clay v1.5 multispectral GeoTIFF chip embeddings.
- `scripts/run_clay_embeddings.py` - Clay embedding experiments.
- `scripts/clay_delta_detector.py` - delta-based Clay detector work.
- `scripts/clay_reconstruction.py` - tests Clay reconstruction/change signal.
- `scripts/validate_temporal.py` - temporal candidate checks.
- `scripts/validate_temporal_candidates.py` - temporal validation for candidate directories.
- `scripts/build_delta_review.py` - builds review assets for delta comparisons.

Event/data ingestion and review:

- `scripts/build_event_catalog.py` - combines DFO/manual/track events into GeoJSON.
- `scripts/search_known_events.py` - builds known-event scene search requests.
- `scripts/run_gee_search.py` - searches Sentinel-2 scenes for known events.
- `scripts/fetch_gee_thumbnails.py` - downloads Earth Engine thumbnails.
- `scripts/download_and_review.py` - batch thumbnail download plus review page generation.
- `scripts/build_interactive_review.py` - interactive review page builder.
- `scripts/build_combined_report.py` - Clay and DINOv2 side-by-side report.
- `scripts/build_labeler.py` - static label/review page builder used by newer ingressed datasets.
- `scripts/label_images.py` - terminal-based labeling tool.
- `scripts/label_fiftyone.py` - FiftyOne labeling support.
- `scripts/ingress_external_data.py` - external event ingestion.
- `scripts/ingress_dfo_gee_search.py` - DFO event GEE search, thumbnail download, manifest, and review page generation.

## Data Assets

High-value labeled data:

- `data/samples/positive/` - positive spawn thumbnails used for training. The current final-sweep human-reviewed subset is the 5 Rose-verified files listed below.
- `data/samples/negative/` - confirmed non-spawn thumbnails used for training.
- `data/samples/training_manifest.json` - current final-sweep training manifest: 5 rose-verified positives and 50 negatives.
- `data/candidates_v2/rose_super_review.json`, `rose_200_labels.json`, `rose_training_verify.json`, `ai_labels.json` - human/AI review labels and verification aids.
- `data/candidates_knn/silo_labels.json`, `data/sog_candidates/silo_labels.json`, `data/ingressed/silo_labels.json` - Silo-style labels from review workflows.

Current final-sweep human-reviewed positives:

- `qualicum_2024-03-18_score0.01_49.254865_-124.497442_20240318.png`
- `tofino_2024-03-16_score0.00_49.114865_-125.806603_20240316.png`
- `tofino_2024-03-16_score0.01_49.194865_-126.026603_20240316.png`
- `nootka-sound_2024-03-16_score0.00_49.584865_-126.528503_20240316.png`
- `nootka-sound_2024-02-12_score0.00_49.564865_-126.508503_20240212.png`

Do not treat model-ranked candidates, high-confidence score buckets, temporal positives, or generated review pages as human-reviewed positives unless they also appear in an explicit human label file. When promoting labels into training manifests, preserve reviewer/source provenance such as Rose or Silo.

Candidate and review outputs:

- `data/candidates_v2/` - earlier SVM candidate set, review pages, labels, temporal outputs, and cached temporal review artifacts.
- `data/candidates_knn/` - current KNN scan output. Summary: 205 training samples, 2,863 points scanned, 725 candidates, 0 download errors, about 852 seconds elapsed.
- `data/candidates_final/` - final SVM sweep outputs. Commit lightweight reports and logs; upload generated PNGs and review pages to Hugging Face.
- `data/sog_candidates/` - Strait of Georgia candidate thumbnails from 333 filtered SOG records. Summary: 452 thumbnails, 241 records with scenes, 0 download errors.
- `data/ingressed/` - ingressed DFO/external events, downloaded thumbnails, manifests, and review pages. The DFO GEE search selected 200 events and downloaded 128 clear thumbnails.
- `data/candidates_salmon_coast*/` - related salmon coast scan outputs by year.
- `https://huggingface.co/datasets/dfichuk/herring-spawn-candidates` - public home for generated candidate thumbnails, review pages, and other image-heavy scan outputs.

Model artifacts:

- `data/models/dinov2_svm.pkl` - current DINOv2 SVM model.
- `data/models/dinov2_svm.summary.json` - current SVM summary: DINOv2 ViT-S/14, RBF kernel, 55 train samples, 5 positives, 50 negatives, 94.5% mean CV accuracy, 1.8658 separation.
- `data/models/improved_model.pkl` - improved feature model.
- `data/models/improved_model.summary.json` - improved model summary: 67 train samples, 17 positives, 50 negatives, 85.1% mean CV accuracy, 97.0% combined-feature accuracy.

Large local artifacts:

- `data/candidates_v2/temporal_review.html` and `data/candidates_v2/temporal_results.json` can exceed GitHub's normal 100 MB file limit.
- `checkpoints/v1.5/clay-v1.5.ckpt` is a multi-GB local checkpoint.
- Do not add oversized artifacts to normal Git history unless Git LFS is enabled and the team explicitly wants them versioned.
- Use `scripts/upload_hf_dataset.py` for generated candidate imagery and review assets that belong in the Hugging Face dataset.

## Recommended Workflow For New Agents

1. Read `AGENTS.md`, `README.md`, `docs/technical_recommendation.md`, and this file.
2. Check `git status --short` before editing. The repo often contains generated data and review artifacts.
3. If using Earth Engine, authenticate locally and initialize project `redd-fish`.
4. Prefer existing labeled review files over weak automatic labels.
5. Use DINOv2/KNN/SVM for candidate ranking, not final truth.
6. Require temporal repeatability or human review before calling a candidate a confirmed spawn.
7. Avoid committing raw checkpoints, virtualenv files, caches, or artifacts above GitHub's file-size limit.
8. Keep image-heavy generated outputs in the public Hugging Face dataset, not GitHub.

## Common Commands

Setup and verification:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -v
ruff check .
```

Earth Engine connectivity:

```bash
source .venv/bin/activate
python - <<'PY'
import ee
ee.Initialize(project="redd-fish")
print(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").limit(1).size().getInfo())
PY
```

KNN BC coast scan:

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
```

DFO GEE ingress review:

```bash
source .venv/bin/activate
python scripts/ingress_dfo_gee_search.py
python -m http.server 8766 --directory data/ingressed
```

Serve a candidate review directory:

```bash
python -m http.server 8766 --directory data/candidates_knn
```

Upload generated candidate datasets to Hugging Face:

```bash
source .venv/bin/activate
python -m pip install huggingface_hub
huggingface-cli login
python scripts/upload_hf_dataset.py --repo-id dfichuk/herring-spawn-candidates
```

## Classification Guidance

Visual spawn cues:

- Bright turquoise or milky plume in shallow nearshore water.
- Plume follows shoreline or bathymetry rather than cloud shape.
- Signal repeats across nearby points or dates around known spawn windows.
- Strongest events are often near known DFO/SOG/WA/AK event records.

Common false positives:

- Surf and breaking waves.
- Glacial/sediment plumes.
- Cloud haze, cloud shadow, and sun glint.
- Bright beaches and exposed sandbars.
- Kelp/vegetation edges or shoreline texture learned by single-image models.

Confidence rule of thumb:

- High: clear plume plus temporal support or human-verified label.
- Medium: clear plume but only one scene/date.
- Low: model score only, ambiguous visual evidence, or likely surf/sediment.

## Open Risks And Next Work

- The positive dataset is still small and label quality varies by source.
- Single-image DINOv2 models can overfit location and shoreline appearance.
- Generated review pages and JSON outputs can become too large for normal Git pushes.
- The most important next technical step is a paired pre-spawn versus spawn-season multispectral delta pipeline with robust review labels.
- The most important data step is consolidating human labels into one canonical manifest with provenance and confidence.
