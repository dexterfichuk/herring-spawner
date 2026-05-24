# Few-Shot And Zero-Shot Herring Spawn Detection Roadmap

## Executive Summary

The highest-probability path is candidate generation plus few-shot ranking plus human verification, not a fully supervised detector. Herring spawn is a shoreline-attached, seasonal, optically complex event whose most visible satellite signal is milky/turquoise water from milt. The roadmap prioritizes spectral and physics baselines, temporal deltas, frozen foundation embeddings, anomaly detection, and active learning.

The current repo already has a useful candidate-generation workflow, but the confirmed final-sweep positive set is tiny. Treat the existing DINOv2/KNN/SVM outputs as review prioritization. Do not treat model-ranked candidates, high-confidence score buckets, temporal positives, or generated review pages as confirmed spawn unless they are linked to explicit human-label provenance.

## Current Ground Truth And Constraints

- Current final-sweep human-reviewed positives are exactly the 5 Rose-verified files in `data/samples/training_manifest.json`.
- Those positives are:
- `qualicum_2024-03-18_score0.01_49.254865_-124.497442_20240318.png`
- `tofino_2024-03-16_score0.00_49.114865_-125.806603_20240316.png`
- `tofino_2024-03-16_score0.01_49.194865_-126.026603_20240316.png`
- `nootka-sound_2024-03-16_score0.00_49.584865_-126.528503_20240316.png`
- `nootka-sound_2024-02-12_score0.00_49.564865_-126.508503_20240212.png`
- The current final-sweep training manifest has 5 positives and 50 negatives.
- Candidate review outputs and score buckets are not confirmed positives unless linked to explicit human label files.
- Current SVM summary: 55 train samples, 5 positives, 50 negatives, 94.5% mean CV accuracy, 1.8658 separation.
- KNN scan summary: 205 training labels, 2,863 scanned points, 725 candidates.
- Final sweep summary: 2,863 grid points, 13 regions, 878 candidates.
- The improved feature model summary reports 67 train samples, 17 positives, 50 negatives, 85.1% mean CV accuracy, and 97.0% combined-feature accuracy, but those extra positives are not the current final-sweep human-reviewed positive authority.

## Current Repository Baseline

The operational baseline is a Sentinel-2 thumbnail candidate pipeline. `scripts/scan_bc_coast_knn.py` scans BC habitat regions with DINOv2 ViT-S/14 embeddings and KNN voting. `scripts/scan_bc_coast.py` and `scripts/final_bc_sweep.py` provide the earlier SVM-based and final-sweep workflows. DFO/external event ingress is handled by `scripts/ingress_dfo_gee_search.py`, with review pages and manifests under `data/ingressed/`.

The current models are useful triage models, not final detectors. `data/models/dinov2_svm.summary.json` records a strong cross-validation number on a very small set, but the known failure mode is shoreline, surf, sediment, beach, and cloud-haze bias. `docs/technical_recommendation.md` therefore recommends Clay or other multispectral embedding deltas on paired pre-spawn and spawn-season chips, with temporal repeatability and human review before detection claims.

Useful current assets include `data/candidates_knn/`, `data/candidates_final/`, `data/sog_candidates/`, and `data/ingressed/`. Their review pages, candidate manifests, and model scores are candidate evidence. They should feed review queues, hard-negative mining, temporal checks, and active learning.

## Data And Provenance Map

GitHub should contain source code, tests, docs, manifests, labels, lightweight summaries, and research notes. Generated imagery and heavy review artifacts belong in the public Hugging Face dataset `dfichuk/herring-spawn-candidates`. Use `scripts/upload_hf_dataset.py` as the upload path for image-heavy candidate directories.

Human labels should be promoted only from explicit provenance sources such as Rose, Silo, Koko, or other named reviewers. When labels are consolidated, preserve reviewer/source names, source file paths, review date if available, and whether the label is positive, negative, uncertain, or candidate-only. Candidate manifests and score-ranked lists are not labels by themselves.

Recommended canonical buckets:

- Confirmed positive: explicit human-reviewed spawn label, currently the 5 Rose-verified final-sweep positives for this training manifest.
- Confirmed negative: explicit non-spawn label or curated hard negative.
- Uncertain: human-reviewed ambiguous, partial, or low-confidence entries.
- Candidate-only: model-ranked, temporal, generated-review, or ingressed candidates with no explicit human confirmation.
- External-prior-only: DFO/PSF/CRIMS records or known spawn windows used to guide search, not direct image labels unless matched and reviewed.

## Evaluation Strategy

This is rare-event alerting. Overall pixel accuracy or thumbnail accuracy is secondary to whether the system finds real events while reducing review burden.

Primary metrics:

- Event recall on known DFO and human-reviewed sites.
- Precision at a fixed human-review budget, such as top 50, 100, or 200 candidates per season.
- False alerts per coastline-km/day.
- Temporal holdout performance by year.
- Spatial holdout performance by region.
- Calibration of scores into review priority bands.
- Review-hours saved versus the current candidate queue.

Recommended validation splits:

- Leave-one-region-out to test spatial generalization.
- Leave-one-date-window-out to test temporal robustness.
- Known-event replay where the model must rank DFO/verified sites above background candidates.
- Hard-negative challenge sets covering surf, sediment, glint, cloud haze, bright beach, kelp, shallow bottom, and shoreline texture.

## Technology Matrix

| Option | Category | Why Try It | Minimum Viable Build | Promotion Criteria | Main Risk |
|---|---|---|---|---|---|
| SHSI / green-minus-red spectral threshold | Zero-shot spectral | Published/UVic-style physics baseline; strong with tiny labels | Compute green-red/SHSI-like scores on Sentinel-2 or Landsat chips with water masks | Finds known events with manageable review volume | Surf, sediment, cloud haze, shallow bottom |
| Local same-scene water anomaly | Zero-shot anomaly | Detects abrupt bright water relative to nearby baseline | Compare candidate water pixels to local non-candidate water in same scene | Improves precision over global threshold | Local background contaminated by plume |
| DINOv2 patch/prototype | Few-shot RGB | Current repo already uses DINOv2; patch features may reduce shoreline bias | Embed patches around 5 positives and hard negatives, rank by prototype distance | Beats current thumbnail-level KNN/SVM at same recall | RGB-only shoreline/surf bias |
| CLAY multispectral delta | Few-shot multispectral | Current best research direction; delta removes static shoreline bias | Export pre/post multispectral chips and train linear probe on delta embeddings | Higher precision than single-image DINOv2 | GeoTIFF export and checkpoint cost |
| Prithvi-EO / HLS embeddings | Few-shot temporal | HLS time-series pretraining fits multi-date coastal change | Build HLS chip stack and frozen embedding linear probe | Better temporal generalization | Heavier integration stack |
| RemoteCLIP / RS VLM prompts | Zero/few-shot VLM | Text prompts allow no-label triage and semantic retrieval | Score RGB chips with spawn/confounder prompts; compare to prototype classifier | Useful review queue ranking | Prompt sensitivity; generic semantics too coarse |
| SatMAE / SeCo / SSL4EO-S12 | Few-shot RS embeddings | RS-specific self-supervised alternatives | Extract frozen embeddings and train linear/prototype heads | Beats DINOv2 or CLAY on heldout regions | Model availability and input mismatch |
| PatchCore / PaDiM / SubspaceAD | Zero/few-shot anomaly | Finds novel shoreline anomalies from normal coast model | Fit normal background from non-spawn coastal tiles, score residuals | Finds plausible new-site candidates with lower false alert burden | Many optical false positives |
| PEFT / LoRA adapters | Later few-shot | Adds task specificity after labels grow | Adapter fine-tune CLAY/Prithvi/RemoteCLIP after hundreds of reviewed labels | Improves calibrated ranking without overfitting | Premature complexity with only 5 positives |
| PlanetScope | Commercial sensor | Daily 3 m imagery improves cadence and shoreline resolution | Pilot one high-priority region if access exists | Captures events missed by S2/Landsat | Cost/licensing |
| Maxar / WorldView | Commercial verification | High-res validation and label expansion | Use sparingly for ambiguous/high-value candidates | Confirms labels and plume boundaries | Cost/licensing/tasking friction |
| UAV / hyperspectral | Field calibration | Best label/spectral library source | Target known positives and hard confounders | Produces higher-confidence labels | Field logistics |

## Experiment Cards

### SHSI / Green-Minus-Red Spectral Baseline

Purpose: establish a transparent zero-shot baseline for the milky/turquoise water signal.

Input data: Sentinel-2 and Landsat chips, water masks, shoreline buffers, current positives, curated hard negatives, and DFO event priors.

Minimum viable build: compute SHSI-like, green-minus-red, brightness, turbidity, and normalized color scores on candidate chips; rank by shoreline-attached water anomaly.

Metrics: event recall, precision at review budget, false alerts per coastline-km/day, and region holdout.

Expected failure modes: surf, sediment plumes, shallow bright bottom, haze, glint, cloud edge, and beaches.

Promote or defer: promote if it recovers known positives with a review queue smaller than the current DINOv2 sweep; defer if false alerts remain dominated by surf/sediment at every threshold.

### Local Same-Scene Water Anomaly

Purpose: reduce global threshold fragility by comparing candidate water to local nearby water from the same scene.

Input data: Sentinel-2 chips, water masks, shoreline buffers, local background rings, and candidate centroids.

Minimum viable build: estimate local non-candidate water color/spectral distribution and score pixels or chips by positive deviation in milky/turquoise bands.

Metrics: precision improvement over SHSI at fixed recall and false alerts per coastline-km/day.

Expected failure modes: background contaminated by the plume, large sediment fields, local bathymetry, and cloud haze.

Promote or defer: promote if it removes region-specific brightness bias; defer if it suppresses broad real plumes.

### DINOv2 Patch/Prototype

Purpose: improve the existing RGB embedding workflow by focusing on plume-like patches rather than whole thumbnails that include shoreline texture.

Input data: current 5 Rose-verified positives, curated negatives, candidate thumbnails, and hard confounders.

Minimum viable build: extract DINOv2 patch features or crop-level features; build positive prototypes and negative prototypes; rank candidates by nearest-prototype margin.

Metrics: top-k recall of known positives, precision at review budget, and region holdout versus current thumbnail KNN/SVM.

Expected failure modes: RGB-only surf and beach confusion, overfitting to the 5 positives, and location-specific backgrounds.

Promote or defer: promote if it beats thumbnail-level KNN/SVM at the same review budget; defer if prototypes mostly learn coastline appearance.

### CLAY Multispectral Delta

Purpose: test the preferred research direction: spawn-season change relative to a pre-spawn baseline using satellite-native multispectral embeddings.

Input data: paired pre-spawn and spawn-season GeoTIFF chips, Sentinel-2 bands, current positives, negatives, and candidate locations.

Minimum viable build: export paired chips, run CLAY v1.5 embeddings, compute delta vectors, and train a small linear or prototype classifier.

Metrics: precision at review budget, event recall, temporal holdout, spatial holdout, and improvement over single-image DINOv2.

Expected failure modes: registration mismatch, clouds, tide differences, sensor/atmosphere artifacts, and limited positive labels.

Promote or defer: promote if deltas reduce shoreline bias and improve heldout precision; defer if export/checkpoint cost blocks reproducible runs.

### Prithvi-EO / HLS Embeddings

Purpose: test a temporal foundation model whose pretraining aligns with multi-date Earth observation stacks.

Input data: HLS/Sentinel-Landsat aligned chip stacks, pre-spawn and spawn-season windows, masks, and current labels.

Minimum viable build: create a small chip stack for known positives and hard negatives, extract frozen embeddings, and train a linear probe or prototype scorer.

Metrics: temporal generalization, region holdout, and review-budget precision.

Expected failure modes: heavier data engineering, mismatch between HLS resolution and nearshore plume scale, and cloud/tide artifacts.

Promote or defer: promote if it handles multi-date change better than CLAY or DINOv2; defer if integration cost exceeds likely gains before labels grow.

### RemoteCLIP / RS VLM Prompts

Purpose: evaluate whether remote-sensing vision-language embeddings can provide zero-shot prompt ranking or few-shot semantic retrieval.

Input data: RGB thumbnails, spawn prompts, confounder prompts, positive prototypes, negative prototypes, and review labels.

Minimum viable build: score candidates against prompts such as herring spawn milt plume, sediment plume, surf, cloud haze, shallow water, and bright beach; compare prompt ranking to embedding kNN.

Metrics: top-k enrichment, false alerts per coastline-km/day, and reviewer usefulness.

Expected failure modes: generic scene semantics too coarse, prompt sensitivity, and poor handling of small nearshore plumes.

Promote or defer: promote if it improves review ordering or hard-negative separation; defer if results are unstable across prompt wording.

### SatMAE / SeCo / SSL4EO-S12 Frozen Embeddings

Purpose: compare remote-sensing-specific self-supervised embeddings against DINOv2 and CLAY.

Input data: Sentinel-2 or RGB chips depending on model input, labels, and hard negatives.

Minimum viable build: extract frozen embeddings and evaluate linear, prototype, and kNN heads.

Metrics: heldout top-k precision, region holdout, and calibration.

Expected failure modes: model input mismatch, weaker representation for water-color changes, and engineering overhead.

Promote or defer: promote only if a model clearly beats DINOv2/CLAY on heldout candidates.

### PatchCore / PaDiM / SubspaceAD

Purpose: detect unusual nearshore water appearances from a normal-coast background model without requiring many positives.

Input data: non-spawn coastal tiles, candidate-season tiles, water/shoreline masks, and review labels for triage.

Minimum viable build: fit normal background features from non-spawn coastline imagery and score candidate chips or patches by anomaly distance.

Metrics: new-candidate discovery rate, precision at review budget, and false alerts per coastline-km/day.

Expected failure modes: many optical anomalies unrelated to spawn, seasonal turbidity, cloud artifacts, and infrastructure/shoreline changes.

Promote or defer: promote if it discovers plausible new-site candidates with lower review burden; defer if anomaly scores are mostly weather and surf.

### PEFT / LoRA Adapters

Purpose: add task specificity after the label set grows enough to support fine-tuning.

Input data: hundreds of reviewed positives/negatives with provenance and hard-negative coverage.

Minimum viable build: adapter-tune CLAY, Prithvi, or RemoteCLIP on a train/validation split with region holdout.

Metrics: calibrated ranking, heldout region performance, and precision at fixed review budget.

Expected failure modes: overfitting, unstable validation with tiny positives, and expensive experiments before the data supports them.

Promote or defer: defer until material label growth; promote only after frozen probes saturate.

### PlanetScope

Purpose: evaluate whether daily 3 m imagery improves event capture and shoreline resolution.

Input data: one high-priority region and season, known spawn windows, current candidates, and review labels.

Minimum viable build: pilot a small region if access exists and compare missed/extra events versus Sentinel-2/Landsat.

Metrics: event recall, review burden, and added detections per cost.

Expected failure modes: licensing, cost, cloudy coast limitations, and integration overhead.

Promote or defer: promote if access exists and it recovers events missed by open sensors; otherwise treat as partner/commercial option.

### Maxar / WorldView

Purpose: provide targeted high-resolution verification for ambiguous or high-value candidates.

Input data: top candidates, external event priors, and limited high-resolution scenes.

Minimum viable build: manually inspect a small number of ambiguous candidates and use results for label expansion.

Metrics: confirmation rate and improvement to training labels.

Expected failure modes: cost, licensing, timing, and tasking friction.

Promote or defer: use sparingly for validation, not broad scanning.

### UAV / Hyperspectral Calibration

Purpose: collect high-confidence plume and confounder spectra for future supervised or physics-informed models.

Input data: field observations, known spawn windows, hyperspectral or UAV imagery, and matching satellite scenes.

Minimum viable build: partner-led field campaign over a few known events and confounders.

Metrics: label quality, spectral separability, and satellite transfer usefulness.

Expected failure modes: logistics, weather, timing, and limited spatial coverage.

Promote or defer: promote through partnerships; defer as a dependency for current open-data experiments.

## Recommended Priority Order

1. Spectral/SHSI baseline.
2. CLAY multispectral delta linear probe.
3. DINOv2 patch/prototype refinement.
4. Normal-coast anomaly detection with PatchCore/PaDiM/SubspaceAD.
5. RemoteCLIP/RS VLM prompt and embedding retrieval.
6. Prithvi-EO/HLS temporal embedding trial.
7. Active learning and label manifest consolidation.
8. PEFT/LoRA or segmentation only after material label growth.

## Implementation Notes For Current Labels

- Use `data/samples/training_manifest.json` as the final-sweep human-reviewed positive source.
- Preserve Rose/Silo/Koko/reviewer names when promoting labels.
- Treat `data/candidates_final/report.json`, `data/candidates_knn/summary.json`, and candidate manifests as candidate metadata, not truth.
- Use hard negatives from surf, sediment, glint, cloud haze, bright beach, kelp, and shallow-bottom examples.
- Upload generated imagery to Hugging Face rather than committing it to GitHub.
- Keep `dfichuk/herring-spawn-candidates` as the public image-heavy dataset boundary.
- Prefer methods that produce inspectable review queues and provenance-rich label updates.
