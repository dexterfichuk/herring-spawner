# Few-Shot And Zero-Shot Research Epic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a detailed repo roadmap and one GitHub epic issue for zero-shot and few-shot Pacific herring spawn detection experiments.

**Architecture:** Keep GitHub as the progress tracker and the repository as the durable technical reference. The roadmap synthesizes three downloaded research reports plus current repo state; the issue summarizes the roadmap into checkbox tasks that can be used as a long-running epic.

**Tech Stack:** Markdown, GitHub CLI (`gh`), Git, existing repo docs and data manifests.

---

## File Structure

- Create: `docs/research/few-shot-zero-shot-roadmap.md` — detailed technical roadmap and experiment guide.
- Create: `/var/folders/hd/3rqqvs_j7ns9lp665gvzv0_w0000gn/T/opencode/herring-spawner-epic-body.md` — temporary GitHub issue body used by `gh issue create`; do not commit.
- Existing reference: `docs/superpowers/specs/2026-05-24-few-shot-zero-shot-research-epic-design.md` — approved design spec.

## Required Source Materials

- `/Users/dexterfichuk/Downloads/deep-research-report.md`
- `/Users/dexterfichuk/Downloads/Remote Sensing and Foundation-Model Approaches for Detecting Pacific Herring Spawn on the BC Coast-2.md`
- `/Users/dexterfichuk/Downloads/Herring Spawn Detection System.md`
- `AGENTS.md`
- `docs/agent_handoff.md`
- `docs/technical_recommendation.md`
- `data/samples/training_manifest.json`
- `data/candidates_final/report.json`
- `data/models/dinov2_svm.summary.json`
- `data/models/improved_model.summary.json`

## Content Requirements

The roadmap and issue must state:

- Current final-sweep human-reviewed positives are exactly the 5 Rose-verified files in `data/samples/training_manifest.json`.
- Model-ranked candidates, high-confidence score buckets, temporal positives, and generated review pages are not confirmed spawn unless explicit human-label provenance exists.
- Generated imagery belongs in Hugging Face dataset `dfichuk/herring-spawn-candidates`, not normal Git history.
- Evaluation should focus on rare-event alerting: event recall, precision at fixed review budget, false alerts per coastline-km/day, temporal generalization, spatial holdout generalization, and review-hours saved.

### Task 1: Push Approved Design Spec

**Files:**
- Existing committed file: `docs/superpowers/specs/2026-05-24-few-shot-zero-shot-research-epic-design.md`

- [ ] **Step 1: Confirm branch is ahead only by the design spec commit**

Run: `/usr/bin/git status --branch --short`

Expected: `## main...origin/main [ahead 1]`

- [ ] **Step 2: Push the design spec commit**

Run: `/usr/bin/git push`

Expected: push succeeds and updates `main -> main`.

- [ ] **Step 3: Verify branch is clean before implementation**

Run: `/usr/bin/git status --branch --short`

Expected: `## main...origin/main`

### Task 2: Create Detailed Roadmap Markdown

**Files:**
- Create: `docs/research/few-shot-zero-shot-roadmap.md`

- [ ] **Step 1: Ensure docs research directory exists**

Run: `/bin/ls docs`

Expected: `docs` exists. If `docs/research` does not exist, create it with `/bin/mkdir -p docs/research`.

- [ ] **Step 2: Draft `docs/research/few-shot-zero-shot-roadmap.md`**

Create the file with these sections and concrete content synthesized from the required source materials:

```markdown
# Few-Shot And Zero-Shot Herring Spawn Detection Roadmap

## Executive Summary

The highest-probability path is candidate generation plus few-shot ranking plus human verification, not a fully supervised detector. Herring spawn is a shoreline-attached, seasonal, optically complex event whose most visible satellite signal is milky/turquoise water from milt. The roadmap prioritizes spectral/physics baselines, temporal deltas, frozen foundation embeddings, anomaly detection, and active learning.

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

## Current Repository Baseline

Summarize current scripts, data locations, model artifacts, Hugging Face dataset, and known failure modes from `docs/agent_handoff.md`.

## Data And Provenance Map

Explain GitHub-tracked code/docs/manifests/model summaries versus Hugging Face-hosted generated imagery. Include `dfichuk/herring-spawn-candidates` and note that `scripts/upload_hf_dataset.py` is the upload path.

## Evaluation Strategy

Use event recall, precision at fixed review budget, false alerts per coastline-km/day, temporal holdout performance, region holdout performance, calibration, and review-hours saved. Overall pixel accuracy is secondary because this is rare-event alerting.

## Technology Matrix

| Option | Category | Why Try It | Minimum Viable Build | Promotion Criteria | Main Risk |
|---|---|---|---|---|---|
| SHSI / green-minus-red spectral threshold | Zero-shot spectral | Published/UVic-style physics baseline; strong with tiny labels | Compute green-red/SHSI-like scores on S2/Landsat chips with water masks | Finds known events with manageable review volume | Surf, sediment, cloud haze, shallow bottom |
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

For each technology in the matrix, add a short card with: purpose, input data, minimum viable build, metrics, expected failure modes, and promote/defer rule. Use the technology matrix entries as source content, expanded enough for a future agent to start implementation.

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
```

- [ ] **Step 3: Verify roadmap contains required anchors**

Run:

```bash
/usr/bin/grep -E "5 Rose-verified|SHSI|CLAY multispectral delta|RemoteCLIP|PatchCore|dfichuk/herring-spawn-candidates|false alerts per coastline-km/day" docs/research/few-shot-zero-shot-roadmap.md
```

Expected: each required anchor appears at least once.

### Task 3: Create GitHub Epic Body

**Files:**
- Create temporary file only: `/var/folders/hd/3rqqvs_j7ns9lp665gvzv0_w0000gn/T/opencode/herring-spawner-epic-body.md`

- [ ] **Step 1: Write the issue body file**

Create `/var/folders/hd/3rqqvs_j7ns9lp665gvzv0_w0000gn/T/opencode/herring-spawner-epic-body.md` with:

```markdown
## Goal

Track zero-shot and few-shot research paths for detecting Pacific herring spawn on the BC coast, using the current human-reviewed positive set, generated candidate datasets, and remote-sensing research reports.

## Current Truth And Constraints

- Current final-sweep human-reviewed positives: exactly 5 Rose-verified files in `data/samples/training_manifest.json`.
- Current final-sweep training set: 5 positives + 50 negatives.
- Generated imagery lives in Hugging Face: https://huggingface.co/datasets/dfichuk/herring-spawn-candidates
- Model-ranked candidates, high-confidence buckets, temporal positives, and generated review pages are candidates, not confirmed spawn, unless explicit human-label provenance exists.
- Detailed roadmap: `docs/research/few-shot-zero-shot-roadmap.md`

## Success Metrics

- [ ] Event recall on known DFO/human-reviewed sites
- [ ] Precision at fixed human-review budget
- [ ] False alerts per coastline-km/day
- [ ] Temporal holdout performance by year
- [ ] Spatial holdout performance by region
- [ ] Review-hours saved versus current candidate queue

## Phase 0: Data And Provenance

- [ ] Consolidate human label manifests with reviewer/source provenance
- [ ] List canonical positive, negative, uncertain, and candidate-only sources
- [ ] Build hard-negative/confounder catalog: surf, sediment, glint, cloud haze, bright beach, kelp, shallow bottom
- [ ] Document DFO/PSF/CRIMS spatial priors and region-specific spawn windows
- [ ] Verify GitHub/Hugging Face storage boundaries for generated imagery

## Zero-Shot Experiments

- [ ] SHSI / green-minus-red spectral threshold baseline on Sentinel-2 and Landsat
- [ ] Local same-scene water anomaly score against nearby background water
- [ ] RemoteCLIP or RS VLM prompt scoring for spawn and confounder prompts
- [ ] PatchCore / PaDiM / SubspaceAD normal-coast anomaly detector
- [ ] Shoreline morphology filter for narrow, shoreline-attached plume candidates

## Few-Shot Experiments

- [ ] DINOv2 patch/prototype baseline using current positives and hard negatives
- [ ] CLAY multispectral pre/post delta linear probe
- [ ] Prithvi-EO / HLS frozen embedding probe
- [ ] RemoteCLIP embedding kNN/prototype classifier
- [ ] SatMAE / SeCo / SSL4EO-S12 frozen embedding trial
- [ ] PEFT / LoRA adapter trial only after material label growth

## Temporal And Context Upgrades

- [ ] Pre-spawn versus spawn-season paired deltas
- [ ] Tide normalization and scene filtering
- [ ] Bathymetry/depth and shoreline-buffer masks
- [ ] Cloud, shadow, haze, and sunglint QA
- [ ] Turbidity, chlorophyll, SPM, algae, and sediment confounder flags
- [ ] Active-learning review loop prioritizing uncertainty and diversity

## Stretch / Partner / Commercial Options

- [ ] PlanetScope daily monitoring pilot for a high-priority region
- [ ] Maxar / WorldView targeted verification of ambiguous candidates
- [ ] UAV or hyperspectral calibration/label-expansion campaign
- [ ] Partner data workflow with DFO, PSF, First Nations, or community monitoring groups

## Notes

Do not treat this issue as a claim that any candidate is confirmed spawn. Confirmation requires explicit human label provenance or external validation.
```

- [ ] **Step 2: Verify issue body has checkbox tracker sections**

Run:

```bash
/usr/bin/grep -E "## Zero-Shot Experiments|## Few-Shot Experiments|\- \[ \] CLAY multispectral|\- \[ \] SHSI|5 Rose-verified" /var/folders/hd/3rqqvs_j7ns9lp665gvzv0_w0000gn/T/opencode/herring-spawner-epic-body.md
```

Expected: all section anchors appear.

### Task 4: Commit Roadmap And Plan

**Files:**
- Add: `docs/research/few-shot-zero-shot-roadmap.md`
- Add: `docs/superpowers/plans/2026-05-24-few-shot-zero-shot-research-epic.md`

- [ ] **Step 1: Stage roadmap and plan**

Run:

```bash
/usr/bin/git add docs/research/few-shot-zero-shot-roadmap.md docs/superpowers/plans/2026-05-24-few-shot-zero-shot-research-epic.md
```

Expected: files are staged.

- [ ] **Step 2: Verify staged diff**

Run: `/usr/bin/git diff --cached --stat`

Expected: only the roadmap and plan are staged.

- [ ] **Step 3: Check Markdown whitespace**

Run: `/usr/bin/git diff --cached --check`

Expected: no output and exit 0.

- [ ] **Step 4: Commit roadmap and plan**

Run: `/usr/bin/git commit -m "docs: add few-shot research roadmap"`

Expected: commit succeeds.

### Task 5: Create GitHub Epic Issue

**Files:**
- Uses temporary file: `/var/folders/hd/3rqqvs_j7ns9lp665gvzv0_w0000gn/T/opencode/herring-spawner-epic-body.md`

- [ ] **Step 1: Verify GitHub CLI authentication**

Run: `/opt/homebrew/bin/gh auth status`

Expected: authenticated for `github.com`. If `gh` is elsewhere, use `/usr/bin/which gh` to locate it and run that binary.

- [ ] **Step 2: Create the epic issue**

Run:

```bash
gh issue create \
  --title "Epic: zero-shot and few-shot herring spawn detection experiments" \
  --body-file /var/folders/hd/3rqqvs_j7ns9lp665gvzv0_w0000gn/T/opencode/herring-spawner-epic-body.md
```

Expected: command prints the new GitHub issue URL.

- [ ] **Step 3: If issue creation fails**

If authentication or permissions fail, do not keep retrying. Save the issue body path in the final response and state that manual issue creation is needed.

### Task 6: Push And Final Verification

**Files:**
- No new file edits expected.

- [ ] **Step 1: Push commits**

Run: `/usr/bin/git push`

Expected: `main -> main` push succeeds.

- [ ] **Step 2: Verify branch status**

Run: `/usr/bin/git status --branch --short`

Expected: `## main...origin/main` and no changed files.

- [ ] **Step 3: Verify latest commit**

Run: `/usr/bin/git log --oneline -2`

Expected: includes `docs: add few-shot research roadmap` and `docs: design few-shot research epic`.

- [ ] **Step 4: Report final state**

Final response must include:

- Roadmap path.
- GitHub issue URL, or explicit issue-creation blocker plus temporary issue body path.
- Commit hashes for the design spec and roadmap commits.
- Confirmation that branch is clean and pushed.
