# Few-Shot And Zero-Shot Research Epic Design

## Goal

Create a durable research tracker for Pacific herring spawn detection experiments, with one GitHub epic issue for progress tracking and one detailed repository Markdown document for implementation context.

## Inputs

Use these source materials:

- `/Users/dexterfichuk/Downloads/deep-research-report.md`
- `/Users/dexterfichuk/Downloads/Remote Sensing and Foundation-Model Approaches for Detecting Pacific Herring Spawn on the BC Coast-2.md`
- `/Users/dexterfichuk/Downloads/Herring Spawn Detection System.md`
- Existing repo context in `AGENTS.md`, `docs/agent_handoff.md`, `docs/technical_recommendation.md`, and current model/data artifacts.

The repo currently has exactly 5 final-sweep Rose-verified human-reviewed positives in `data/samples/training_manifest.json`. Model-ranked candidates, high-confidence buckets, temporal positives, and generated review pages must be treated as candidates unless they appear in explicit human label files.

## Deliverables

1. Create a detailed repo document at `docs/research/few-shot-zero-shot-roadmap.md`.
2. Create one GitHub epic issue using `gh issue create`.
3. The issue body should be a progress tracker with checkboxes and enough context to stand alone.
4. The repo document should be the deeper technical reference for how to build experiments using the current positive human-labelled samples and candidate datasets.

## Recommended Structure

Use one GitHub epic plus one detailed Markdown roadmap.

The GitHub epic is the operational tracker. It should be concise enough for progress management but detailed enough that a future agent understands what each checkbox means.

The repo Markdown file is the full technical context. It should synthesize the downloaded reports and repo state, explain why each technology is worth trying, define evaluation metrics, and describe how to use existing human-reviewed positives, negatives, candidate manifests, and Hugging Face-hosted generated imagery.

## GitHub Epic Content

The epic should include these sections:

- **Current truth and constraints**: 5 Rose-verified final-sweep positives, 50 final-sweep negatives, HF dataset location, and the rule that score-only candidates are not confirmed spawn.
- **Success metrics**: event recall on known sites, precision at fixed review budget, false alerts per coastline-km/day, temporal generalization, spatial holdout generalization, and review-hours saved.
- **Phase 0 data/provenance tasks**: consolidate label manifests, preserve reviewer/source provenance, build hard-negative/confounder catalog, verify DFO/PSF/CRIMS priors, and record HF/GitHub storage boundaries.
- **Zero-shot experiments**: SHSI/spectral threshold baseline, local same-scene water anomaly scoring, RemoteCLIP or RS VLM prompt scoring, SubspaceAD/PatchCore/PaDiM normal-coast anomaly detection, and shoreline morphology filters.
- **Few-shot experiments**: DINOv2 patch/prototype baseline, CLAY multispectral delta linear probe, Prithvi-EO/HLS frozen embedding probe, RemoteCLIP embedding kNN/prototype classifier, SatMAE/SeCo/SSL4EO-S12 embedding trials, and PEFT/LoRA only after label growth.
- **Temporal/context upgrades**: pre/post seasonal deltas, tide normalization, bathymetry/depth masks, shoreline buffers, cloud/shadow/sunglint QA, turbidity/chlorophyll/confounder flags, and active-learning review loops.
- **Stretch/commercial options**: PlanetScope daily monitoring, Maxar/WorldView targeted validation, UAV/hyperspectral label expansion, and partner data from DFO/PSF/First Nations/community monitoring.

## Repo Roadmap Content

The detailed Markdown roadmap should include:

- Executive summary of the research direction.
- Current repository baseline and known failure modes.
- Human-reviewed positive section listing the 5 Rose-verified files.
- Data map for GitHub-tracked manifests/models and Hugging Face-hosted imagery.
- Technology matrix covering SHSI, DINOv2, CLAY, Prithvi-EO, RemoteCLIP, RS VLMs, SatMAE/SeCo/SSL4EO-S12, PatchCore/PaDiM/SubspaceAD, PEFT/LoRA, PlanetScope, Maxar, UAV, and hyperspectral sources.
- Experiment cards for each option, with purpose, input data, minimum viable build, metrics, expected failure modes, and when to promote/defer the approach.
- Evaluation strategy focused on rare-event alerting rather than overall accuracy.
- Implementation notes for using current positives, negatives, candidate manifests, HF data, and review labels.
- A recommended priority order: spectral/SHSI baseline, CLAY delta, DINOv2 patch/prototype, anomaly detection, RemoteCLIP prompts, Prithvi/HLS, active learning, then PEFT or segmentation only after label growth.

## Non-Goals

- Do not implement new detectors in this task.
- Do not train or evaluate models in this task.
- Do not move generated imagery back into GitHub.
- Do not claim any candidate is confirmed spawn without explicit human label provenance.

## Validation

After implementation, verify:

- `docs/research/few-shot-zero-shot-roadmap.md` exists and contains all required sections.
- The GitHub issue exists and includes checkbox sections for the experiment families.
- The issue links to the roadmap file or commit.
- The roadmap and issue both state the 5-human-positive constraint clearly.
- `git status` is clean after commit and push.

## Open Question For Implementation

Use one GitHub issue unless issue creation fails or the user explicitly asks for many child issues. If GitHub issue creation fails due to authentication or permission, commit the repo roadmap and provide the exact issue body for manual creation.
