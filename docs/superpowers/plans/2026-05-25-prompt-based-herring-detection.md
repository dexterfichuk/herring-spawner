# Prompt-Based Herring Spawn Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a RAM-safe prompt-based satellite scorer for herring spawn detection with optional SenCLIP support, contrastive prompt sets, multi-scale crop scoring, and a benchmark over the canonical golden set.

**Architecture:** Keep the existing RemoteCLIP pipeline intact. Add a new focused prompt-scoring module that owns prompt configuration, model loading, image/crop scoring, and benchmark reporting. The module should be backend-agnostic enough to support optional SenCLIP first, with a cheap fallback path that can reuse existing RemoteCLIP helpers when available.

**Tech Stack:** Python, Pillow, numpy, torch, optional open_clip / huggingface_hub / sklearn, pytest.

---

### Task 1: Lock down pure prompt and crop logic with tests

**Files:**
- Create: `tests/test_prompt_detector.py`

- [ ] **Step 1: Write the failing test**

```python
from scripts.prompt_detector import (
    PromptSet,
    contrastive_score,
    generate_sliding_crops,
    aggregate_crop_scores,
)

def test_contrastive_score_uses_spawn_mean_minus_max_confounder():
    spawn = [0.9, 0.7, 0.8]
    confounders = {
        "foam": [0.2, 0.3],
        "glint": [0.5, 0.4],
    }
    assert contrastive_score(spawn, confounders) == 0.25

def test_generate_sliding_crops_limits_count_and_keeps_full_image():
    crops = generate_sliding_crops((1024, 768), crop_size=256, stride=256, max_crops=8)
    assert crops[0] == (0, 0, 1024, 768)
    assert len(crops) == 8
    assert all(len(box) == 4 for box in crops)

def test_aggregate_crop_scores_supports_max_and_topk_mean():
    scores = [0.1, 0.8, 0.3, 0.6]
    assert aggregate_crop_scores(scores, mode="max") == 0.8
    assert aggregate_crop_scores(scores, mode="topk_mean", top_k=2) == 0.7
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/test_prompt_detector.py -v`

Expected: import/function failures before implementation exists.

- [ ] **Step 3: Implement only the pure functions needed to satisfy the tests**

```python
def contrastive_score(spawn_scores, confounder_scores_by_group):
    return float(np.mean(spawn_scores) - max(np.mean(v) for v in confounder_scores_by_group.values()))

def generate_sliding_crops(image_size, crop_size=256, stride=256, max_crops=8):
    ...

def aggregate_crop_scores(scores, mode="topk_mean", top_k=2):
    ...
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `pytest tests/test_prompt_detector.py -v`

Expected: PASS.

---

### Task 2: Add the new prompt-scoring module with optional SenCLIP support

**Files:**
- Create: `scripts/prompt_detector.py`

- [ ] **Step 1: Write the failing test for backend loading and prompt-set structure**

```python
from scripts.prompt_detector import PromptBank, load_backend, PROMPT_BANKS

def test_prompt_bank_has_spawn_and_confounder_groups():
    assert "default" in PROMPT_BANKS
    bank = PROMPT_BANKS["default"]
    assert bank.spawn
    assert set(bank.confounders) == {"foam_waves", "glint", "sediment", "clouds", "surf_beach"}

def test_load_backend_fails_clearly_when_senclip_unavailable():
    with pytest.raises(RuntimeError, match="SenCLIP|pallavijainpj/SenCLIP"):
        load_backend("senclip", device="cpu")
```

- [ ] **Step 2: Run the test to confirm the missing module/backend failure**

Run: `pytest tests/test_prompt_detector.py::test_load_backend_fails_clearly_when_senclip_unavailable -v`

- [ ] **Step 3: Implement a RAM-safe backend layer and scoring API**

```python
@dataclass(frozen=True)
class PromptBank:
    name: str
    spawn: tuple[str, ...]
    confounders: dict[str, tuple[str, ...]]

def load_backend(name: str, device: str = "cpu") -> PromptBackend:
    ...

def score_image(path: str, backend: PromptBackend, prompt_bank: PromptBank, *, device: str = "cpu") -> dict:
    ...

def score_directory(image_dir: str, backend: PromptBackend, prompt_bank: PromptBank, *, batch_size: int = 1, max_crops: int = 8, crop_mode: str = "topk_mean") -> list[dict]:
    ...
```

Implementation constraints:
- load one model at a time
- default to CPU
- process one image at a time
- compute prompt text embeddings once per prompt bank
- keep crop inference sequential or small-batch only
- release large tensors after each image/crop loop
- if SenCLIP cannot load, raise an actionable error that names the checkpoint and expected optional dependencies

- [ ] **Step 4: Run module-focused tests until they pass**

Run: `pytest tests/test_prompt_detector.py -v`

Expected: PASS.

---

### Task 3: Add the benchmark CLI and outputs for the golden set

**Files:**
- Create: `scripts/benchmark_prompt_models.py`
- Modify: `scripts/prompt_detector.py` (reuse public scoring helpers)
- Create: `data/prompt_benchmarks/` at runtime

- [ ] **Step 1: Write the failing test for benchmark summary shape**

```python
from scripts.benchmark_prompt_models import build_benchmark_summary

def test_build_benchmark_summary_includes_all_requested_rows():
    rows = [
        {"model": "senclip", "mode": "full", "score": 0.5, "label": 1},
        {"model": "remoteclip", "mode": "multicrop", "score": -0.1, "label": 0},
    ]
    summary = build_benchmark_summary(rows)
    assert "rows" in summary
    assert "metrics" in summary
```

- [ ] **Step 2: Run the benchmark test to confirm it fails first**

Run: `pytest tests/test_prompt_detector.py tests/test_benchmark_prompt_models.py -v`

- [ ] **Step 3: Implement benchmark orchestration**

```python
def build_benchmark_summary(rows):
    ...

def main(argv=None):
    # evaluate the canonical golden set from data/samples/training_manifest.json
    # run: full-image contrastive, multi-crop contrastive, RemoteCLIP if present, SenCLIP if loadable
    # save summary.json plus CSV/HTML review outputs under data/prompt_benchmarks/
```

Metrics behavior:
- prefer AUROC/AP/accuracy when sklearn is available
- otherwise save ranked scores and threshold-derived metrics
- keep the benchmark resilient: unavailable backends become skipped rows, not fatal failures

- [ ] **Step 4: Run a tiny dry-run first, then the real benchmark if feasible**

Run:
`python scripts/benchmark_prompt_models.py --limit 5 --output-dir data/prompt_benchmarks/_dry_run`

Then:
`python scripts/benchmark_prompt_models.py --output-dir data/prompt_benchmarks`

Expected: JSON summary plus CSV or HTML review page.

---

### Task 4: Update repository guidance for the new prompt path

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/agent_handoff.md`

- [ ] **Step 1: Add the new guidance in prose**

Include these points:
- SenCLIP (`pallavijainpj/SenCLIP`) is the first prompt model to try
- score = spawn similarity aggregate minus the maximum confounder similarity group
- multi-scale multi-crop is recommended for small plumes
- use RAM-safe sequential inference and small crop batches
- BioCLIP/BiomedCLIP are not recommended for nadir Sentinel-2 imagery

- [ ] **Step 2: Keep the wording consistent with existing docs and note the fallback behavior**

Mention that prompt scoring is additive and does not replace the current RemoteCLIP pipeline.

- [ ] **Step 3: Run doc lint/spot checks manually**

Run: `python - <<'PY'\nfrom pathlib import Path\nfor p in [Path('AGENTS.md'), Path('docs/agent_handoff.md')]:\n    print(p, 'ok' if p.exists() else 'missing')\nPY`

Expected: both files exist and mention the new prompt path.

---

### Verification

- Run `pytest tests/test_prompt_detector.py -v`
- Run `pytest tests/test_benchmark_prompt_models.py -v` if created
- Run a dry benchmark on `--limit 5`
- Run the full benchmark only if model downloads and time permit
- Confirm `git status --short` shows only the intended source/doc/test files plus generated benchmark outputs (if any)
