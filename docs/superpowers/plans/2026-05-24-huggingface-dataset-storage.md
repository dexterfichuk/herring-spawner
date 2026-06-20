# Hugging Face Dataset Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move herring-spawn image-heavy artifacts to the public Hugging Face dataset `dfichuk/herring-spawn-candidates` while keeping GitHub focused on code, docs, tests, and lightweight metadata.

**Architecture:** The Git repo retains scripts, documentation, labels, manifests, and model summaries. A dedicated upload script publishes selected image/review/data directories to Hugging Face using `huggingface_hub`, and `.gitignore` prevents regenerated image payloads from being staged into normal Git history.

**Tech Stack:** Python 3.11+, `huggingface_hub`, Git, Hugging Face Datasets repositories.

---

### Task 1: Remove Image Payloads From The Unpushed Git Commit

**Files:**
- Modify Git history only: reset the unpushed commit while preserving the working tree.

- [ ] **Step 1: Confirm branch is ahead by only the unpushed image-heavy commit**

Run: `git status --branch --short`

Expected: `## main...origin/main [ahead 1]`

- [ ] **Step 2: Preserve working tree while undoing the commit**

Run: `git reset --mixed HEAD~1`

Expected: files from the commit return as unstaged working-tree changes.

- [ ] **Step 3: Confirm the commit was removed locally**

Run: `git status --branch --short`

Expected: branch is no longer ahead, with local unstaged changes visible.

### Task 2: Ignore Hugging Face-Owned Image Artifacts

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Update `.gitignore` with HF-owned image directories**

Add these patterns if absent:

```gitignore
data/candidates_knn/
data/sog_candidates/thumbnails/
data/ingressed/thumbnails/
data/candidates_salmon_coast/
data/candidates_salmon_coast_2023/
data/candidates_salmon_coast_2025/
data/candidates_salmon_coast_2026/
```

- [ ] **Step 2: Verify image directories are ignored**

Run: `git status --short --ignored`

Expected: image-heavy directories appear with `!!` and are not stageable by `git add -A`.

### Task 3: Add Hugging Face Upload Script

**Files:**
- Create: `scripts/upload_hf_dataset.py`

- [ ] **Step 1: Create upload script**

Create `scripts/upload_hf_dataset.py` with:

```python
#!/usr/bin/env python3
"""Upload generated herring-spawn datasets to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


DEFAULT_REPO_ID = "dfichuk/herring-spawn-candidates"
DEFAULT_PATHS = [
    Path("data/candidates_knn"),
    Path("data/candidates_final"),
    Path("data/sog_candidates"),
    Path("data/ingressed"),
    Path("data/candidates_salmon_coast"),
    Path("data/candidates_salmon_coast_2023"),
    Path("data/candidates_salmon_coast_2025"),
    Path("data/candidates_salmon_coast_2026"),
]


def existing_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def upload_dataset(repo_id: str, paths: list[Path], private: bool, dry_run: bool) -> list[str]:
    selected = existing_paths(paths)
    if not selected:
        raise SystemExit("No selected dataset paths exist")

    uploaded: list[str] = []
    if dry_run:
        return [str(path) for path in selected]

    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api = HfApi()
    for path in selected:
        upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(path),
            path_in_repo=str(path),
            commit_message=f"upload {path}",
            ignore_patterns=["**/.DS_Store", "**/__pycache__/**"],
        )
        uploaded.append(str(path))
    api.update_repo_visibility(repo_id, private=private, repo_type="dataset")
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_PATHS)
    args = parser.parse_args()

    uploaded = upload_dataset(args.repo_id, args.paths, args.private, args.dry_run)
    action = "Would upload" if args.dry_run else "Uploaded"
    for path in uploaded:
        print(f"{action}: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify dry run works**

Run: `source .venv/bin/activate && python scripts/upload_hf_dataset.py --dry-run`

Expected: prints selected existing dataset paths without contacting Hugging Face.

### Task 4: Document Dataset Location

**Files:**
- Modify: `docs/agent_handoff.md`
- Modify: `AGENTS.md`
- Modify: `README.md`

- [ ] **Step 1: Add HF dataset references**

Add `https://huggingface.co/datasets/dfichuk/herring-spawn-candidates` as the public home for image-heavy generated assets.

- [ ] **Step 2: Document upload command**

Add:

```bash
source .venv/bin/activate
python -m pip install huggingface_hub
huggingface-cli login
python scripts/upload_hf_dataset.py --repo-id dfichuk/herring-spawn-candidates
```

### Task 5: Upload To Hugging Face

**Files:**
- No repo file changes expected.

- [ ] **Step 1: Check authentication**

Run: `source .venv/bin/activate && python -c "from huggingface_hub import HfApi; print(HfApi().whoami().get('name'))"`

Expected: prints the authenticated Hugging Face username. If it errors with authentication failure, ask the user for a token.

- [ ] **Step 2: Upload public dataset**

Run: `source .venv/bin/activate && python scripts/upload_hf_dataset.py --repo-id dfichuk/herring-spawn-candidates`

Expected: creates or updates a public dataset repo and prints uploaded paths.

### Task 6: Commit And Push Lightweight Git Repo

**Files:**
- Commit only code, docs, tests, labels, manifests, and small model artifacts.

- [ ] **Step 1: Stage non-image changes**

Run: `git add -A`

Expected: ignored image directories remain unstaged.

- [ ] **Step 2: Verify no large files are staged**

Run: `python3 - <<'PY'
import subprocess
from pathlib import Path
files = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).splitlines()
bad = []
for name in files:
    path = Path(name)
    if path.exists() and path.is_file() and path.stat().st_size > 50_000_000:
        bad.append((path.stat().st_size, name))
print('\n'.join(f'{size} {name}' for size, name in bad))
raise SystemExit(1 if bad else 0)
PY`

Expected: exits 0 and prints nothing.

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && rtk pytest -v`

Expected: all tests pass.

- [ ] **Step 4: Commit lightweight repo changes**

Run: `git commit -m "feat: document herring dataset handoff"`

Expected: commit succeeds.

- [ ] **Step 5: Push to GitHub**

Run: `git push`

Expected: branch is up to date with `origin/main`.
