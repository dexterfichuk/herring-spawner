#!/usr/bin/env python3
"""Upload generated herring-spawn datasets to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import create_repo, upload_folder


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

    if dry_run:
        return [str(path) for path in selected]

    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    uploaded: list[str] = []
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
