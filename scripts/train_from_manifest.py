#!/usr/bin/env python3
"""Train SVM classifier from training_manifest.json — specific 8 positives + all negatives.

Reads the manifest to get exact positive filenames (to exclude rose-verified
spawns that are also in the directory), loads all negatives, trains DINOv2+SVM.

Usage:
    python scripts/train_from_manifest.py \
        --manifest data/samples/training_manifest.json \
        --positive-dir data/samples/positive \
        --negative-dir data/samples/negative \
        --output-model data/models/svm_8pos_126neg.pkl
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import transforms

MODEL_NAME = "dinov2_vits14"
EMBED_DIM = 384

DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_embeddings_from_manifest(
    manifest_path: Path,
    pos_dir: Path,
    neg_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load embeddings for files specified in manifest + all negatives.

    Returns (embeddings, labels, filenames, errors).
    labels[i] = 1 for positive (spawn), 0 for negative (no spawn).
    """
    manifest = json.loads(manifest_path.read_text())
    pos_fnames: list[str] = manifest["positives"]

    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    filenames: list[str] = []
    errors: list[str] = []

    # Load only manifest-specified positives
    for fname in pos_fnames:
        fpath = pos_dir / fname
        if not fpath.exists():
            errors.append(f"{fname}: file not found in {pos_dir}")
            continue
        try:
            img = Image.open(fpath).convert("RGB")
            tensor = DINO_TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor)
            emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
            embeddings.append(emb)
            labels.append(1)
            filenames.append(fname)
        except Exception as exc:
            errors.append(f"{fname}: {exc}")

    # Load all negatives
    for fpath in sorted(neg_dir.glob("*.png")):
        try:
            img = Image.open(fpath).convert("RGB")
            tensor = DINO_TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor)
            emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
            embeddings.append(emb)
            labels.append(0)
            filenames.append(fpath.name)
        except Exception as exc:
            errors.append(f"{fpath.name}: {exc}")

    return np.array(embeddings), np.array(labels), filenames, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train SVM from training manifest (specific positives + all negatives)",
    )
    parser.add_argument("--manifest", default="data/samples/training_manifest.json")
    parser.add_argument("--positive-dir", default="data/samples/positive")
    parser.add_argument("--negative-dir", default="data/samples/negative")
    parser.add_argument("--output-model", default="data/models/svm_8pos_126neg.pkl")
    parser.add_argument("--kernel", default="rbf", choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = repo_root / args.manifest
    pos_dir = repo_root / args.positive_dir
    neg_dir = repo_root / args.negative_dir
    model_path = repo_root / args.output_model

    # 1. Load DINOv2
    print("=" * 60)
    print("  Loading DINOv2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval()
    model = model.to(device)
    print(f"  Model: {MODEL_NAME} ({EMBED_DIM}-dim)")

    # 2. Load embeddings from manifest
    print(f"\n{'=' * 60}")
    print("  Loading labeled samples from manifest...")
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text())
    print(f"  Manifest: {manifest['positive_count']} positives, {manifest['negative_count']} negatives")

    X, y, fnames, errors = load_embeddings_from_manifest(
        manifest_path, pos_dir, neg_dir, model, device
    )
    n_pos = int(y.sum())
    n_neg = int(len(y) - y.sum())
    print(f"  Loaded {len(X)} samples: {n_pos} positive, {n_neg} negative")
    print(f"  Embedding dim: {X.shape[1]}")
    if errors:
        print(f"  WARNING: {len(errors)} errors:")
        for e in errors[:5]:
            print(f"    - {e}")

    if len(X) < 10:
        print("ERROR: Too few samples (<10)")
        return 1
    if n_pos < 2 or n_neg < 2:
        print("ERROR: Need >=2 per class")
        return 1

    # 3. Train SVM
    print(f"\n{'=' * 60}")
    print(f"  Training SVM (kernel={args.kernel}, class_weight='balanced')...")
    svm = SVC(
        kernel=args.kernel,
        class_weight="balanced",
        probability=True,
        random_state=args.random_state,
        gamma="scale",
    )
    svm.fit(X, y)

    # Full-dataset metrics
    y_pred = svm.predict(X)
    full_acc = accuracy_score(y, y_pred)
    y_decision = svm.decision_function(X)
    pos_scores = y_decision[y == 1]
    neg_scores = y_decision[y == 0]
    separation = float(np.mean(pos_scores) - np.mean(neg_scores))

    print(f"\n  FULL DATASET RESULTS")
    print(f"  Accuracy: {full_acc:.4f}")
    print(classification_report(y, y_pred, target_names=["negative", "positive"]))
    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                Neg   Pos")
    print(f"  Actual Neg    {cm[0][0]:<5} {cm[0][1]:<5}")
    print(f"         Pos    {cm[1][0]:<5} {cm[1][1]:<5}")
    print(f"  Separation: {separation:.4f}")

    # 4. Cross-validation
    n_folds = min(args.cv_folds, min(n_pos, n_neg))
    if n_folds >= 3:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.random_state)
        cv_svm = SVC(
            kernel=args.kernel,
            class_weight="balanced",
            random_state=args.random_state,
            gamma="scale",
        )
        cv_scores = cross_val_score(cv_svm, X, y, cv=cv, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        print(f"\n  CV ({n_folds}-fold): accuracy = {cv_mean:.4f} +/- {cv_std:.4f}")
    else:
        cv_mean = cv_std = 0.0
        print("\n  CV skipped (too few per class)")

    # 5. Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "svm": svm,
        "embed_dim": EMBED_DIM,
        "model_name": MODEL_NAME,
        "kernel": args.kernel,
        "class_weight": "balanced",
        "n_train": len(X),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "full_accuracy": float(full_acc),
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "separation": separation,
        "manifest_source": str(manifest_path),
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    summary_path = model_path.with_suffix(".summary.json")
    summary_data = {k: v for k, v in model_data.items() if k != "svm"}
    summary_path.write_text(json.dumps(summary_data, indent=2))

    print(f"\n  Model saved:   {model_path}")
    print(f"  Summary saved: {summary_path}")
    print(f"\n{'=' * 60}")
    print(f"  Summary: SVM {args.kernel.upper()} | CV {cv_mean:.1%} +/- {cv_std:.1%} | "
          f"Full {full_acc:.1%} | Sep {separation:.4f}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
