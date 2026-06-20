#!/usr/bin/env python3
"""Train an SVM classifier on DINOv2 embeddings from labeled samples.

Trains on the full labeled dataset with class-balanced RBF SVM.
Uses stratified cross-validation for evaluation (no held-out test set,
since we only have 70 samples). Reports accuracy, precision, recall,
confusion matrix, and separation.

Usage:
    python scripts/train_classifier.py \
        --positive-dir data/samples/positive \
        --negative-dir data/samples/negative \
        --output-model data/models/dinov2_svm.pkl \
        --output-vectors data/embeddings/training_vectors.npz
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


def load_labels_and_embeddings(
    pos_dir: Path, neg_dir: Path, model: torch.nn.Module, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load all labeled PNGs, compute DINOv2 embeddings.

    Returns (embeddings, labels, filenames, errors).
    labels[i] = 1 for positive (spawn), 0 for negative (no spawn).
    """
    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    filenames: list[str] = []
    errors: list[str] = []

    for label_val, search_dir in [(1, pos_dir), (0, neg_dir)]:
        paths = sorted(search_dir.glob("*.png"))
        if not paths:
            print(f"  WARNING: No samples found in {search_dir}")
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                tensor = DINO_TRANSFORM(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(tensor)
                emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
                embeddings.append(emb)
                labels.append(label_val)
                filenames.append(p.name)
            except Exception as exc:
                errors.append(f"{p.name}: {exc}")

    return np.array(embeddings), np.array(labels), filenames, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train SVM classifier on DINOv2 embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--positive-dir", default="data/samples/positive")
    parser.add_argument("--negative-dir", default="data/samples/negative")
    parser.add_argument("--output-model", default="data/models/dinov2_svm.pkl")
    parser.add_argument("--output-vectors", default="data/embeddings/training_vectors.npz")
    parser.add_argument("--kernel", default="rbf", choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    pos_dir = repo_root / args.positive_dir
    neg_dir = repo_root / args.negative_dir
    model_path = repo_root / args.output_model
    vectors_path = repo_root / args.output_vectors

    # ------------------------------------------------------------------
    # 1. Load DINOv2 model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Loading DINOv2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    try:
        model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
        model.eval()
        model = model.to(device)
    except Exception as exc:
        print(f"ERROR: Failed to load DINOv2: {exc}")
        return 1
    print(f"  Model: {MODEL_NAME} ({EMBED_DIM}-dim embeddings)")

    # ------------------------------------------------------------------
    # 2. Load labeled samples and compute embeddings
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Loading labeled samples and computing embeddings...")
    if not pos_dir.exists():
        print(f"ERROR: Positive samples directory not found: {pos_dir}")
        return 1
    if not neg_dir.exists():
        print(f"ERROR: Negative samples directory not found: {neg_dir}")
        return 1

    X, y, filenames, errors = load_labels_and_embeddings(pos_dir, neg_dir, model, device)
    n_pos = int(y.sum())
    n_neg = int(len(y) - y.sum())
    print(f"  Loaded {len(X)} labeled samples: {n_pos} positive, {n_neg} negative")
    print(f"  Embedding dimension: {X.shape[1]}")
    if errors:
        print(f"  WARNING: {len(errors)} sample(s) failed to load:")
        for err in errors[:5]:
            print(f"    - {err}")

    if len(X) < 10:
        print("ERROR: Too few samples to train a classifier (need at least 10).")
        return 1
    if n_pos < 2 or n_neg < 2:
        print("ERROR: Need at least 2 samples per class.")
        return 1

    # ------------------------------------------------------------------
    # 3. Save training vectors for reproducibility
    # ------------------------------------------------------------------
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(vectors_path, embeddings=X, labels=y, filenames=filenames)
    print(f"\n  Saved training vectors to {vectors_path}")

    # ------------------------------------------------------------------
    # 4. Train SVM on full dataset with class balancing
    # ------------------------------------------------------------------
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

    # Full-dataset predictions
    y_pred = svm.predict(X)
    full_acc = accuracy_score(y, y_pred)

    # Decision function scores for all samples
    y_decision = svm.decision_function(X)
    pos_scores = y_decision[y == 1]
    neg_scores = y_decision[y == 0]
    separation = float(np.mean(pos_scores) - np.mean(neg_scores))

    print(f"\n  FULL DATASET RESULTS")
    print(f"  {'-' * 40}")
    print(f"  Accuracy:  {full_acc:.4f}")
    print(f"\n  Classification Report:")
    print(f"  {classification_report(y, y_pred, target_names=['negative', 'positive'])}")
    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                Neg   Pos")
    print(f"  Actual Neg    {cm[0][0]:<5} {cm[0][1]:<5}")
    print(f"         Pos    {cm[1][0]:<5} {cm[1][1]:<5}")
    print(f"\n  Separation (pos_mean - neg_mean decision): {separation:.4f}")

    # ------------------------------------------------------------------
    # 5. Cross-validation on full dataset (honest evaluation)
    # ------------------------------------------------------------------
    n_folds = min(args.cv_folds, min(n_pos, n_neg))
    if n_folds >= 3:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.random_state)
        # Use a fresh estimator for CV (class_weight='balanced' same as final)
        cv_svm = SVC(
            kernel=args.kernel,
            class_weight="balanced",
            random_state=args.random_state,
            gamma="scale",
        )
        cv_scores = cross_val_score(cv_svm, X, y, cv=cv, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        print(f"\n  Cross-validation ({n_folds}-fold): accuracy = {cv_mean:.4f} +/- {cv_std:.4f}")
    else:
        cv_mean = cv_std = 0.0
        print("\n  Cross-validation skipped (too few samples per class).")

    # Also evaluate at threshold 0 for the similarity baseline comparison
    n_above = int((y_decision >= 0).sum())
    print(f"\n  Decision >= 0: {n_above}/{len(y_decision)} ({100.0 * n_above / len(y_decision):.1f}%)")

    # ------------------------------------------------------------------
    # 6. Save model and metadata
    # ------------------------------------------------------------------
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "svm": svm,
        "embed_dim": EMBED_DIM,
        "model_name": MODEL_NAME,
        "kernel": args.kernel,
        "class_weight": "balanced",
        "n_train": len(X),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "full_accuracy": float(full_acc),
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "separation": separation,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    summary_path = model_path.with_suffix(".summary.json")
    summary_data = {k: v for k, v in model_data.items() if k != "svm"}
    summary_path.write_text(json.dumps(summary_data, indent=2))

    print(f"\n  Model saved to:   {model_path}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Vectors saved to: {vectors_path}")
    print(f"\n  {'=' * 60}")
    print(f"  Summary: SVM {args.kernel.upper()} | CV {cv_mean:.1%} +/- {cv_std:.1%} | "
          f"Full {full_acc:.1%} | Sep {separation:.4f}")
    print(f"  Usage:  python scripts/scan_bc_coast.py --classifier svm")
    print(f"  {'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
