#!/usr/bin/env python3
"""
Train a classifier on GeoRSCLIP embeddings from labeled spawn/no-spawn images.

1. Loads all labeled PNGs, extracts 512-d GeoRSCLIP embeddings
2. Trains a small classifier (Random Forest + optional MLP)
3. Scores all unlabeled images, saves soft labels
4. Can extend training: you label more → retrain → get better scores

Usage:
  .venv/bin/python3 scripts/train_georsclip_classifier.py              # train + score
  .venv/bin/python3 scripts/train_georsclip_classifier.py --retrain     # reload previous model, add new labels
  .venv/bin/python3 scripts/train_georsclip_classifier.py --export /path/to/new_candidates.csv
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import open_clip
import torch

IMAGE_DIR = Path("/Volumes/Z Slim/herring-spawn-data/candidates_fresh")
MANIFEST_PATH = IMAGE_DIR / "manifest.json"
LABELS_PATH = IMAGE_DIR / "labels.json"
EMBEDDINGS_PATH = IMAGE_DIR / "georsclip_embeddings.npz"
MODEL_PATH = IMAGE_DIR / "georsclip_classifier.pkl"
SCALER_PATH = IMAGE_DIR / "georsclip_scaler.pkl"

IMG_SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model():
    """Load GeoRSCLIP model."""
    ckpt = str(Path.home() / ".cache/huggingface/hub/models--Zilun--GeoRSCLIP/snapshots/"
               "4920188e6eba4e711ef9848cfd7cb77e874ee33f/ckpt/RS5M_ViT-B-32.pt")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model = model.to(device).eval()
    return model


def encode_image(model, img: Image.Image) -> np.ndarray:
    """512-d normalized GeoRSCLIP embedding."""
    img = img.convert("RGB")
    w, h = img.size
    size = min(w, h)
    img = img.crop(((w - size) // 2, (h - size) // 2, (w + size) // 2, (h + size) // 2))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()


def get_embeddings(model, fnames: list[str], cache: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, list[str]]:
    """Get embeddings for a list of filenames, using cache when possible."""
    embeddings, valid_fnames = [], []
    cache = cache or {}

    for fname in fnames:
        if fname in cache:
            embeddings.append(cache[fname])
            valid_fnames.append(fname)
            continue

        img_path = IMAGE_DIR / fname
        if not img_path.exists():
            continue

        try:
            img = Image.open(str(img_path))
            emb = encode_image(model, img)
            embeddings.append(emb)
            valid_fnames.append(fname)
        except Exception:
            continue

    return np.array(embeddings, dtype=np.float32), valid_fnames


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--export", type=str, default=None, help="Export soft-labeled candidates to CSV")
    args = parser.parse_args()

    if not LABELS_PATH.exists():
        print("No labels found. Label some images first.")
        sys.exit(1)

    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Load model
    model = load_model()

    # Load or compute embeddings
    fnames_all = [e["filename"] for e in manifest if e.get("filename") and e.get("status") == "ok"
                  and Path(IMAGE_DIR / e["filename"]).exists()]

    if EMBEDDINGS_PATH.exists() and not args.retrain:
        cached = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        emb_cache = dict(zip(cached["fnames"], cached["embeddings"]))
        print(f"Loaded {len(emb_cache)} cached embeddings")
    else:
        emb_cache = {}

    # Compute any missing embeddings
    missing = [f for f in fnames_all if f not in emb_cache]
    if missing:
        print(f"Computing embeddings for {len(missing)} images...")
        batch_size = 50
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i+batch_size]
            for fname in batch:
                try:
                    img = Image.open(str(IMAGE_DIR / fname))
                    emb_cache[fname] = encode_image(model, img)
                except Exception:
                    pass
                time.sleep(0.02)
            print(f"  {min(i+batch_size, len(missing))}/{len(missing)}")
        # Save cache
        fnames_arr = np.array(list(emb_cache.keys()))
        embs_arr = np.array(list(emb_cache.values()))
        np.savez_compressed(EMBEDDINGS_PATH, fnames=fnames_arr, embeddings=embs_arr)
        print("Embeddings cached.")

    # Build training set from labeled images
    X_train, y_train = [], []
    train_fnames = []
    for fname, label in labels.items():
        if label == "skip" or fname not in emb_cache:
            continue
        X_train.append(emb_cache[fname])
        y_train.append(1 if label == "spawn" else 0)
        train_fnames.append(fname)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    print(f"\nTraining: {len(X_train)} labeled ({int(n_pos)} spawn, {int(n_neg)} no-spawn)")

    if len(X_train) < 10:
        print("Need at least 10 labeled images.")
        sys.exit(1)

    # Train classifier
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Try RF first
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y_train)

    # Cross-val score
    cv_scores = cross_val_score(rf, X_scaled, y_train, cv=min(5, len(X_train) // 3), scoring="roc_auc")
    print(f"RF cross-val ROC-AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

    # Train/val split for report
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train)
    rf.fit(X_tr, y_tr)
    val_preds = rf.predict_proba(X_val)[:, 1]
    val_binary = (val_preds > 0.5).astype(int)
    print(f"\nValidation report (20% holdout):")
    print(classification_report(y_val, val_binary, target_names=["no-spawn", "spawn"]))
    val_roc = roc_auc_score(y_val, val_preds)
    print(f"ROC-AUC: {val_roc:.3f}")

    # Retrain on full set
    rf.fit(X_scaled, y_train)

    # Score all unlabeled images
    unlabeled = [f for f in fnames_all if f not in labels]
    if unlabeled:
        X_unlabeled = np.array([emb_cache[f] for f in unlabeled])
        X_unlabeled_scaled = scaler.transform(X_unlabeled)
        scores = rf.predict_proba(X_unlabeled_scaled)[:, 1]

        print(f"\nScored {len(unlabeled)} unlabeled images")
        top_n = min(20, len(unlabeled))
        top_idx = np.argsort(-scores)[:top_n]
        print(f"Top-{top_n} spawn probabilities:")
        for i in top_idx:
            entry = next(e for e in manifest if e.get("filename") == unlabeled[i])
            print(f"  {scores[i]:.3f} — {entry.get('region','?')} {unlabeled[i]}")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nModel saved to {MODEL_PATH}")

    # Reorder manifest by spawn probability (highest first)
    new_manifest = []
    seen = set()

    if unlabeled:
        scored = sorted(zip(unlabeled, scores), key=lambda x: -x[1])
        for fname, score in scored:
            entry = next(e for e in manifest if e.get("filename") == fname)
            entry["spawn_score"] = float(score)
            entry["rank_source"] = "georsclip_rf"
            new_manifest.append(entry)
            seen.add(id(entry))

    for entry in manifest:
        if id(entry) not in seen:
            new_manifest.append(entry)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(new_manifest, f, indent=2)
    print("Manifest reordered by spawn probability.")

    # Export soft-labeled candidates if requested
    if args.export:
        import csv
        out_path = Path(args.export)
        rows = []
        for fname, score in sorted(zip(unlabeled, scores), key=lambda x: -x[1]):
            entry = next(e for e in manifest if e.get("filename") == fname)
            rows.append({
                "filename": fname,
                "soft_label": "spawn" if score > 0.5 else "no-spawn",
                "confidence": round(score, 4),
                "region": entry.get("region", ""),
                "lat": entry.get("lat", ""),
                "lon": entry.get("lon", ""),
                "dfo_date": entry.get("date", ""),
                "scene_date": entry.get("scene_date", ""),
            })

        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Soft-labeled candidates exported to {out_path}")

    print(f"\nNext steps:")
    print(f"  1. Relaunch label app:  .venv/bin/python3 scripts/label_app.py")
    print(f"  2. Train temporal:      .venv/bin/python3 scripts/train_tempcnn.py")
    print(f"  3. Retrain classifier:  .venv/bin/python3 scripts/train_georsclip_classifier.py --retrain")


if __name__ == "__main__":
    main()
