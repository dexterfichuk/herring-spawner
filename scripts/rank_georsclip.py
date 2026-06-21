#!/usr/bin/env python3
"""
GeoRSCLIP embeddings + KNN to rank unlabeled spawn candidates.

For each labeled image (spawn/no-spawn), extract a GeoRSCLIP visual embedding.
Then for each unlabeled image, find its nearest neighbors among the labeled set
and score it by how many spawn neighbors it has.

Ranked manifest is saved — relaunch label_app.py to review top candidates.

Usage:
  .venv/bin/python3 scripts/rank_georsclip.py
  .venv/bin/python3 scripts/label_app.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors

IMAGE_DIR = Path("/Volumes/Z Slim/herring-spawn-data/candidates_fresh")
MANIFEST_PATH = IMAGE_DIR / "manifest.json"
LABELS_PATH = IMAGE_DIR / "labels.json"
EMBEDDINGS_CACHE = IMAGE_DIR / "georsclip_embeddings.npz"

# ImageNet mean/std used by open_clip
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
IMG_SIZE = 224

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def preprocess(img: Image.Image) -> torch.Tensor:
    """Resize, center-crop, normalize to 224x224 tensor."""
    img = img.convert("RGB")
    w, h = img.size
    # Center crop to square
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / np.float32(255.0)
    arr = (arr - MEAN) / STD
    # (H, W, C) -> (C, H, W) -> batch dim
    arr = arr.astype(np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def load_model():
    ckpt_path = os.path.join(
        os.path.expanduser("~"),
        ".cache/huggingface/hub/models--Zilun--GeoRSCLIP/snapshots",
        "4920188e6eba4e711ef9848cfd7cb77e874ee33f/ckpt/RS5M_ViT-B-32.pt",
    )
    print("Loading GeoRSCLIP backbone...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded.")
    return model


def encode_image(model, img: Image.Image) -> np.ndarray:
    """Compute normalized GeoRSCLIP embedding vector for a PIL image."""
    tensor = preprocess(img)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()


def main():
    if not LABELS_PATH.exists():
        print("No labels found. Label some images first.")
        sys.exit(1)

    with open(LABELS_PATH) as f:
        labels = json.load(f)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Build list of all entries with images that need embedding
    entries_to_embed = []
    for entry in manifest:
        fname = entry.get("filename")
        if not fname or entry.get("status") != "ok":
            continue
        img_path = IMAGE_DIR / fname
        if img_path.exists():
            entries_to_embed.append((fname, entry))

    n_total = len(entries_to_embed)
    print(f"Total images to embed: {n_total}")

    # Determine which are labeled
    labeled_indices = [
        i for i, (fname, _) in enumerate(entries_to_embed) if fname in labels and labels[fname] != "skip"
    ]
    unlabeled_indices = [
        i for i, (fname, _) in enumerate(entries_to_embed) if fname not in labels
    ]
    print(f"Labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")

    if len(labeled_indices) < 5:
        print("Need at least 5 labeled images (excluding skips).")
        sys.exit(1)

    # Load model
    model = load_model()

    # Compute all embeddings (with cache)
    cache_path = str(EMBEDDINGS_CACHE)
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        cached_fnames = list(cached["fnames"])
        cached_embs = cached["embeddings"]
        # Map cached fname -> embedding
        emb_map = dict(zip(cached_fnames, cached_embs))
        print(f"Loaded {len(emb_map)} cached embeddings")
    else:
        emb_map = {}

    embeddings = []
    new_count = 0
    for fname, entry in entries_to_embed:
        if fname in emb_map:
            embeddings.append(emb_map[fname])
        else:
            img_path = IMAGE_DIR / fname
            img = Image.open(img_path)
            emb = encode_image(model, img)
            embeddings.append(emb)
            emb_map[fname] = emb
            new_count += 1
            if new_count % 100 == 0:
                print(f"  Encoded {new_count}/{n_total}...")
                # Save checkpoint
                fnames_arr = np.array(list(emb_map.keys()))
                embs_arr = np.array(list(emb_map.values()))
                np.savez_compressed(cache_path, fnames=fnames_arr, embeddings=embs_arr)

    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save all embeddings
    fnames_arr = np.array([fn for fn, _ in entries_to_embed])
    np.savez_compressed(cache_path, fnames=fnames_arr, embeddings=embeddings)
    print("Embeddings cached.")

    # KNN: for each unlabeled, find labeled neighbors
    labeled_embs = embeddings[labeled_indices]
    labeled_labels = np.array([
        1 if entries_to_embed[i][0] in labels and labels[entries_to_embed[i][0]] == "spawn" else 0
        for i in labeled_indices
    ])

    n_labeled_pos = labeled_labels.sum()
    n_labeled_neg = len(labeled_labels) - n_labeled_pos
    print(f"Labeled: {n_labeled_pos:.0f} spawn, {n_labeled_neg} no-spawn")

    # Fit KNN on labeled set
    knn = NearestNeighbors(n_neighbors=min(15, len(labeled_indices)), metric="cosine")
    knn.fit(labeled_embs)

    # Score each unlabeled
    unlabeled_embs = embeddings[unlabeled_indices]
    distances, neighbor_indices = knn.kneighbors(unlabeled_embs)

    # Score = fraction of neighbors that are spawn
    scores = np.zeros(len(unlabeled_indices))
    for i, neighbors in enumerate(neighbor_indices):
        neighbor_labels = labeled_labels[neighbors]
        # Weight closer neighbors more
        weights = 1.0 / (distances[i] + 0.001)
        scores[i] = np.average(neighbor_labels, weights=weights)

    # Build score map
    score_map = {}
    for idx_in_list, score in zip(unlabeled_indices, scores):
        fname = entries_to_embed[idx_in_list][0]
        score_map[fname] = float(score)

    # Also store spawn_distance (how far the nearest labeled spawn is)
    spawn_indices = [i for i in labeled_indices if labeled_labels[labeled_indices.index(i)] == 1]
    if spawn_indices:
        spawn_embs = embeddings[spawn_indices]
        spawn_knn = NearestNeighbors(n_neighbors=1, metric="cosine")
        spawn_knn.fit(spawn_embs)
        spawn_dists, _ = spawn_knn.kneighbors(unlabeled_embs)
        for idx_in_list, dist in zip(unlabeled_indices, spawn_dists):
            fname = entries_to_embed[idx_in_list][0]
            score_map[fname] = score_map.get(fname, 0) * (1 - float(dist[0]) * 0.5)  # penalize far from spawn

    print(f"\nScored {len(scores)} images")
    sorted_scores = sorted(score_map.values(), reverse=True)
    print(f"Score range: {sorted_scores[-1]:.3f} - {sorted_scores[0]:.3f}")
    print(f"Top-5: {sorted_scores[:5]}")

    # Reorder manifest: highest score first
    reindexed = []
    seen = set()

    # 1. Scored unlabeled by descending score
    scored_pairs = [(score_map.get(e[0], 0), i) for i, e in enumerate(entries_to_embed) if e[0] in score_map]
    scored_pairs.sort(key=lambda x: -x[0])
    for _, idx in scored_pairs:
        reindexed.append(entries_to_embed[idx][1])
        seen.add(idx)

    # 2. Already labeled, then failures
    for i, (fname, entry) in enumerate(entries_to_embed):
        if i not in seen:
            reindexed.append(entry)
            seen.add(i)

    # Add remaining (no-scene, download_failed, etc.)
    all_indices = {i for i in range(len(manifest))}
    seen_indices = set()
    for entry in reindexed:
        for i, e in enumerate(manifest):
            if e is entry:
                seen_indices.add(i)
                break

    for i, entry in enumerate(manifest):
        if i not in seen_indices:
            reindexed.append(entry)

    # Attach scores to manifest entries
    for entry in reindexed:
        fname = entry.get("filename")
        if fname and fname in score_map:
            entry["spawn_score"] = score_map[fname]
            entry["rank_source"] = "georsclip_knn"

    with open(MANIFEST_PATH, "w") as f:
        json.dump(reindexed, f, indent=2)

    print(f"\nDone. Manifest reordered by GeoRSCLIP+KNN similarity to your labeled spawns.")
    print(f"Top spawn-like candidates are now first.")
    print(f"\nRelaunch:  .venv/bin/python3 scripts/label_app.py")


if __name__ == "__main__":
    main()
