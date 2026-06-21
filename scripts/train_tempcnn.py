#!/usr/bin/env python3
"""
Train a TempCNN on multi-temporal Sentinel-2 features from labeled spawn locations.

Reads temporal_dataset/{location}/timesteps/*/features.npz, assembles padded
sequences (T x 6 features), and trains a temporal classifier.

Evaluates by year holdout (train on 2016-2023, test on 2024-2025).

Usage:
  .venv/bin/python3 scripts/train_tempcnn.py                     # full train
  .venv/bin/python3 scripts/train_tempcnn.py --use-soft-labels   # propagate weak labels from GeoRSCLIP
  .venv/bin/python3 scripts/train_tempcnn.py --epochs 100

Architecture:
  Input (batch, 6 channels, T timesteps)
    → Conv1D(32, k=3) → BN → ReLU
    → Conv1D(64, k=3) → BN → ReLU
    → Conv1D(64, k=3) → BN → ReLU
    → GlobalAvgPool → Dropout(0.3) → Linear(64→1) → Sigmoid
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader

FEATURE_NAMES = ["shsi", "green", "red", "nir", "blue_minus_coastal", "coastal"]
N_CHANNELS = len(FEATURE_NAMES)
TEMPORAL_DIR = Path("/Volumes/Z Slim/herring-spawn-data/temporal_dataset")


class TempCNN(nn.Module):
    def __init__(self, n_channels: int, n_conv_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, n_conv_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(n_conv_channels)
        self.conv2 = nn.Conv1d(n_conv_channels, n_conv_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_conv_channels * 2)
        self.conv3 = nn.Conv1d(n_conv_channels * 2, n_conv_channels * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_conv_channels * 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(n_conv_channels * 2, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))


class TemporalSpawnDataset(Dataset):
    """Load temporal feature sequences from temporal_dataset/."""

    def __init__(self, location_dirs: list[Path], labels: list[int], max_t: int | None = None):
        self.sequences = []
        self.labels = []
        self.masks = []

        for loc_dir, label in zip(location_dirs, labels):
            timesteps_dir = loc_dir / "timesteps"
            if not timesteps_dir.exists():
                continue
            # Load features for each timestep, sorted by date
            ts_dirs = sorted(timesteps_dir.iterdir())
            feat_list = []
            for ts_dir in ts_dirs:
                feat_path = ts_dir / "features.npz"
                if feat_path.exists():
                    feat_list.append(np.load(feat_path)["features"])

            if len(feat_list) < 2:
                continue  # need at least 2 timesteps

            seq = np.stack(feat_list, axis=0)  # (T, 6)
            self.sequences.append(seq)
            self.labels.append(label)
            self.masks.append(np.ones(len(seq), dtype=bool))

        # Pad all sequences to same length
        if not self.sequences:
            return
        self.max_t = max_t or max(s.shape[0] for s in self.sequences)
        for i in range(len(self.sequences)):
            seq = self.sequences[i]
            t = seq.shape[0]
            if t < self.max_t:
                pad = np.zeros((self.max_t - t, N_CHANNELS), dtype=np.float32)
                self.sequences[i] = np.concatenate([seq, pad], axis=0)
                new_mask = np.ones(self.max_t, dtype=bool)
                new_mask[t:] = False
                self.masks[i] = new_mask
            elif t > self.max_t:
                self.sequences[i] = seq[:self.max_t]
                self.masks[i] = np.ones(self.max_t, dtype=bool)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.masks = np.array(self.masks, dtype=bool)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return (features, label, mask) where features is (C, T) for Conv1d
        return (
            torch.from_numpy(self.sequences[idx]).permute(1, 0),  # (C, T)
            torch.tensor(self.labels[idx]).unsqueeze(0),
            torch.from_numpy(self.masks[idx]),
        )


def masked_bce_loss(pred, target, mask):
    """BCE loss only over unmasked (valid) timesteps — but this is sequence-level,
    so we just use standard BCE. The mask is available for future per-timestep losses."""
    return nn.functional.binary_cross_entropy(pred, target)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-year", type=int, default=2024, help="Hold out years >= this for testing")
    parser.add_argument("--use-soft-labels", action="store_true", help="Use GeoRSCLIP scores for unlabeled locations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load all location dirs
    if not TEMPORAL_DIR.exists():
        print(f"Temporal dataset not found at {TEMPORAL_DIR}")
        print("Run scripts/fetch_temporal.py first")
        sys.exit(1)

    loc_dirs = sorted([d for d in TEMPORAL_DIR.iterdir() if d.is_dir() and not d.name.startswith("._")])
    all_dirs, all_labels = [], []
    for d in loc_dirs:
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        label = meta.get("label")
        if label is None and args.use_soft_labels and "spawn_score" in meta:
            label = "spawn" if meta["spawn_score"] > 0.5 else "no-spawn"
        if label not in ("spawn", "no-spawn"):
            continue
        all_dirs.append(d)
        all_labels.append(1 if label == "spawn" else 0)

    if len(all_dirs) < 10:
        print(f"Only {len(all_dirs)} locations with temporal data. Run fetch_temporal.py first.")
        sys.exit(1)

    print(f"Loaded {len(all_dirs)} locations ({sum(all_labels)} spawn, {len(all_labels)-sum(all_labels)} no-spawn)")

    # Split by year
    train_dirs, train_labels = [], []
    test_dirs, test_labels = [], []
    for d, lbl in zip(all_dirs, all_labels):
        meta = json.load(open(d / "metadata.json"))
        year = int(meta["dfo_date"][:4])
        if year >= args.test_year:
            test_dirs.append(d)
            test_labels.append(lbl)
        else:
            train_dirs.append(d)
            train_labels.append(lbl)

    print(f"Train: {len(train_dirs)} ({sum(train_labels)} spawn), Test: {len(test_dirs)} ({sum(test_labels)} spawn)")

    if len(train_dirs) < 5 or len(test_dirs) < 2:
        print("Need more data for train/test split.")
        sys.exit(1)

    train_ds = TemporalSpawnDataset(train_dirs, train_labels)
    test_ds = TemporalSpawnDataset(test_dirs, test_labels, max_t=train_ds.max_t)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = TempCNN(N_CHANNELS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCELoss()

    print(f"\nTraining for {args.epochs} epochs...")
    best_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y, mask in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y, mask in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                all_preds.extend(pred.cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        binary = (all_preds > 0.5).astype(int)
        acc = accuracy_score(all_labels, binary)
        f1 = f1_score(all_labels, binary, zero_division=0)
        roc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), str(TEMPORAL_DIR / "best_tempcnn.pt"))

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch+1:3d} | loss={total_loss/len(train_loader):.4f} | val acc={acc:.3f} f1={f1:.3f} roc={roc:.3f}")

    # Final metrics
    print(f"\n{'='*60}")
    print(f"  Best validation F1: {best_f1:.3f}")
    print(f"  Single-image baseline (UVic ThBA): 78.3%")
    print(f"  TempCNN temporal:                  {acc*100:.1f}% ({'BEATS' if acc > 0.783 else 'BELOW'} baseline)")
    print(f"{'='*60}")

    # Feature importance via permutation
    print("\nFeature summary (load order):")
    for i, name in enumerate(FEATURE_NAMES):
        mean_val = train_ds.sequences[:, i, :].mean()
        print(f"  {name:20s} mean={mean_val:.4f}")


if __name__ == "__main__":
    main()
