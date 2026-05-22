#!/usr/bin/env python3
"""
Improved herring spawn detector using delta-based scoring and multi-modal features.

Builds on the insight that spawn is best detected by comparing spawn-season
imagery to off-season baseline at the same location. Combines:

  - DINOv2 embedding deltas (via SVM)
  - HSV color feature deltas (turquoise, white foam, green sediment fractions)
  - Texture feature deltas (edge density, GLCM contrast/uniformity)

Usage modes:

  # 1. Train/validate the improved model:
  python scripts/improved_detector.py --mode train \
      --positive-dir data/samples/positive \
      --negative-dir data/samples/negative \
      --output-model data/models/improved_model.pkl

  # 2. Score individual image pairs (delta mode):
  python scripts/improved_detector.py --mode score-pair \
      --spawn-img <path> --off-img <path>

  # 3. Scan BC coast using delta approach (requires GEE auth):
  python scripts/improved_detector.py --mode scan \
      --output data/candidates_v3 \
      --start 2024-02-01 --end 2024-04-30 \
      --off-start 2024-06-01 --off-end 2024-08-31 \
      --max-cloud 50 --workers 4

  # 4. Generate review page from scan results:
  python scripts/improved_detector.py --mode review \
      --candidates-dir data/candidates_v3

  # 5. Clean training data based on rose labels:
  python scripts/improved_detector.py --mode clean-data
"""

import argparse
import glob as glob_mod
import hashlib
import html
import io
import json
import math
import os
import pickle
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn.functional as F
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "dinov2_vits14"
EMBED_DIM = 384

DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Feature weights for final score (from the spec)
WEIGHT_SVM_DELTA = 0.5
WEIGHT_TURQUOISE_DELTA = 0.3
WEIGHT_EDGE_DELTA = 0.2

# Detection thresholds
# Spawn plumes typically cover 0.2-5% of the image frame (rest is dark ocean/land)
MIN_TURQUOISE_FRACTION = 0.002
SCORE_THRESHOLD = 0.0

# BC coast herring habitat regions
REGIONS: list[dict[str, Any]] = [
    {"name": "qualicum", "lat": 49.35, "lon": -124.45, "radius_km": 15},
    {"name": "nanaimo", "lat": 49.15, "lon": -123.85, "radius_km": 15},
    {"name": "comox", "lat": 49.68, "lon": -124.88, "radius_km": 15},
    {"name": "denman-island", "lat": 49.55, "lon": -124.80, "radius_km": 10},
    {"name": "tofino", "lat": 49.15, "lon": -125.90, "radius_km": 15},
    {"name": "ucluelet", "lat": 48.94, "lon": -125.55, "radius_km": 10},
    {"name": "nootka-sound", "lat": 49.60, "lon": -126.60, "radius_km": 15},
    {"name": "quatsino-sound", "lat": 50.50, "lon": -128.00, "radius_km": 15},
    {"name": "spiller-channel", "lat": 52.30, "lon": -128.30, "radius_km": 15},
    {"name": "milbanke-sound", "lat": 52.50, "lon": -128.80, "radius_km": 15},
    {"name": "prince-rupert", "lat": 54.30, "lon": -130.40, "radius_km": 20},
    {"name": "haida-gwaii-south", "lat": 52.40, "lon": -131.40, "radius_km": 15},
    {"name": "masset-inlet", "lat": 53.70, "lon": -132.90, "radius_km": 15},
]


# ===================================================================
# Data Classes
# ===================================================================

@dataclass
class ImageFeatures:
    """Multi-modal features extracted from a single RGB image."""
    dinov2_embedding: np.ndarray  # 384-dim normalized vector
    turquoise_fraction: float     # % pixels bright turquoise (spawn signature)
    bright_cyan_fraction: float   # % pixels bright cyan (broader water color)
    white_foam_fraction: float    # % pixels with high V, low S (surf)
    green_sediment_fraction: float  # % pixels with hue 60-120, low S
    edge_density: float           # mean of Canny edge detection (0-1)
    glcm_contrast: float          # GLCM contrast
    glcm_uniformity: float        # GLCM uniformity (energy)


@dataclass
class DeltaFeatures:
    """Delta features between spawn-season and off-season."""
    svm_spawn: float = 0.0
    svm_off: float = 0.0
    turquoise_spawn: float = 0.0
    turquoise_off: float = 0.0
    bright_cyan_spawn: float = 0.0
    bright_cyan_off: float = 0.0
    white_foam_spawn: float = 0.0
    white_foam_off: float = 0.0
    green_sediment_spawn: float = 0.0
    green_sediment_off: float = 0.0
    edge_density_spawn: float = 0.0
    edge_density_off: float = 0.0

    @property
    def svm_delta(self) -> float:
        return self.svm_spawn - self.svm_off

    @property
    def turquoise_delta(self) -> float:
        return self.turquoise_spawn - self.turquoise_off

    @property
    def edge_density_delta(self) -> float:
        return self.edge_density_spawn - self.edge_density_off

    def final_score(self) -> float:
        return (WEIGHT_SVM_DELTA * self.svm_delta
                + WEIGHT_TURQUOISE_DELTA * self.turquoise_delta
                + WEIGHT_EDGE_DELTA * self.edge_density_delta)


# ===================================================================
# Color Feature Extraction
# ===================================================================

def compute_hsv_features(img: np.ndarray) -> dict[str, float]:
    """Compute HSV-based color features from an RGB image array (HWC, uint8).
    
    Returns fractions of pixels matching each spectral signature.
    """
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0].astype(float), hsv[:, :, 1].astype(float), hsv[:, :, 2].astype(float)
    
    # Normalize
    h = h / 180.0  # OpenCV hue is 0-180
    s = s / 255.0
    v = v / 255.0
    
    total_pixels = h.shape[0] * h.shape[1]
    if total_pixels == 0:
        return {"turquoise": 0.0, "white_foam": 0.0, "green_sediment": 0.0}
    
    # Turquoise spawn signature (bright cyan-turquoise patches):
    #   - Hue 0.39-0.67 normalized (OpenCV ~70-120 / 180) → spans 140-240° in 360° space
    #     Herring spawn appears as milky cyan-turquoise, not narrow-band
    #   - V > 0.2 (brighter than typical dark ocean water, V~0.07)
    #   - S > 0.15 (moderate saturation — milky, not pure cyan or grey)
    # NOTE: OpenCV H is 0-180, S is 0-255, V is 0-255; normalized to 0-1 here
    turquoise_mask = (
        (h >= 0.39) & (h <= 0.67)  # Hue ~70-120 in OpenCV
        & (s > 0.15) & (v > 0.2)   # Moderately bright, somewhat saturated
    )
    turquoise_frac = float(np.mean(turquoise_mask))
    
    # Bright cyan (slightly broader, for milky/glacial water / sediment)
    bright_cyan_mask = (
        (h >= 0.33) & (h <= 0.72)  # Hue ~60-130 in OpenCV
        & (s > 0.1) & (v > 0.15)   # Slightly lower thresholds
    )
    bright_cyan_frac = float(np.mean(bright_cyan_mask))
    
    # White foam: very high value, very low saturation (surf/breaking waves)
    white_mask = (v > 0.7) & (s < 0.15)
    white_frac = float(np.mean(white_mask))
    
    # Green sediment / turbid water: hue ~60-120° (0.167-0.333), low-mid sat
    green_mask = (h >= 0.167) & (h <= 0.333) & (s < 0.3)
    green_frac = float(np.mean(green_mask))
    
    return {
        "turquoise": turquoise_frac,
        "bright_cyan": bright_cyan_frac,
        "white_foam": white_frac,
        "green_sediment": green_frac,
    }


# ===================================================================
# Texture Feature Extraction
# ===================================================================

def compute_texture_features(img: np.ndarray) -> dict[str, float]:
    """Compute texture features from an RGB image array (HWC, uint8).
    
    Returns edge density and GLCM metrics.
    """
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Edge density via Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))
    
    # GLCM features (on downsampled image for speed)
    h, w = gray.shape
    if h > 128 or w > 128:
        scale = min(128.0 / h, 128.0 / w)
        new_w = max(32, int(w * scale))
        new_h = max(32, int(h * scale))
        gray_small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        gray_small = gray
    
    # Quantize to 32 levels for GLCM
    gray_quant = (gray_small // 8).astype(np.uint8)
    
    try:
        glcm = graycomatrix(gray_quant, distances=[1], angles=[0],
                            levels=32, symmetric=True, normed=True)
        contrast = float(graycoprops(glcm, 'contrast')[0, 0])
        uniformity = float(graycoprops(glcm, 'energy')[0, 0])  # energy = sqrt(uniformity)
    except Exception:
        contrast = 0.0
        uniformity = 0.0
    
    return {
        "edge_density": edge_density,
        "glcm_contrast": contrast,
        "glcm_uniformity": uniformity,
    }


# ===================================================================
# Full Feature Extraction Pipeline
# ===================================================================

def extract_features(
    image: Image.Image | np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> ImageFeatures:
    """Extract all features from a single RGB image."""
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image
    
    # DINOv2 embedding
    tensor = DINO_TRANSFORM(Image.fromarray(img_np)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
    
    # Color features
    hsv = compute_hsv_features(img_np)
    
    # Texture features
    tex = compute_texture_features(img_np)
    
    return ImageFeatures(
        dinov2_embedding=emb,
        turquoise_fraction=hsv["turquoise"],
        bright_cyan_fraction=hsv["bright_cyan"],
        white_foam_fraction=hsv["white_foam"],
        green_sediment_fraction=hsv["green_sediment"],
        edge_density=tex["edge_density"],
        glcm_contrast=tex["glcm_contrast"],
        glcm_uniformity=tex["glcm_uniformity"],
    )


def extract_features_from_bytes(
    png_bytes: bytes,
    model: torch.nn.Module,
    device: torch.device,
) -> ImageFeatures | None:
    """Extract features from raw PNG bytes."""
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return extract_features(img, model, device)
    except Exception as exc:
        print(f"    Feature extraction error: {exc}")
        return None


# ===================================================================
# Improved Detector Model
# ===================================================================

class ImprovedDetector:
    """Combined detector using DINOv2 SVM + color features + texture features."""
    
    def __init__(self):
        self.svm: SVC | None = None
        self.mean_pos: np.ndarray | None = None
        self.mean_neg: np.ndarray | None = None
        self.use_svm: bool = True
        self.training_stats: dict[str, Any] = {}
    
    def train(
        self,
        pos_dir: Path,
        neg_dir: Path,
        model: torch.nn.Module,
        device: torch.device,
        kernel: str = "rbf",
        cv_folds: int = 5,
    ) -> dict[str, Any]:
        """Train the detector on labeled samples.
        
        Returns training statistics.
        """
        print("=" * 60)
        print("  Training Improved Detector")
        print("=" * 60)
        
        # Extract features from all samples
        print("\n  Extracting features from training samples...")
        features_list: list[ImageFeatures] = []
        labels: list[int] = []
        filenames: list[str] = []
        errors: list[str] = []
        
        for label_val, search_dir in [(1, pos_dir), (0, neg_dir)]:
            paths = sorted(search_dir.glob("*.png"))
            if not paths:
                print(f"  WARNING: No samples found in {search_dir}")
            for p in paths:
                try:
                    feats = extract_features(Image.open(p).convert("RGB"), model, device)
                    features_list.append(feats)
                    labels.append(label_val)
                    filenames.append(p.name)
                except Exception as exc:
                    errors.append(f"{p.name}: {exc}")
        
        if len(features_list) < 10:
            raise RuntimeError(f"Too few samples: {len(features_list)} (need >= 10)")
        
        n_pos = int(sum(labels))
        n_neg = int(len(labels) - n_pos)
        print(f"  Loaded {len(features_list)} samples: {n_pos} positive, {n_neg} negative")
        if errors:
            print(f"  WARNING: {len(errors)} samples failed:")
            for err in errors[:5]:
                print(f"    - {err}")
        
        # Extract feature arrays
        dinov2_embs = np.array([f.dinov2_embedding for f in features_list])
        turquoise_vals = np.array([f.turquoise_fraction for f in features_list])
        bright_cyan_vals = np.array([f.bright_cyan_fraction for f in features_list])
        foam_vals = np.array([f.white_foam_fraction for f in features_list])
        sediment_vals = np.array([f.green_sediment_fraction for f in features_list])
        edge_vals = np.array([f.edge_density for f in features_list])
        glcm_contrast_vals = np.array([f.glcm_contrast for f in features_list])
        glcm_uniformity_vals = np.array([f.glcm_uniformity for f in features_list])
        labels_arr = np.array(labels)
        
        # Compute mean reference vectors for similarity scoring
        pos_embs = dinov2_embs[labels_arr == 1]
        neg_embs = dinov2_embs[labels_arr == 0]
        self.mean_pos = np.mean(pos_embs, axis=0) if len(pos_embs) > 0 else None
        self.mean_neg = np.mean(neg_embs, axis=0) if len(neg_embs) > 0 else None
        if self.mean_pos is not None:
            self.mean_pos /= np.linalg.norm(self.mean_pos)
        if self.mean_neg is not None:
            self.mean_neg /= np.linalg.norm(self.mean_neg)
        
        # Compute training set statistics for color/texture features
        pos_turq = turquoise_vals[labels_arr == 1]
        neg_turq = turquoise_vals[labels_arr == 0]
        pos_cyan = bright_cyan_vals[labels_arr == 1]
        neg_cyan = bright_cyan_vals[labels_arr == 0]
        pos_edge = edge_vals[labels_arr == 1]
        neg_edge = edge_vals[labels_arr == 0]
        pos_foam = foam_vals[labels_arr == 1]
        neg_foam = foam_vals[labels_arr == 0]
        pos_sed = sediment_vals[labels_arr == 1]
        neg_sed = sediment_vals[labels_arr == 0]
        
        stats = {
            "n_train": len(features_list),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "mean_turquoise_pos": float(np.mean(pos_turq)) if len(pos_turq) > 0 else 0,
            "mean_turquoise_neg": float(np.mean(neg_turq)) if len(neg_turq) > 0 else 0,
            "mean_bright_cyan_pos": float(np.mean(pos_cyan)) if len(pos_cyan) > 0 else 0,
            "mean_bright_cyan_neg": float(np.mean(neg_cyan)) if len(neg_cyan) > 0 else 0,
            "mean_edge_pos": float(np.mean(pos_edge)) if len(pos_edge) > 0 else 0,
            "mean_edge_neg": float(np.mean(neg_edge)) if len(neg_edge) > 0 else 0,
            "mean_foam_pos": float(np.mean(pos_foam)) if len(pos_foam) > 0 else 0,
            "mean_foam_neg": float(np.mean(neg_foam)) if len(neg_foam) > 0 else 0,
            "mean_sediment_pos": float(np.mean(pos_sed)) if len(pos_sed) > 0 else 0,
            "mean_sediment_neg": float(np.mean(neg_sed)) if len(neg_sed) > 0 else 0,
        }
        print(f"\n  Color/Texture Training Statistics:")
        for k, v in stats.items():
            if k.startswith("mean_"):
                print(f"    {k}: {v:.4f}")
        
        # Train SVM on DINOv2 embeddings
        print(f"\n  Training SVM (kernel={kernel}, class_weight='balanced')...")
        self.svm = SVC(
            kernel=kernel,
            class_weight="balanced",
            probability=True,
            random_state=42,
            gamma="scale",
        )
        self.svm.fit(dinov2_embs, labels_arr)
        
        # Evaluate
        y_pred = self.svm.predict(dinov2_embs)
        full_acc = accuracy_score(labels_arr, y_pred)
        y_decision = self.svm.decision_function(dinov2_embs)
        pos_scores_svm = y_decision[labels_arr == 1]
        neg_scores_svm = y_decision[labels_arr == 0]
        separation = float(np.mean(pos_scores_svm) - np.mean(neg_scores_svm))
        
        print(f"\n  SVM FULL DATASET RESULTS")
        print(f"  Accuracy: {full_acc:.4f}")
        print(f"  Separation: {separation:.4f}")
        print(f"\n  Classification Report:")
        print(f"  {classification_report(labels_arr, y_pred, target_names=['negative', 'positive'])}")
        cm = confusion_matrix(labels_arr, y_pred)
        print(f"  Confusion Matrix:")
        print(f"                Neg   Pos")
        print(f"  Actual Neg    {cm[0][0]:<5} {cm[0][1]:<5}")
        print(f"         Pos    {cm[1][0]:<5} {cm[1][1]:<5}")
        
        # Cross-validation
        n_folds = min(cv_folds, min(n_pos, n_neg))
        if n_folds >= 3:
            cv_svm = SVC(
                kernel=kernel, class_weight="balanced",
                random_state=42, gamma="scale",
            )
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(cv_svm, dinov2_embs, labels_arr, cv=cv, scoring="accuracy")
            stats["cv_accuracy_mean"] = float(cv_scores.mean())
            stats["cv_accuracy_std"] = float(cv_scores.std())
            print(f"\n  Cross-validation ({n_folds}-fold): accuracy = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        else:
            stats["cv_accuracy_mean"] = 0.0
            stats["cv_accuracy_std"] = 0.0
        
        stats["full_accuracy"] = float(full_acc)
        stats["svm_separation"] = separation
        
        # Now validate the combined delta scoring on what we have
        # Use a simple heuristic to compute "pseudo-delta" scores
        # (we don't have paired data, so this is approximate)
        print(f"\n  Computing delta-combined scores...")
        combined_scores = []
        for i, f in enumerate(features_list):
            svm_score = float(y_decision[i])
            # Pseudo-delta: compare to negative mean features
            turq_delta = f.turquoise_fraction - stats["mean_turquoise_neg"]
            edge_delta = -(f.edge_density - stats["mean_edge_neg"])  # spawn has lower edge density
            
            # Normalize ranges roughly
            combined = (WEIGHT_SVM_DELTA * np.tanh(svm_score / 2.0)
                        + WEIGHT_TURQUOISE_DELTA * np.clip(turq_delta * 5, -1, 1)
                        + WEIGHT_EDGE_DELTA * np.clip(edge_delta * 5, -1, 1))
            combined_scores.append(float(combined))
        
        combined_arr = np.array(combined_scores)
        pos_combined = combined_arr[labels_arr == 1]
        neg_combined = combined_arr[labels_arr == 0]
        combined_sep = float(np.mean(pos_combined) - np.mean(neg_combined))
        combined_thresh = 0.0
        combined_pred = (combined_arr > combined_thresh).astype(int)
        combined_acc = accuracy_score(labels_arr, combined_pred)
        
        print(f"  Combined delta score separation: {combined_sep:.4f}")
        print(f"  Combined delta accuracy (threshold=0): {combined_acc:.4f}")
        
        stats["combined_separation"] = combined_sep
        stats["combined_accuracy"] = combined_acc
        self.training_stats = stats
        
        return stats
    
    def score_svm(self, embedding: np.ndarray) -> float:
        """Get SVM decision function score for a DINOv2 embedding."""
        if self.svm is not None:
            return float(self.svm.decision_function(embedding.reshape(1, -1))[0])
        elif self.mean_pos is not None and self.mean_neg is not None:
            pos_sim = float(np.dot(self.mean_pos, embedding))
            neg_sim = float(np.dot(self.mean_neg, embedding))
            return pos_sim - neg_sim
        return 0.0
    
    def compute_delta(self, spawn_feats: ImageFeatures, off_feats: ImageFeatures) -> DeltaFeatures:
        """Compute delta features between spawn-season and off-season."""
        svm_spawn = self.score_svm(spawn_feats.dinov2_embedding)
        svm_off = self.score_svm(off_feats.dinov2_embedding)
        
        return DeltaFeatures(
            svm_spawn=svm_spawn,
            svm_off=svm_off,
            turquoise_spawn=spawn_feats.turquoise_fraction,
            turquoise_off=off_feats.turquoise_fraction,
            bright_cyan_spawn=spawn_feats.bright_cyan_fraction,
            bright_cyan_off=off_feats.bright_cyan_fraction,
            white_foam_spawn=spawn_feats.white_foam_fraction,
            white_foam_off=off_feats.white_foam_fraction,
            green_sediment_spawn=spawn_feats.green_sediment_fraction,
            green_sediment_off=off_feats.green_sediment_fraction,
            edge_density_spawn=spawn_feats.edge_density,
            edge_density_off=off_feats.edge_density,
        )
    
    def is_candidate(self, delta: DeltaFeatures) -> tuple[bool, float]:
        """Determine if a location is a spawn candidate.
        
        Returns (is_candidate, final_score).
        """
        score = delta.final_score()
        if score > SCORE_THRESHOLD and delta.turquoise_spawn > MIN_TURQUOISE_FRACTION:
            return True, score
        return False, score
    
    def save(self, path: Path) -> None:
        """Save the trained model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "svm": self.svm,
            "mean_pos": self.mean_pos,
            "mean_neg": self.mean_neg,
            "use_svm": self.use_svm,
            "training_stats": self.training_stats,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "ImprovedDetector":
        """Load a trained model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        detector = cls()
        detector.svm = data["svm"]
        detector.mean_pos = data.get("mean_pos")
        detector.mean_neg = data.get("mean_neg")
        detector.use_svm = data.get("use_svm", True)
        detector.training_stats = data.get("training_stats", {})
        return detector


# ===================================================================
# Data Cleaning
# ===================================================================

def clean_training_data(
    repo_root: Path,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Clean the training data based on rose verification labels.
    
    Strategy (prioritizing verified labels over filename heuristics):
    - Removes rose-verified false positives (correct=false from rose_training_verify.json)
    - Adds rose-confirmed spawns from rose_200_labels.json (spawn=true)
    - Keeps all verified-correct positives regardless of filename
    - Removes only the unverified false-positive cloud images (those not already
      verified as correct by Rose) — the "cloud in filename" heuristic is used
      ONLY for images NOT covered by rose verification
    
    NOTE: Images with "cloud" in the filename that were verified as correct
    spawns by Rose (e.g., dfo-verified-*-cloud*.png) are PRESERVED because
    they contain clear turquoise signals despite atmospheric haze.
    """
    pos_dir = repo_root / "data" / "samples" / "positive"
    neg_dir = repo_root / "data" / "samples" / "negative"
    candidates_dir = repo_root / "data" / "candidates_v2"
    
    verify_path = candidates_dir / "rose_training_verify.json"
    labels_path = candidates_dir / "rose_200_labels.json"
    
    results = {
        "removed_false_positives": [],
        "removed_unverified_cloud": [],
        "added_confirmed_spawns": [],
        "kept_verified_positives": [],
        "preserved_verified_cloud_images": [],
    }
    
    # 1. Load rose verification data
    if verify_path.exists():
        with open(verify_path) as f:
            verify_data = json.load(f)
    else:
        print(f"  WARNING: {verify_path} not found")
        verify_data = []
    
    # 2. Load rose 200 labels
    if labels_path.exists():
        with open(labels_path) as f:
            labels_data = json.load(f)
    else:
        print(f"  WARNING: {labels_path} not found")
        labels_data = []
    
    # 3. Identify verified true/false positives
    verified_false = {entry["filename"] for entry in verify_data if not entry.get("correct", True)}
    verified_true = {entry["filename"] for entry in verify_data if entry.get("correct", False)}
    
    # 4. Find images with 'cloud' in filename in positive set
    #    Only remove those NOT verified as correct
    cloud_in_pos = set()
    for p in pos_dir.glob("*.png"):
        if "cloud" in p.name.lower():
            cloud_in_pos.add(p.name)
    
    # Preserve verified-correct cloud images
    unverified_cloud = cloud_in_pos - verified_true
    
    # 5. Find confirmed spawns from rose_200_labels
    confirmed_spawns = []
    for entry in labels_data:
        if entry.get("spawn", False):
            confirmed_spawns.append(entry["filename"])
    
    # 6. Determine what to remove: false positives + unverified cloud images
    files_to_remove = verified_false | unverified_cloud
    
    results["removed_false_positives"] = sorted(verified_false)
    results["removed_unverified_cloud"] = sorted(unverified_cloud)
    results["added_confirmed_spawns"] = confirmed_spawns
    results["kept_verified_positives"] = sorted(verified_true - cloud_in_pos)
    results["preserved_verified_cloud_images"] = sorted(cloud_in_pos & verified_true)
    
    if not dry_run:
        # Remove false positives and unverified cloud images from positive set
        for fname in files_to_remove:
            fpath = pos_dir / fname
            if fpath.exists():
                fpath.unlink()
                print(f"  Removed: {fname}")
        
        # Copy confirmed spawns from candidates_v2 to positive set
        import shutil
        for fname in confirmed_spawns:
            src = candidates_dir / fname
            if src.exists():
                dst = pos_dir / fname
                if not dst.exists():
                    shutil.copy2(src, dst)
                    print(f"  Added: {fname}")
            else:
                print(f"  WARNING: Confirmed spawn not found in candidates_v2: {fname}")
    
    total_after = len(list(pos_dir.glob("*.png"))) if not dry_run else "N/A (dry run)"
    print(f"\n  Data cleaning summary:")
    print(f"    Rose-verified false positives removed: {len(verified_false)}")
    print(f"    Unverified cloud images removed:       {len(unverified_cloud)}")
    print(f"    Verified-correct cloud images kept:    {len(cloud_in_pos & verified_true)}")
    print(f"    Confirmed spawns added:                {len(confirmed_spawns)}")
    print(f"    Verified non-cloud positives kept:     {len(verified_true - cloud_in_pos)}")
    print(f"    Total positives after cleanup:         {total_after}")
    print(f"    (dry run: {dry_run})")
    
    return results


# ===================================================================
# GEE Integration (reused from scan_bc_coast.py)
# ===================================================================

def find_best_scene(
    ee_module: Any,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_cloud: float,
) -> dict[str, Any] | None:
    """Find the single best Sentinel-2 scene for a point."""
    try:
        collection = ee_module.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        point = ee_module.Geometry.Point(lon, lat)
        scenes = (
            collection
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee_module.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        scene_ids = scenes.aggregate_array("system:index").getInfo()
        clouds = scenes.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        if not scene_ids:
            return None
        best_idx = 0
        sid = scene_ids[best_idx]
        return {
            "scene_id": sid,
            "cloud": float(clouds[best_idx]),
            "date": f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}",
            "lat": lat,
            "lon": lon,
        }
    except Exception as exc:
        print(f"    GEE search error at ({lat:.4f}, {lon:.4f}): {exc}")
        return None


def download_thumbnail(
    ee_module: Any,
    lat: float,
    lon: float,
    scene_id: str,
) -> bytes | None:
    """Download a 512×512 RGB thumbnail from a Sentinel-2 scene."""
    try:
        scene_img = ee_module.Image(f"COPERNICUS/S2_SR_HARMONIZED/{scene_id}")
        rgb = scene_img.select(["B4", "B3", "B2"])
        region = ee_module.Geometry.Point(lon, lat).buffer(1280).bounds()
        url = rgb.getThumbURL({
            "min": 0,
            "max": 3000,
            "region": region,
            "dimensions": 512,
            "format": "png",
        })
        resp = __import__("requests").get(url, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        print(f"    Download failed for {scene_id}: {exc}")
        return None


# ===================================================================
# Grid Point Generation
# ===================================================================

def generate_grid_points(
    regions: list[dict[str, Any]],
    spacing_deg: float,
) -> list[dict[str, Any]]:
    """Generate grid points within each region's circular buffer."""
    points: list[dict[str, Any]] = []
    for region in regions:
        lat, lon = region["lat"], region["lon"]
        radius_km = region["radius_km"]
        radius_deg_lat = radius_km / 111.0
        radius_deg_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
        n_steps_lat = max(1, int(2 * radius_deg_lat / spacing_deg))
        n_steps_lon = max(1, int(2 * radius_deg_lon / spacing_deg))
        region_points = 0
        for i in range(n_steps_lat + 1):
            p_lat = lat - radius_deg_lat + i * spacing_deg
            if abs(p_lat - lat) > radius_deg_lat + spacing_deg * 0.5:
                continue
            for j in range(n_steps_lon + 1):
                p_lon = lon - radius_deg_lon + j * spacing_deg
                if abs(p_lon - lon) > radius_deg_lon + spacing_deg * 0.5:
                    continue
                dlat = (p_lat - lat) * 111.0
                dlon = (p_lon - lon) * 111.0 * math.cos(math.radians(lat))
                dist_km = math.sqrt(dlat**2 + dlon**2)
                if dist_km <= radius_km:
                    points.append({
                        "region": region["name"],
                        "lat": round(p_lat, 6),
                        "lon": round(p_lon, 6),
                    })
                    region_points += 1
        print(f"  {region['name']}: {region_points} grid points")
    return points


# ===================================================================
# Scanning Logic
# ===================================================================

_stats_lock = threading.Lock()


def scan_point(
    point: dict[str, Any],
    detector: ImprovedDetector,
    model: torch.nn.Module,
    device: torch.device,
    ee_module: Any,
    args: argparse.Namespace,
    idx: int,
    total: int,
    output_dir: Path,
) -> dict[str, int]:
    """Process a single grid point: download both seasons, compute delta, score."""
    result = {"processed": 1, "candidates": 0, "no_scene": 0, "no_offscene": 0,
              "download_errors": 0, "low_score": 0}
    
    lat, lon = point["lat"], point["lon"]
    region = point["region"]
    
    # Find spawn-season scene
    spawn_scene = find_best_scene(
        ee_module, lat, lon, args.start, args.end, args.max_cloud,
    )
    if spawn_scene is None:
        with _stats_lock:
            _print_progress(idx, total, region, lat, lon, "no spawn scene", 0)
        result["no_scene"] = 1
        return result
    
    # Find off-season scene
    off_scene = find_best_scene(
        ee_module, lat, lon, args.off_start, args.off_end, args.max_cloud,
    )
    if off_scene is None:
        with _stats_lock:
            _print_progress(idx, total, region, lat, lon, "no off-season scene", 0)
        result["no_offscene"] = 1
        return result
    
    # Download spawn thumbnail
    spawn_bytes = download_thumbnail(ee_module, lat, lon, spawn_scene["scene_id"])
    if spawn_bytes is None:
        with _stats_lock:
            _print_progress(idx, total, region, lat, lon, "spawn download error", 0)
        result["download_errors"] = 1
        return result
    
    # Download off-season thumbnail
    off_bytes = download_thumbnail(ee_module, lat, lon, off_scene["scene_id"])
    if off_bytes is None:
        with _stats_lock:
            _print_progress(idx, total, region, lat, lon, "off-season download error", 0)
        result["download_errors"] = 1
        return result
    
    # Extract features
    spawn_feats = extract_features_from_bytes(spawn_bytes, model, device)
    off_feats = extract_features_from_bytes(off_bytes, model, device)
    if spawn_feats is None or off_feats is None:
        with _stats_lock:
            _print_progress(idx, total, region, lat, lon, "feature extraction error", 0)
        result["download_errors"] = 1
        return result
    
    # Compute delta and score
    delta = detector.compute_delta(spawn_feats, off_feats)
    is_cand, final_score = detector.is_candidate(delta)
    
    if is_cand and final_score > args.threshold:
        # Save candidate (spawn image only)
        info = {
            "region": region,
            "lat": lat,
            "lon": lon,
            "date": spawn_scene["date"],
            "off_date": off_scene["date"],
            "scene_id": spawn_scene["scene_id"],
            "off_scene_id": off_scene["scene_id"],
            "cloud": spawn_scene["cloud"],
            "off_cloud": off_scene["cloud"],
            "score": round(final_score, 4),
            "svm_delta": round(delta.svm_delta, 4),
            "turquoise_delta": round(delta.turquoise_delta, 4),
            "bright_cyan_spawn": round(delta.bright_cyan_spawn, 4),
            "bright_cyan_delta": round(delta.bright_cyan_spawn - delta.bright_cyan_off, 4),
            "edge_density_delta": round(delta.edge_density_delta, 4),
            "turquoise_spawn": round(delta.turquoise_spawn, 4),
            "turquoise_off": round(delta.turquoise_off, 4),
            "svm_spawn": round(delta.svm_spawn, 4),
            "svm_off": round(delta.svm_off, 4),
        }
        fname = _save_candidate_spawn(output_dir, spawn_bytes, info, final_score)
        
        with _stats_lock:
            # Update manifest
            manifest_path = output_dir / "manifest.json"
            entries: list[dict] = []
            if manifest_path.exists():
                try:
                    entries = json.loads(manifest_path.read_text())
                    if not isinstance(entries, list):
                        entries = []
                except Exception:
                    entries = []
            entries.append({**info, "thumbnail_path": fname})
            manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            
            _print_progress(idx, total, region, lat, lon,
                           f"CANDIDATE score={final_score:.4f} turq={delta.turquoise_spawn:.3f} {fname}", 0)
        result["candidates"] = 1
    else:
        reason = f"below threshold ({final_score:.4f}, turq={delta.turquoise_spawn:.3f})"
        with _stats_lock:
            _print_progress(idx, total, region, lat, lon, reason, 0)
        result["low_score"] = 1
    
    return result


def _print_progress(
    idx: int, total: int, region: str, lat: float, lon: float,
    status: str, elapsed: float,
) -> None:
    pct = 100.0 * (idx + 1) / total
    eta_str = "?"
    if idx > 0 and elapsed > 0:
        rate = idx / elapsed
        remaining_s = (total - idx) / rate if rate > 0 else 0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining_s))
    print(f"  [{idx + 1}/{total}] ({pct:.0f}%) {region} ({lat:.4f}, {lon:.4f}) | {status} | ETA {eta_str}")


def _save_candidate_spawn(output_dir: Path, png_bytes: bytes, info: dict, score: float) -> str:
    """Save candidate spawn thumbnail PNG."""
    region = info["region"]
    date = info["date"]
    lat = info["lat"]
    lon = info["lon"]
    scene_id = info["scene_id"]
    scene_short = scene_id[:8] if len(scene_id) >= 8 else scene_id
    fname = f"{region}_{date}_score{score:.2f}_{lat}_{lon}_{scene_short}.png"
    # Sanitize: keep only alphanumeric, dot, underscore, minus
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    fname = "".join(c if c in safe_chars else "_" for c in fname)
    fpath = output_dir / fname
    fpath.write_bytes(png_bytes)
    return fname


# ===================================================================
# Review Page Generation
# ===================================================================

def generate_review_page(candidates_dir: Path) -> str:
    """Generate an HTML review page for candidates."""
    manifest_path = candidates_dir / "manifest.json"
    candidates: list[dict] = []
    
    if manifest_path.exists():
        with open(manifest_path) as f:
            candidates = json.load(f)
    
    if not candidates:
        # Fallback: scan directory for PNGs and build entries
        png_files = sorted(candidates_dir.glob("*.png"))
        if not png_files:
            print(f"ERROR: No candidates or PNGs found in {candidates_dir}")
            return ""
        print(f"  No manifest found. Building from {len(png_files)} PNGs in directory.")
        for p in png_files:
            candidates.append({
                "thumbnail_path": p.name,
                "score": 0.0,
                "region": "unknown",
                "lat": 0.0,
                "lon": 0.0,
                "date": "?",
                "off_date": "?",
                "svm_delta": 0.0,
                "turquoise_delta": 0.0,
                "turquoise_spawn": 0.0,
                "edge_density_delta": 0.0,
            })
    
    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)
    
    cards = []
    for i, cand in enumerate(candidates):
        thumb_path = cand["thumbnail_path"]
        score = cand["score"]
        svm_delta = cand.get("svm_delta", 0)
        turq_delta = cand.get("turquoise_delta", 0)
        edge_delta = cand.get("edge_density_delta", 0)
        turq_spawn = cand.get("turquoise_spawn", 0)
        
        # Color coding
        if score > 0.5:
            card_class = "good"
        elif score > 0.1:
            card_class = "mid"
        else:
            card_class = "low"
        
        cards.append(f"""
    <div class="card {card_class}">
        <img src="{html.escape(thumb_path)}" alt="Candidate {i}" loading="lazy">
        <div class="body">
            <div class="info"><strong>{html.escape(cand.get("region", "?"))}</strong> &middot; {cand.get("lat", 0):.4f}, {cand.get("lon", 0):.4f}</div>
            <div class="info">Date: {cand.get("date", "?")} &middot; Off: {cand.get("off_date", "?")}</div>
            <div class="scores">
                <span class="badge score-badge">Score: {score:.4f}</span>
                <span class="badge svm-badge">SVM &Delta;: {svm_delta:+.4f}</span>
                <span class="badge turq-badge">Turq &Delta;: {turq_delta:+.4f}</span>
                <span class="badge edge-badge">Edge &Delta;: {edge_delta:+.4f}</span>
                <span class="badge turq-val-badge">Turq: {turq_spawn:.3f}</span>
            </div>
        </div>
    </div>""")
    
    cards_joined = "\n".join(cards)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Improved Detector — Candidates Review</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #111; color: #eee; }}
.bar {{ background: #1a1a2e; padding: 12px 20px; position: sticky; top: 0; z-index: 99; border-bottom: 1px solid #2a2a4e; }}
.bar h1 {{ margin: 0; font-size: 18px; }}
.bar .sub {{ font-size: 13px; color: #888; margin-top: 4px; }}
.g {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; padding: 14px; }}
.card {{ background: #1e1e2e; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.4); }}
.good {{ border-left: 4px solid #4CAF50; }}
.mid {{ border-left: 4px solid #FFC107; }}
.low {{ border-left: 4px solid #f44336; }}
.card img {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; display: block; }}
.body {{ padding: 10px 12px; }}
.info {{ font-size: 11px; color: #999; margin-bottom: 3px; }}
.info strong {{ color: #ddd; }}
.scores {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; }}
.score-badge {{ background: #1b3a2b; color: #4CAF50; }}
.svm-badge {{ background: #1b2a3a; color: #64B5F6; }}
.turq-badge {{ background: #1b2a1b; color: #4DD0E1; }}
.edge-badge {{ background: #2a1b2a; color: #CE93D8; }}
.turq-val-badge {{ background: #2a2a1b; color: #FFD54F; }}
.summary {{ background: #1a1a2e; margin: 14px; padding: 14px 18px; border-radius: 8px; font-size: 13px; color: #aaa; line-height: 1.5; }}
.summary strong {{ color: #fff; }}
.summary-grid {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; }}
.summary-stat {{ text-align: center; padding: 8px 14px; background: #0a0a14; border-radius: 6px; }}
.summary-stat .num {{ font-size: 22px; font-weight: 700; color: #fff; }}
.summary-stat .lbl {{ font-size: 10px; color: #888; text-transform: uppercase; margin-top: 2px; }}
</style>
</head>
<body>
<div class="bar">
    <h1>&#x1f41f; Improved Detector — {len(candidates)} Candidates</h1>
    <div class="sub">
        Delta score = 0.5&times;SVM&Delta; + 0.3&times;Turq&Delta; + 0.2&times;Edge&Delta;
        &middot; Requires turquoise &gt; {MIN_TURQUOISE_FRACTION}
    </div>
</div>
<div class="summary">
    <strong>Summary:</strong> {len(candidates)} candidates found.
    Delta-based scoring compares spawn-season vs off-season features.
    <div class="summary-grid">
        <div class="summary-stat"><div class="num">{len(candidates)}</div><div class="lbl">Candidates</div></div>
        <div class="summary-stat"><div class="num">{sum(1 for c in candidates if c.get('score', 0) > 0.5)}</div><div class="lbl">Score &gt; 0.5</div></div>
        <div class="summary-stat"><div class="num">{sum(1 for c in candidates if c.get('turquoise_spawn', 0) > 0.1)}</div><div class="lbl">Turq &gt; 0.1</div></div>
    </div>
</div>
<div class="g">
{cards_joined}
</div>
</body>
</html>"""
    
    review_path = candidates_dir / "review.html"
    review_path.write_text(html_content, encoding="utf-8")
    print(f"  Review page: file://{review_path.resolve()}")
    return str(review_path)


# ===================================================================
# CLI
# ===================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improved herring spawn detector with delta-based scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", default="train",
                        choices=["train", "score-pair", "scan", "review", "clean-data"],
                        help="Operation mode (default: train)")
    parser.add_argument("--positive-dir", default="data/samples/positive")
    parser.add_argument("--negative-dir", default="data/samples/negative")
    parser.add_argument("--output-model", default="data/models/improved_model.pkl")
    parser.add_argument("--output", "--candidates-dir", default="data/candidates_v3")
    parser.add_argument("--spawn-img", help="Path to spawn-season image (for score-pair mode)")
    parser.add_argument("--off-img", help="Path to off-season image (for score-pair mode)")
    parser.add_argument("--start", default="2024-02-01", help="Spawn season start")
    parser.add_argument("--end", default="2024-04-30", help="Spawn season end")
    parser.add_argument("--off-start", default="2024-06-01", help="Off-season start")
    parser.add_argument("--off-end", default="2024-08-31", help="Off-season end")
    parser.add_argument("--max-cloud", type=float, default=50, help="Max cloud %")
    parser.add_argument("--threshold", type=float, default=0.0, help="Score threshold")
    parser.add_argument("--grid-spacing", type=float, default=0.01, help="Grid spacing in degrees")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute (for clean-data, scan)")
    parser.add_argument("--kernel", default="rbf", choices=["linear", "rbf", "poly", "sigmoid"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    
    # ==============================================================
    # MODE: clean-data
    # ==============================================================
    if args.mode == "clean-data":
        print("\n=== Cleaning Training Data ===")
        results = clean_training_data(repo_root, dry_run=args.dry_run)
        print(f"\n  To actually apply changes, run without --dry-run")
        return 0
    
    # ==============================================================
    # MODE: review
    # ==============================================================
    if args.mode == "review":
        candidates_dir = repo_root / args.output
        print(f"\n=== Generating Review Page ===")
        print(f"  Directory: {candidates_dir}")
        review_path = generate_review_page(candidates_dir)
        if review_path:
            print(f"  Review page generated: {review_path}")
            return 0
        return 1
    
    # Load DINOv2 model (needed for all remaining modes)
    print(f"\n=== Loading DINOv2 model ({MODEL_NAME}) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    try:
        model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
        model.eval()
        model = model.to(device)
    except Exception as exc:
        print(f"ERROR: Failed to load DINOv2: {exc}")
        return 1
    
    # ==============================================================
    # MODE: train
    # ==============================================================
    if args.mode == "train":
        pos_dir = repo_root / args.positive_dir
        neg_dir = repo_root / args.negative_dir
        output_model = repo_root / args.output_model
        
        if not pos_dir.exists():
            print(f"ERROR: Positive dir not found: {pos_dir}")
            return 1
        if not neg_dir.exists():
            print(f"ERROR: Negative dir not found: {neg_dir}")
            return 1
        
        detector = ImprovedDetector()
        detector.train(pos_dir, neg_dir, model, device, kernel=args.kernel)
        detector.save(output_model)
        
        # Save summary
        summary_path = output_model.with_suffix(".summary.json")
        summary_data = {k: v for k, v in detector.training_stats.items()}
        summary_path.write_text(json.dumps(summary_data, indent=2))
        print(f"  Summary saved to {summary_path}")
        return 0
    
    # ==============================================================
    # MODE: score-pair
    # ==============================================================
    if args.mode == "score-pair":
        if not args.spawn_img or not args.off_img:
            print("ERROR: --spawn-img and --off-img required for score-pair mode")
            return 1
        
        spawn_path = Path(args.spawn_img)
        off_path = Path(args.off_img)
        if not spawn_path.exists():
            print(f"ERROR: {spawn_path} not found")
            return 1
        if not off_path.exists():
            print(f"ERROR: {off_path} not found")
            return 1
        
        # Try to load model
        model_path = repo_root / args.output_model
        if model_path.exists():
            detector = ImprovedDetector.load(model_path)
            print(f"  Loaded model from {model_path}")
        else:
            print(f"  WARNING: No model found at {model_path}, using similarity scoring")
            print(f"  Train one first: python scripts/improved_detector.py --mode train")
            return 1
        
        spawn_feats = extract_features(Image.open(spawn_path).convert("RGB"), model, device)
        off_feats = extract_features(Image.open(off_path).convert("RGB"), model, device)
        
        delta = detector.compute_delta(spawn_feats, off_feats)
        score = delta.final_score()
        
        print(f"\n  {'=' * 50}")
        print(f"  Delta Score Results")
        print(f"  {'=' * 50}")
        print(f"  SVM spawn:          {delta.svm_spawn:.4f}")
        print(f"  SVM off:            {delta.svm_off:.4f}")
        print(f"  SVM delta:          {delta.svm_delta:+.4f}")
        print(f"  Turquoise spawn:    {delta.turquoise_spawn:.4f}")
        print(f"  Turquoise off:      {delta.turquoise_off:.4f}")
        print(f"  Turquoise delta:    {delta.turquoise_delta:+.4f}")
        print(f"  Edge density spawn: {delta.edge_density_spawn:.4f}")
        print(f"  Edge density off:   {delta.edge_density_off:.4f}")
        print(f"  Edge density delta: {delta.edge_density_delta:+.4f}")
        print(f"  {'=' * 50}")
        print(f"  FINAL SCORE:        {score:.4f}")
        
        is_cand, _ = detector.is_candidate(delta)
        print(f"  Candidate:          {'YES' if is_cand else 'NO'}")
        print(f"  {'=' * 50}")
        return 0
    
    # ==============================================================
    # MODE: scan
    # ==============================================================
    if args.mode == "scan":
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = repo_root / args.output
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load trained model
        model_path = repo_root / args.output_model
        if model_path.exists():
            detector = ImprovedDetector.load(model_path)
            print(f"\n  Loaded model from {model_path}")
            if detector.training_stats:
                print(f"  Training stats: {detector.training_stats.get('n_train', '?')} samples, "
                      f"SVM sep: {detector.training_stats.get('svm_separation', '?'):.4f}")
        else:
            print(f"\n  WARNING: No model at {model_path}, training from samples first...")
            pos_dir = repo_root / "data" / "samples" / "positive"
            neg_dir = repo_root / "data" / "samples" / "negative"
            detector = ImprovedDetector()
            try:
                detector.train(pos_dir, neg_dir, model, device)
                detector.save(model_path)
            except RuntimeError as e:
                print(f"ERROR: {e}")
                return 1
        
        # Generate grid points
        print(f"\n=== Generating Grid Points ===")
        points = generate_grid_points(REGIONS, args.grid_spacing)
        print(f"  Total: {len(points)}")
        
        if not points:
            print("ERROR: No grid points generated.")
            return 1
        
        # Initialize GEE
        print(f"\n=== Initializing GEE ===")
        try:
            import ee
            ee.Initialize(project="redd-fish")
            print("  GEE initialized")
        except Exception as exc:
            print(f"ERROR: GEE init failed: {exc}")
            return 1
        
        if args.dry_run:
            print(f"\n=== Dry Run ===")
            print(f"  Points: {len(points)}")
            print(f"  Spawn: {args.start} to {args.end}")
            print(f"  Off: {args.off_start} to {args.off_end}")
            print(f"  Max cloud: {args.max_cloud}%")
            print(f"  Threshold: {args.threshold}")
            print(f"  Workers: {args.workers}")
            print(f"  Output: {output_dir}")
            return 0
        
        # Scan
        print(f"\n=== Scanning {len(points)} points ===")
        start_time = time.time()
        processed = candidates = no_scene = no_off = dl_errors = low_score = 0
        
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        scan_point, point, detector, model, device, ee, args,
                        idx, len(points), output_dir
                    ): idx
                    for idx, point in enumerate(points)
                }
                for future in as_completed(futures):
                    r = future.result()
                    processed += r["processed"]
                    candidates += r["candidates"]
                    no_scene += r["no_scene"]
                    no_off += r["no_offscene"]
                    dl_errors += r["download_errors"]
                    low_score += r["low_score"]
        except KeyboardInterrupt:
            print("\n\nInterrupted! Partial results saved.")
        except Exception as exc:
            print(f"\n\nError: {exc}")
            import traceback
            traceback.print_exc()
        
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        
        print(f"\n{'=' * 60}")
        print("  Scan Complete")
        print(f"  {'=' * 60}")
        print(f"  Total points:     {len(points)}")
        print(f"  Processed:        {processed}")
        print(f"  Candidates:       {candidates}")
        print(f"  No spawn scene:   {no_scene}")
        print(f"  No off scene:     {no_off}")
        print(f"  DL errors:        {dl_errors}")
        print(f"  Below threshold:  {low_score}")
        print(f"  Elapsed:          {elapsed:.1f}s")
        print(f"  Rate:             {rate:.1f} pts/s")
        print(f"  Output:           {output_dir}")
        
        if candidates > 0:
            print(f"\n  Generate review page:")
            print(f"    python scripts/improved_detector.py --mode review --candidates-dir {output_dir}")
        
        return 0
    
    print(f"Unknown mode: {args.mode}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
