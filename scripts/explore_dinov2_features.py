#!/usr/bin/env python3
"""DINOv2 PCA feature visualization for herring spawn detection.

Extracts patch-level features from DINOv2 ViT-S/14, applies PCA,
and visualizes feature maps to identify spatial patterns that distinguish
spawn from non-spawn satellite imagery.

Usage:
    python scripts/explore_dinov2_features.py \
        --n-samples 16 \
        --output-dir data/review/dinov2_features

This generates:
  - One PNG per image with original | PCA feature map | attention proxy | overlay
  - data/review/dinov2_features/class_comparison.png — mean spawn vs non-spawn
  - data/review/dinov2_features/index.html — browsable gallery
  - data/review/dinov2_features/analysis.json — quantitative metrics
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
import os
os.environ['MPLBACKEND'] = 'Agg'

# ──────────────────────────── helpers ────────────────────────────


def fmt(x):
    """Format a float for display."""
    return f"{x:.4f}"


def minmax_01(arr):
    """Min-max normalize each channel of an (H, W, C) array to [0, 1]."""
    result = arr.copy().astype(np.float64)
    for i in range(arr.shape[-1]):
        col = result[..., i]
        mn, mx = col.min(), col.max()
        if mx - mn > 1e-8:
            result[..., i] = (col - mn) / (mx - mn)
        else:
            result[..., i] = 0.5
    return np.clip(result, 0, 1)


def project_to_pca(pca, patches):
    """Project patch tokens to PCA space and normalize for display."""
    proj = pca.transform(patches)  # (N, n_components)
    n = int(proj.shape[1])
    # If n_components < 3, pad with zeros
    if n < 3:
        padded = np.zeros((proj.shape[0], 3))
        padded[:, :n] = proj
        proj = padded
    return minmax_01(proj.reshape(16, 16, 3))


def patch_norm_heatmap(patch_tokens):
    """Compute L2 norm of each patch token as an 'attention' proxy."""
    norms = np.linalg.norm(patch_tokens, axis=1)  # (256,)
    return norms.reshape(16, 16)


def cls_patch_similarity(patch_tokens, cls_token):
    """Cosine similarity between CLS token and each patch token."""
    patch_norm = patch_tokens / (np.linalg.norm(patch_tokens, axis=1, keepdims=True) + 1e-8)
    cls_norm = cls_token / (np.linalg.norm(cls_token) + 1e-8)
    sims = patch_norm @ cls_norm  # (256,)
    return sims.reshape(16, 16)


# ──────────────────────────── main ────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 PCA feature visualization for spawn detection"
    )
    parser.add_argument(
        "--pos-dir",
        default="data/samples/positive",
        help="Directory with positive (spawn) samples",
    )
    parser.add_argument(
        "--neg-dir",
        default="data/samples/negative",
        help="Directory with negative (non-spawn) samples",
    )
    parser.add_argument(
        "--output-dir",
        default="data/review/dinov2_features",
        help="Where to save visualizations",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=16,
        help="Number of samples per class (use all if more available)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="0=last layer, 1=second-to-last, etc.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ──────────────────────────────────────────

    print("Loading DINOv2 ViT-S/14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    n_layers = len(model.blocks)
    print(f"  Device: {device}, Layers: {n_layers}")

    # Standard ImageNet normalization (used by DINOv2)
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── 2. Load images ─────────────────────────────────────────

    pos_dir = Path(args.pos_dir)
    neg_dir = Path(args.neg_dir)

    pos_files = sorted(pos_dir.glob("*.png"))
    neg_files = sorted(neg_dir.glob("*.png"))

    # Use the specified number, respecting available count
    n_pos = min(args.n_samples, len(pos_files))
    n_neg = min(args.n_samples, len(neg_files))

    pos_files = pos_files[:n_pos]
    neg_files = neg_files[:n_neg]

    # Alternate for balanced ordering in gallery
    entries = []
    max_n = max(len(pos_files), len(neg_files))
    for i in range(max_n):
        if i < len(pos_files):
            entries.append((pos_files[i], "spawn"))
        if i < len(neg_files):
            entries.append((neg_files[i], "nospawn"))

    print(f"Loaded {n_pos} spawn + {n_neg} non-spawn = {len(entries)} images")

    # ── 3. Extract patch features ─────────────────────────────

    # Storage
    all_features = {}   # fname -> {patch_tokens, cls_token, image_pil}
    all_patches = []    # for cross-image PCA

    for img_path, label in entries:
        fname = img_path.name
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            # get_intermediate_layers with reshape=True returns patch tokens
            # in spatial format: [B, D, H, W] = [1, 384, 16, 16]
            # cls_tokens: [B, D] = [1, 384]
            # fmt: off
            patch_tokens, cls_tokens = model.get_intermediate_layers(
                img_tensor, n=1, reshape=True, return_class_token=True
            )[0]
            # fmt: on

        # Reshape from [B, D, H, W] to [B, N, D] then [N, D]
        pt = patch_tokens.flatten(2).transpose(1, 2).squeeze(0).cpu().numpy()  # (256, 384)
        ct = cls_tokens.squeeze(0).cpu().numpy()  # (384,)

        all_features[fname] = {
            "patch_tokens": pt,
            "cls_token": ct,
            "image_pil": img_pil,
            "label": label,
            "path": str(img_path),
        }
        all_patches.append(pt)

    # ── 4. Fit cross-image PCA on ALL patch tokens ────────────

    combined = np.vstack(all_patches)  # (N_images * 256, 384)
    print(f"Fitting PCA on {combined.shape[0]} patch tokens ({combined.shape[1]} dims)...")

    pca = PCA(n_components=3)
    pca.fit(combined)
    var_ratio = pca.explained_variance_ratio_
    print(f"  Explained variance: PC1={var_ratio[0]:.4f}, PC2={var_ratio[1]:.4f}, PC3={var_ratio[2]:.4f}")
    print(f"  Total: {var_ratio.sum():.4f}")

    # Also fit a per-image PCA comparison + 10-component PCA for richer analysis
    pca_10 = PCA(n_components=10)
    pca_10.fit(combined)
    var_10 = pca_10.explained_variance_ratio_
    print(f"  Top-10 explained variance: {var_10.sum():.4f}")

    # ── 5. Generate per-image visualizations ──────────────────

    print(f"\nGenerating visualizations for {len(entries)} images...")

    pca_components_path = out_dir / "pca_components.png"

    # We'll collect data for class comparison
    spawn_projs = []
    nospawn_projs = []
    spawn_norms = []
    nospawn_norms = []
    spawn_sims = []
    nospawn_sims = []
    gallery_rows = []

    for img_path, label in entries:
        fname = img_path.name
        feat = all_features[fname]
        pt = feat["patch_tokens"]
        ct = feat["cls_token"]
        img_pil = feat["image_pil"]
        display_label = "SPAWN" if label == "spawn" else "NO SPAWN"

        # Cross-image PCA projection
        pca_map = project_to_pca(pca, pt)

        # Per-image PCA (for comparison)
        pca_img = PCA(n_components=3)
        pca_img_proj = pca_img.fit_transform(pt)
        pca_img_map = minmax_01(pca_img_proj.reshape(16, 16, 3))

        # Attention proxy: CLS-patch similarity
        sim_map = cls_patch_similarity(pt, ct)

        # Feature norm heatmap (secondary attention proxy)
        norm_map = patch_norm_heatmap(pt)

        # Resized original image
        img_224 = np.array(img_pil.resize((224, 224)))

        # ── Create figure ──
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle(f"{display_label}: {fname}", fontsize=13, fontweight="bold", y=0.98)

        # Row 0: Original, Cross PCA, Per-image PCA, Overlay
        axes[0, 0].imshow(img_224)
        axes[0, 0].set_title("Original (224×224)", fontsize=10)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(pca_map, interpolation="nearest")
        axes[0, 1].set_title("Cross-image PCA (top-3)", fontsize=10)
        axes[0, 1].axis("off")

        axes[0, 2].imshow(pca_img_map, interpolation="nearest")
        axes[0, 2].set_title("Per-image PCA", fontsize=10)
        axes[0, 2].axis("off")

        # Overlay: blend cross-PCA with original
        pca_rgb = (pca_map * 255).astype(np.uint8)
        pca_large = np.array(Image.fromarray(pca_rgb).resize((224, 224), Image.NEAREST))
        overlay = (img_224.astype(np.float32) * 0.5 + pca_large.astype(np.float32) * 0.5).astype(np.uint8)
        axes[0, 3].imshow(overlay)
        axes[0, 3].set_title("Overlay (orig × PCA)", fontsize=10)
        axes[0, 3].axis("off")

        # Row 1: CLS similarity heatmap, Norm heatmap, Feature diff, Legend
        im1 = axes[1, 0].imshow(sim_map, cmap="inferno", interpolation="nearest")
        axes[1, 0].set_title("CLS-Patch Similarity", fontsize=10)
        axes[1, 0].axis("off")
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im2 = axes[1, 1].imshow(norm_map, cmap="viridis", interpolation="nearest")
        axes[1, 1].set_title("Patch Feature Norm", fontsize=10)
        axes[1, 1].axis("off")
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # Per-channel PCA components
        comps_ch = []
        for ch in range(3):
            ch_map = pca_proj_3ch(pt, ch, pca)
            comps_ch.append(ch_map)
        ch_grid = np.concatenate(comps_ch, axis=1)
        im3 = axes[1, 2].imshow(ch_grid, cmap="RdBu_r", interpolation="nearest",
                                 vmin=-1, vmax=1)
        axes[1, 2].set_title("PCA channels (PC1|PC2|PC3)", fontsize=10)
        axes[1, 2].axis("off")
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

        axes[1, 3].axis("off")
        # Summary stats text
        stats_text = (
            f"CLS sim: μ={sim_map.mean():.4f} σ={sim_map.std():.4f}\n"
            f"Norm: μ={norm_map.mean():.3f} σ={norm_map.std():.3f}\n"
            f"PCA var: {var_ratio[0]:.2%}+{var_ratio[1]:.2%}+{var_ratio[2]:.2%}"
        )
        axes[1, 3].text(0.1, 0.5, stats_text, transform=axes[1, 3].transAxes,
                        fontsize=11, verticalalignment="center",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # Save
        safe_name = fname.replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"{label}_{safe_name}"
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close()

        # Collect for class comparison
        proj = pca.transform(pt)  # (256, 3)
        if label == "spawn":
            spawn_projs.append(proj)
            spawn_norms.append(norm_map)
            spawn_sims.append(sim_map)
        else:
            nospawn_projs.append(proj)
            nospawn_norms.append(norm_map)
            nospawn_sims.append(sim_map)

        gallery_rows.append({
            "filename": fname,
            "label": label,
            "display_label": display_label,
            "image_path": out_path.name,
            "cls_sim_mean": float(sim_map.mean()),
            "cls_sim_std": float(sim_map.std()),
            "norm_mean": float(norm_map.mean()),
            "norm_std": float(norm_map.std()),
        })

        print(f"  [{display_label:>8}] {fname}")

    # ── 6. Class comparison ────────────────────────────────────

    print("\nComputing class comparison...")

    # Mean PCA maps per class
    spawn_mean_proj = np.mean(spawn_projs, axis=0).reshape(16, 16, 3)
    nospawn_mean_proj = np.mean(nospawn_projs, axis=0).reshape(16, 16, 3)
    diff_proj = spawn_mean_proj - nospawn_mean_proj

    # Mean CLS-patch similarity per class (as spatial average)
    spawn_sim_mean = np.mean(spawn_sims, axis=0)
    nospawn_sim_mean = np.mean(nospawn_sims, axis=0)
    sim_diff = spawn_sim_mean - nospawn_sim_mean

    # Mean norm per class
    spawn_norm_mean = np.mean(spawn_norms, axis=0)
    nospawn_norm_mean = np.mean(nospawn_norms, axis=0)
    norm_diff = spawn_norm_mean - nospawn_norm_mean

    # Normalize for visualization
    spawn_mean_viz = minmax_01(
        (spawn_mean_proj - spawn_mean_proj.min(axis=(0, 1), keepdims=True))
        / (spawn_mean_proj.max(axis=(0, 1), keepdims=True)
           - spawn_mean_proj.min(axis=(0, 1), keepdims=True) + 1e-8)
    )
    nospawn_mean_viz = minmax_01(
        (nospawn_mean_proj - nospawn_mean_proj.min(axis=(0, 1), keepdims=True))
        / (nospawn_mean_proj.max(axis=(0, 1), keepdims=True)
           - nospawn_mean_proj.min(axis=(0, 1), keepdims=True) + 1e-8)
    )

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Class Comparison: Spawn vs Non-Spawn Feature Maps", fontsize=14, fontweight="bold")

    # Row 0: Mean PCA maps
    axes[0, 0].imshow(spawn_mean_viz, interpolation="nearest")
    axes[0, 0].set_title(f"Mean PCA — Spawn ({len(spawn_projs)} images)", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(nospawn_mean_viz, interpolation="nearest")
    axes[0, 1].set_title(f"Mean PCA — No Spawn ({len(nospawn_projs)} images)", fontsize=10)
    axes[0, 1].axis("off")

    # Difference: use diverging colormap
    diff_max = max(abs(diff_proj.min()), abs(diff_proj.max()))
    for ch in range(3):
        d = diff_proj[:, :, ch]
        mx = max(abs(d.min()), abs(d.max()))
        if mx > 0:
            diff_proj[:, :, ch] = d / mx
    im3 = axes[0, 2].imshow(diff_proj, cmap="RdBu_r", interpolation="nearest", vmin=-1, vmax=1)
    axes[0, 2].set_title("Difference (Spawn − No Spawn)", fontsize=10)
    axes[0, 2].axis("off")
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Per-channel breakdown of difference
    ch_plots = []
    for ch in range(3):
        d = diff_proj[:, :, ch]
        mx = max(abs(d.min()), abs(d.max())) + 1e-8
        ch_plots.append(d / mx)
    ch_grid_diff = np.concatenate(ch_plots, axis=1)
    im4 = axes[0, 3].imshow(ch_grid_diff, cmap="RdBu_r", interpolation="nearest", vmin=-1, vmax=1)
    axes[0, 3].set_title("Diff per channel (PC1|PC2|PC3)", fontsize=10)
    axes[0, 3].axis("off")
    plt.colorbar(im4, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # Row 1: CLS similarity comparison, Norm comparison
    mx_sim = max(abs(sim_diff.min()), abs(sim_diff.max())) + 1e-8
    im5 = axes[1, 0].imshow(sim_diff / mx_sim, cmap="RdBu_r", interpolation="nearest", vmin=-1, vmax=1)
    axes[1, 0].set_title("CLS-Sim Difference (spawn−nospawn)", fontsize=10)
    axes[1, 0].axis("off")
    plt.colorbar(im5, ax=axes[1, 0], fraction=0.046, pad=0.04)

    mx_norm = max(abs(norm_diff.min()), abs(norm_diff.max())) + 1e-8
    im6 = axes[1, 1].imshow(norm_diff / mx_norm, cmap="RdBu_r", interpolation="nearest", vmin=-1, vmax=1)
    axes[1, 1].set_title("Feature Norm Difference (spawn−nospawn)", fontsize=10)
    axes[1, 1].axis("off")
    plt.colorbar(im6, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Spawn vs nospawn mean CLS similarity per image (bar chart)
    spawn_sim_vals = [r["cls_sim_mean"] for r in gallery_rows if r["label"] == "spawn"]
    nospawn_sim_vals = [r["cls_sim_mean"] for r in gallery_rows if r["label"] == "nospawn"]
    axes[1, 2].bar(0, np.mean(spawn_sim_vals), yerr=np.std(spawn_sim_vals),
                   color="green", alpha=0.7, capsize=5, label="Spawn")
    axes[1, 2].bar(1, np.mean(nospawn_sim_vals), yerr=np.std(nospawn_sim_vals),
                   color="red", alpha=0.7, capsize=5, label="No Spawn")
    axes[1, 2].set_xticks([0, 1])
    axes[1, 2].set_xticklabels(["Spawn", "No Spawn"])
    axes[1, 2].set_ylabel("Mean CLS-Patch Similarity")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_title("CLS-Sim by Class", fontsize=10)

    # Feature norm comparison bar
    spawn_norm_vals = [r["norm_mean"] for r in gallery_rows if r["label"] == "spawn"]
    nospawn_norm_vals = [r["norm_mean"] for r in gallery_rows if r["label"] == "nospawn"]
    axes[1, 3].bar(0, np.mean(spawn_norm_vals), yerr=np.std(spawn_norm_vals),
                   color="green", alpha=0.7, capsize=5, label="Spawn")
    axes[1, 3].bar(1, np.mean(nospawn_norm_vals), yerr=np.std(nospawn_norm_vals),
                   color="red", alpha=0.7, capsize=5, label="No Spawn")
    axes[1, 3].set_xticks([0, 1])
    axes[1, 3].set_xticklabels(["Spawn", "No Spawn"])
    axes[1, 3].set_ylabel("Mean Patch Feature Norm")
    axes[1, 3].legend(fontsize=8)
    axes[1, 3].set_title("Feature Norm by Class", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "class_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved class_comparison.png")

    # ── 7. PCA component visualization ─────────────────────────

    # Show per-class distribution of PCA projection values
    # Collect all projection values per class from stored projections
    spawn_pc_vals = {i: [] for i in range(3)}
    nospawn_pc_vals = {i: [] for i in range(3)}
    for proj in spawn_projs:
        for i in range(3):
            spawn_pc_vals[i].extend(proj[:, i].tolist())
    for proj in nospawn_projs:
        for i in range(3):
            nospawn_pc_vals[i].extend(proj[:, i].tolist())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        axes[i].hist(spawn_pc_vals[i], bins=50, alpha=0.6, color="green",
                     label=f"Spawn (μ={np.mean(spawn_pc_vals[i]):.3f})")
        axes[i].hist(nospawn_pc_vals[i], bins=50, alpha=0.6, color="red",
                     label=f"No Spawn (μ={np.mean(nospawn_pc_vals[i]):.3f})")
        axes[i].set_title(f"PC{i+1} projection distribution", fontsize=10)
        axes[i].set_xlabel("Projection value")
        axes[i].set_ylabel("Frequency")
        axes[i].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(pca_components_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved pca_components.png")

    # ── 8. Spatial pattern analysis ────────────────────────────

    print("\nComputing spatial pattern analysis...")
    # For each image, compute spatial statistics on the PCA feature maps
    spawn_spatial = {"pc_variance": [], "bottom_activation": [], "spatial_entropy": []}
    nospawn_spatial = {"pc_variance": [], "bottom_activation": [], "spatial_entropy": []}

    for proj_3d, label in zip(spawn_projs + nospawn_projs,
                               ["spawn"] * len(spawn_projs) + ["nospawn"] * len(nospawn_projs)):
        target = spawn_spatial if label == "spawn" else nospawn_spatial

        # Reshape to spatial grid
        spatial = proj_3d.reshape(16, 16, 3)  # (16, 16, 3)

        # 1. Spatial variance per PCA channel (mean across channels)
        pc_var = np.var(spatial, axis=(0, 1)).mean()
        target["pc_variance"].append(float(pc_var))

        # 2. Bottom-half vs top-half activation (water vs land proxy)
        # Bottom 8 rows = water area in typical coastal imagery
        bottom = spatial[8:, :, :]  # (8, 16, 3)
        top = spatial[:8, :, :]     # (8, 16, 3)
        bottom_act = np.abs(bottom).mean()
        top_act = np.abs(top).mean()
        bottom_ratio = bottom_act / (top_act + 1e-8)
        target["bottom_activation"].append(float(bottom_ratio))

        # 3. Spatial entropy: measure of how "structured" the feature map is
        # Use the std of the spatial gradient as a structure proxy
        grad_h = np.abs(np.diff(spatial, axis=0)).mean()
        grad_v = np.abs(np.diff(spatial, axis=1)).mean()
        entropy_proxy = float(np.sqrt(grad_h**2 + grad_v**2))
        target["spatial_entropy"].append(entropy_proxy)

    # Statistical tests on spatial features
    for metric_name in ["pc_variance", "bottom_activation", "spatial_entropy"]:
        sv = spawn_spatial[metric_name]
        nv = nospawn_spatial[metric_name]
        t_stat, p_val = ttest_ind(sv, nv)
        sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
        print(f"  {metric_name}: spawn={np.mean(sv):.4f} nospawn={np.mean(nv):.4f} "
              f"t={t_stat:.3f} p={p_val:.4f} {sig}")

    # Spawn vs nospawn spatial metrics bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (metric, title) in enumerate([
        ("pc_variance", "PCA Spatial Variance"),
        ("bottom_activation", "Bottom/Top Activation Ratio"),
        ("spatial_entropy", "Spatial Gradient Magnitude"),
    ]):
        sv = spawn_spatial[metric]
        nv = nospawn_spatial[metric]
        axes[i].bar(0, np.mean(sv), yerr=np.std(sv), color="green", alpha=0.7,
                    capsize=5, label=f"Spawn (n={len(sv)})")
        axes[i].bar(1, np.mean(nv), yerr=np.std(nv), color="red", alpha=0.7,
                    capsize=5, label=f"No Spawn (n={len(nv)})")
        _, pv = ttest_ind(sv, nv)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Spawn", "No Spawn"])
        axes[i].set_ylabel(title)
        axes[i].set_title(f"{title}\np={pv:.4f}")
        axes[i].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "spatial_analysis.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved spatial_analysis.png")

    # ── 9. Analysis report ─────────────────────────────────────

    # Quantitative metrics
    cls_sim_t, cls_p = ttest_ind(spawn_sim_vals, nospawn_sim_vals)
    norm_t, norm_p = ttest_ind(spawn_norm_vals, nospawn_norm_vals)

    # Compute significance for spatial metrics
    sp_metrics = {}
    for metric_name in ["pc_variance", "bottom_activation", "spatial_entropy"]:
        sv = spawn_spatial[metric_name]
        nv = nospawn_spatial[metric_name]
        t_stat, p_val = ttest_ind(sv, nv)
        sp_metrics[metric_name] = {
            "spawn_mean": float(np.mean(sv)),
            "spawn_std": float(np.std(sv)),
            "nospawn_mean": float(np.mean(nv)),
            "nospawn_std": float(np.std(nv)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
        }

    analysis = {
        "n_spawn": len(spawn_projs),
        "n_nospawn": len(nospawn_projs),
        "pca_explained_variance": {
            "pc1": float(var_ratio[0]),
            "pc2": float(var_ratio[1]),
            "pc3": float(var_ratio[2]),
            "top3_total": float(var_ratio.sum()),
            "top10_total": float(var_10.sum()),
        },
        "cls_similarity": {
            "spawn_mean": float(np.mean(spawn_sim_vals)),
            "spawn_std": float(np.std(spawn_sim_vals)),
            "nospawn_mean": float(np.mean(nospawn_sim_vals)),
            "nospawn_std": float(np.std(nospawn_sim_vals)),
            "t_statistic": float(cls_sim_t),
            "p_value": float(cls_p),
            "significant": bool(cls_p < 0.05),
        },
        "feature_norm": {
            "spawn_mean": float(np.mean(spawn_norm_vals)),
            "spawn_std": float(np.std(spawn_norm_vals)),
            "nospawn_mean": float(np.mean(nospawn_norm_vals)),
            "nospawn_std": float(np.std(nospawn_norm_vals)),
            "t_statistic": float(norm_t),
            "p_value": float(norm_p),
            "significant": bool(norm_p < 0.05),
        },
        "spatial_analysis": sp_metrics,
        "images": gallery_rows,
    }

    json_path = out_dir / "analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2))
    print(f"  Saved analysis.json")

    # ── 9. HTML gallery ────────────────────────────────────────

    print("Generating HTML gallery...")

    # Image rows sorted: spawn first, then nospawn
    spawn_rows = [r for r in gallery_rows if r["label"] == "spawn"]
    nospawn_rows = [r for r in gallery_rows if r["label"] == "nospawn"]

    def make_card(row):
        return f"""
        <div class="card {row['label']}">
            <div class="badge {row['label']}">{row['display_label']}</div>
            <img src="{row['image_path']}" loading="lazy"
                 onclick="toggleExpanded(this)" title="Click to expand">
            <div class="caption">{row['filename']}</div>
            <div class="stats">
                CLS-sim: {row['cls_sim_mean']:.4f} ± {row['cls_sim_std']:.4f}<br>
                Norm: {row['norm_mean']:.3f} ± {row['norm_std']:.3f}
            </div>
        </div>"""

    pos_cards = "\n".join(make_card(r) for r in spawn_rows)
    neg_cards = "\n".join(make_card(r) for r in nospawn_rows)

    # Significance indicators
    cls_sig = "\u2705" if analysis["cls_similarity"]["significant"] else "\u274c"
    norm_sig = "\u2705" if analysis["feature_norm"]["significant"] else "\u274c"

    # Build spatial analysis summary rows for HTML
    sp_rows_html = ""
    metric_labels = {
        "pc_variance": "PCA Spatial Variance",
        "bottom_activation": "Bottom/Top Activation Ratio",
        "spatial_entropy": "Spatial Gradient Magnitude",
    }
    for key, label in metric_labels.items():
        sm = analysis["spatial_analysis"][key]
        sig = "\u2705" if sm["significant"] else "\u274c"
        sp_rows_html += f"""
<tr>
    <td>{label}</td>
    <td>{fmt(sm['spawn_mean'])} ± {fmt(sm['spawn_std'])}</td>
    <td>{fmt(sm['nospawn_mean'])} ± {fmt(sm['nospawn_std'])}</td>
    <td>{fmt(sm['spawn_mean'] - sm['nospawn_mean'])}</td>
    <td class="{'sig-yes' if sm['significant'] else 'sig-no'}">
        {sig} p={sm['p_value']:.4f}</td>
</tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DINOv2 Feature Visualization — Spawn Detection</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
h1 {{ font-size: 22px; margin-bottom: 5px; }}
h2 {{ font-size: 18px; margin: 20px 0 10px; border-bottom: 1px solid #444; padding-bottom: 5px; }}
.subtitle {{ color: #aaa; font-size: 14px; margin-bottom: 20px; }}
.summary {{ background: #16213e; border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
.summary table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.summary th, .summary td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #333; }}
.summary th {{ color: #8cf; }}
.sig-yes {{ color: #4caf50; font-weight: bold; }}
.sig-no {{ color: #f44336; font-weight: bold; }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
           gap: 15px; }}
.card {{ background: #16213e; border-radius: 8px; overflow: hidden; position: relative; }}
.card.spawn {{ border-left: 4px solid #4caf50; }}
.card.nospawn {{ border-left: 4px solid #f44336; }}
.badge {{ position: absolute; top: 8px; right: 8px; padding: 3px 10px;
          border-radius: 4px; font-size: 11px; font-weight: bold; z-index: 2; }}
.badge.spawn {{ background: #4caf50; color: #fff; }}
.badge.nospawn {{ background: #f44336; color: #fff; }}
.card img {{ width: 100%; height: auto; display: block; cursor: pointer;
             transition: transform 0.2s; }}
.card img:hover {{ transform: scale(1.02); }}
.card img.expanded {{ transform: scale(1.8); transform-origin: top left; }}
.caption {{ padding: 8px 12px; font-size: 12px; color: #aaa;
            word-break: break-all; }}
.stats {{ padding: 0 12px 10px; font-size: 11px; color: #888; line-height: 1.5; }}
</style>
</head>
<body>

<h1>DINOv2 PCA Feature Visualization</h1>
<p class="subtitle">ViT-S/14 patch-level features to PCA spatial maps |
    {n_pos} spawn + {n_neg} non-spawn samples

<div class="summary">
<table>
<tr><th>Metric</th><th>Spawn</th><th>No Spawn</th><th>Difference</th><th>Significant?</th></tr>
<tr>
    <td>CLS-Patch Similarity</td>
    <td>{fmt(analysis["cls_similarity"]["spawn_mean"])} &plusmn; {fmt(analysis["cls_similarity"]["spawn_std"])}</td>
    <td>{fmt(analysis["cls_similarity"]["nospawn_mean"])} &plusmn; {fmt(analysis["cls_similarity"]["nospawn_std"])}</td>
    <td>{fmt(analysis["cls_similarity"]["spawn_mean"] - analysis["cls_similarity"]["nospawn_mean"])}</td>
    <td class="{'sig-yes' if analysis['cls_similarity']['significant'] else 'sig-no'}">
        {cls_sig} p={analysis["cls_similarity"]["p_value"]:.4f}</td>
</tr>
<tr>
    <td>Patch Feature Norm</td>
    <td>{fmt(analysis["feature_norm"]["spawn_mean"])} &plusmn; {fmt(analysis["feature_norm"]["spawn_std"])}</td>
    <td>{fmt(analysis["feature_norm"]["nospawn_mean"])} &plusmn; {fmt(analysis["feature_norm"]["nospawn_std"])}</td>
    <td>{fmt(analysis["feature_norm"]["spawn_mean"] - analysis["feature_norm"]["nospawn_mean"])}</td>
    <td class="{'sig-yes' if analysis['feature_norm']['significant'] else 'sig-no'}">
        {norm_sig} p={analysis["feature_norm"]["p_value"]:.4f}</td>
</tr>
{sp_rows_html}
</table>
<p style="margin-top:10px; font-size:12px; color:#aaa;">
    PCA explained variance: PC1={var_ratio[0]:.1%}, PC2={var_ratio[1]:.1%}, PC3={var_ratio[2]:.1%}
    (top-3: {var_ratio.sum():.1%}, top-10: {var_10.sum():.1%})
</p>
</div>

<h2>Class Comparison &amp; Spatial Analysis</h2>
<div style="margin-bottom:20px; display:flex; flex-wrap:wrap; gap:10px;">
    <img src="class_comparison.png" style="flex:1; min-width:400px; max-width:800px; border-radius:8px;">
    <img src="spatial_analysis.png" style="flex:1; min-width:400px; max-width:600px; border-radius:8px;">
</div>
<div style="margin-bottom:20px;">
    <img src="pca_components.png" style="width:100%; max-width:900px; border-radius:8px;">
</div>

<h2>Spawn Samples</h2>
<div class="gallery">{pos_cards}</div>

<h2>No-Spawn Samples</h2>
<div class="gallery">{neg_cards}</div>

<script>
function toggleExpanded(img) {{
    img.classList.toggle('expanded');
    if (img.classList.contains('expanded')) {{
        img.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    }}
}}
</script>

</body>
</html>"""

    html_path = out_dir / "index.html"
    html_path.write_text(html)
    print(f"  Saved index.html")

    # ── 10. Print final report ─────────────────────────────────

    print(f"\n{'='*65}")
    print(f"  DINOv2 FEATURE VISUALIZATION — RESULTS")
    print(f"{'='*65}")
    print(f"  Samples: {n_pos} spawn, {n_neg} non-spawn")
    print(f"  PCA explained variance (top-3): {var_ratio.sum():.2%}")
    print(f"  PCA explained variance (top-10): {var_10.sum():.2%}")
    print()
    c = analysis["cls_similarity"]
    print(f"  CLS-Patch Similarity:")
    print(f"    Spawn:    {c['spawn_mean']:.4f} ± {c['spawn_std']:.4f}")
    print(f"    No Spawn: {c['nospawn_mean']:.4f} ± {c['nospawn_std']:.4f}")
    print(f"    t={c['t_statistic']:.3f}, p={c['p_value']:.4f} {'SIGNIFICANT' if c['significant'] else 'not significant'}")
    print()
    f = analysis["feature_norm"]
    print(f"  Patch Feature Norm:")
    print(f"    Spawn:    {f['spawn_mean']:.3f} ± {f['spawn_std']:.3f}")
    print(f"    No Spawn: {f['nospawn_mean']:.3f} ± {f['nospawn_std']:.3f}")
    print(f"    t={f['t_statistic']:.3f}, p={f['p_value']:.4f} {'SIGNIFICANT' if f['significant'] else 'not significant'}")
    print()
    print(f"  Spatial Pattern Analysis:")
    for key in ["pc_variance", "bottom_activation", "spatial_entropy"]:
        sm = analysis["spatial_analysis"][key]
        print(f"    {key}:")
        print(f"      Spawn:    {sm['spawn_mean']:.4f} ± {sm['spawn_std']:.4f}")
        print(f"      No Spawn: {sm['nospawn_mean']:.4f} ± {sm['nospawn_std']:.4f}")
        print(f"      t={sm['t_statistic']:.3f}, p={sm['p_value']:.4f} {'SIGNIFICANT' if sm['significant'] else 'not significant'}")
    print()
    print(f"  Output: {out_dir.resolve()}")
    print(f"  Gallery: file://{out_dir.resolve() / 'index.html'}")
    print(f"{'='*65}")


def pca_proj_3ch(patches, channel, pca):
    """Project patches onto a single PCA component and return as 16×16 map."""
    proj = pca.transform(patches)  # (256, 3)
    ch = proj[:, channel]
    mx = max(abs(ch.min()), abs(ch.max())) + 1e-8
    return (ch / mx).reshape(16, 16)


def ttest_ind(a, b):
    """Independent t-test. Returns (t_statistic, p_value)."""
    from scipy import stats as scipy_stats
    # Use scipy if available
    try:
        res = scipy_stats.ttest_ind(a, b)
        return float(res.statistic), float(res.pvalue)
    except ImportError:
        pass

    # Fallback: manual computation
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0
    m1, m2 = np.mean(a), np.mean(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    if se < 1e-15:
        return 0.0, 1.0
    t = (m1 - m2) / se
    # Welch-Satterthwaite df
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else 1.0
    # Approximate p-value using normal distribution for large df
    # Use simplified: |t| > 2 => roughly p < 0.05 for df > 10
    from math import erf
    p = 1.0 - erf(abs(t) / np.sqrt(2))
    return float(t), float(p)


if __name__ == "__main__":
    main()
