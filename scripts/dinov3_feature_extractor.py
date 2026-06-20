"""DINOv3 ViT-L/16 SAT-493M feature extractor for herring spawn segmentation.

Computes cosine-similarity heatmaps from click points to guide SAM-based
segmentation. Uses the satellite-pretrained DINOv3 model via timm.

This model uses a 256x256 input resolution (patch_size=16 → 16x16 grid)
and includes 4 register tokens (num_prefix_tokens=5).

Usage:
    extractor = DINOv3FeatureExtractor(device="cuda")
    mask = extractor.get_similarity_map(image, (click_x, click_y))
"""

import threading
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

MODEL_ID = "hf-hub:timm/vit_large_patch16_dinov3.sat493m"
PATCH_SIZE = 16
# Model expects 256x256 input -> 16x16 grid (256/16 = 16)
INPUT_SIZE = 256
GRID_SIZE = INPUT_SIZE // PATCH_SIZE  # 16
N_PATCHES = GRID_SIZE * GRID_SIZE  # 256
EMBED_DIM = 1024

_model_lock = threading.Lock()
_global_model = None
_global_transform = None
_global_device = None


def _load_dinov3(device: str = "cpu"):
    global _global_model, _global_transform, _global_device
    if _global_model is not None and _global_device == device:
        return _global_model, _global_transform
    with _model_lock:
        if _global_model is None or _global_device != device:
            model = timm.create_model(MODEL_ID, pretrained=True, num_classes=0)
            model.eval()
            model = model.to(device)
            data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
            transform = timm.data.create_transform(**data_cfg)
            _global_model = model
            _global_transform = transform
            _global_device = device
    return _global_model, _global_transform


class DINOv3FeatureExtractor:
    """Extract DINOv3 patch tokens and compute cosine-similarity heatmaps.

    The heatmap highlights image regions that are semantically similar to
    the clicked point, serving as a coarse mask for SAM refinement.
    """

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._model = None
        self._transform = None
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._model, self._transform = _load_dinov3(self._device)
            self._loaded = True

    def get_patch_tokens(self, image: Image.Image) -> np.ndarray:
        """Extract patch tokens from an RGB image.

        Returns:
            np.ndarray shape (GRID_SIZE**2, EMBED_DIM) — normalized embeddings.
        """
        self._ensure_loaded()
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.forward_features(tensor)
        if features.dim() == 3:
            # Skip CLS + register tokens; patch tokens start at num_prefix_tokens
            n_prefix = getattr(self._model, 'num_prefix_tokens', 1)
            patches = features[:, n_prefix:, :]
        else:
            patches = features.flatten(2).transpose(1, 2)
        patches = F.normalize(patches, dim=-1)
        return patches.squeeze(0).cpu().numpy().astype(np.float32)

    def get_similarity_map(
        self, image: Image.Image, click_xy: tuple[int, int],
    ) -> np.ndarray:
        """Compute cosine-similarity heatmap from a click point.

        Args:
            image: RGB PIL image at any size.
            click_xy: (x, y) pixel coordinates in the original image space.

        Returns:
            np.ndarray of shape (H, W) — binary mask where 1 = similar to click.
        """
        model_to_use = self._model
        transform_to_use = self._transform
        if model_to_use is None:
            self._ensure_loaded()
            model_to_use = self._model
            transform_to_use = self._transform

        if transform_to_use is None:
            # Fallback for mock/test scenarios where _model is injected
            # but _transform is not set: just resize and normalize.
            from torchvision import transforms as T
            transform_to_use = T.Compose([
                T.Resize((INPUT_SIZE, INPUT_SIZE), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        orig_w, orig_h = image.width, image.height

        # The model transform already resizes to INPUT_SIZE; don't pre-resize.
        tensor = transform_to_use(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = model_to_use.forward_features(tensor)

        if features.dim() == 3:
            n_prefix = getattr(model_to_use, 'num_prefix_tokens', 1)
            patches = features[:, n_prefix:, :]
        else:
            patches = features.flatten(2).transpose(1, 2)

        patches = F.normalize(patches, dim=-1)

        # Derive grid size from actual number of patch tokens
        n_patches = patches.shape[1]
        grid_size = int(np.sqrt(n_patches))
        max_coord = INPUT_SIZE - 1

        scale_x = INPUT_SIZE / orig_w
        scale_y = INPUT_SIZE / orig_h
        cx = int(np.clip(click_xy[0] * scale_x, 0, max_coord))
        cy = int(np.clip(click_xy[1] * scale_y, 0, max_coord))
        px = min(cx // PATCH_SIZE, grid_size - 1)
        py = min(cy // PATCH_SIZE, grid_size - 1)
        patch_idx = py * grid_size + px

        clicked_feat = patches[0, patch_idx]
        similarity = (patches[0] @ clicked_feat).cpu().numpy()

        heatmap_grid = similarity.reshape(grid_size, grid_size)
        heatmap_full = np.array(
            Image.fromarray(
                ((heatmap_grid - heatmap_grid.min())
                 / (heatmap_grid.max() - heatmap_grid.min() + 1e-8)
                 * 255).astype(np.uint8)
            ).resize((orig_w, orig_h), Image.BILINEAR)
        ).astype(np.float32) / 255.0

        threshold = heatmap_full.mean() + 1.5 * heatmap_full.std()
        mask = (heatmap_full > threshold).astype(np.uint8)

        return mask
