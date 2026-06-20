"""Tests for DINOv3 feature extractor."""

import numpy as np
import pytest
from PIL import Image


class TestDINOv3FeatureExtractor:
    """Test the DINOv3 similarity map extraction."""

    def test_import(self):
        from scripts.dinov3_feature_extractor import DINOv3FeatureExtractor
        assert DINOv3FeatureExtractor is not None

    def test_extractor_creates_correct_size_mask(self):
        """Given a mock extractor, similarity map produces mask matching input size."""
        from scripts.dinov3_feature_extractor import DINOv3FeatureExtractor

        extractor = DINOv3FeatureExtractor(device="cpu")
        import torch
        extractor._model = _MockDINOv3Model()
        extractor._loaded = True

        img = Image.new("RGB", (512, 256), color=(100, 150, 200))
        mask = extractor.get_similarity_map(img, (256, 128))

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (256, 512)
        assert mask.dtype == np.uint8
        assert mask.max() <= 1

    def test_click_to_patch_mapping(self):
        """Click coordinates correctly map to patch indices."""
        from scripts.dinov3_feature_extractor import DINOv3FeatureExtractor

        extractor = DINOv3FeatureExtractor(device="cpu")
        extractor._model = _MockDINOv3Model()
        extractor._loaded = True

        img = Image.new("RGB", (256, 256))
        mask_tl = extractor.get_similarity_map(img, (0, 0))
        mask_br = extractor.get_similarity_map(img, (255, 255))

        assert mask_tl.shape == (256, 256)
        assert mask_br.shape == (256, 256)

    def test_similarity_self_clicks_produce_strong_center(self):
        """Clicking the same location should yield high similarity there."""
        from scripts.dinov3_feature_extractor import DINOv3FeatureExtractor

        extractor = DINOv3FeatureExtractor(device="cpu")
        extractor._model = _MockDINOv3Model()
        extractor._loaded = True

        img = Image.new("RGB", (256, 256))
        mask = extractor.get_similarity_map(img, (128, 128))

        center = mask[112:144, 112:144].mean()
        edge = np.concatenate([mask[0:10, :].flatten(), mask[-10:, :].flatten(),
                               mask[:, 0:10].flatten(), mask[:, -10:].flatten()]).mean()
        assert center > edge, f"center={center:.3f}, edge={edge:.3f}"


class _MockDINOv3Model:
    """Fake model that returns identity-like patch tokens so self-similarity
    peaks at the clicked patch. Matches DINOv3 SAT-493M: 256 patches (16x16),
    4 register tokens (num_prefix_tokens=5)."""
    num_prefix_tokens = 5  # 1 CLS + 4 register tokens

    def eval(self):
        return self
    def to(self, device):
        return self

    def forward_features(self, x):
        import torch
        N_PATCHES = 256
        D = 1024
        N_PREFIX = 5  # 1 CLS + 4 register tokens
        tokens = torch.eye(N_PATCHES, D)[:N_PATCHES, :D]
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
        prefix = torch.zeros(N_PREFIX, D)
        full = torch.cat([prefix, tokens], dim=0)  # (261, 1024)
        return full.unsqueeze(0)  # (1, 261, 1024)
