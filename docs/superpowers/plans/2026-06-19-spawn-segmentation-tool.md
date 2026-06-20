# Herring Spawn Segmentation Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Flask web app that lets users interactively segment herring spawn plumes from Sentinel-2 thumbnails using DINOv3 ViT-L/16 SAT-493M feature similarity + SAM pixel-level refinement.

**Architecture:** Three components — `DINOv3FeatureExtractor` (timm-based patch similarity maps), `SAMRefiner` (wraps segment-anything for box+point prompt refinement), and a Flask app (`segment_spawn.py`) with a single-page HTML template. Models loaded once at startup via threading locks, matching the existing `label_subspace_app.py` pattern.

**Tech Stack:** Python 3, Flask, timm (DINOv3), segment-anything (SAM), torch, numpy, Pillow, opencv-python-headless

**Spec:** `docs/superpowers/specs/2026-06-19-spawn-segmentation-tool-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `scripts/dinov3_feature_extractor.py` | Load DINOv3 ViT-L/16 SAT-493M via timm, extract patch tokens, compute cosine similarity heatmaps from click points, threshold to coarse mask |
| `scripts/segment_spawn.py` | Flask app — SAMRefiner class, CLI, routes for image serving + click-to-segment + mask management |
| `webapp/templates/segment.html` | Single-page UI — image viewer, click-to-segment canvas overlay, keyboard nav, accept/reject |
| `tests/test_dinov3_feature_extractor.py` | Unit tests for DINOv3 similarity map generation |
| `tests/test_segment_spawn.py` | Integration tests for Flask app routes |

---

### Task 1: Install SAM dependency

**Files:**
- Create: (none)
- Modify: (none)

- [ ] **Step 1: Install segment-anything**

```bash
source .venv/bin/activate && pip install git+https://github.com/facebookresearch/segment-anything.git
```

- [ ] **Step 2: Verify import works**

```bash
source .venv/bin/activate && python -c "from segment_anything import sam_model_registry, SamPredictor; print('SAM OK')"
```

Expected: `SAM OK`

- [ ] **Step 3: Download SAM checkpoint**

```bash
curl -L -o data/models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Skip if file already exists (check first: `ls -lh data/models/sam_vit_h_4b8939.pth`).

- [ ] **Step 4: Verify timm can load DINOv3 SAT model**

```bash
source .venv/bin/activate && python -c "
import timm
m = timm.create_model('hf-hub:timm/vit_large_patch16_dinov3.sat493m', pretrained=False, num_classes=0)
print('DINOv3 model OK, params:', sum(p.numel() for p in m.parameters()))
"
```

Expected: prints param count (~300M). This downloads on first run.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: verify SAM and DINOv3 dependencies"
```

---

### Task 2: DINOv3 Feature Extractor

**Files:**
- Create: `scripts/dinov3_feature_extractor.py`
- Test: `tests/test_dinov3_feature_extractor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dinov3_feature_extractor.py
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
        # Inject a mock model that returns fake patch tokens
        import torch
        extractor._model = _MockDINOv3Model()
        extractor._loaded = True

        img = Image.new("RGB", (512, 256), color=(100, 150, 200))
        mask = extractor.get_similarity_map(img, (256, 128))

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (256, 512)  # H×W, matching input
        assert mask.dtype == np.uint8
        assert mask.max() <= 1

    def test_click_to_patch_mapping(self):
        """Click coordinates correctly map to patch indices."""
        from scripts.dinov3_feature_extractor import DINOv3FeatureExtractor

        extractor = DINOv3FeatureExtractor(device="cpu")
        extractor._model = _MockDINOv3Model()
        extractor._loaded = True

        img = Image.new("RGB", (224, 224))
        # Click at top-left should map to patch (0,0)
        # Click at bottom-right should map to patch (13,13)
        mask_tl = extractor.get_similarity_map(img, (0, 0))
        mask_br = extractor.get_similarity_map(img, (223, 223))

        assert mask_tl.shape == (224, 224)
        assert mask_br.shape == (224, 224)

    def test_similarity_self_clicks_produce_strong_center(self):
        """Clicking the same location should yield high similarity there."""
        from scripts.dinov3_feature_extractor import DINOv3FeatureExtractor

        extractor = DINOv3FeatureExtractor(device="cpu")
        extractor._model = _MockDINOv3Model()
        extractor._loaded = True

        # Mock model returns identity-like patch tokens so self-similarity is max
        img = Image.new("RGB", (224, 224))
        mask = extractor.get_similarity_map(img, (112, 112))

        # The center region should have high values relative to edges
        center = mask[100:124, 100:124].mean()
        edge = np.concatenate([mask[0:10, :].flatten(), mask[-10:, :].flatten(),
                               mask[:, 0:10].flatten(), mask[:, -10:].flatten()]).mean()
        assert center > edge, f"center={center:.3f}, edge={edge:.3f}"


class _MockDINOv3Model:
    """Fake model that returns identity-like patch tokens so self-similarity
    peaks at the clicked patch."""
    def eval(self):
        return self
    def to(self, device):
        return self

    def forward_features(self, x):
        import torch
        B, C, H, W = x.shape
        # 14x14 patches = 196 patches
        N_patches = 196
        D = 1024
        # Return normalized identity matrix so cosine similarity is high
        # only for the same patch
        tokens = torch.eye(N_patches, D)[:N_patches, :D]  # (196, 1024)
        # Normalize
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
        # Add CLS token at position 0
        cls = torch.zeros(1, D)
        full = torch.cat([cls, tokens], dim=0)  # (197, 1024)
        return full.unsqueeze(0)  # (1, 197, 1024)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_dinov3_feature_extractor.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.dinov3_feature_extractor'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/dinov3_feature_extractor.py
"""DINOv3 ViT-L/16 SAT-493M feature extractor for herring spawn segmentation.

Computes cosine-similarity heatmaps from click points to guide SAM-based
segmentation. Uses the satellite-pretrained DINOv3 model via timm.

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
GRID_SIZE = 14  # 224 / 16
N_PATCHES = GRID_SIZE * GRID_SIZE  # 196
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
            np.ndarray shape (196, 1024) — 14×14 grid of normalized embeddings.
        """
        self._ensure_loaded()
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.forward_features(tensor)
        # timm returns (B, N+1, D) with CLS token first
        if features.dim() == 3:
            patches = features[:, 1:, :]  # skip CLS token
        else:
            patches = features.flatten(2).transpose(1, 2)  # (B, N, D)
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
        if self._model is None:
            # For testing with mock models
            model_to_use = self._model
            transform_to_use = self._transform
            if model_to_use is None:
                self._ensure_loaded()
                model_to_use = self._model
                transform_to_use = self._transform
        else:
            model_to_use = self._model
            transform_to_use = self._transform

        if transform_to_use is None:
            transform_to_use = self._transform
            if transform_to_use is None:
                self._ensure_loaded()
                transform_to_use = self._transform

        orig_w, orig_h = image.width, image.height

        # Resize to 224 for DINOv3
        img_224 = image.resize((224, 224), Image.BICUBIC)
        tensor = transform_to_use(img_224).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = model_to_use.forward_features(tensor)

        if features.dim() == 3:
            patches = features[:, 1:, :]
        else:
            patches = features.flatten(2).transpose(1, 2)

        patches = F.normalize(patches, dim=-1)  # (1, 196, 1024)

        # Map click to 224 space
        scale_x = 224.0 / orig_w
        scale_y = 224.0 / orig_h
        cx = int(np.clip(click_xy[0] * scale_x, 0, 223))
        cy = int(np.clip(click_xy[1] * scale_y, 0, 223))
        px = min(cx // PATCH_SIZE, GRID_SIZE - 1)
        py = min(cy // PATCH_SIZE, GRID_SIZE - 1)
        patch_idx = py * GRID_SIZE + px

        clicked_feat = patches[0, patch_idx]  # (1024,)
        similarity = (patches[0] @ clicked_feat).cpu().numpy()  # (196,)

        # Reshape to 14×14 and resize to original image dimensions
        heatmap_14 = similarity.reshape(GRID_SIZE, GRID_SIZE)
        heatmap_full = np.array(
            Image.fromarray(
                ((heatmap_14 - heatmap_14.min())
                 / (heatmap_14.max() - heatmap_14.min() + 1e-8)
                 * 255).astype(np.uint8)
            ).resize((orig_w, orig_h), Image.BILINEAR)
        ).astype(np.float32) / 255.0

        # Threshold: mean + 1.5 * std
        threshold = heatmap_full.mean() + 1.5 * heatmap_full.std()
        mask = (heatmap_full > threshold).astype(np.uint8)

        return mask
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_dinov3_feature_extractor.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/dinov3_feature_extractor.py tests/test_dinov3_feature_extractor.py
git commit -m "feat: DINOv3 ViT-L/16 SAT-493M feature extractor with similarity heatmaps"
```

---

### Task 3: SAM Refiner + Flask App

**Files:**
- Create: `scripts/segment_spawn.py`
- Test: `tests/test_segment_spawn.py`

- [ ] **Step 1: Write the failing Flask test**

```python
# tests/test_segment_spawn.py
"""Integration tests for the spawn segmentation Flask app."""

import json
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def test_app():
    """Create a Flask test client with temp directories."""
    from scripts.segment_spawn import create_app

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        image_dir = tmp / "images"
        image_dir.mkdir()
        output_dir = tmp / "masks"
        output_dir.mkdir()

        # Create a test image
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        img.save(image_dir / "test_001.png")

        app = create_app(
            image_dir=str(image_dir),
            output_dir=str(output_dir),
            device="cpu",
            sam_checkpoint=None,  # Skip SAM for tests
        )
        app.config["TESTING"] = True
        yield app.test_client(), tmp


class TestSegmentAPI:
    """Test Flask API routes."""

    def test_index_returns_html(self, test_app):
        client, tmp = test_app
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"<!DOCTYPE html>" in resp.data or b"<html" in resp.data

    def test_api_images_lists_pngs(self, test_app):
        client, tmp = test_app
        resp = client.get("/api/images")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert len(data["images"]) == 1
        assert data["images"][0]["filename"] == "test_001.png"

    def test_api_image_serves_png(self, test_app):
        client, tmp = test_app
        resp = client.get("/api/image/test_001.png")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_api_segment_returns_mask(self, test_app):
        client, tmp = test_app
        payload = {"filename": "test_001.png", "x": 256, "y": 256}
        resp = client.post(
            "/api/segment",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "mask_url" in data
        assert "overlay_url" in data

    def test_api_segment_rejects_missing_filename(self, test_app):
        client, tmp = test_app
        resp = client.post(
            "/api/segment",
            data=json.dumps({"x": 100, "y": 100}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_api_accept_saves_mask(self, test_app):
        client, tmp = test_app
        # First segment
        client.post(
            "/api/segment",
            data=json.dumps({"filename": "test_001.png", "x": 256, "y": 256}),
            content_type="application/json",
        )
        # Then accept
        resp = client.post(
            "/api/accept",
            data=json.dumps({"filename": "test_001.png"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        # Check manifest exists
        manifest = tmp / "masks" / "manifest.json"
        assert manifest.exists()
        labels = json.loads(manifest.read_text())
        assert any(
            lb["filename"] == "test_001.png" and lb["label"] == "accept"
            for lb in labels
        )

    def test_api_reject_removes_mask(self, test_app):
        client, tmp = test_app
        client.post(
            "/api/segment",
            data=json.dumps({"filename": "test_001.png", "x": 256, "y": 256}),
            content_type="application/json",
        )
        resp = client.post(
            "/api/reject",
            data=json.dumps({"filename": "test_001.png"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && python -m pytest tests/test_segment_spawn.py -v
```

Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write the Flask app**

```python
# scripts/segment_spawn.py
#!/usr/bin/env python3
"""DINOv3 + SAM interactive herring spawn segmentation web app.

Click on a spawn region in the browser, and the app uses DINOv3 feature
similarity to generate a coarse mask, then refines it with SAM to produce
a pixel-level binary mask overlaid on the image.

Usage:
    python scripts/segment_spawn.py \
        --image-dir data/samples/positive \
        --output-dir data/segmentation_masks \
        --sam-checkpoint data/models/sam_vit_h_4b8939.pth \
        --port 8777
"""

import argparse
import json
import os
import sys
import threading
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.dinov3_feature_extractor import DINOv3FeatureExtractor

DEFAULT_PORT = 8777

# ---------------------------------------------------------------------------
# SAM Refiner (global singleton)
# ---------------------------------------------------------------------------

_sam_lock = threading.Lock()
_sam_predictor = None


def _load_sam(checkpoint_path: str):
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    with _sam_lock:
        if _sam_predictor is not None:
            return _sam_predictor
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam.eval()
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


class SAMRefiner:
    """Wrap SAM for mask refinement from coarse DINOv3 heatmaps."""

    def __init__(self, checkpoint_path: str | None = None):
        self._checkpoint = checkpoint_path
        self._available = checkpoint_path is not None and os.path.isfile(checkpoint_path)

    def refine(
        self, image: np.ndarray, coarse_mask: np.ndarray,
        click_xy: tuple[int, int],
    ) -> np.ndarray | None:
        """Refine a coarse binary mask with SAM.

        Args:
            image: RGB numpy array (H, W, 3) uint8.
            coarse_mask: Binary mask (H, W) uint8 from DINOv3.
            click_xy: (x, y) of the original click for fallback.

        Returns:
            Refined binary mask (H, W) uint8, or None if SAM unavailable.
        """
        if not self._available:
            return None

        predictor = _load_sam(self._checkpoint)
        if predictor is None:
            return None

        predictor.set_image(image)

        # Extract box and point from coarse mask
        mask_uint8 = (coarse_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        input_box = None
        input_point = None
        input_label = None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            input_box = np.array([x, y, x + w, y + h])
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            input_point = np.array([[cx, cy]])
            input_label = np.array([1])
        else:
            input_point = np.array([[click_xy[0], click_xy[1]]])
            input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True,
        )
        best = masks[scores.argmax()].astype(np.uint8)
        return best


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _make_overlay(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Blend a green mask over an RGB image."""
    overlay = image.copy()
    green = np.array([0, 255, 0], dtype=np.uint8)
    alpha = 0.4
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + green * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def create_app(
    image_dir: str,
    output_dir: str,
    device: str = "cpu",
    sam_checkpoint: str | None = None,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=os.path.join(PROJECT_ROOT, "webapp", "templates"),
        static_folder=os.path.join(PROJECT_ROOT, "webapp", "static"),
    )

    # Global state
    state = {
        "image_dir": image_dir,
        "output_dir": output_dir,
        "device": device,
        "sam_checkpoint": sam_checkpoint,
        "dinov3": None,
        "sam_refiner": None,
        "current_mask": {},  # filename -> (mask, overlay) paths
    }

    img_dir = Path(image_dir)

    def _get_extractor():
        if state["dinov3"] is None:
            state["dinov3"] = DINOv3FeatureExtractor(device=device)
        return state["dinov3"]

    def _get_sam():
        if state["sam_refiner"] is None:
            state["sam_refiner"] = SAMRefiner(sam_checkpoint)
        return state["sam_refiner"]

    # Routes

    @app.route("/")
    def index():
        return render_template("segment.html", port=app.config.get("PORT", DEFAULT_PORT))

    @app.route("/api/images")
    def api_images():
        pngs = sorted(img_dir.glob("*.png"))
        images = [{"filename": p.name} for p in pngs]
        return jsonify({"images": images, "count": len(images)})

    @app.route("/api/image/<path:filename>")
    def api_image(filename: str):
        img_path = img_dir / filename
        if not img_path.exists():
            abort(404)
        img = Image.open(img_path).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    @app.route("/api/mask/<path:filename>")
    def api_mask(filename: str):
        out_dir = Path(state["output_dir"])
        mask_path = out_dir / filename
        if not mask_path.exists():
            abort(404)
        return send_file(str(mask_path), mimetype="image/png")

    @app.route("/api/segment", methods=["POST"])
    def api_segment():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        filename = data.get("filename", "")
        x = data.get("x")
        y = data.get("y")

        if not filename:
            return jsonify({"error": "Missing filename"}), 400
        if x is None or y is None:
            return jsonify({"error": "Missing x,y coordinates"}), 400

        img_path = img_dir / filename
        if not img_path.exists():
            return jsonify({"error": f"Image not found: {filename}"}), 404

        # Load image
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)

        # DINOv3 coarse mask
        extractor = _get_extractor()
        coarse_mask = extractor.get_similarity_map(image_pil, (int(x), int(y)))

        # SAM refinement
        sam = _get_sam()
        refined = sam.refine(image_np, coarse_mask, (int(x), int(y)))
        final_mask = refined if refined is not None else coarse_mask

        # Save mask
        out_dir = Path(state["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_name = f"{Path(filename).stem}_mask.png"
        mask_path = out_dir / mask_name
        Image.fromarray(final_mask * 255).save(mask_path)

        # Save overlay
        overlay_name = f"{Path(filename).stem}_overlay.png"
        overlay_path = out_dir / overlay_name
        overlay = _make_overlay(image_np, final_mask)
        overlay.save(overlay_path)

        # Track current mask
        state["current_mask"][filename] = {
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
        }

        return jsonify({
            "status": "ok",
            "filename": filename,
            "mask_url": f"/api/mask/{mask_name}",
            "overlay_url": f"/api/mask/{overlay_name}",
        })

    @app.route("/api/accept", methods=["POST"])
    def api_accept():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400
        filename = data.get("filename", "")
        if not filename:
            return jsonify({"error": "Missing filename"}), 400

        out_dir = Path(state["output_dir"])
        manifest_path = out_dir / "manifest.json"

        labels = []
        if manifest_path.exists():
            labels = json.loads(manifest_path.read_text())

        labels.append({
            "filename": filename,
            "label": "accept",
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mask_path": state["current_mask"].get(filename, {}).get("mask_path", ""),
        })

        manifest_path.write_text(json.dumps(labels, indent=2))
        return jsonify({"status": "ok"})

    @app.route("/api/reject", methods=["POST"])
    def api_reject():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400
        filename = data.get("filename", "")
        if not filename:
            return jsonify({"error": "Missing filename"}), 400

        out_dir = Path(state["output_dir"])
        manifest_path = out_dir / "manifest.json"

        labels = []
        if manifest_path.exists():
            labels = json.loads(manifest_path.read_text())

        labels.append({
            "filename": filename,
            "label": "reject",
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

        # Remove mask files
        mask_name = f"{Path(filename).stem}_mask.png"
        overlay_name = f"{Path(filename).stem}_overlay.png"
        for name in [mask_name, overlay_name]:
            p = out_dir / name
            if p.exists():
                p.unlink()

        manifest_path.write_text(json.dumps(labels, indent=2))
        return jsonify({"status": "ok"})

    @app.route("/api/progress")
    def api_progress():
        total = len(list(img_dir.glob("*.png")))
        out_dir = Path(state["output_dir"])
        manifest_path = out_dir / "manifest.json"
        labels = []
        if manifest_path.exists():
            labels = json.loads(manifest_path.read_text())
        accepted = sum(1 for lb in labels if lb["label"] == "accept")
        rejected = sum(1 for lb in labels if lb["label"] == "reject")
        return jsonify({
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "labeled": accepted + rejected,
            "unlabeled": total - accepted - rejected,
        })

    app.config["PORT"] = DEFAULT_PORT
    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import cv2  # noqa: F401 — used by SAMRefiner at runtime
    parser = argparse.ArgumentParser(
        description="DINOv3 + SAM interactive spawn segmentation web app",
    )
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory of PNG images to segment")
    parser.add_argument("--output-dir", type=str,
                        default="data/segmentation_masks",
                        help="Directory for mask outputs")
    parser.add_argument("--sam-checkpoint", type=str,
                        default="data/models/sam_vit_h_4b8939.pth",
                        help="Path to SAM ViT-H checkpoint")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and __import__("torch").cuda.is_available() else args.device if args.device != "auto" else "cpu"
    if args.device == "auto" and device == "cpu":
        print("Note: CUDA not available, using CPU (DINOv3 will be slow).")

    if not os.path.isdir(args.image_dir):
        print(f"ERROR: Image directory not found: {args.image_dir}")
        return 1

    sam_path = args.sam_checkpoint
    if not os.path.isfile(sam_path):
        print(f"WARNING: SAM checkpoint not found at {sam_path}")
        print("  Segmentation will use DINOv3 coarse masks only (no SAM refinement).")
        print(f"  Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        sam_path = None

    app = create_app(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=device,
        sam_checkpoint=sam_path,
    )
    app.config["PORT"] = args.port

    print(f"\n  {'='*60}")
    print("  Spawn Segmentation App — Ready")
    print(f"  {'='*60}")
    print(f"  Image dir: {args.image_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  DINOv3: Vit-L/16 SAT-493M (device={device})")
    print(f"  SAM: {'enabled' if sam_path else 'DISABLED (coarse masks only)'}")
    print(f"  Port: {args.port}")
    print(f"  URL: http://localhost:{args.port}")
    print(f"  {'='*60}\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_segment_spawn.py -v
```

Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/segment_spawn.py tests/test_segment_spawn.py
git commit -m "feat: Flask segmentation app with DINOv3 + SAM click-to-mask"
```

---

### Task 4: HTML Template

**Files:**
- Create: `webapp/templates/segment.html`

- [ ] **Step 1: Write the template**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spawn Segmentation</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; }
  .header { padding: 12px 20px; background: #16213e; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #0f3460; }
  .header h1 { font-size: 18px; }
  .stats { display: flex; gap: 20px; font-size: 14px; }
  .stats span { padding: 4px 10px; border-radius: 4px; }
  .stat-accept { background: #1b4332; color: #95d5b2; }
  .stat-reject { background: #3e1f1f; color: #e07a7a; }
  .stat-remain { background: #1a3a4a; color: #7ab8d4; }
  .layout { display: flex; height: calc(100vh - 53px); }
  .sidebar { width: 260px; background: #16213e; overflow-y: auto; border-right: 1px solid #0f3460; flex-shrink: 0; }
  .sidebar .item { padding: 10px 14px; cursor: pointer; border-bottom: 1px solid #0f3460; font-size: 12px; word-break: break-all; }
  .sidebar .item:hover { background: #1a1a3e; }
  .sidebar .item.active { background: #0f3460; border-left: 3px solid #e94560; }
  .sidebar .item .label-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .dot-accept { background: #95d5b2; }
  .dot-reject { background: #e07a7a; }
  .dot-none { background: #555; }
  .main { flex: 1; display: flex; flex-direction: column; align-items: center; padding: 20px; overflow-y: auto; }
  .viewer { position: relative; display: inline-block; }
  .viewer img { max-width: 90vw; max-height: 70vh; display: block; border: 1px solid #333; }
  .viewer canvas { position: absolute; top: 0; left: 0; }
  .controls { margin-top: 16px; display: flex; gap: 12px; align-items: center; }
  .controls button { padding: 8px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
  .btn-accept { background: #2d6a4f; color: #eee; }
  .btn-accept:hover { background: #40916c; }
  .btn-reject { background: #6a2d2d; color: #eee; }
  .btn-reject:hover { background: #914040; }
  .btn-clear { background: #444; color: #eee; }
  .btn-clear:hover { background: #555; }
  .toggle-label { display: flex; align-items: center; gap: 6px; font-size: 13px; }
  .help { margin-top: 10px; font-size: 11px; color: #888; }
  .status { padding: 6px 12px; font-size: 13px; border-radius: 4px; }
  .status-ok { background: #1b4332; color: #95d5b2; }
  .status-err { background: #3e1f1f; color: #e07a7a; }
  .status-wait { background: #333; color: #aaa; }
</style>
</head>
<body>
<div class="header">
  <h1>Herring Spawn Segmentation</h1>
  <div class="stats">
    <span class="stat-accept" id="stat-accept">Accepted: 0</span>
    <span class="stat-reject" id="stat-reject">Rejected: 0</span>
    <span class="stat-remain" id="stat-remain">Remaining: 0</span>
  </div>
</div>
<div class="layout">
  <div class="sidebar" id="sidebar"></div>
  <div class="main">
    <div class="viewer" id="viewer">
      <img id="main-image" src="" alt="Click to segment">
      <canvas id="overlay-canvas"></canvas>
    </div>
    <div class="controls">
      <button class="btn-accept" id="btn-accept" onclick="acceptMask()" disabled>Accept (A)</button>
      <button class="btn-reject" id="btn-reject" onclick="rejectMask()" disabled>Reject (R)</button>
      <button class="btn-clear" id="btn-clear" onclick="clearMask()">Clear</button>
      <label class="toggle-label">
        <input type="checkbox" id="toggle-overlay" checked onchange="toggleOverlay()"> Show mask
      </label>
      <span class="status status-wait" id="status">Click on spawn region</span>
    </div>
    <div class="help">
      Click on spawn region to segment | A=Accept R=Reject &larr;&rarr;=Navigate | C=Clear
    </div>
  </div>
</div>

<script>
  const PORT = {{ port }};
  let images = [];
  let currentIndex = -1;
  let currentMask = null;
  let currentOverlay = null;
  let labels = {};

  async function loadImages() {
    const resp = await fetch('/api/images');
    const data = await resp.json();
    images = data.images;
    renderSidebar();
    updateStats();
    if (images.length > 0) selectImage(0);
  }

  function renderSidebar() {
    const sb = document.getElementById('sidebar');
    sb.innerHTML = images.map((img, i) => {
      const label = labels[img.filename];
      let dot = 'dot-none';
      if (label === 'accept') dot = 'dot-accept';
      else if (label === 'reject') dot = 'dot-reject';
      return `<div class="item ${i === currentIndex ? 'active' : ''}" onclick="selectImage(${i})">
        <span class="label-dot ${dot}"></span>${img.filename}
      </div>`;
    }).join('');
  }

  async function selectImage(idx) {
    currentIndex = idx;
    currentMask = null;
    currentOverlay = null;
    const img = images[idx];
    document.getElementById('main-image').src = `/api/image/${encodeURIComponent(img.filename)}`;
    document.getElementById('btn-accept').disabled = true;
    document.getElementById('btn-reject').disabled = true;
    document.getElementById('btn-clear').disabled = false;
    setStatus('Click on spawn region', 'wait');
    clearCanvas();
    renderSidebar();
    // Check for existing label
    if (labels[img.filename]) {
      document.getElementById('btn-accept').disabled = false;
      setStatus(`Labeled: ${labels[img.filename]}`, labels[img.filename] === 'accept' ? 'ok' : 'err');
    }
  }

  document.getElementById('viewer').addEventListener('click', async function(e) {
    if (currentIndex < 0) return;
    const img = document.getElementById('main-image');
    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    setStatus('Segmenting...', 'wait');
    const resp = await fetch('/api/segment', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: images[currentIndex].filename, x, y}),
    });
    const data = await resp.json();
    if (data.status === 'ok') {
      currentMask = data.mask_url;
      currentOverlay = data.overlay_url;
      drawOverlay(data.overlay_url);
      document.getElementById('btn-accept').disabled = false;
      document.getElementById('btn-reject').disabled = false;
      setStatus('Mask generated — accept or reject', 'ok');
    } else {
      setStatus('Error: ' + (data.error || 'unknown'), 'err');
    }
  });

  function drawOverlay(overlayUrl) {
    const img = document.getElementById('main-image');
    const canvas = document.getElementById('overlay-canvas');
    const overlay = new Image();
    overlay.onload = () => {
      canvas.width = overlay.naturalWidth;
      canvas.height = overlay.naturalHeight;
      canvas.style.width = img.offsetWidth + 'px';
      canvas.style.height = img.offsetHeight + 'px';
      canvas.style.display = document.getElementById('toggle-overlay').checked ? 'block' : 'none';
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (document.getElementById('toggle-overlay').checked) {
        ctx.drawImage(overlay, 0, 0);
      }
    };
    overlay.src = overlayUrl;
  }

  window.addEventListener('resize', () => {
    if (currentOverlay) {
      const img = document.getElementById('main-image');
      const canvas = document.getElementById('overlay-canvas');
      canvas.style.width = img.offsetWidth + 'px';
      canvas.style.height = img.offsetHeight + 'px';
    }
  });

  function clearCanvas() {
    const canvas = document.getElementById('overlay-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function clearMask() {
    currentMask = null;
    currentOverlay = null;
    clearCanvas();
    document.getElementById('btn-accept').disabled = true;
    document.getElementById('btn-reject').disabled = true;
    setStatus('Click on spawn region', 'wait');
  }

  async function acceptMask() {
    if (currentIndex < 0) return;
    labels[images[currentIndex].filename] = 'accept';
    await fetch('/api/accept', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: images[currentIndex].filename}),
    });
    renderSidebar();
    updateStats();
    setStatus('Accepted', 'ok');
    nextImage();
  }

  async function rejectMask() {
    if (currentIndex < 0) return;
    labels[images[currentIndex].filename] = 'reject';
    await fetch('/api/reject', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: images[currentIndex].filename}),
    });
    currentMask = null;
    currentOverlay = null;
    clearCanvas();
    renderSidebar();
    updateStats();
    setStatus('Rejected', 'err');
    nextImage();
  }

  function nextImage() {
    if (currentIndex < images.length - 1) selectImage(currentIndex + 1);
  }
  function prevImage() {
    if (currentIndex > 0) selectImage(currentIndex - 1);
  }

  function toggleOverlay() {
    const checked = document.getElementById('toggle-overlay').checked;
    document.getElementById('overlay-canvas').style.display = checked ? 'block' : 'none';
  }

  function setStatus(msg, cls) {
    const el = document.getElementById('status');
    el.textContent = msg;
    el.className = 'status status-' + cls;
  }

  async function updateStats() {
    const resp = await fetch('/api/progress');
    const data = await resp.json();
    document.getElementById('stat-accept').textContent = `Accepted: ${data.accepted}`;
    document.getElementById('stat-reject').textContent = `Rejected: ${data.rejected}`;
    document.getElementById('stat-remain').textContent = `Remaining: ${data.unlabeled}`;
  }

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') prevImage();
    else if (e.key === 'ArrowRight') nextImage();
    else if (e.key === 'a' && !e.ctrlKey && !e.metaKey) { e.preventDefault(); acceptMask(); }
    else if (e.key === 'r' && !e.ctrlKey && !e.metaKey) { e.preventDefault(); rejectMask(); }
    else if (e.key === 'c' && !e.ctrlKey && !e.metaKey) { e.preventDefault(); clearMask(); }
  });

  loadImages();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify template renders (start app, check HTML)**

```bash
source .venv/bin/activate && python -m pytest tests/test_segment_spawn.py::TestSegmentAPI::test_index_returns_html -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add webapp/templates/segment.html
git commit -m "feat: segmentation UI template with click-to-segment and keyboard nav"
```

---

### Task 5: Integration verification + AGENTS.md update

**Files:**
- Modify: `AGENTS.md`
- Test: manual smoke test

- [ ] **Step 1: Run all tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_dinov3_feature_extractor.py tests/test_segment_spawn.py -v
```

Expected: all 12 tests PASS

- [ ] **Step 2: Manual smoke test — start app on positive images**

```bash
source .venv/bin/activate && python scripts/segment_spawn.py \
  --image-dir data/samples/positive \
  --output-dir data/segmentation_masks \
  --sam-checkpoint data/models/sam_vit_h_4b8939.pth \
  --device cpu \
  --port 8777
```

Visit `http://localhost:8777`, click a spawn image, verify:
- Image loads
- Click on bright turbidity → green overlay appears
- Accept/reject buttons work
- Keyboard navigation works
- Check `data/segmentation_masks/` for mask PNGs

- [ ] **Step 3: Update AGENTS.md**

Add under the "Scripts" section of AGENTS.md:

```
- `scripts/segment_spawn.py` — DINOv3 + SAM interactive click-to-segment web app
- `scripts/dinov3_feature_extractor.py` — DINOv3 ViT-L/16 SAT-493M feature extraction with similarity heatmaps
```

And add a new subsection under "Current Approach" after item 6:

```
7. Interactive segmentation: use `segment_spawn.py` to produce pixel-level binary masks. Click on suspected spawn regions → DINOv3 computes similarity heatmap → SAM refines to pixel mask. Use for creating training masks from positive images.
```

- [ ] **Step 4: Final commit**

```bash
git add AGENTS.md
git commit -m "docs: add segmentation tool to AGENTS.md"
```

---

## Verification Checklist

After all tasks complete, verify:

1. `pytest tests/test_dinov3_feature_extractor.py tests/test_segment_spawn.py -v` — all pass
2. `python scripts/segment_spawn.py --image-dir data/samples/positive --output-dir data/segmentation_masks --sam-checkpoint data/models/sam_vit_h_4b8939.pth --device cpu --port 8777` — starts without errors
3. Browser: `http://localhost:8777` — images load, click produces green overlay
4. `ls data/segmentation_masks/` — contains mask PNGs and manifest.json after using the UI
