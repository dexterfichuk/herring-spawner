#!/usr/bin/env python3
"""DINOv3 + SAM interactive herring spawn segmentation web app.

Click on a spawn region in the browser, and the app uses DINOv3 feature
similarity to generate a coarse mask, then refines it with SAM to produce
a pixel-level binary mask overlaid on the image.

Usage:
    python scripts/segment_spawn.py \\
        --image-dir data/samples/positive \\
        --output-dir data/segmentation_masks \\
        --sam-checkpoint data/models/sam_vit_h_4b8939.pth \\
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


def _load_sam(checkpoint_path: str, device: str = "cpu"):
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    with _sam_lock:
        if _sam_predictor is not None:
            return _sam_predictor
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam.eval()
        sam = sam.to(device)
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


class SAMRefiner:
    """Wrap SAM for mask refinement from coarse DINOv3 heatmaps,
    plus direct point-prompt accumulation for interactive refinement."""

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu"):
        self._checkpoint = checkpoint_path
        self._device = device
        self._available = checkpoint_path is not None and os.path.isfile(checkpoint_path)

    def refine(
        self, image: np.ndarray, coarse_mask: np.ndarray,
        click_xy: tuple[int, int],
    ) -> np.ndarray | None:
        """Refine a coarse binary mask with SAM (initial DINOv3-guided pass)."""
        if not self._available:
            return None
        predictor = _load_sam(self._checkpoint, device=self._device)
        if predictor is None:
            return None
        predictor.set_image(image)

        import cv2
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
        return masks[scores.argmax()].astype(np.uint8)

    def predict_from_points(
        self, image: np.ndarray, points: list[list[int]],
        labels: list[int],
    ) -> np.ndarray | None:
        """Run SAM directly with accumulated foreground/background points.

        Args:
            image: RGB numpy array (H, W, 3) uint8.
            points: List of [x, y] coordinates.
            labels: List of 1 (foreground) or 0 (background).

        Returns:
            Binary mask (H, W) uint8, or None if SAM unavailable.
        """
        if not self._available or not points:
            return None
        predictor = _load_sam(self._checkpoint, device=self._device)
        if predictor is None:
            return None
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=True,
        )
        return masks[scores.argmax()].astype(np.uint8)


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

    state = {
        "image_dir": image_dir,
        "output_dir": output_dir,
        "device": device,
        "sam_checkpoint": sam_checkpoint,
        "dinov3": None,
        "sam_refiner": None,
        "current_mask": {},
        "sam_prompts": {},  # filename -> {"points": [[x,y],...], "labels": [1/0,...], "image_np": ...}
    }

    img_dir = Path(image_dir)

    def _get_extractor():
        if state["dinov3"] is None:
            state["dinov3"] = DINOv3FeatureExtractor(device=device)
        return state["dinov3"]

    def _get_sam():
        if state["sam_refiner"] is None:
            state["sam_refiner"] = SAMRefiner(sam_checkpoint, device=device)
        return state["sam_refiner"]

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
        label = data.get("label", 1)  # 1=foreground, 0=background

        if not filename:
            return jsonify({"error": "Missing filename"}), 400
        if x is None or y is None:
            return jsonify({"error": "Missing x,y coordinates"}), 400

        img_path = img_dir / filename
        if not img_path.exists():
            return jsonify({"error": f"Image not found: {filename}"}), 404

        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)
        out_dir = Path(state["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        sam = _get_sam()

        # Check if this is a first click (no existing prompts) or a refinement
        prompts = state["sam_prompts"].get(filename)
        is_first = prompts is None

        if is_first:
            # --- First click: DINOv3 heatmap → SAM refinment ---
            extractor = _get_extractor()
            coarse_mask = extractor.get_similarity_map(image_pil, (int(x), int(y)))
            refined = sam.refine(image_np, coarse_mask, (int(x), int(y)))
            final_mask = refined if refined is not None else coarse_mask

            # Initialize SAM prompts with the click point
            state["sam_prompts"][filename] = {
                "points": [[int(x), int(y)]],
                "labels": [1],  # first click is always foreground
                "image_np": image_np,
            }
            region_count = 1
        else:
            # --- Refinement click: add point to SAM, skip DINOv3 ---
            prompts["points"].append([int(x), int(y)])
            prompts["labels"].append(label)
            prompts["image_np"] = image_np  # may differ if image changed
            final_mask = sam.predict_from_points(
                image_np, prompts["points"], prompts["labels"])
            if final_mask is None:
                return jsonify({"error": "SAM unavailable"}), 500
            region_count = len(prompts["points"])

        # Save mask and overlay
        mask_name = f"{Path(filename).stem}_mask.png"
        mask_path = out_dir / mask_name
        Image.fromarray(final_mask * 255).save(mask_path)

        overlay_name = f"{Path(filename).stem}_overlay.png"
        overlay_path = out_dir / overlay_name
        overlay = _make_overlay(image_np, final_mask)
        overlay.save(overlay_path)

        state["current_mask"][filename] = {
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
        }

        return jsonify({
            "status": "ok",
            "filename": filename,
            "mask_url": f"/api/mask/{mask_name}",
            "overlay_url": f"/api/mask/{overlay_name}",
            "region_count": region_count,
            "is_refinement": not is_first,
        })

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400
        filename = data.get("filename", "")
        if not filename:
            return jsonify({"error": "Missing filename"}), 400
        state["sam_prompts"].pop(filename, None)
        state["current_mask"].pop(filename, None)
        return jsonify({"status": "ok"})

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
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Resolve device
    device = args.device
    if args.device == "auto":
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        if device == "cpu":
            print("Note: CUDA/MPS not available, using CPU (DINOv3 will be slow).")

    # Resolve paths relative to project root
    def resolve_path(p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str(Path(PROJECT_ROOT) / path)

    image_dir = resolve_path(args.image_dir)
    output_dir = resolve_path(args.output_dir)

    if not os.path.isdir(image_dir):
        print(f"ERROR: Image directory not found: {image_dir}")
        return 1

    sam_path = resolve_path(args.sam_checkpoint) if args.sam_checkpoint else None
    if not os.path.isfile(sam_path):
        print(f"WARNING: SAM checkpoint not found at {sam_path}")
        print("  Segmentation will use DINOv3 coarse masks only (no SAM refinement).")
        print("  Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        sam_path = None

    app = create_app(
        image_dir=image_dir,
        output_dir=output_dir,
        device=device,
        sam_checkpoint=sam_path,
    )
    app.config["PORT"] = args.port

    print(f"\n  {'='*60}")
    print("  Spawn Segmentation App — Ready")
    print(f"  {'='*60}")
    print(f"  Image dir: {image_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  DINOv3: Vit-L/16 SAT-493M (device={device})")
    print(f"  SAM: {'enabled' if sam_path else 'DISABLED (coarse masks only)'}")
    print(f"  Port: {args.port}")
    print(f"  URL: http://localhost:{args.port}")
    print(f"  {'='*60}\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
