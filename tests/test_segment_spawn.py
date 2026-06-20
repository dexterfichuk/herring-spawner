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

        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        img.save(image_dir / "test_001.png")

        app = create_app(
            image_dir=str(image_dir),
            output_dir=str(output_dir),
            device="cpu",
            sam_checkpoint=None,
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
        client.post(
            "/api/segment",
            data=json.dumps({"filename": "test_001.png", "x": 256, "y": 256}),
            content_type="application/json",
        )
        resp = client.post(
            "/api/accept",
            data=json.dumps({"filename": "test_001.png"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
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
