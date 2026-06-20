import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rs_foundation_probe import evaluate_linear_probe


class DummyBackbone:
    name = "dummy-rs"
    model_id = None

    def encode_image(self, image: Image.Image):
        rgb = image.convert("RGB")
        array = __import__("numpy").asarray(rgb, dtype=float)
        r = float(array[:, :, 0].mean())
        g = float(array[:, :, 1].mean())
        return [r, g]


def _write_solid_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (16, 16), color=color).save(path)


def test_evaluate_linear_probe_supports_dummy_backbone(tmp_path):
    pos_a = tmp_path / "spawn_a.png"
    pos_b = tmp_path / "spawn_b.png"
    neg_a = tmp_path / "noise_a.png"
    neg_b = tmp_path / "noise_b.png"
    _write_solid_image(pos_a, (255, 0, 0))
    _write_solid_image(pos_b, (240, 10, 0))
    _write_solid_image(neg_a, (0, 255, 0))
    _write_solid_image(neg_b, (0, 240, 10))

    rows = [
        {"image_path": str(pos_a), "label": 1},
        {"image_path": str(pos_b), "label": 1},
        {"image_path": str(neg_a), "label": 0},
        {"image_path": str(neg_b), "label": 0},
    ]

    summary = evaluate_linear_probe(rows, backbone=DummyBackbone(), classifier="logreg", cv="auto")

    assert summary["row_count"] == 4
    assert summary["metrics"]["accuracy"] == pytest.approx(1.0)
    assert summary["backend"] == "dummy-rs"
    assert len(summary["rows"]) == 4
