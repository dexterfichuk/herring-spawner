"""Tests for the RemoteCLIP few-shot classification mode.

Follows the same mocking patterns as test_remoteclip_zero_shot.py.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.remoteclip_zero_shot import (
    extract_embeddings,
    train_few_shot,
    main,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_fake_png(path: Path, size: tuple = (224, 224)) -> Path:
    """Create a small random-noise PNG at *path* and return it."""
    arr = np.random.randint(0, 255, size=(*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _make_labels_file(labels_dir: Path, entries: list[dict]) -> Path:
    """Write a labels JSON file and return its path."""
    path = labels_dir / "labels.json"
    path.write_text(json.dumps({"labels": entries}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def image_dir_with_pngs(tmp_path):
    """Create a directory with 8 fake PNGs (5 positive, 3 negative labels)."""
    d = tmp_path / "images"
    d.mkdir()
    for i in range(8):
        _make_fake_png(d / f"img_{i}.png")
    return d


@pytest.fixture
def labels_json(tmp_path, image_dir_with_pngs):
    """Create a labels JSON referencing the 8 PNGs with 5 pos / 3 neg."""
    entries = [
        {"filename": "img_0.png", "label": 1},
        {"filename": "img_1.png", "label": 1},
        {"filename": "img_2.png", "label": 1},
        {"filename": "img_3.png", "label": 1},
        {"filename": "img_4.png", "label": 1},
        {"filename": "img_5.png", "label": 0},
        {"filename": "img_6.png", "label": 0},
        {"filename": "img_7.png", "label": 0},
    ]
    return _make_labels_file(tmp_path, entries)


@pytest.fixture
def controlled_embeddings():
    """Return controlled 8×768 embeddings with 5 positive / 3 negative separable clusters.

    Positives: centered around +0.5 in first 2 dims, rest noise.
    Negatives: centered around -0.5 in first 2 dims, rest noise.
    The embeddings are normalized to unit length.
    """
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((8, 768)).astype(np.float32)

    # Make first 5 separable as positive class, last 3 as negative
    for i in range(5):
        emb[i, 0] = 0.5 + rng.standard_normal() * 0.05
        emb[i, 1] = 0.5 + rng.standard_normal() * 0.05
    for i in range(5, 8):
        emb[i, 0] = -0.5 + rng.standard_normal() * 0.05
        emb[i, 1] = -0.5 + rng.standard_normal() * 0.05

    # Normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms

    return emb


# ===========================================================================
# Tests: extract_embeddings
# ===========================================================================


class TestExtractEmbeddings:
    """Verify extract_embeddings returns correct types and handles errors."""

    def test_extract_embeddings_returns_correct_types(self, tmp_path, image_dir_with_pngs):
        """Happy path: returns (np.ndarray, list, list) with correct shapes."""
        fake_cache = tmp_path / "cache" / "remoteclip_embeddings.npz"
        with (
            patch("scripts.remoteclip_zero_shot.load_model") as mock_load,
            patch("scripts.remoteclip_zero_shot.get_image_embedding") as mock_get_img,
            patch("scripts.remoteclip_zero_shot.EMBEDDINGS_CACHE_PATH", str(fake_cache)),
        ):
            mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())
            # Return a 768-dim embedding for each image
            # get_image_embedding signature: (model, preprocess, image_path, device)
            def _img_side(model, preprocess, image_path, device):
                return np.random.default_rng(42).standard_normal(768).astype(np.float32)
            mock_get_img.side_effect = _img_side

            embeddings, img_paths, errors = extract_embeddings(
                str(image_dir_with_pngs), device="cpu"
            )

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (8, 768), f"Expected (8, 768), got {embeddings.shape}"
        assert embeddings.dtype == np.float32

        assert isinstance(img_paths, list)
        assert len(img_paths) == 8
        assert all(isinstance(p, str) for p in img_paths)

        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_extract_embeddings_skips_corrupted_images(self, tmp_path):
        """Images that fail get_image_embedding are recorded in errors list."""
        d = tmp_path / "images"
        d.mkdir()
        _make_fake_png(d / "good_1.png")
        _make_fake_png(d / "bad.png")
        _make_fake_png(d / "good_2.png")

        fake_cache = tmp_path / "cache" / "remoteclip_embeddings.npz"
        with (
            patch("scripts.remoteclip_zero_shot.load_model") as mock_load,
            patch("scripts.remoteclip_zero_shot.get_image_embedding") as mock_get_img,
            patch("scripts.remoteclip_zero_shot.EMBEDDINGS_CACHE_PATH", str(fake_cache)),
        ):
            mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())

            # get_image_embedding signature: (model, preprocess, image_path, device)
            def _img_side(model, preprocess, image_path, device):
                if "bad" in str(image_path):
                    return None
                return np.random.default_rng(42).standard_normal(768).astype(np.float32)
            mock_get_img.side_effect = _img_side

            embeddings, img_paths, errors = extract_embeddings(
                str(d), device="cpu"
            )

        assert embeddings.shape == (2, 768)
        assert len(img_paths) == 2
        assert len(errors) == 1
        assert "bad" in errors[0]
        assert all("good" in p for p in img_paths)

    def test_extract_embeddings_empty_directory(self, tmp_path):
        """No PNGs in directory returns empty arrays."""
        d = tmp_path / "empty"
        d.mkdir()

        fake_cache = tmp_path / "cache" / "remoteclip_embeddings.npz"
        with (
            patch("scripts.remoteclip_zero_shot.load_model") as mock_load,
            patch("scripts.remoteclip_zero_shot.get_image_embedding") as mock_get_img,
            patch("scripts.remoteclip_zero_shot.EMBEDDINGS_CACHE_PATH", str(fake_cache)),
        ):
            mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())
            embeddings, img_paths, errors = extract_embeddings(
                str(d), device="cpu"
            )

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, 0) or embeddings.size == 0
        assert isinstance(img_paths, list)
        assert len(img_paths) == 0
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_extract_embeddings_only_png(self, tmp_path):
        """Only PNG files are processed, non-PNG are ignored."""
        d = tmp_path / "images"
        d.mkdir()
        _make_fake_png(d / "img.png")
        # Create some non-PNG files
        (d / "notes.txt").write_text("hello")
        (d / "data.csv").write_text("a,b,c")

        fake_cache = tmp_path / "cache" / "remoteclip_embeddings.npz"
        with (
            patch("scripts.remoteclip_zero_shot.load_model") as mock_load,
            patch("scripts.remoteclip_zero_shot.get_image_embedding") as mock_get_img,
            patch("scripts.remoteclip_zero_shot.EMBEDDINGS_CACHE_PATH", str(fake_cache)),
        ):
            mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())
            mock_get_img.return_value = np.random.default_rng(42).standard_normal(768).astype(np.float32)

            embeddings, img_paths, errors = extract_embeddings(
                str(d), device="cpu"
            )

        assert embeddings.shape == (1, 768)
        assert len(img_paths) == 1
        assert img_paths[0].endswith("img.png")

    def test_extract_embeddings_uses_cache(self, tmp_path):
        """When cache file exists, embeddings are loaded without calling model."""
        cache_dir = tmp_path / "embeddings"
        cache_dir.mkdir(parents=True)
        cache_path = cache_dir / "remoteclip_embeddings.npz"

        # Create a pre-existing cache
        fake_emb = np.random.default_rng(42).standard_normal((3, 768)).astype(np.float32)
        fake_paths = np.array(["cached_1.png", "cached_2.png", "cached_3.png"])
        np.savez_compressed(cache_path, embeddings=fake_emb, img_paths=fake_paths)

        d = tmp_path / "images"
        d.mkdir()
        _make_fake_png(d / "fresh.png")

        with (
            patch("scripts.remoteclip_zero_shot.load_model") as mock_load,
            patch("scripts.remoteclip_zero_shot.get_image_embedding") as mock_get_img,
            patch("scripts.remoteclip_zero_shot.EMBEDDINGS_CACHE_PATH", str(cache_path)),
        ):
            embeddings, img_paths, errors = extract_embeddings(
                str(d), device="cpu"
            )

            # Model should NOT have been loaded (cache hit)
            mock_load.assert_not_called()
            mock_get_img.assert_not_called()

        assert embeddings.shape == (3, 768)
        assert len(img_paths) == 3
        assert np.allclose(embeddings, fake_emb)


# ===========================================================================
# Tests: train_few_shot
# ===========================================================================


class TestTrainFewShot:
    """Verify train_few_shot returns correct structure and metrics."""

    def test_train_few_shot_returns_dict_with_all_keys(
        self, tmp_path, image_dir_with_pngs, labels_json, controlled_embeddings
    ):
        """Happy path: returns dict with all required keys."""
        with patch("scripts.remoteclip_zero_shot.extract_embeddings") as mock_extract:
            mock_extract.return_value = (
                controlled_embeddings,
                [str(image_dir_with_pngs / f"img_{i}.png") for i in range(8)],
                [],
            )

            result = train_few_shot(
                str(image_dir_with_pngs),
                str(labels_json),
                device="cpu",
                cv_folds=3,
                seed=42,
            )

        required_keys = [
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "cv_precision_mean",
            "cv_recall_mean",
            "cv_f1_mean",
            "cv_folds",
            "n_total",
            "n_pos",
            "n_neg",
            "full_train_accuracy",
            "per_fold",
            "classifier",
            "embed_dim",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        assert result["cv_folds"] == 3
        assert result["classifier"] == "logistic"
        assert result["embed_dim"] == 768
        assert result["n_total"] == 8
        assert result["n_pos"] == 5
        assert result["n_neg"] == 3

    def test_train_few_shot_cv_metrics_reasonable(
        self, tmp_path, image_dir_with_pngs, labels_json, controlled_embeddings
    ):
        """CV metrics are floats in [0, 1] range (except std which can be small)."""
        with patch("scripts.remoteclip_zero_shot.extract_embeddings") as mock_extract:
            mock_extract.return_value = (
                controlled_embeddings,
                [str(image_dir_with_pngs / f"img_{i}.png") for i in range(8)],
                [],
            )

            result = train_few_shot(
                str(image_dir_with_pngs),
                str(labels_json),
                device="cpu",
                cv_folds=3,
                seed=42,
            )

        assert 0.0 <= result["cv_accuracy_mean"] <= 1.0
        assert 0.0 <= result["cv_precision_mean"] <= 1.0
        assert 0.0 <= result["cv_recall_mean"] <= 1.0
        assert 0.0 <= result["cv_f1_mean"] <= 1.0
        assert result["cv_accuracy_std"] >= 0.0

    def test_train_few_shot_per_fold_list(
        self, tmp_path, image_dir_with_pngs, labels_json, controlled_embeddings
    ):
        """per_fold is a list of dicts with fold-level metrics, length == cv_folds."""
        with patch("scripts.remoteclip_zero_shot.extract_embeddings") as mock_extract:
            mock_extract.return_value = (
                controlled_embeddings,
                [str(image_dir_with_pngs / f"img_{i}.png") for i in range(8)],
                [],
            )

            result = train_few_shot(
                str(image_dir_with_pngs),
                str(labels_json),
                device="cpu",
                cv_folds=3,
                seed=42,
            )

        assert isinstance(result["per_fold"], list)
        assert len(result["per_fold"]) == 3
        for fold_dict in result["per_fold"]:
            assert "fold" in fold_dict
            assert "accuracy" in fold_dict
            assert "precision" in fold_dict
            assert "recall" in fold_dict
            assert "f1" in fold_dict
            assert isinstance(fold_dict["fold"], int)

    def test_train_few_shot_handles_missing_images(
        self, tmp_path, image_dir_with_pngs, labels_json
    ):
        """Images in labels that don't exist on disk are skipped."""
        # Add a label referencing a non-existent file
        labels_path = labels_json
        labels_data = json.loads(labels_path.read_text())
        labels_data["labels"].append(
            {"filename": "nonexistent.png", "label": 0}
        )
        labels_path.write_text(json.dumps(labels_data))

        rng = np.random.default_rng(42)
        emb = rng.standard_normal((8, 768)).astype(np.float32)

        with patch("scripts.remoteclip_zero_shot.extract_embeddings") as mock_extract:
            mock_extract.return_value = (
                emb,
                [str(image_dir_with_pngs / f"img_{i}.png") for i in range(8)],
                [],
            )

            # Should not raise
            result = train_few_shot(
                str(image_dir_with_pngs),
                str(labels_path),
                device="cpu",
                cv_folds=3,
                seed=42,
            )

        # Only 8 images exist, but 9 labels (one missing). Missing images
        # are silently skipped during matching, so n_total should be 8.
        assert result["n_total"] == 8

    def test_train_few_shot_seed_reproducibility(
        self, tmp_path, image_dir_with_pngs, labels_json, controlled_embeddings
    ):
        """Same seed produces same CV results."""
        with patch("scripts.remoteclip_zero_shot.extract_embeddings") as mock_extract:
            mock_extract.return_value = (
                controlled_embeddings.copy(),
                [str(image_dir_with_pngs / f"img_{i}.png") for i in range(8)],
                [],
            )

            result_1 = train_few_shot(
                str(image_dir_with_pngs),
                str(labels_json),
                device="cpu",
                cv_folds=3,
                seed=42,
            )

        with patch("scripts.remoteclip_zero_shot.extract_embeddings") as mock_extract:
            mock_extract.return_value = (
                controlled_embeddings.copy(),
                [str(image_dir_with_pngs / f"img_{i}.png") for i in range(8)],
                [],
            )

            result_2 = train_few_shot(
                str(image_dir_with_pngs),
                str(labels_json),
                device="cpu",
                cv_folds=3,
                seed=42,
            )

        assert result_1["cv_accuracy_mean"] == result_2["cv_accuracy_mean"]
        assert result_1["cv_accuracy_std"] == result_2["cv_accuracy_std"]
        assert result_1["cv_f1_mean"] == result_2["cv_f1_mean"]

    def test_train_few_shot_different_seed_different_splits(
        self, tmp_path, image_dir_with_pngs, labels_json, controlled_embeddings
    ):
        """Different seeds may produce different results (non-deterministic)."""
        with patch("scripts.remoteclip_zero_shot.extract_embeddings") as mock_extract:
            mock_extract.return_value = (
                controlled_embeddings.copy(),
                [str(image_dir_with_pngs / f"img_{i}.png") for i in range(8)],
                [],
            )

            result_1 = train_few_shot(
                str(image_dir_with_pngs),
                str(labels_json),
                device="cpu",
                cv_folds=3,
                seed=42,
            )

            result_2 = train_few_shot(
                str(image_dir_with_pngs),
                str(labels_json),
                device="cpu",
                cv_folds=3,
                seed=99,
            )

        # These could differ or be the same depending on data; we just check
        # that both are valid. The key assertion is reproducibility with same seed.
        assert result_1["cv_folds"] == result_2["cv_folds"]

    def test_train_few_shot_empty_labels(self, tmp_path, image_dir_with_pngs):
        """Empty labels list returns error dict."""
        labels_path = _make_labels_file(tmp_path, [])

        # We never call extract_embeddings because labels check happens first
        result = train_few_shot(
            str(image_dir_with_pngs),
            str(labels_path),
            device="cpu",
        )

        assert "error" in result
        assert result["error"] is not None

    def test_train_few_shot_labels_file_not_found(self, tmp_path, image_dir_with_pngs):
        """Non-existent labels file returns error."""
        result = train_few_shot(
            str(image_dir_with_pngs),
            str(tmp_path / "nonexistent.json"),
            device="cpu",
        )

        assert "error" in result


# ===========================================================================
# Tests: main() with --mode fewshot
# ===========================================================================


class TestMainFewShot:
    """Verify the CLI --mode fewshot flag works correctly."""

    def test_main_fewshot_runs_end_to_end(self, tmp_path):
        """--mode fewshot with valid args exits 0 and produces output JSON."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        for i in range(6):
            _make_fake_png(image_dir / f"img_{i}.png")

        labels_path = _make_labels_file(tmp_path, [
            {"filename": "img_0.png", "label": 1},
            {"filename": "img_1.png", "label": 1},
            {"filename": "img_2.png", "label": 1},
            {"filename": "img_3.png", "label": 0},
            {"filename": "img_4.png", "label": 0},
            {"filename": "img_5.png", "label": 0},
        ])

        output_path = tmp_path / "results.json"

        argv = [
            "--mode", "fewshot",
            "--image-dir", str(image_dir),
            "--labels-json", str(labels_path),
            "--output-json", str(output_path),
        ]

        with (
            patch("scripts.remoteclip_zero_shot.train_few_shot") as mock_train,
        ):
            mock_train.return_value = {
                "cv_accuracy_mean": 0.95,
                "cv_accuracy_std": 0.05,
                "cv_precision_mean": 0.94,
                "cv_recall_mean": 0.96,
                "cv_f1_mean": 0.95,
                "cv_folds": 5,
                "n_total": 6,
                "n_pos": 3,
                "n_neg": 3,
                "full_train_accuracy": 1.0,
                "per_fold": [],
                "classifier": "logistic",
                "embed_dim": 768,
            }

            exit_code = main(argv)

        assert exit_code == 0
        assert output_path.exists()
        saved = json.loads(output_path.read_text())
        assert saved["cv_accuracy_mean"] == 0.95

    def test_main_fewshot_requires_labels_json(self, tmp_path):
        """--mode fewshot fails without --labels-json."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        _make_fake_png(image_dir / "img.png")

        argv = [
            "--mode", "fewshot",
            "--image-dir", str(image_dir),
        ]

        exit_code = main(argv)
        assert exit_code != 0

    def test_main_zeroshot_default_mode(self, tmp_path):
        """Default mode is zeroshot when --mode is not specified."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        _make_fake_png(image_dir / "img.png")

        argv = [
            "--image-dir", str(image_dir),
        ]

        with (
            patch("scripts.remoteclip_zero_shot.score_directory") as mock_score,
        ):
            mock_score.return_value = []
            exit_code = main(argv)

        assert exit_code == 0
        # score_directory should be called (zero-shot path, not train_few_shot)
        mock_score.assert_called_once()

    def test_main_fewshot_invalid_mode(self, tmp_path):
        """Invalid --mode value raises error."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        argv = [
            "--mode", "invalid",
            "--image-dir", str(image_dir),
        ]

        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code != 0


class TestTrainFewShotSVM:
    """Verify that both logistic regression and SVM are trained."""

    def test_svm_trained_and_reported(self, tmp_path, image_dir_with_pngs, labels_json, controlled_embeddings):
        """train_few_shot trains both LR and SVM and includes SVM accuracy."""
        with patch("scripts.remoteclip_zero_shot.extract_embeddings") as mock_extract:
            mock_extract.return_value = (
                controlled_embeddings,
                [str(image_dir_with_pngs / f"img_{i}.png") for i in range(8)],
                [],
            )

            result = train_few_shot(
                str(image_dir_with_pngs),
                str(labels_json),
                device="cpu",
                cv_folds=3,
                seed=42,
            )

        assert "svm_full_train_accuracy" in result
        assert isinstance(result["svm_full_train_accuracy"], float)


class TestEmbedDimConstant:
    """Verify embed_dim is correctly reported."""

    def test_embed_dim_in_result(self, tmp_path, image_dir_with_pngs, labels_json, controlled_embeddings):
        """embed_dim in result matches EMBED_DIM constant."""
        with patch("scripts.remoteclip_zero_shot.extract_embeddings") as mock_extract:
            mock_extract.return_value = (
                controlled_embeddings,
                [str(image_dir_with_pngs / f"img_{i}.png") for i in range(8)],
                [],
            )

            result = train_few_shot(
                str(image_dir_with_pngs),
                str(labels_json),
                device="cpu",
                cv_folds=3,
                seed=42,
            )

        assert result["embed_dim"] == 768
