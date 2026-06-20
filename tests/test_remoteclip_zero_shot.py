"""Tests for the RemoteCLIP zero-shot classifier.

Tests use mocks extensively so they can be written and pass before the
implementation exists, verifying the API contract.
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
    NEGATIVE_PROMPTS,
    POSITIVE_PROMPTS,
    get_image_embedding,
    get_text_embedding,
    load_model,
    score_directory,
    score_image,
    validate,
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


def _make_mock_embedding_tensor(shape, rng=None):
    """Return a MagicMock that quacks like a torch Tensor with *shape*.

    The mock supports .float(), .cpu(), .detach(), .numpy(), division,
    and matrix-multiplication so that real get_text_embedding /
    get_image_embedding implementations can operate on it.
    """
    t = MagicMock(spec=["float", "cpu", "detach", "numpy", "shape", "norm",
                         "__truediv__", "__matmul__", "__getitem__",
                         "unsqueeze", "expand", "clone", "to", "device"])
    t.shape = shape
    t.float.return_value = t
    t.cpu.return_value = t
    t.detach.return_value = t
    t.to.return_value = t
    t.unsqueeze.return_value = t
    t.expand.return_value = t
    t.clone.return_value = t

    if rng is None:
        rng = np.random.default_rng(42)
    data = rng.standard_normal(shape).astype(np.float32)
    t.numpy.return_value = data

    # norm: return a scalar mock (or a per-row norm mock)
    if len(shape) == 1:
        norm_val = np.linalg.norm(data)
        norm_mock = MagicMock(spec=["item"])
        norm_mock.item.return_value = norm_val
    else:
        # 2D: return a column-vector mock for keepdim=True
        norms = np.linalg.norm(data, axis=-1, keepdims=True)
        norm_mock = MagicMock(spec=["float", "cpu", "detach", "numpy",
                                     "__getitem__", "shape"])
        norm_mock.shape = norms.shape
        norm_mock.float.return_value = norm_mock
        norm_mock.cpu.return_value = norm_mock
        norm_mock.detach.return_value = norm_mock

        def _norm_numpy():
            return norms

        norm_mock.numpy = _norm_numpy

        def _norm_getitem(key):
            # Return the scalar for that row
            if isinstance(key, tuple):
                row = key[0]
                return float(norms[row, 0])
            return float(norms[0, 0])

        norm_mock.__getitem__ = _norm_getitem
    t.norm.return_value = norm_mock

    # __truediv__: return self (enough for chaining)
    def _div(*args):
        return t

    t.__truediv__ = _div

    # __matmul__: return a mock that can be converted to list/float
    def _matmul(*args):
        mm = MagicMock(spec=["tolist", "item", "cpu", "numpy", "float",
                              "__getitem__", "shape"])
        mm.tolist.return_value = [0.5] * shape[0] if len(shape) == 2 else [0.5]
        mm.item.return_value = 0.5
        mm.cpu.return_value = mm
        mm.numpy.return_value = np.array([0.5])
        mm.float.return_value = mm
        mm.shape = (shape[0],) if len(shape) == 2 else (1,)

        def _getitem(*args):
            # Return a mock that supports .cpu().tolist()
            key = args[-1]
            all_vals = [0.5] * shape[0] if len(shape) == 2 else [0.5]
            if isinstance(key, slice):
                vals = all_vals[key]
            else:
                vals = [all_vals[key]]
            sm = MagicMock()
            sm.tolist.return_value = vals
            sm.cpu.return_value = sm
            sm.numpy.return_value = np.array(vals)
            sm.float.return_value = sm
            sm.shape = (len(vals),)
            return sm

        mm.__getitem__ = _getitem
        return mm

    t.__matmul__ = _matmul

    return t


# ---------------------------------------------------------------------------
# mock model factory
# ---------------------------------------------------------------------------

def _make_mock_model():
    """Return (model, preprocess, tokenize) backed by MagicMock.

    ``model.encode_text(texts)`` returns a mock tensor of shape (B, 512).
    ``model.encode_image(images)`` returns a mock tensor of shape (512,).
    ``preprocess(pil_image)`` returns a mock tensor of shape (3, 224, 224).
    ``tokenize(texts)`` returns a mock tensor.
    """
    model = MagicMock()
    rng = np.random.default_rng(42)

    # tokenize: returns a mock whose shape encodes the batch size
    def _tokenize_side_effect(texts):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, (list, tuple)):
            B = len(texts)
        else:
            B = 5
        return _make_mock_embedding_tensor((B, 77))

    def _text_side_effect(tokens):
        # tokens is the output of tokenize().to(device) — a MagicMock with shape
        B = tokens.shape[0] if (hasattr(tokens, 'shape') and isinstance(tokens.shape, tuple)) else 9
        return _make_mock_embedding_tensor((B, 512), rng)

    model.encode_text = MagicMock(side_effect=_text_side_effect)

    def _img_side_effect(images):
        return _make_mock_embedding_tensor((512,), rng)

    model.encode_image = MagicMock(side_effect=_img_side_effect)

    preprocess = MagicMock(return_value=_make_mock_embedding_tensor((3, 224, 224)))
    tokenize = MagicMock(side_effect=_tokenize_side_effect)

    return model, preprocess, tokenize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """Return a basic mock (model, preprocess, tokenize)."""
    return _make_mock_model()


@pytest.fixture
def mock_device():
    return "cpu"


# ===========================================================================
# Tests
# ===========================================================================


class TestImports:
    """Verify the module can be imported and all required functions exist."""

    def test_imports(self):
        assert callable(load_model)
        assert callable(get_text_embedding)
        assert callable(get_image_embedding)
        assert callable(score_image)
        assert callable(score_directory)
        assert callable(validate)

    def test_prompt_constants(self):
        assert len(POSITIVE_PROMPTS) == 12
        assert len(NEGATIVE_PROMPTS) == 12
        assert all(isinstance(p, str) for p in POSITIVE_PROMPTS)
        assert all(isinstance(p, str) for p in NEGATIVE_PROMPTS)


class TestScoreImageOutputKeys:
    """Verify score_image returns a dict with all required keys and correct types."""

    def test_score_image_output_keys(self, tmp_path, mock_model, mock_device):
        img_path = _make_fake_png(tmp_path / "test.png")
        model, preprocess, tokenize = mock_model

        # Patch the internal embedding functions so score_image gets
        # deterministic embeddings regardless of model internals.
        fake_img_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        with (
            patch("scripts.remoteclip_zero_shot.get_text_embedding") as mock_get_text,
            patch("scripts.remoteclip_zero_shot.get_image_embedding") as mock_get_img,
        ):
            # get_text_embedding returns shape (n_texts, 3)
            all_texts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS  # 24 texts
            text_emb = _make_mock_embedding_tensor((len(all_texts), 3))
            mock_get_text.return_value = text_emb
            mock_get_img.return_value = fake_img_emb

            result = score_image(
                model, preprocess, tokenize, str(img_path),
                POSITIVE_PROMPTS, NEGATIVE_PROMPTS, mock_device,
            )

        assert isinstance(result, dict), "result should be a dict"
        assert "pos_scores" in result
        assert "neg_scores" in result
        assert "score" in result
        assert "pos_mean" in result
        assert "neg_mean" in result
        assert "prediction" in result
        assert "image_path" in result

        assert isinstance(result["pos_scores"], list)
        assert len(result["pos_scores"]) == 12, "should match POSITIVE_PROMPTS length"
        assert all(isinstance(v, float) for v in result["pos_scores"])

        assert isinstance(result["neg_scores"], list)
        assert len(result["neg_scores"]) == 12, "should match NEGATIVE_PROMPTS length"
        assert all(isinstance(v, float) for v in result["neg_scores"])

        assert isinstance(result["score"], float)
        assert isinstance(result["pos_mean"], float)
        assert isinstance(result["neg_mean"], float)
        assert result["prediction"] in (0, 1)
        assert result["image_path"] == str(img_path)

    def test_score_image_calls_embedding_functions(self, tmp_path, mock_model, mock_device):
        """Verify score_image delegates to get_text_embedding and get_image_embedding."""
        img_path = _make_fake_png(tmp_path / "test.png")
        model, preprocess, tokenize = mock_model

        with (
            patch("scripts.remoteclip_zero_shot.get_text_embedding") as mock_get_text,
            patch("scripts.remoteclip_zero_shot.get_image_embedding") as mock_get_img,
        ):
            text_emb = _make_mock_embedding_tensor((24, 3))
            mock_get_text.return_value = text_emb
            mock_get_img.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            score_image(
                model, preprocess, tokenize, str(img_path),
                POSITIVE_PROMPTS, NEGATIVE_PROMPTS, mock_device,
            )

            # get_text_embedding should be called with all texts combined
            calls = mock_get_text.call_args_list
            assert len(calls) >= 1
            # The call should include all 24 texts (12 positive + 12 negative)
            call_texts = calls[0][0][2]  # third positional arg
            assert len(call_texts) == 24

            # get_image_embedding should be called with our image path
            mock_get_img.assert_called_once()
            img_call_args = mock_get_img.call_args[0]
            assert str(img_path) in img_call_args or img_path.name in str(img_call_args)


class TestScoreImagePrediction:
    """With controlled embeddings, verify correct predictions."""

    def test_prediction_positive(self, tmp_path, mock_model, mock_device):
        img_path = _make_fake_png(tmp_path / "pos.png")
        model, preprocess, tokenize = mock_model

        # Control embeddings: image matches positive better
        # pos_scores: 0.9 down to 0.35 (12 values), neg_scores: 0.1 down to -0.45 (12 values)
        # pos_mean = 0.625, neg_mean = -0.175, score = 0.8 > 0 => prediction = 1
        pos_vals = [0.9 - i * 0.05 for i in range(12)]  # 0.9 down to 0.35
        neg_vals = [0.1 - i * 0.05 for i in range(12)]  # 0.1 down to -0.45
        all_vals = pos_vals + neg_vals
        with (
            patch("scripts.remoteclip_zero_shot.get_text_embedding") as mock_get_text,
            patch("scripts.remoteclip_zero_shot.get_image_embedding") as mock_get_img,
        ):
            text_emb_mock = MagicMock()
            # __matmul__ returns a MagicMock whose .tolist() gives our scores
            def _matmul_side(*args):
                mm = MagicMock()
                # Similarity with positive: high; negative: low
                mm.tolist.return_value = all_vals
                mm.cpu.return_value = mm
                mm.numpy.return_value = np.array(all_vals)
                mm.float.return_value = mm
                mm.shape = (24,)

                def _getitem(*args):
                    key = args[-1]
                    if isinstance(key, slice):
                        sliced = all_vals[key]
                    else:
                        sliced = [all_vals[key]]
                    sm = MagicMock()
                    sm.tolist.return_value = sliced
                    sm.cpu.return_value = sm
                    sm.numpy.return_value = np.array(sliced)
                    sm.float.return_value = sm
                    sm.shape = (len(sliced),)
                    return sm

                mm.__getitem__ = _getitem
                return mm

            text_emb_mock.__matmul__ = _matmul_side
            mock_get_text.return_value = text_emb_mock
            mock_get_img.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            result = score_image(
                model, preprocess, tokenize, str(img_path),
                POSITIVE_PROMPTS, NEGATIVE_PROMPTS, mock_device,
            )

        assert result["prediction"] == 1, "should predict positive"
        assert result["score"] > 0
        assert result["pos_mean"] > result["neg_mean"]

    def test_prediction_negative(self, tmp_path, mock_model, mock_device):
        img_path = _make_fake_png(tmp_path / "neg.png")
        model, preprocess, tokenize = mock_model

        with (
            patch("scripts.remoteclip_zero_shot.get_text_embedding") as mock_get_text,
            patch("scripts.remoteclip_zero_shot.get_image_embedding") as mock_get_img,
        ):
            pos_vals = [-0.1 - i * 0.05 for i in range(12)]  # -0.1 down to -0.65
            neg_vals = [0.8 - i * 0.05 for i in range(12)]   # 0.8 down to 0.25
            all_vals = pos_vals + neg_vals
            text_emb_mock = MagicMock()

            def _matmul_side(*args):
                mm = MagicMock()
                # Negative prompts score higher than positive
                mm.tolist.return_value = all_vals
                mm.cpu.return_value = mm
                mm.numpy.return_value = np.array(all_vals)
                mm.float.return_value = mm
                mm.shape = (24,)

                def _getitem(*args):
                    key = args[-1]
                    if isinstance(key, slice):
                        sliced = all_vals[key]
                    else:
                        sliced = [all_vals[key]]
                    sm = MagicMock()
                    sm.tolist.return_value = sliced
                    sm.cpu.return_value = sm
                    sm.numpy.return_value = np.array(sliced)
                    sm.float.return_value = sm
                    sm.shape = (len(sliced),)
                    return sm

                mm.__getitem__ = _getitem
                return mm

            text_emb_mock.__matmul__ = _matmul_side
            mock_get_text.return_value = text_emb_mock
            mock_get_img.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            result = score_image(
                model, preprocess, tokenize, str(img_path),
                POSITIVE_PROMPTS, NEGATIVE_PROMPTS, mock_device,
            )

        assert result["prediction"] == 0, "should predict negative"
        assert result["score"] < 0
        assert result["pos_mean"] < result["neg_mean"]


class TestTextEmbeddingShape:
    """Verify get_text_embedding returns a tensor-like with correct first dimension."""

    def test_text_embedding_shape(self, mock_model, mock_device):
        model, _, tokenize = mock_model
        embed = get_text_embedding(model, tokenize, POSITIVE_PROMPTS, mock_device)
        assert hasattr(embed, "shape"), "should have a shape attribute"
        assert embed.shape[0] == 12, "first dim should match number of texts"


class TestImageEmbeddingShape:
    """Verify get_image_embedding returns a numpy 1-D array."""

    def test_image_embedding_shape(self, tmp_path, mock_model, mock_device):
        img_path = _make_fake_png(tmp_path / "img.png")
        model, preprocess, _ = mock_model

        embed = get_image_embedding(model, preprocess, str(img_path), mock_device)
        assert isinstance(embed, np.ndarray), "should return a numpy array"
        assert embed.ndim == 1, "should be a 1-D vector"


class TestValidateFunction:
    """Verify validate returns correct metrics when all predictions are correct."""

    def test_validate_all_correct(self, tmp_path):
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        _make_fake_png(image_dir / "pos1.png")
        _make_fake_png(image_dir / "pos2.png")
        _make_fake_png(image_dir / "neg1.png")
        _make_fake_png(image_dir / "neg2.png")

        labels_path = _make_labels_file(tmp_path, [
            {"filename": "pos1.png", "label": 1},
            {"filename": "pos2.png", "label": 1},
            {"filename": "neg1.png", "label": 0},
            {"filename": "neg2.png", "label": 0},
        ])

        def _score_side_effect(model, preprocess, tokenize, image_path,
                               positive_texts, negative_texts, device):
            fname = Path(image_path).name
            scores = {
                "pos1.png": (0.5, 1),
                "pos2.png": (0.3, 1),
                "neg1.png": (-0.2, 0),
                "neg2.png": (-0.5, 0),
            }
            score_val, pred_val = scores[fname]
            return {
                "pos_scores": [0.6] * 12,
                "neg_scores": [-0.1] * 12,
                "score": score_val,
                "pos_mean": 0.4,
                "neg_mean": -0.25,
                "prediction": pred_val,
                "image_path": image_path,
            }

        with patch("scripts.remoteclip_zero_shot.score_image") as mock_score:
            mock_score.side_effect = _score_side_effect
            result = validate(str(labels_path), str(image_dir), device="cpu")

        assert result["accuracy"] == pytest.approx(1.0)
        assert result["n_total"] == 4
        assert result["n_pos"] == 2
        assert result["n_neg"] == 2
        assert result["confusion_matrix"] == [[2, 0], [0, 2]]
        assert "best_accuracy" in result
        assert "best_threshold" in result
        assert "auc_roc" in result
        assert "avg_precision" in result
        assert len(result["per_sample"]) == 4
        for sample in result["per_sample"]:
            assert "score" in sample
            assert "prediction" in sample
            assert "true_label" in sample
            assert "filename" in sample


class TestValidateMixedPredictions:
    """Verify validate returns correct metrics with mixed correct/incorrect."""

    def test_validate_mixed(self, tmp_path):
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        _make_fake_png(image_dir / "pos1.png")
        _make_fake_png(image_dir / "pos2.png")
        _make_fake_png(image_dir / "neg1.png")
        _make_fake_png(image_dir / "neg2.png")

        labels_path = _make_labels_file(tmp_path, [
            {"filename": "pos1.png", "label": 1},
            {"filename": "pos2.png", "label": 1},
            {"filename": "neg1.png", "label": 0},
            {"filename": "neg2.png", "label": 0},
        ])

        def _score_side_effect(model, preprocess, tokenize, image_path,
                               positive_texts, negative_texts, device):
            fname = Path(image_path).name
            scores = {
                "pos1.png": (0.5, 1),    # correct
                "pos2.png": (-0.1, 0),   # incorrect (should be positive)
                "neg1.png": (-0.2, 0),   # correct
                "neg2.png": (0.1, 1),    # incorrect (should be negative)
            }
            score_val, pred_val = scores[fname]
            return {
                "pos_scores": [0.6] * 12,
                "neg_scores": [-0.1] * 12,
                "score": score_val,
                "pos_mean": 0.4,
                "neg_mean": -0.25,
                "prediction": pred_val,
                "image_path": image_path,
            }

        with patch("scripts.remoteclip_zero_shot.score_image") as mock_score:
            mock_score.side_effect = _score_side_effect
            result = validate(str(labels_path), str(image_dir), device="cpu")

        assert result["accuracy"] == pytest.approx(0.5)
        # confusion_matrix: [[true_neg, false_pos], [false_neg, true_pos]]
        assert result["confusion_matrix"] == [[1, 1], [1, 1]]
        assert result["n_total"] == 4
        assert result["n_pos"] == 2
        assert result["n_neg"] == 2


class TestScoreDirectory:
    """Verify score_directory returns list sorted by score descending."""

    def test_score_directory_sorted(self, tmp_path):
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        _make_fake_png(image_dir / "a.png")
        _make_fake_png(image_dir / "b.png")
        _make_fake_png(image_dir / "c.png")

        # Scores sorted descending: c (0.9), a (0.5), b (0.1)
        def _score_side_effect(model, preprocess, tokenize, image_path,
                               positive_texts, negative_texts, device):
            fname = Path(image_path).name
            scores_map = {
                "a.png": 0.5,
                "b.png": 0.1,
                "c.png": 0.9,
            }
            s = scores_map[fname]
            return {
                "pos_scores": [s] * 12,
                "neg_scores": [-0.1] * 12,
                "score": s,
                "pos_mean": s,
                "neg_mean": -0.1,
                "prediction": 1 if s > 0 else 0,
                "image_path": image_path,
            }

        with patch("scripts.remoteclip_zero_shot.score_image") as mock_score:
            mock_score.side_effect = _score_side_effect
            results = score_directory(str(image_dir), device="cpu", batch_size=16)

        assert len(results) == 3
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "should be sorted descending"
        assert results[0]["image_path"].endswith("c.png")
        assert results[2]["image_path"].endswith("b.png")
