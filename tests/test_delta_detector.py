"""Tests for delta_detector.py."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.delta_detector import (
    compute_delta,
    load_locations_from_manifest,
    parse_location_from_filename,
    train_delta_classifier,
)


def test_parse_location_from_filename_standard():
    """Parse standard filename format: region_date_score_lat_lon_scenedate.png."""
    result = parse_location_from_filename(
        "SoG_2021-03-11_score0.00_49.5175_-124.577222_20210311.png"
    )
    assert result is not None
    assert result["lat"] == pytest.approx(49.5175)
    assert result["lon"] == pytest.approx(-124.577222)
    assert result["date"] == "2021-03-11"
    assert result["scene_date"] == "20210311"
    assert result["region"] == "SoG"


def test_parse_location_from_filename_no_match():
    """Return None for unparseable filenames."""
    assert parse_location_from_filename("random_file.png") is None
    assert parse_location_from_filename("dfo-verified_2024-03-16_cloud0.png") is None


def test_parse_location_from_filename_varied():
    """Parse various standard-format filenames from the negative set."""
    result = parse_location_from_filename(
        "SoG_2016-03-05_score0.00_49.21326_-123.940398_20160305.png"
    )
    assert result is not None
    assert result["lat"] == pytest.approx(49.21326)
    assert result["lon"] == pytest.approx(-123.940398)
    assert result["date"] == "2016-03-05"


def test_parse_location_from_filename_nootka():
    """Parse nootka-sound format with negative lat/lon."""
    result = parse_location_from_filename(
        "nootka-sound_2024-03-16_score0.00_49.584865_-126.528503_20240316.png"
    )
    assert result is not None
    assert result["lat"] == pytest.approx(49.584865)
    assert result["lon"] == pytest.approx(-126.528503)
    assert result["date"] == "2024-03-16"
    assert result["region"] == "nootka-sound"


def test_parse_location_from_filename_qualicum():
    """Parse qualicum format."""
    result = parse_location_from_filename(
        "qualicum_2024-03-18_score0.01_49.254865_-124.497442_20240318.png"
    )
    assert result is not None
    assert result["lat"] == pytest.approx(49.254865)
    assert result["lon"] == pytest.approx(-124.497442)
    assert result["date"] == "2024-03-18"


def test_compute_delta_vectors():
    """Delta = spawn_emb - baseline_emb with L2 norm."""
    baseline = np.array([1.0, 0.0, 0.0])
    spawn = np.array([0.0, 1.0, 0.0])
    delta, mag = compute_delta(spawn, baseline)
    assert np.allclose(delta, np.array([-1.0, 1.0, 0.0]))
    assert mag == pytest.approx(np.sqrt(2.0))


def test_compute_delta_identical():
    """Identical embeddings produce zero delta."""
    emb = np.array([0.5, 0.5, 0.5])
    delta, mag = compute_delta(emb, emb)
    assert np.allclose(delta, np.zeros(3))
    assert mag == pytest.approx(0.0)


def test_compute_delta_magnitude():
    """Delta magnitude equals L2 norm of difference."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([4.0, 0.0, 0.0])
    delta, mag = compute_delta(a, b)
    assert mag == pytest.approx(3.0)
    assert np.allclose(delta, np.array([-3.0, 0.0, 0.0]))


def test_load_locations_from_manifest(tmp_path):
    """Load locations from a small training manifest."""
    pos_dir = tmp_path / "positive"
    pos_dir.mkdir(parents=True)
    fname = "SoG_2021-03-11_score0.00_49.5175_-124.577222_20210311.png"
    (pos_dir / fname).write_text("fake")

    manifest = {
        "description": "test",
        "positive_count": 1,
        "negative_count": 0,
        "positives": [fname],
    }
    mfile = tmp_path / "training_manifest.json"
    mfile.write_text(json.dumps(manifest))

    neg_dir = tmp_path / "negative"
    neg_dir.mkdir()

    locs = load_locations_from_manifest(str(mfile), str(neg_dir))
    assert len(locs["positives"]) == 1
    assert locs["positives"][0]["lat"] == pytest.approx(49.5175)
    assert locs["positives"][0]["lon"] == pytest.approx(-124.577222)
    assert locs["positives"][0]["date"] == "2021-03-11"


def test_load_locations_skips_unparseable(tmp_path):
    """Skip negative files that don't have parseable lat/lon."""
    pos_dir = tmp_path / "positive"
    pos_dir.mkdir(parents=True)
    manifest = {"positives": []}
    mfile = tmp_path / "training_manifest.json"
    mfile.write_text(json.dumps(manifest))

    neg_dir = tmp_path / "negative"
    neg_dir.mkdir()
    # Unparseable name
    (neg_dir / "dfo-verified_2024-03-16_cloud0.png").write_text("fake")

    locs = load_locations_from_manifest(str(mfile), str(neg_dir))
    assert len(locs["positives"]) == 0
    assert len(locs["negatives"]) == 0


def test_load_locations_negative_with_parseable_name(tmp_path):
    """Parse negative files that DO have standard-format names."""
    pos_dir = tmp_path / "positive"
    pos_dir.mkdir(parents=True)
    manifest = {"positives": []}
    mfile = tmp_path / "training_manifest.json"
    mfile.write_text(json.dumps(manifest))

    neg_dir = tmp_path / "negative"
    neg_dir.mkdir()
    fname = "SoG_2016-03-05_score0.00_49.21326_-123.940398_20160305.png"
    (neg_dir / fname).write_text("fake")

    locs = load_locations_from_manifest(str(mfile), str(neg_dir))
    assert len(locs["positives"]) == 0
    assert len(locs["negatives"]) == 1
    assert locs["negatives"][0]["lat"] == pytest.approx(49.21326)
    assert locs["negatives"][0]["date"] == "2016-03-05"


def test_train_delta_classifier_separable_data():
    """SVM should perfectly separate very different deltas."""
    # Spawn deltas: large positive vector
    spawn = [np.array([1.0, 0.0]) * 5 for _ in range(4)]
    # Non-spawn deltas: zero vector
    nonspawn = [np.array([0.0, 0.0]) for _ in range(4)]

    result = train_delta_classifier(spawn, nonspawn)
    assert result["train_accuracy"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [[4, 0], [0, 4]]


def test_train_delta_classifier_accuracy_stats():
    """Classifier returns useful stats even for mixed data."""
    np.random.seed(42)
    spawn = [np.random.randn(10) * 0.5 + 1.0 for _ in range(5)]
    nonspawn = [np.random.randn(10) * 0.5 for _ in range(5)]

    result = train_delta_classifier(spawn, nonspawn)
    assert "train_accuracy" in result
    assert "confusion_matrix" in result
    assert "spawn_mean_magnitude" in result
    assert "nonspawn_mean_magnitude" in result
    assert "separation_in_magnitude" in result
    assert result["n_spawn"] == 5
    assert result["n_nonspawn"] == 5
    assert result["model"] is not None
