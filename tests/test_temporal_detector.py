from datetime import datetime

import numpy as np
import pytest

from scripts.temporal_detector import (
    compute_timeseries_features,
    compute_triplet_features,
    parse_acquisition_datetime,
    select_tide_matched_scene,
)


def test_parse_acquisition_datetime_extracts_granule_time():
    parsed = parse_acquisition_datetime("20260410T192909_20260410T194001_T09UXS")

    assert parsed == datetime(2026, 4, 10, 19, 29, 9)


def test_select_tide_matched_scene_prefers_closest_tide():
    scenes = [
        {"scene_id": "a", "tide_height": 1.8, "cloud": 20.0},
        {"scene_id": "b", "tide_height": 0.9, "cloud": 5.0},
        {"scene_id": "c", "tide_height": 1.0, "cloud": 40.0},
    ]

    chosen = select_tide_matched_scene(scenes, target_tide_height=1.05)

    assert chosen["scene_id"] == "c"


def test_compute_triplet_features_scores_transient_change():
    before = np.array([0.0, 0.0])
    during = np.array([3.0, 0.0])
    after = np.array([0.1, 0.0])

    features = compute_triplet_features(before, during, after, tide_before=1.2, tide_during=1.1, tide_after=1.0)

    assert features["spawn_score"] == pytest.approx(3.0)
    assert features["temporal_score"] == pytest.approx(2.9)
    assert features["reversal_ok"] is True
    assert features["tide_span"] == pytest.approx(0.2)


def test_compute_timeseries_features_detects_rise_fall_bump():
    embeddings = [
        np.array([0.0, 0.0]),
        np.array([0.2, 0.0]),
        np.array([1.2, 0.0]),
        np.array([0.3, 0.0]),
        np.array([0.1, 0.0]),
    ]
    dates = [
        "2026-01-05",
        "2026-02-01",
        "2026-03-15",
        "2026-04-05",
        "2026-05-12",
    ]
    tides = [1.0, 1.0, 1.1, 1.0, 1.0]

    features = compute_timeseries_features(embeddings, dates, tides)

    assert features["peak_distance"] > 1.0
    assert features["rise_rate"] > 0.0
    assert features["fall_rate"] > 0.0
    assert features["bump_score"] > 0.0
