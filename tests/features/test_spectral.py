import numpy as np

from herring_spawner.features.spectral import compute_visible_features


def test_compute_visible_features_from_band_arrays():
    bands = {
        "blue": np.array([[0.10, 0.20], [0.10, 0.20]]),
        "green": np.array([[0.30, 0.40], [0.30, 0.40]]),
        "red": np.array([[0.10, 0.20], [0.10, 0.20]]),
    }
    features = compute_visible_features(bands)
    assert features["mean_blue"] == 0.15
    assert features["mean_green"] == 0.35
    assert features["mean_red"] == 0.15
    assert round(features["green_blue_ratio"], 3) == 2.333
    assert round(features["green_red_ratio"], 3) == 2.333
