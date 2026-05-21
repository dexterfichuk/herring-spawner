import numpy as np


def compute_visible_features(bands: dict[str, np.ndarray]) -> dict[str, float]:
    blue = _finite(bands["blue"])
    green = _finite(bands["green"])
    red = _finite(bands["red"])
    mean_blue = float(np.mean(blue))
    mean_green = float(np.mean(green))
    mean_red = float(np.mean(red))
    epsilon = 1e-9
    return {
        "mean_blue": round(mean_blue, 6),
        "mean_green": round(mean_green, 6),
        "mean_red": round(mean_red, 6),
        "visible_brightness": round(float(np.mean([mean_blue, mean_green, mean_red])), 6),
        "green_blue_ratio": round(mean_green / (mean_blue + epsilon), 6),
        "green_red_ratio": round(mean_green / (mean_red + epsilon), 6),
    }


def _finite(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    return values[np.isfinite(values)]
