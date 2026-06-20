from datetime import date, datetime
from pathlib import Path

import numpy as np

from scripts.environmental_matcher import (
    Observation,
    Sample,
    benchmark_environmental_matcher,
    environmental_bucket_key,
    score_environmental_outliers,
)


def make_sample(filename: str, label: int, day: str, location_key: str) -> Sample:
    d = date.fromisoformat(day)
    return Sample(
        filename=filename,
        label=label,
        image_path=Path(filename),
        region_key=location_key,
        location_key=location_key,
        observation_date=d,
        acquisition_dt=datetime.combine(d, datetime.min.time()),
        lat=49.0,
        lon=-123.0,
    )


def make_obs(
    filename: str, label: int, day: str, location_key: str, *, emb, sun_elev, sun_azim, tide
):
    return Observation(
        sample=make_sample(filename, label, day, location_key),
        embedding=np.asarray(emb, dtype=float),
        sun_elevation=sun_elev,
        sun_azimuth=sun_azim,
        tide_height=tide,
        tide_source="actual",
    )


def test_environmental_bucket_key_rounds_sun_and_tide():
    key_a = environmental_bucket_key(49.0, 151.0, 1.12, tide_source="actual")
    key_b = environmental_bucket_key(51.9, 149.0, 1.18, tide_source="actual")

    assert key_a == key_b


def test_score_environmental_outliers_prefers_embedding_outlier_within_group():
    observations = [
        make_obs(
            "a.png",
            0,
            "2024-03-01",
            "loc-a",
            emb=[0.0, 0.0],
            sun_elev=31.2,
            sun_azim=151.0,
            tide=1.2,
        ),
        make_obs(
            "b.png",
            0,
            "2024-03-08",
            "loc-a",
            emb=[0.1, 0.0],
            sun_elev=31.4,
            sun_azim=150.2,
            tide=1.1,
        ),
        make_obs(
            "spawn.png",
            1,
            "2024-03-15",
            "loc-a",
            emb=[3.0, 0.0],
            sun_elev=31.1,
            sun_azim=149.7,
            tide=1.15,
        ),
    ]

    rows, summary = score_environmental_outliers(observations)

    scored = {row["filename"]: row for row in rows}
    assert scored["spawn.png"]["score"] > scored["a.png"]["score"]
    assert scored["spawn.png"]["group_size"] == 3
    assert summary["evaluated_count"] == 3


def test_benchmark_environmental_matcher_reports_metrics_and_coverage():
    observations = [
        make_obs(
            "p1.png",
            1,
            "2024-03-15",
            "loc-p",
            emb=[3.0, 0.0],
            sun_elev=30.0,
            sun_azim=150.0,
            tide=1.0,
        ),
        make_obs(
            "p2.png",
            0,
            "2024-03-08",
            "loc-p",
            emb=[0.1, 0.0],
            sun_elev=30.1,
            sun_azim=149.8,
            tide=1.0,
        ),
        make_obs(
            "p3.png",
            0,
            "2024-03-22",
            "loc-p",
            emb=[0.0, 0.0],
            sun_elev=30.2,
            sun_azim=150.2,
            tide=1.0,
        ),
        make_obs(
            "n1.png",
            0,
            "2024-03-15",
            "loc-n",
            emb=[0.0, 0.0],
            sun_elev=35.0,
            sun_azim=160.0,
            tide=2.0,
        ),
        make_obs(
            "n2.png",
            0,
            "2024-03-22",
            "loc-n",
            emb=[0.1, 0.0],
            sun_elev=35.1,
            sun_azim=159.8,
            tide=2.0,
        ),
        make_obs(
            "n3.png",
            0,
            "2024-03-29",
            "loc-n",
            emb=[0.0, 0.05],
            sun_elev=35.2,
            sun_azim=160.2,
            tide=2.0,
        ),
    ]

    _, summary = benchmark_environmental_matcher(observations)

    assert summary["evaluated_positive_count"] == 1
    assert summary["average_scenes_per_location"] == 3.0
    assert summary["metrics"]["accuracy"] >= 0.5
    assert "auroc" in summary["metrics"]
