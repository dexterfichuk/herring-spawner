from datetime import date, datetime
from pathlib import Path

import numpy as np

from scripts.temporal_v2 import Observation, Sample, load_local_samples, score_mode


def make_sample(filename: str, label: int, day: str, location_key: str | None = None) -> Sample:
    d = date.fromisoformat(day)
    return Sample(
        filename=filename,
        label=label,
        image_path=Path(filename),
        region_key=location_key or filename.split("_")[0],
        location_key=location_key or filename.split("_")[0],
        observation_date=d,
        acquisition_dt=datetime.combine(d, datetime.min.time()),
        lat=None,
        lon=None,
    )


def make_obs(filename: str, label: int, day: str, location_key: str, *, emb, bands=None, cloud=0.0):
    return Observation(
        sample=make_sample(filename, label, day, location_key),
        embedding=np.asarray(emb, dtype=float),
        bands=bands or {},
        cloud_fraction=cloud,
        source="local",
    )


def test_load_local_samples_prefers_existing_files(tmp_path):
    manifest = tmp_path / "training_manifest.json"
    pos_dir = tmp_path / "positive"
    neg_dir = tmp_path / "negative"
    pos_dir.mkdir()
    neg_dir.mkdir()
    pos_file = pos_dir / "pos_2024-03-01_49.0_-123.0_20240301.png"
    neg_file = neg_dir / "neg_2024-03-02_49.1_-123.1_20240302.png"
    pos_file.write_bytes(b"x")
    neg_file.write_bytes(b"x")
    manifest.write_text('{"positives": ["pos_2024-03-01_49.0_-123.0_20240301.png"], "rejected": ["neg_2024-03-02_49.1_-123.1_20240302.png"]}')

    samples = load_local_samples(manifest, pos_dir, neg_dir)

    assert [s.filename for s in samples] == [pos_file.name, neg_file.name]
    assert [s.label for s in samples] == [1, 0]


def test_outlier_mode_scores_march_outlier_against_janfeb_baseline():
    observations = [
        make_obs("a_jan.png", 0, "2024-01-05", "loc-a", emb=[0.0, 0.0]),
        make_obs("a_feb.png", 0, "2024-02-05", "loc-a", emb=[0.1, 0.0]),
        make_obs("a_mar.png", 1, "2024-03-15", "loc-a", emb=[3.0, 0.0]),
        make_obs("b_mar.png", 0, "2024-03-15", "loc-b", emb=[0.0, 0.0]),
    ]

    rows, summary = score_mode("outlier", observations)

    scored = {row["filename"]: row for row in rows}
    assert scored["a_mar.png"]["score"] > scored["b_mar.png"]["score"]
    assert summary["evaluated_positive_count"] == 1


def test_yoy_mode_uses_multi_year_march_norm():
    observations = [
        make_obs("y2019.png", 0, "2019-03-20", "loc-y", emb=[0.0, 0.0]),
        make_obs("y2020.png", 0, "2020-03-20", "loc-y", emb=[0.1, 0.0]),
        make_obs("y2021.png", 1, "2021-03-20", "loc-y", emb=[3.0, 0.0]),
        make_obs("other.png", 0, "2021-03-20", "loc-z", emb=[0.0, 0.0]),
    ]

    rows, summary = score_mode("yoy", observations)

    scored = {row["filename"]: row for row in rows}
    assert scored["y2021.png"]["score"] > scored["y2019.png"]["score"]
    assert summary["evaluated_positive_count"] == 1


def test_spectral_mode_prefers_spawn_like_blue_green_shift():
    observations = [
        make_obs("s_base.png", 0, "2024-02-01", "loc-s", emb=[0.0, 0.0], bands={"blue": 0.20, "green": 0.22, "red": 0.18, "coastal": 0.18, "nir": 0.30}),
        make_obs("s_spawn.png", 1, "2024-03-15", "loc-s", emb=[0.0, 0.0], bands={"blue": 0.45, "green": 0.25, "red": 0.18, "coastal": 0.40, "nir": 0.28}),
    ]

    rows, summary = score_mode("spectral", observations)

    scored = {row["filename"]: row for row in rows}
    assert scored["s_spawn.png"]["score"] > scored["s_base.png"]["score"]
    assert summary["evaluated_positive_count"] == 1


def test_trajectory_and_cloud_fusion_modes_return_scores():
    observations = [
        make_obs("t1.png", 0, "2024-01-01", "loc-t", emb=[0.0, 0.0], cloud=20.0),
        make_obs("t2.png", 0, "2024-02-01", "loc-t", emb=[0.1, 0.0], cloud=30.0),
        make_obs("t3.png", 1, "2024-03-01", "loc-t", emb=[2.0, 0.0], cloud=40.0),
        make_obs("t4.png", 0, "2024-04-01", "loc-t", emb=[0.2, 0.0], cloud=10.0),
        make_obs("t5.png", 0, "2024-05-01", "loc-t", emb=[0.1, 0.0], cloud=0.0),
        make_obs("u1.png", 0, "2024-01-01", "loc-u", emb=[0.0, 0.0], cloud=0.0),
        make_obs("u2.png", 0, "2024-02-01", "loc-u", emb=[0.0, 0.0], cloud=0.0),
        make_obs("u3.png", 0, "2024-03-01", "loc-u", emb=[0.0, 0.0], cloud=0.0),
        make_obs("u4.png", 0, "2024-04-01", "loc-u", emb=[0.0, 0.0], cloud=0.0),
        make_obs("u5.png", 0, "2024-05-01", "loc-u", emb=[0.0, 0.0], cloud=0.0),
    ]

    traj_rows, traj_summary = score_mode("trajectory", observations)
    fused_rows, fused_summary = score_mode("cloud_fusion", observations)

    traj_scores = {row["filename"]: row for row in traj_rows}
    fused_scores = {row["filename"]: row for row in fused_rows}
    assert traj_scores["t3.png"]["score"] != traj_scores["u3.png"]["score"]
    assert fused_scores["t3.png"]["score"] > fused_scores["u3.png"]["score"]
    assert traj_summary["evaluated_positive_count"] == 1
    assert fused_summary["evaluated_positive_count"] == 1
