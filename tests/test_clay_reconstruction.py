from pathlib import Path

import pytest

from scripts.clay_reconstruction import parse_label_id, resolve_thumbnail_path, summarize_results


def test_parse_label_id_extracts_region_coordinates_and_date():
    record = parse_label_id("cand:tofino:49.134865:-125.946603:2023-04-28")

    assert record.region == "tofino"
    assert record.lat == 49.134865
    assert record.lon == -125.946603
    assert record.date == "2023-04-28"


def test_resolve_thumbnail_path_matches_manifest_row():
    manifest_rows = [
        {
            "region": "tofino",
            "lat": 49.134865,
            "lon": -125.946603,
            "date": "2023-04-28",
            "thumbnail_path": "tofino_2023-04-28_score0.34_49.134865_-125.946603_20230428.png",
        }
    ]

    path = resolve_thumbnail_path(
        "cand:tofino:49.134865:-125.946603:2023-04-28",
        manifest_rows,
        Path("/tmp/candidates_v2"),
    )

    assert path.name == "tofino_2023-04-28_score0.34_49.134865_-125.946603_20230428.png"


def test_summarize_results_reports_mean_difference_and_separation():
    spawn = [0.40, 0.50, 0.60, 0.70]
    nospawn = [0.10, 0.20, 0.30]

    summary = summarize_results(spawn, nospawn)

    assert summary["spawn_mean"] == pytest.approx(0.55)
    assert summary["nospawn_mean"] == pytest.approx(0.2)
    assert summary["mean_difference"] == pytest.approx(0.35)
    assert summary["separation"] == pytest.approx(0.35)
