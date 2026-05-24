from pathlib import Path

import pytest

from scripts.dinov2_consensus import (
    build_report_html,
    compute_consensus_metrics,
    match_candidate_image,
    parse_candidate_id,
)


def test_parse_candidate_id_extracts_fields():
    parsed = parse_candidate_id("cand:tofino:49.134865:-125.946603:2023-04-28")

    assert parsed["site"] == "tofino"
    assert parsed["lat"] == pytest.approx(49.134865)
    assert parsed["lon"] == pytest.approx(-125.946603)
    assert parsed["date"] == "2023-04-28"
    assert parsed["date_compact"] == "20230428"


def test_match_candidate_image_finds_expected_file():
    candidate_dir = Path("/tmp/candidates")
    image = match_candidate_image(
        candidate_dir,
        "cand:tofino:49.134865:-125.946603:2023-04-28",
        [candidate_dir / "tofino_2023-04-28_score0.54_49.134865_-125.946603_20230428.png"],
    )

    assert image == candidate_dir / "tofino_2023-04-28_score0.54_49.134865_-125.946603_20230428.png"


def test_compute_consensus_metrics_returns_mean_variance_and_consensus():
    metrics = compute_consensus_metrics([0.2, 0.4, 0.6])

    assert metrics["mean"] == pytest.approx(0.4)
    assert metrics["variance"] == pytest.approx(0.0266666667)
    assert metrics["consensus"] == pytest.approx(0.3733333333)


def test_build_report_html_includes_before_after_summary():
    html = build_report_html(
        {
            "baseline": {"spawn": 0.6, "nospawn": 0.2, "separation": 0.4},
            "consensus": {"spawn": 0.7, "nospawn": 0.1, "separation": 0.6},
        },
        [
            {
                "id": "cand:tofino:49.134865:-125.946603:2023-04-28",
                "label": "spawn",
                "baseline": 0.51,
                "mean": 0.55,
                "variance": 0.01,
                "consensus": 0.54,
            }
        ],
    )

    assert "Augmentation Consensus" in html
    assert "Baseline separation" in html
    assert "Consensus separation" in html
    assert "before/after" in html.lower()
