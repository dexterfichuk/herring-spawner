from pathlib import Path

import numpy as np
import pytest

from scripts.knn_detector import (
    build_report_html,
    compute_baseline_metrics,
    evaluate_leave_one_out_knn,
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


def test_evaluate_leave_one_out_knn_prefers_smaller_cluster():
    positive = np.array([[1.0, 0.0]] * 5, dtype=float)
    negative = np.array([[-1.0, 0.0]] * 5, dtype=float)
    embeddings = np.vstack([positive, negative])
    labels = np.array([1] * 5 + [0] * 5, dtype=int)

    metrics = evaluate_leave_one_out_knn(embeddings, labels, ks=[3, 5, 7, 10, 15])

    assert metrics[3]["accuracy"] == pytest.approx(1.0)
    assert metrics[5]["accuracy"] == pytest.approx(1.0)
    assert metrics[10]["effective_k"] == 9
    assert metrics[15]["effective_k"] == 9


def test_compute_baseline_metrics_reports_accuracy_and_confusion_matrix():
    positive = np.array([[1.0, 0.0]] * 4, dtype=float)
    negative = np.array([[-1.0, 0.0]] * 4, dtype=float)
    embeddings = np.vstack([positive, negative])
    labels = np.array([1] * 4 + [0] * 4, dtype=int)

    metrics = compute_baseline_metrics(embeddings, labels)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[4, 0], [0, 4]]


def test_build_report_html_includes_comparison_summary():
    html = build_report_html(
        {
            "baseline": {"accuracy": 0.80, "confusion_matrix": [[8, 1], [2, 9]]},
            "knn": {"best_k": 5, "accuracy": 0.90, "confusion_matrix": [[9, 0], [2, 9]]},
            "k_results": {3: {"accuracy": 0.88}, 5: {"accuracy": 0.90}},
        },
        [
            {
                "id": "cand:tofino:49.134865:-125.946603:2023-04-28",
                "label": "spawn",
                "baseline_score": 0.41,
                "predictions": {3: 1, 5: 1},
            }
        ],
    )

    assert "KNN Voting Classifier" in html
    assert "Best K" in html
    assert "Baseline accuracy" in html
    assert "Confusion Matrix" in html
