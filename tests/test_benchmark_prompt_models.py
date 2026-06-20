import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_prompt_models import build_benchmark_summary, sweep_thresholds


def test_build_benchmark_summary_reports_metrics_and_rows():
    rows = [
        {"model": "senclip", "mode": "full", "score": 0.9, "label": 1},
        {"model": "senclip", "mode": "full", "score": -0.2, "label": 0},
    ]
    summary = build_benchmark_summary(rows)
    assert summary["row_count"] == 2
    assert len(summary["rows"]) == 2
    assert summary["metrics"]["accuracy"] == 1.0


def test_sweep_thresholds_finds_best_accuracy_and_f1():
    rows = [
        {"score": 0.9, "label": 1},
        {"score": 0.8, "label": 1},
        {"score": -0.1, "label": 0},
        {"score": -0.2, "label": 0},
    ]

    summary = sweep_thresholds(rows)

    assert summary["baseline"]["accuracy"] == 1.0
    assert summary["best_accuracy"]["accuracy"] == 1.0
    assert summary["best_f1"]["f1"] == 1.0
    assert summary["best_accuracy"]["threshold"] > 0.1
