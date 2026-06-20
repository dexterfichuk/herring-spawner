import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prompt_detector import (
    PROMPT_BANKS,
    aggregate_crop_scores,
    contrastive_score,
    generate_sliding_crops,
    score_prompt_groups,
)


def test_prompt_bank_structure():
    assert "default" in PROMPT_BANKS
    bank = PROMPT_BANKS["default"]
    assert len(bank.spawn) >= 4
    assert set(bank.confounders) == {
        "foam_waves",
        "glint",
        "sediment",
        "clouds",
        "surf_beach",
    }
    assert all(isinstance(prompt, str) for prompt in bank.spawn)


def test_contrastive_score_uses_spawn_mean_minus_max_confounder():
    score = contrastive_score(
        spawn_scores=[0.9, 0.7, 0.8],
        confounder_scores_by_group={
            "foam_waves": [0.2, 0.3],
            "glint": [0.5, 0.4],
        },
    )
    assert score == pytest.approx(0.35)


def test_score_prompt_groups_reports_margin_and_max_confounder():
    class DummyBackend:
        name = "dummy"

    prompt_embeddings = type(
        "PromptEmbeddingsLike",
        (),
        {
            "spawn": __import__("numpy").array([[1.0, 0.0], [0.5, 0.5]], dtype=float),
            "confounders": {
                "foam_waves": __import__("numpy").array([[0.0, 1.0]], dtype=float),
                "glint": __import__("numpy").array([[0.2, 0.8]], dtype=float),
            },
        },
    )
    result = score_prompt_groups(prompt_embeddings, __import__("numpy").array([1.0, 0.0]))

    assert result["spawn_mean"] == pytest.approx(0.75)
    assert result["max_confounder_group"] == "glint"
    assert result["max_confounder_mean"] == pytest.approx(0.2)
    assert result["margin"] == pytest.approx(0.55)
    assert result["score"] == pytest.approx(0.55)


def test_generate_sliding_crops_limits_count_and_keeps_full_image():
    crops = generate_sliding_crops(
        (1024, 768), crop_size=256, stride=256, max_crops=8
    )
    assert crops[0] == (0, 0, 1024, 768)
    assert len(crops) == 8
    assert all(len(box) == 4 for box in crops)
    assert all(box[2] > box[0] and box[3] > box[1] for box in crops)


def test_aggregate_crop_scores_supports_max_and_topk_mean():
    scores = [0.1, 0.8, 0.3, 0.6]
    assert aggregate_crop_scores(scores, mode="max") == pytest.approx(0.8)
    assert aggregate_crop_scores(scores, mode="topk_mean", top_k=2) == pytest.approx(0.7)
