import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fewshot_subspace_variants import (
    aggregate_patch_scores,
    combine_positive_aware_scores,
    safe_pca_components,
)


def test_safe_pca_components_caps_positive_pca_for_small_few_shot_sets():
    assert safe_pca_components(15, 384, positive_side=True) == 7
    assert safe_pca_components(164, 384, positive_side=False) == 32


def test_positive_aware_score_multiplication_rewards_positive_similarity():
    residuals = np.array([0.9, 0.1], dtype=np.float32)
    similarities = np.array([0.8, -0.5], dtype=np.float32)

    scores = combine_positive_aware_scores(residuals, similarities, mode="residual_x_pos_sim")

    assert scores[0] > scores[1]
    assert scores[1] == 0.0


def test_positive_aware_score_zsum_is_ordered_for_matching_examples():
    residuals = np.array([0.9, 0.1, 0.2], dtype=np.float32)
    similarities = np.array([0.8, -0.5, -0.4], dtype=np.float32)

    scores = combine_positive_aware_scores(residuals, similarities, mode="zsum")

    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_patch_aggregation_supports_topk_max_and_percentile():
    patch_scores = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    assert aggregate_patch_scores(patch_scores, method="max") == 4.0
    assert aggregate_patch_scores(patch_scores, method="topk_mean", top_k=2) == 3.5
    assert aggregate_patch_scores(patch_scores, method="percentile", percentile=75.0) == 3.25
