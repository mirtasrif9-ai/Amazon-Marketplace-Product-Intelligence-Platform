from pathlib import Path

import pytest

from src.features.product_recommendation import (
    ProductRecommendationEngine,
)
from src.features.product_recommendation.models import (
    RecommendationResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
)


@pytest.fixture(scope="module")
def recommendation_engine():
    """Create recommendation engine once per test module."""

    return ProductRecommendationEngine(
        model_dir=MODEL_DIR
    )


@pytest.fixture
def sample_asin():
    return "B0HCNYJM8Z"


def test_engine_loads_artifacts(recommendation_engine):
    """Engine should load all persisted artifacts."""

    assert recommendation_engine.vectorizer is not None
    assert recommendation_engine.product_matrix is not None
    assert recommendation_engine.product_metadata is not None


def test_recommend_returns_requested_count(
    recommendation_engine,
    sample_asin,
):
    """Engine should return exactly top_k recommendations."""

    top_k = 5

    results = recommendation_engine.recommend(
        product_asin=sample_asin,
        top_k=top_k,
    )

    assert len(results) == top_k


def test_recommend_returns_correct_result_type(
    recommendation_engine,
    sample_asin,
):
    """All returned items should be RecommendationResult."""

    results = recommendation_engine.recommend(
        product_asin=sample_asin,
        top_k=5,
    )

    assert all(
        isinstance(result, RecommendationResult)
        for result in results
    )


def test_query_product_not_recommended(
    recommendation_engine,
    sample_asin,
):
    """The query product must not recommend itself."""

    results = recommendation_engine.recommend(
        product_asin=sample_asin,
        top_k=10,
    )

    recommended_asins = [
        result.asin
        for result in results
    ]

    assert sample_asin not in recommended_asins


def test_recommendations_are_sorted_by_similarity(
    recommendation_engine,
    sample_asin,
):
    """Recommendations must be sorted descending by similarity."""

    results = recommendation_engine.recommend(
        product_asin=sample_asin,
        top_k=10,
    )

    scores = [
        result.similarity_score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_recommendations_are_deterministic(
    recommendation_engine,
    sample_asin,
):
    """Same input must always produce identical results."""

    results_1 = recommendation_engine.recommend(
        product_asin=sample_asin,
        top_k=5,
    )

    results_2 = recommendation_engine.recommend(
        product_asin=sample_asin,
        top_k=5,
    )

    assert results_1 == results_2


def test_invalid_asin_raises_error(
    recommendation_engine,
):
    """Unknown ASIN should raise ValueError."""

    with pytest.raises(
        ValueError,
        match="not found",
    ):
        recommendation_engine.recommend(
            product_asin="INVALID_ASIN",
            top_k=5,
        )


@pytest.mark.parametrize(
    "invalid_top_k",
    [0, -1, -10],
)
def test_invalid_top_k_raises_error(
    recommendation_engine,
    sample_asin,
    invalid_top_k,
):
    """Non-positive top_k should raise ValueError."""

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        recommendation_engine.recommend(
            product_asin=sample_asin,
            top_k=invalid_top_k,
        )


def test_similarity_scores_are_valid(
    recommendation_engine,
    sample_asin,
):
    """Similarity scores should be between 0 and 1."""

    results = recommendation_engine.recommend(
        product_asin=sample_asin,
        top_k=10,
    )

    for result in results:
        assert 0 <= result.similarity_score <= 1


def test_recommendation_asins_are_unique(
    recommendation_engine,
    sample_asin,
):
    """Returned recommendations should not contain duplicates."""

    results = recommendation_engine.recommend(
        product_asin=sample_asin,
        top_k=10,
    )

    asins = [
        result.asin
        for result in results
    ]

    assert len(asins) == len(set(asins))