from pathlib import Path

import pytest

from src.application.product_repository import ProductRepository
from src.application.recommendation_service import (
    RecommendationService,
)
from src.features.product_recommendation.models import (
    RecommendationResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_products.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
)


@pytest.fixture(scope="module")
def repository():

    return ProductRepository(
        DATASET_PATH
    )


@pytest.fixture(scope="module")
def recommendation_service(repository):

    return RecommendationService(
        repository=repository,
        model_dir=MODEL_DIR,
    )


def test_recommendation_artifacts_exist():

    assert MODEL_DIR.exists()

    assert (
        MODEL_DIR / "tfidf_vectorizer.joblib"
    ).exists()

    assert (
        MODEL_DIR / "product_tfidf_matrix.joblib"
    ).exists()

    assert (
        MODEL_DIR / "product_metadata.csv"
    ).exists()


def test_recommendations_for_product(
    repository,
    recommendation_service,
):

    product = repository.get_all_products()[0]

    results = (
        recommendation_service
        .get_recommendations(
            product.asin,
            top_k=5,
        )
    )

    assert len(results) == 5

    assert all(
        isinstance(
            result,
            RecommendationResult,
        )
        for result in results
    )

    assert all(
        result.asin != product.asin
        for result in results
    )


def test_recommendations_are_sorted(
    repository,
    recommendation_service,
):

    product = repository.get_all_products()[0]

    results = (
        recommendation_service
        .get_recommendations(
            product.asin,
            top_k=5,
        )
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
    repository,
    recommendation_service,
):

    product = repository.get_all_products()[0]

    first = (
        recommendation_service
        .get_recommendations(
            product.asin,
            top_k=5,
        )
    )

    second = (
        recommendation_service
        .get_recommendations(
            product.asin,
            top_k=5,
        )
    )

    assert first == second


def test_invalid_product_asin(
    recommendation_service,
):

    with pytest.raises(
        ValueError,
        match="was not found",
    ):

        recommendation_service.get_recommendations(
            "INVALID_ASIN_12345"
        )


def test_invalid_top_k(
    repository,
    recommendation_service,
):

    product = repository.get_all_products()[0]

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):

        recommendation_service.get_recommendations(
            product.asin,
            top_k=0,
        )