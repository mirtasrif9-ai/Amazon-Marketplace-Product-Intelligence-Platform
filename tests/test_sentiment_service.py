from pathlib import Path

import pytest

from src.application.product_repository import ProductRepository
from src.application.sentiment_service import SentimentService


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
    / "sentiment"
    / "final_sentiment_model"
)


@pytest.fixture(scope="module")
def repository():

    return ProductRepository(
        DATASET_PATH
    )


@pytest.fixture(scope="module")
def sentiment_service(repository):

    return SentimentService(
        repository=repository,
        model_dir=MODEL_DIR,
    )


def test_sentiment_model_directory_exists():

    assert MODEL_DIR.exists(), (
        f"Sentiment model directory not found:\n"
        f"{MODEL_DIR}"
    )

    assert (
        MODEL_DIR / "model.safetensors"
    ).exists(), (
        "model.safetensors not found."
    )

    assert (
        MODEL_DIR / "thresholds.json"
    ).exists(), (
        "thresholds.json not found."
    )

    assert (
        MODEL_DIR / "label_mapping.json"
    ).exists(), (
        "label_mapping.json not found."
    )


def test_product_sentiment_summary(
    repository,
    sentiment_service,
):

    # Find a product that actually has reviews
    product = next(
        product
        for product in repository.get_all_products()
        if product.reviews
    )

    summary = (
        sentiment_service
        .get_product_sentiment_summary(
            product.asin
        )
    )

    assert summary["asin"] == product.asin

    assert (
        summary["total_reviews"]
        == len(product.reviews)
    )

    for sentiment in [
        "negative",
        "neutral",
        "positive",
    ]:

        assert sentiment in summary

        assert "count" in summary[sentiment]

        assert "percentage" in summary[sentiment]

    total_count = sum(
        summary[sentiment]["count"]
        for sentiment in [
            "negative",
            "neutral",
            "positive",
        ]
    )

    assert (
        total_count
        == summary["total_reviews"]
    )


def test_category_sentiment_summary(
    repository,
    sentiment_service,
):

    keywords = (
        repository.get_search_keywords()
    )

    keyword = keywords[0]

    products = (
        repository
        .get_products_by_search_keyword(
            keyword
        )
    )

    expected_review_count = sum(
        len(product.reviews)
        for product in products
    )

    summary = (
        sentiment_service
        .get_category_sentiment_summary(
            keyword
        )
    )

    assert (
        summary["search_keyword"]
        == keyword
    )

    assert (
        summary["total_reviews"]
        == expected_review_count
    )

    total_count = sum(
        summary[sentiment]["count"]
        for sentiment in [
            "negative",
            "neutral",
            "positive",
        ]
    )

    assert (
        total_count
        == summary["total_reviews"]
    )


def test_individual_review_sentiments(
    repository,
    sentiment_service,
):

    product = next(
        product
        for product in repository.get_all_products()
        if product.reviews
    )

    results = (
        sentiment_service
        .get_product_review_sentiments(
            product.asin
        )
    )

    assert (
        len(results)
        == len(product.reviews)
    )

    valid_sentiments = {
        "negative",
        "neutral",
        "positive",
    }

    for result in results:

        assert (
            result["sentiment"]
            in valid_sentiments
        )

        assert (
            0.0
            <= result["negative_probability"]
            <= 1.0
        )

        assert (
            0.0
            <= result["neutral_probability"]
            <= 1.0
        )

        assert (
            0.0
            <= result["positive_probability"]
            <= 1.0
        )