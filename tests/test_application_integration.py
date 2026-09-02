from pathlib import Path

import pytest

from src.application.product_repository import ProductRepository
from src.application.recommendation_service import RecommendationService
from src.application.sentiment_service import SentimentService
from src.features.price_tier.price_tier_classifier import PriceTierClassifier
from src.features.thumbnail_grouping.thumbnail_grouping_service import (
    ThumbnailGroupingService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_products.csv"
)

RECOMMENDATION_MODEL_DIR = (
    PROJECT_ROOT / "models" / "recommendation"
)

SENTIMENT_MODEL_DIR = (
    PROJECT_ROOT / "models" / "sentiment" / "final_sentiment_model"
)

PRICE_TIER_MODEL_PATH = (
    PROJECT_ROOT / "models" / "price_tier" / "price_tier_model.joblib"
)


@pytest.fixture(scope="module")
def repository():
    return ProductRepository(DATASET_PATH)


@pytest.fixture(scope="module")
def product(repository):
    return repository.get_all_products()[0]


def test_all_application_features_work_for_same_product(
    repository,
    product,
):
    # Feature A — Similar Product Recommendation
    recommendation_service = RecommendationService(
        repository=repository,
        model_dir=RECOMMENDATION_MODEL_DIR,
    )

    recommendations = recommendation_service.get_recommendations(
        product.asin,
        top_k=5,
    )

    assert len(recommendations) == 5
    assert all(
        recommendation.asin != product.asin
        for recommendation in recommendations
    )

    # Feature B — Review Sentiment
    sentiment_service = SentimentService(
        repository=repository,
        model_dir=SENTIMENT_MODEL_DIR,
    )

    sentiment_summary = (
        sentiment_service.get_product_sentiment_summary(
            product.asin
        )
    )

    assert sentiment_summary["asin"] == product.asin
    assert sentiment_summary["total_reviews"] == len(product.reviews)

    # Feature C — Thumbnail Grouping
    thumbnail_service = ThumbnailGroupingService(
        project_root=PROJECT_ROOT
    )

    thumbnail_group = thumbnail_service.get_product_group(
        product.asin
    )

    assert thumbnail_group is not None
    assert thumbnail_group.asin == product.asin
    assert thumbnail_group.group_size > 0

    # Feature D — Price Tier
    price_tier_classifier = PriceTierClassifier(
        model_path=PRICE_TIER_MODEL_PATH
    )

    price_tier_result = price_tier_classifier.predict(product)

    assert price_tier_result.asin == product.asin
    assert price_tier_result.price_tier in {
        "budget",
        "mid_range",
        "premium",
    }
    assert 0.0 <= price_tier_result.confidence <= 1.0


def test_category_sentiment_works_with_repository(repository):
    sentiment_service = SentimentService(
        repository=repository,
        model_dir=SENTIMENT_MODEL_DIR,
    )

    keyword = repository.get_search_keywords()[0]

    summary = sentiment_service.get_category_sentiment_summary(
        keyword
    )

    assert summary["search_keyword"] == keyword
    assert summary["total_reviews"] >= 0