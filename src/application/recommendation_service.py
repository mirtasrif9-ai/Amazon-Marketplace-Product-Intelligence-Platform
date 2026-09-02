from pathlib import Path

from src.application.product_repository import ProductRepository
from src.features.product_recommendation.models import (
    RecommendationResult,
)
from src.features.product_recommendation.recommendation_engine import (
    ProductRecommendationEngine,
)


class RecommendationService:
    """
    Application-level service for Feature A.

    Uses the existing trained recommendation engine and exposes
    a clean interface for the application layer.
    """

    def __init__(
        self,
        repository: ProductRepository,
        model_dir: str | Path,
    ) -> None:

        self.repository = repository

        self.engine = ProductRecommendationEngine(
            model_dir=model_dir
        )

    def get_recommendations(
        self,
        asin: str,
        top_k: int = 5,
    ) -> list[RecommendationResult]:
        """
        Return similar products for a product ASIN.
        """

        asin = str(asin).strip()

        if not asin:
            raise ValueError(
                "Product ASIN must be a non-empty string."
            )

        # Validate that the product exists in the
        # application's central dataset.
        self.repository.get_product_by_asin(asin)

        return self.engine.recommend(
            product_asin=asin,
            top_k=top_k,
        )