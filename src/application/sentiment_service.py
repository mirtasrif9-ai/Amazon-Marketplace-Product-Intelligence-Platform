from collections import Counter
from pathlib import Path

from src.application.product_repository import ProductRepository
from src.features.review_sentiment.models.sentiment_result import (
    SentimentResult,
)
from src.features.review_sentiment.sentiment_predictor import (
    SentimentPredictor,
)


class SentimentService:
    """Application-level service for Feature B.

    Provides:
    - Precomputed sentiment analytics for existing products
    - Category-level sentiment aggregation
    - Individual review sentiment exploration
    - Real-time sentiment inference for new review text
    """

    def __init__(
        self,
        repository: ProductRepository,
        model_dir: str | Path | None = None,
    ) -> None:
        self.repository = repository
        self.model_dir = (
            Path(model_dir)
            if model_dir is not None
            else None
        )
        self._predictor = None

    # ---------------------------------------------------------
    # Precomputed Product Summary
    # ---------------------------------------------------------

    def get_product_sentiment_summary(
        self,
        asin: str,
    ) -> dict:
        """Return the precomputed sentiment summary for a product."""

        product = self.repository.get_product_by_asin(asin)

        if product.sentiment_summary:
            summary = product.sentiment_summary

            return {
                "asin": product.asin,
                "total_reviews": summary.get(
                    "total_reviews",
                    len(product.reviews),
                ),
                "positive": {
                    "count": summary.get(
                        "positive_count",
                        0,
                    ),
                    "percentage": summary.get(
                        "positive_percentage",
                        0.0,
                    ),
                },
                "neutral": {
                    "count": summary.get(
                        "neutral_count",
                        0,
                    ),
                    "percentage": summary.get(
                        "neutral_percentage",
                        0.0,
                    ),
                },
                "negative": {
                    "count": summary.get(
                        "negative_count",
                        0,
                    ),
                    "percentage": summary.get(
                        "negative_percentage",
                        0.0,
                    ),
                },
                "overall_sentiment": summary.get(
                    "overall_sentiment"
                ),
                "sentiment_score": summary.get(
                    "sentiment_score"
                ),
            }

        return self._build_summary_from_reviews(
            reviews=product.reviews,
            identifier=product.asin,
            identifier_type="asin",
        )

    # ---------------------------------------------------------
    # Precomputed Category Summary
    # ---------------------------------------------------------

    def get_category_sentiment_summary(
        self,
        search_keyword: str,
    ) -> dict:
        """Aggregate persisted review sentiment predictions
        across all products in a search keyword category.
        """

        products = (
            self.repository.get_products_by_search_keyword(
                search_keyword
            )
        )

        all_reviews = []

        for product in products:
            all_reviews.extend(product.reviews)

        return self._build_summary_from_reviews(
            reviews=all_reviews,
            identifier=search_keyword,
            identifier_type="search_keyword",
        )

    # ---------------------------------------------------------
    # Individual Product Review Sentiments
    # ---------------------------------------------------------

    def get_product_review_sentiments(
        self,
        asin: str,
    ) -> list[dict]:
        """Return persisted sentiment predictions for every
        review of a product.
        """

        product = self.repository.get_product_by_asin(asin)

        output = []

        for review in product.reviews:
            probabilities = (
                review.sentiment_probabilities or {}
            )

            output.append(
                {
                    "reviewer_name": review.reviewer_name,
                    "review_title": review.review_title,
                    "review_description": (
                        review.review_description
                    ),
                    "star_rating": review.star_rating,
                    "sentiment": review.sentiment,
                    "negative_probability": probabilities.get(
                        "negative",
                        0.0,
                    ),
                    "neutral_probability": probabilities.get(
                        "neutral",
                        0.0,
                    ),
                    "positive_probability": probabilities.get(
                        "positive",
                        0.0,
                    ),
                }
            )

        return output

    # ---------------------------------------------------------
    # Real-Time Sentiment Prediction
    # ---------------------------------------------------------

    def predict_review_sentiment(
        self,
        review_text: str,
    ) -> SentimentResult:
        """Run real-time sentiment inference for new review text."""

        review_text = str(review_text).strip()

        if not review_text:
            raise ValueError(
                "Review text cannot be empty."
            )

        predictor = self._get_predictor()

        prediction = predictor.predict(
            review_text
        )

        probabilities = prediction.get(
            "probabilities",
            {},
        )

        return SentimentResult(
            sentiment=str(
                prediction.get(
                    "sentiment",
                    "neutral",
                )
            ),
            negative_probability=float(
                probabilities.get(
                    "negative",
                    0.0,
                )
            ),
            neutral_probability=float(
                probabilities.get(
                    "neutral",
                    0.0,
                )
            ),
            positive_probability=float(
                probabilities.get(
                    "positive",
                    0.0,
                )
            ),
        )

    def _get_predictor(
        self,
    ) -> SentimentPredictor:
        """Lazily initialize the trained sentiment model."""

        if self._predictor is None:
            if self.model_dir is None:
                raise ValueError(
                    "model_dir must be provided for "
                    "real-time sentiment prediction."
                )

            self._predictor = SentimentPredictor(
                model_dir=self.model_dir
            )

        return self._predictor

    # ---------------------------------------------------------
    # Analytics Helper
    # ---------------------------------------------------------

    @staticmethod
    def _build_summary_from_reviews(
        reviews,
        identifier: str,
        identifier_type: str,
    ) -> dict:
        """Build a standardized sentiment summary."""

        total_reviews = len(reviews)

        counts = Counter(
            str(review.sentiment).lower()
            for review in reviews
            if review.sentiment
        )

        negative_count = counts.get(
            "negative",
            0,
        )

        neutral_count = counts.get(
            "neutral",
            0,
        )

        positive_count = counts.get(
            "positive",
            0,
        )

        if total_reviews == 0:
            negative_percentage = 0.0
            neutral_percentage = 0.0
            positive_percentage = 0.0

        else:
            negative_percentage = round(
                negative_count
                / total_reviews
                * 100,
                2,
            )

            neutral_percentage = round(
                neutral_count
                / total_reviews
                * 100,
                2,
            )

            positive_percentage = round(
                positive_count
                / total_reviews
                * 100,
                2,
            )

        result = {
            "total_reviews": total_reviews,
            "positive": {
                "count": positive_count,
                "percentage": positive_percentage,
            },
            "neutral": {
                "count": neutral_count,
                "percentage": neutral_percentage,
            },
            "negative": {
                "count": negative_count,
                "percentage": negative_percentage,
            },
        }

        if identifier_type == "asin":
            result["asin"] = identifier

        elif identifier_type == "search_keyword":
            result["search_keyword"] = identifier

        return result