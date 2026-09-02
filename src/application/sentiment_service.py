from collections import Counter
from pathlib import Path

from src.application.product_repository import ProductRepository
from src.features.review_sentiment.models.sentiment_result import (
    SentimentResult,
)
from src.features.review_sentiment.review_sentiment_analyzer import (
    ReviewSentimentAnalyzer,
)
from src.features.review_sentiment.sentiment_predictor import (
    SentimentPredictor,
)


class SentimentService:
    """
    Application-level service for Feature B.

    Provides sentiment analysis and summaries by:
    - individual product
    - category (search_keyword)
    """

    def __init__(
        self,
        repository: ProductRepository,
        model_dir: str | Path,
    ) -> None:

        self.repository = repository

        predictor = SentimentPredictor(
            model_dir=model_dir
        )

        self.analyzer = ReviewSentimentAnalyzer(
            predictor=predictor
        )

    # ---------------------------------------------------------
    # Single Product Summary
    # ---------------------------------------------------------

    def get_product_sentiment_summary(
        self,
        asin: str,
    ) -> dict:
        """
        Analyze all reviews for a single product and return
        sentiment counts and percentages.
        """

        product = self.repository.get_product_by_asin(
            asin
        )

        results = self.analyzer.analyze_reviews(
            product.reviews
        )

        return self._build_summary(
            results=results,
            identifier=asin,
            identifier_type="asin",
        )

    # ---------------------------------------------------------
    # Category Summary
    # Category = search_keyword
    # ---------------------------------------------------------

    def get_category_sentiment_summary(
        self,
        search_keyword: str,
    ) -> dict:
        """
        Analyze reviews from all products belonging to a
        search keyword category.
        """

        products = (
            self.repository.get_products_by_search_keyword(
                search_keyword
            )
        )

        all_reviews = []

        for product in products:
            all_reviews.extend(
                product.reviews
            )

        results = self.analyzer.analyze_reviews(
            all_reviews
        )

        return self._build_summary(
            results=results,
            identifier=search_keyword,
            identifier_type="search_keyword",
        )

    # ---------------------------------------------------------
    # Individual Product Review Results
    # ---------------------------------------------------------

    def get_product_review_sentiments(
        self,
        asin: str,
    ) -> list[dict]:
        """
        Return sentiment prediction for every review of
        a product.
        """

        product = self.repository.get_product_by_asin(
            asin
        )

        results = self.analyzer.analyze_reviews(
            product.reviews
        )

        output = []

        for review, result in zip(
            product.reviews,
            results,
        ):
            output.append({
                "reviewer_name": review.reviewer_name,
                "review_title": review.review_title,
                "review_description": (
                    review.review_description
                ),
                "star_rating": review.star_rating,
                "sentiment": result.sentiment,
                "negative_probability": (
                    result.negative_probability
                ),
                "neutral_probability": (
                    result.neutral_probability
                ),
                "positive_probability": (
                    result.positive_probability
                ),
            })

        return output

    # ---------------------------------------------------------
    # Helper
    # ---------------------------------------------------------

    @staticmethod
    def _build_summary(
        results: list[SentimentResult],
        identifier: str,
        identifier_type: str,
    ) -> dict:
        """
        Build a standardized sentiment summary.
        """

        total_reviews = len(results)

        counts = Counter(
            result.sentiment.lower()
            for result in results
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
                negative_count / total_reviews * 100,
                2,
            )

            neutral_percentage = round(
                neutral_count / total_reviews * 100,
                2,
            )

            positive_percentage = round(
                positive_count / total_reviews * 100,
                2,
            )

        return {
            identifier_type: identifier,
            "total_reviews": total_reviews,
            "negative": {
                "count": negative_count,
                "percentage": negative_percentage,
            },
            "neutral": {
                "count": neutral_count,
                "percentage": neutral_percentage,
            },
            "positive": {
                "count": positive_count,
                "percentage": positive_percentage,
            },
        }