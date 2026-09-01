import json
from collections import Counter
from pathlib import Path
from typing import Any


class SentimentDatasetAnalyzer:
    """
    Provides exploratory analysis utilities for the
    sentiment-enriched product dataset.
    """

    def __init__(
        self,
        input_path: str | Path,
    ) -> None:
        self.input_path = Path(input_path)

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.input_path}"
            )

        self.products = self._load_products()

    def _load_products(
        self,
    ) -> list[dict[str, Any]]:
        """
        Load sentiment-enriched products JSON.
        """

        with self.input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            products = json.load(file)

        if not isinstance(products, list):
            raise ValueError(
                "Expected dataset to be a list "
                "of products."
            )

        return products

    def get_product_sentiment_distribution(
        self,
    ) -> dict[str, dict[str, float]]:
        """
        Calculate product-level overall sentiment
        counts and percentages.
        """

        sentiments = [
            product["sentiment_summary"][
                "overall_sentiment"
            ]
            for product in self.products
            if "sentiment_summary" in product
        ]

        counts = Counter(sentiments)

        total = sum(counts.values())

        distribution = {}

        for sentiment in [
            "positive",
            "neutral",
            "negative",
        ]:

            count = counts.get(sentiment, 0)

            percentage = (
                (count / total * 100)
                if total > 0
                else 0.0
            )

            distribution[sentiment] = {
                "count": count,
                "percentage": percentage,
            }

        return distribution

    def get_sentiment_score_statistics(
        self,
    ) -> dict[str, float]:
        """
        Calculate descriptive statistics for
        product sentiment scores.
        """

        scores = [
            product["sentiment_summary"][
                "sentiment_score"
            ]
            for product in self.products
            if "sentiment_summary" in product
        ]

        if not scores:
            return {}

        scores = sorted(scores)

        count = len(scores)

        mean_score = sum(scores) / count

        median_score = self._calculate_median(
            scores
        )

        return {
            "count": count,
            "minimum": scores[0],
            "maximum": scores[-1],
            "mean": mean_score,
            "median": median_score,
        }

    def get_top_products_by_sentiment(
        self,
        top_n: int = 10,
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Return products ranked by sentiment score.

        ascending=False → most positive
        ascending=True  → most negative
        """

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        products_with_scores = [
            product
            for product in self.products
            if "sentiment_summary" in product
        ]

        sorted_products = sorted(
            products_with_scores,
            key=lambda product: product[
                "sentiment_summary"
            ]["sentiment_score"],
            reverse=not ascending,
        )

        results = []

        for product in sorted_products[:top_n]:

            summary = product[
                "sentiment_summary"
            ]

            results.append(
                {
                    "asin": product.get("asin"),
                    "title": product.get("title"),
                    "search_keyword": product.get(
                        "search_keyword"
                    ),
                    "average_rating": product.get(
                        "average_rating"
                    ),
                    "review_count": product.get(
                        "review_count"
                    ),
                    "sentiment_score": summary.get(
                        "sentiment_score"
                    ),
                    "overall_sentiment": summary.get(
                        "overall_sentiment"
                    ),
                    "positive_percentage": summary.get(
                        "positive_percentage"
                    ),
                    "neutral_percentage": summary.get(
                        "neutral_percentage"
                    ),
                    "negative_percentage": summary.get(
                        "negative_percentage"
                    ),
                }
            )

        return results

    @staticmethod
    def _calculate_median(
        values: list[float],
    ) -> float:
        """
        Calculate median of sorted values.
        """

        length = len(values)

        middle = length // 2

        if length % 2 == 1:
            return values[middle]

        return (
            values[middle - 1]
            + values[middle]
        ) / 2

    def get_review_sentiment_distribution(
        self,
    ) -> dict[str, dict[str, float]]:
        """
        Calculate sentiment distribution across
        all individual reviews.
        """

        sentiments = []

        for product in self.products:

            for review in product.get(
                "reviews",
                [],
            ):

                sentiment = review.get(
                    "sentiment"
                )

                if sentiment in {
                    "positive",
                    "neutral",
                    "negative",
                }:
                    sentiments.append(sentiment)

        counts = Counter(sentiments)

        total = len(sentiments)

        distribution = {}

        for sentiment in [
            "positive",
            "neutral",
            "negative",
        ]:

            count = counts.get(
                sentiment,
                0,
            )

            percentage = (
                count / total * 100
                if total > 0
                else 0.0
            )

            distribution[sentiment] = {
                "count": count,
                "percentage": percentage,
            }

        return distribution

    def get_review_count_statistics(
        self,
    ) -> dict[str, float]:
        """
        Calculate statistics for the number of
        scraped reviews per product.
        """

        review_counts = [
            len(product.get("reviews", []))
            for product in self.products
        ]

        if not review_counts:
            return {}

        review_counts = sorted(review_counts)

        count = len(review_counts)

        mean = sum(review_counts) / count

        median = self._calculate_median(
            review_counts
        )

        return {
            "products": count,
            "total_reviews": sum(review_counts),
            "minimum_reviews_per_product": review_counts[0],
            "maximum_reviews_per_product": review_counts[-1],
            "mean_reviews_per_product": mean,
            "median_reviews_per_product": median,
        }