from src.features.review_sentiment.models.sentiment_result import (
    SentimentResult,
)
from src.features.review_sentiment.models.product_sentiment_summary import (
    ProductSentimentSummary,
)


class ProductSentimentAggregator:
    """
    Aggregates individual review sentiment results into
    a product-level sentiment summary.
    """

    POSITIVE_THRESHOLD = 0.20
    NEGATIVE_THRESHOLD = -0.20

    def aggregate(
        self,
        sentiment_results: list[SentimentResult],
    ) -> ProductSentimentSummary:
        """
        Generate a sentiment summary for a product.
        """

        total_reviews = len(sentiment_results)

        if total_reviews == 0:
            return self._create_empty_summary()

        positive_count = sum(
            result.sentiment == "positive"
            for result in sentiment_results
        )

        neutral_count = sum(
            result.sentiment == "neutral"
            for result in sentiment_results
        )

        negative_count = sum(
            result.sentiment == "negative"
            for result in sentiment_results
        )

        positive_percentage = (
            positive_count / total_reviews
        ) * 100

        neutral_percentage = (
            neutral_count / total_reviews
        ) * 100

        negative_percentage = (
            negative_count / total_reviews
        ) * 100

        sentiment_score = (
            positive_count - negative_count
        ) / total_reviews

        overall_sentiment = (
            self._determine_overall_sentiment(
                sentiment_score
            )
        )

        return ProductSentimentSummary(
            total_reviews=total_reviews,

            positive_count=positive_count,
            neutral_count=neutral_count,
            negative_count=negative_count,

            positive_percentage=round(
                positive_percentage,
                2,
            ),
            neutral_percentage=round(
                neutral_percentage,
                2,
            ),
            negative_percentage=round(
                negative_percentage,
                2,
            ),

            overall_sentiment=overall_sentiment,
            sentiment_score=round(
                sentiment_score,
                4,
            ),
        )

    def _determine_overall_sentiment(
        self,
        sentiment_score: float,
    ) -> str:
        """
        Determine overall product sentiment based on
        the aggregated sentiment score.
        """

        if sentiment_score >= self.POSITIVE_THRESHOLD:
            return "positive"

        if sentiment_score <= self.NEGATIVE_THRESHOLD:
            return "negative"

        return "neutral"

    @staticmethod
    def _create_empty_summary() -> ProductSentimentSummary:
        """
        Return a default summary when no reviews exist.
        """

        return ProductSentimentSummary(
            total_reviews=0,

            positive_count=0,
            neutral_count=0,
            negative_count=0,

            positive_percentage=0.0,
            neutral_percentage=0.0,
            negative_percentage=0.0,

            overall_sentiment="neutral",
            sentiment_score=0.0,
        )