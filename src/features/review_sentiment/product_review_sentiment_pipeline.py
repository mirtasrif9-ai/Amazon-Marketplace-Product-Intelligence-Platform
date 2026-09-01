from src.data_collection.models.product import Product
from src.data_collection.models.review import Review

from src.features.review_sentiment.models.product_review_sentiment_analysis import (
    ProductReviewSentimentAnalysis,
)
from src.features.review_sentiment.product_sentiment_aggregator import (
    ProductSentimentAggregator,
)
from src.features.review_sentiment.review_sentiment_analyzer import (
    ReviewSentimentAnalyzer,
)


class ProductReviewSentimentPipeline:
    """
    End-to-end sentiment analysis pipeline.

    Supports:
    1. Product-level analysis
    2. Direct review-list analysis
    """

    def __init__(
        self,
        review_analyzer: ReviewSentimentAnalyzer,
        sentiment_aggregator: ProductSentimentAggregator,
    ) -> None:
        self.review_analyzer = review_analyzer
        self.sentiment_aggregator = sentiment_aggregator

    def analyze_product(
        self,
        product: Product,
        batch_size: int = 32,
    ) -> ProductReviewSentimentAnalysis:
        """
        Analyze all reviews belonging to a Product object.
        """

        return self.analyze_reviews(
            reviews=product.reviews,
            batch_size=batch_size,
        )

    def analyze_reviews(
        self,
        reviews: list[Review],
        batch_size: int = 32,
    ) -> ProductReviewSentimentAnalysis:
        """
        Analyze a list of reviews and generate both individual
        sentiment results and an aggregated sentiment summary.
        """

        review_sentiments = (
            self.review_analyzer.analyze_reviews(
                reviews=reviews,
                batch_size=batch_size,
            )
        )

        summary = self.sentiment_aggregator.aggregate(
            review_sentiments
        )

        return ProductReviewSentimentAnalysis(
            review_sentiments=review_sentiments,
            summary=summary,
        )