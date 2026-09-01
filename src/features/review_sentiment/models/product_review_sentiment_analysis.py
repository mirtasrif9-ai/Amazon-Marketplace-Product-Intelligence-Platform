from dataclasses import dataclass

from src.features.review_sentiment.models.product_sentiment_summary import (
    ProductSentimentSummary,
)
from src.features.review_sentiment.models.sentiment_result import (
    SentimentResult,
)


@dataclass(frozen=True)
class ProductReviewSentimentAnalysis:
    """
    Complete sentiment analysis result for a product.
    """

    review_sentiments: list[SentimentResult]
    summary: ProductSentimentSummary