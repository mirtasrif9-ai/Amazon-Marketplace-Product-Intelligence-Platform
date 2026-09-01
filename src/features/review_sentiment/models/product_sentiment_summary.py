from dataclasses import dataclass


@dataclass(frozen=True)
class ProductSentimentSummary:
    """
    Aggregated sentiment analysis result for a product.
    """

    total_reviews: int

    positive_count: int
    neutral_count: int
    negative_count: int

    positive_percentage: float
    neutral_percentage: float
    negative_percentage: float

    overall_sentiment: str
    sentiment_score: float