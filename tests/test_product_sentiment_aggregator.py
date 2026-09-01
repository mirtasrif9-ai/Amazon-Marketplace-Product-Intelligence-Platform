from src.features.review_sentiment.models.sentiment_result import (
    SentimentResult,
)

from src.features.review_sentiment.product_sentiment_aggregator import (
    ProductSentimentAggregator,
)


def main():

    print("=" * 70)
    print("PRODUCT SENTIMENT AGGREGATOR TEST")
    print("=" * 70)

    sentiment_results = [
        SentimentResult(
            sentiment="positive",
            negative_probability=0.01,
            neutral_probability=0.02,
            positive_probability=0.97,
        ),
        SentimentResult(
            sentiment="positive",
            negative_probability=0.02,
            neutral_probability=0.03,
            positive_probability=0.95,
        ),
        SentimentResult(
            sentiment="positive",
            negative_probability=0.01,
            neutral_probability=0.05,
            positive_probability=0.94,
        ),
        SentimentResult(
            sentiment="neutral",
            negative_probability=0.20,
            neutral_probability=0.60,
            positive_probability=0.20,
        ),
        SentimentResult(
            sentiment="negative",
            negative_probability=0.90,
            neutral_probability=0.08,
            positive_probability=0.02,
        ),
    ]

    aggregator = ProductSentimentAggregator()

    summary = aggregator.aggregate(
        sentiment_results
    )

    print("\nPRODUCT SENTIMENT SUMMARY")
    print("-" * 70)

    print(
        f"Total Reviews: {summary.total_reviews}"
    )

    print(
        f"Positive: {summary.positive_count} "
        f"({summary.positive_percentage}%)"
    )

    print(
        f"Neutral: {summary.neutral_count} "
        f"({summary.neutral_percentage}%)"
    )

    print(
        f"Negative: {summary.negative_count} "
        f"({summary.negative_percentage}%)"
    )

    print(
        f"\nSentiment Score: "
        f"{summary.sentiment_score}"
    )

    print(
        f"Overall Sentiment: "
        f"{summary.overall_sentiment.upper()}"
    )


if __name__ == "__main__":
    main()