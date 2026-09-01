from pathlib import Path

from src.data_collection.models.review import Review

from src.features.review_sentiment.product_review_sentiment_pipeline import (
    ProductReviewSentimentPipeline,
)
from src.features.review_sentiment.product_sentiment_aggregator import (
    ProductSentimentAggregator,
)
from src.features.review_sentiment.review_sentiment_analyzer import (
    ReviewSentimentAnalyzer,
)
from src.features.review_sentiment.sentiment_predictor import (
    SentimentPredictor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "sentiment"
    / "final_sentiment_model"
)


def main():
    print("=" * 70)
    print("DIRECT REVIEW-LIST SENTIMENT PIPELINE TEST")
    print("=" * 70)

    predictor = SentimentPredictor(
        model_dir=MODEL_DIR
    )

    review_analyzer = ReviewSentimentAnalyzer(
        predictor=predictor
    )

    aggregator = ProductSentimentAggregator()

    pipeline = ProductReviewSentimentPipeline(
        review_analyzer=review_analyzer,
        sentiment_aggregator=aggregator,
    )

    reviews = [
        Review(
            reviewer_name="Alice",
            star_rating=5.0,
            review_title="Excellent!",
            review_description=(
                "Amazing quality and works perfectly."
            ),
        ),
        Review(
            reviewer_name="Bob",
            star_rating=1.0,
            review_title="Terrible",
            review_description=(
                "Stopped working almost immediately."
            ),
        ),
        Review(
            reviewer_name="Charlie",
            star_rating=4.0,
            review_title="Good value",
            review_description=(
                "Works well and is worth the price."
            ),
        ),
    ]

    analysis = pipeline.analyze_reviews(
        reviews=reviews,
        batch_size=32,
    )

    print("\nINDIVIDUAL SENTIMENTS")
    print("-" * 70)

    for review, result in zip(
        reviews,
        analysis.review_sentiments,
    ):
        print(
            f"{review.review_title}: "
            f"{result.sentiment.upper()}"
        )

    summary = analysis.summary

    print("\nPRODUCT-LEVEL SUMMARY")
    print("-" * 70)
    print(f"Total Reviews: {summary.total_reviews}")
    print(f"Positive: {summary.positive_count}")
    print(f"Neutral: {summary.neutral_count}")
    print(f"Negative: {summary.negative_count}")
    print(f"Sentiment Score: {summary.sentiment_score}")
    print(
        f"Overall Sentiment: "
        f"{summary.overall_sentiment.upper()}"
    )


if __name__ == "__main__":
    main()