from pathlib import Path

from src.data_collection.models.review import Review

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
    print("REVIEW SENTIMENT ANALYZER TEST")
    print("=" * 70)

    predictor = SentimentPredictor(
        model_dir=MODEL_DIR
    )

    analyzer = ReviewSentimentAnalyzer(
        predictor=predictor
    )

    reviews = [
        Review(
            reviewer_name="Alice",
            star_rating=5.0,
            review_title="Amazing product!",
            review_description=(
                "Excellent quality and very easy to use. "
                "Highly recommended!"
            ),
        ),
        Review(
            reviewer_name="Bob",
            star_rating=3.0,
            review_title="It's okay",
            review_description=(
                "The product works as expected, "
                "but there is nothing particularly special."
            ),
        ),
        Review(
            reviewer_name="Charlie",
            star_rating=1.0,
            review_title="Terrible quality",
            review_description=(
                "Stopped working after two days. "
                "Very disappointed with this purchase."
            ),
        ),
    ]

    results = analyzer.analyze_reviews(
        reviews
    )

    print("\nRESULTS")
    print("=" * 70)

    for review, result in zip(
        reviews,
        results,
    ):

        print("\nReview:")
        print(
            f"Title: {review.review_title}"
        )

        print(
            f"Description: "
            f"{review.review_description}"
        )

        print(
            f"\nPredicted Sentiment: "
            f"{result.sentiment}"
        )

        print(
            f"Negative: "
            f"{result.negative_probability:.4f}"
        )

        print(
            f"Neutral: "
            f"{result.neutral_probability:.4f}"
        )

        print(
            f"Positive: "
            f"{result.positive_probability:.4f}"
        )


if __name__ == "__main__":
    main()