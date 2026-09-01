from pathlib import Path

from src.data_collection.models.product import Product
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
    print("END-TO-END PRODUCT SENTIMENT PIPELINE TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # Initialize ML Predictor
    # -----------------------------------------------------

    predictor = SentimentPredictor(
        model_dir=MODEL_DIR
    )

    # -----------------------------------------------------
    # Initialize Review Analyzer
    # -----------------------------------------------------

    review_analyzer = ReviewSentimentAnalyzer(
        predictor=predictor
    )

    # -----------------------------------------------------
    # Initialize Aggregator
    # -----------------------------------------------------

    sentiment_aggregator = (
        ProductSentimentAggregator()
    )

    # -----------------------------------------------------
    # Initialize Pipeline
    # -----------------------------------------------------

    pipeline = ProductReviewSentimentPipeline(
        review_analyzer=review_analyzer,
        sentiment_aggregator=sentiment_aggregator,
    )

    # -----------------------------------------------------
    # Create Test Product
    # -----------------------------------------------------

    product = Product(
        product_number=1,
        search_keyword="wireless earbuds",
        asin="TEST123456",
        title="Test Wireless Earbuds",
        product_url="https://www.amazon.com/dp/TEST123456",
        description="Wireless earbuds for testing.",
        brand="TestBrand",
        price="$49.99",
        image="",
        review_count=5,
        average_rating=4.2,
        video_url="",
        reviews=[
            Review(
                reviewer_name="Alice",
                star_rating=5.0,
                review_title="Amazing!",
                review_description=(
                    "Excellent sound quality and very "
                    "comfortable to wear."
                ),
            ),
            Review(
                reviewer_name="Bob",
                star_rating=5.0,
                review_title="Great product",
                review_description=(
                    "Battery life is excellent and setup "
                    "was very easy."
                ),
            ),
            Review(
                reviewer_name="Charlie",
                star_rating=4.0,
                review_title="Pretty good",
                review_description=(
                    "Works well for the price. "
                    "A few minor issues."
                ),
            ),
            Review(
                reviewer_name="David",
                star_rating=3.0,
                review_title="It's okay",
                review_description=(
                    "The product works but nothing "
                    "particularly special."
                ),
            ),
            Review(
                reviewer_name="Eve",
                star_rating=1.0,
                review_title="Disappointed",
                review_description=(
                    "Stopped working after two days. "
                    "Poor build quality."
                ),
            ),
        ],
    )

    # -----------------------------------------------------
    # Run Complete Pipeline
    # -----------------------------------------------------

    analysis = pipeline.analyze_product(
        product
    )

    # -----------------------------------------------------
    # Print Individual Results
    # -----------------------------------------------------

    print("\nINDIVIDUAL REVIEW SENTIMENTS")
    print("-" * 70)

    for index, (
        review,
        sentiment_result,
    ) in enumerate(
        zip(
            product.reviews,
            analysis.review_sentiments,
        ),
        start=1,
    ):

        print(f"\nReview {index}")
        print(f"Title: {review.review_title}")
        print(
            f"Sentiment: "
            f"{sentiment_result.sentiment.upper()}"
        )

    # -----------------------------------------------------
    # Print Product Summary
    # -----------------------------------------------------

    summary = analysis.summary

    print("\n" + "=" * 70)
    print("PRODUCT SENTIMENT SUMMARY")
    print("=" * 70)

    print(f"Total Reviews: {summary.total_reviews}")

    print(
        f"Positive: "
        f"{summary.positive_count} "
        f"({summary.positive_percentage}%)"
    )

    print(
        f"Neutral: "
        f"{summary.neutral_count} "
        f"({summary.neutral_percentage}%)"
    )

    print(
        f"Negative: "
        f"{summary.negative_count} "
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