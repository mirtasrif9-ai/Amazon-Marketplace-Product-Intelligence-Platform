from pathlib import Path
import json

from src.features.review_sentiment.product_review_sentiment_pipeline import (
    ProductReviewSentimentPipeline,
)
from src.features.review_sentiment.product_sentiment_aggregator import (
    ProductSentimentAggregator,
)
from src.features.review_sentiment.product_sentiment_batch_processor import (
    ProductSentimentBatchProcessor,
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
    print("PRODUCT SENTIMENT BATCH PROCESSOR TEST")
    print("=" * 70)

    # Initialize dependencies once

    predictor = SentimentPredictor(
        model_dir=MODEL_DIR
    )

    analyzer = ReviewSentimentAnalyzer(
        predictor=predictor
    )

    aggregator = ProductSentimentAggregator()

    pipeline = ProductReviewSentimentPipeline(
        review_analyzer=analyzer,
        sentiment_aggregator=aggregator,
    )

    processor = ProductSentimentBatchProcessor(
        sentiment_pipeline=pipeline,
        batch_size=32,
    )

    # Simulate one row from cleaned_products.csv

    product = {
        "product_number": 1,
        "asin": "TEST123456",
        "search_keyword": "wireless earbuds",
        "title": "Test Wireless Earbuds",
        "price": 49.99,
        "reviews": json.dumps([
            {
                "reviewer_name": "Alice",
                "star_rating": 5.0,
                "review_title": "Excellent!",
                "review_description": (
                    "Amazing sound quality and "
                    "very comfortable."
                ),
            },
            {
                "reviewer_name": "Bob",
                "star_rating": 1.0,
                "review_title": "Terrible",
                "review_description": (
                    "Stopped working after two days."
                ),
            },
            {
                "reviewer_name": "Charlie",
                "star_rating": 4.0,
                "review_title": "Good",
                "review_description": (
                    "Works well and worth the price."
                ),
            },
        ]),
    }

    result = processor.process_product(
        product
    )

    print("\nENRICHED REVIEWS")
    print("-" * 70)

    for review in result["reviews"]:

        print(
            f"\nTitle: "
            f"{review['review_title']}"
        )

        print(
            f"Sentiment: "
            f"{review['sentiment'].upper()}"
        )

        print(
            "Probabilities: "
            f"{review['sentiment_probabilities']}"
        )

    print("\n" + "=" * 70)
    print("SENTIMENT SUMMARY")
    print("=" * 70)

    for key, value in result[
        "sentiment_summary"
    ].items():

        print(f"{key}: {value}")


if __name__ == "__main__":
    main()