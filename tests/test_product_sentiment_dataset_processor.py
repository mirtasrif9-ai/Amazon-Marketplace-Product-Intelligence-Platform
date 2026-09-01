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

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_products.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "products_with_sentiment_test.json"
)

FAILED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "failed_sentiment_products_test.json"
)


def main():

    print("=" * 70)
    print("PRODUCT SENTIMENT DATASET PROCESSOR TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Initialize dependencies
    # --------------------------------------------------

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
        progress_interval=2,
    )

    # --------------------------------------------------
    # Process small dataset subset
    # --------------------------------------------------

    batch_result = processor.process_dataset(
        input_path=DATASET_PATH,
        limit=5,
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    processor.save_results(
        batch_result=batch_result,
        output_path=OUTPUT_PATH,
        failed_output_path=FAILED_OUTPUT_PATH,
    )

    # --------------------------------------------------
    # Validate output files
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("OUTPUT VALIDATION")
    print("=" * 70)

    print(f"Output exists: {OUTPUT_PATH.exists()}")
    print(
        f"Failed output exists: "
        f"{FAILED_OUTPUT_PATH.exists()}"
    )

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_products = json.load(file)

    print(
        f"\nProducts saved: {len(saved_products)}"
    )

    print(
        f"Expected successful products: "
        f"{batch_result.successful_count}"
    )

    assert (
        len(saved_products)
        == batch_result.successful_count
    )

    # --------------------------------------------------
    # Validate first product structure
    # --------------------------------------------------

    if saved_products:

        first_product = saved_products[0]

        print("\nFIRST PRODUCT VALIDATION")
        print("-" * 70)

        print(
            "ASIN:",
            first_product.get("asin"),
        )

        print(
            "Title:",
            first_product.get("title"),
        )

        print(
            "Review count:",
            len(
                first_product.get(
                    "reviews",
                    [],
                )
            ),
        )

        print(
            "Has sentiment summary:",
            "sentiment_summary"
            in first_product,
        )

        summary = first_product.get(
            "sentiment_summary",
            {},
        )

        print(
            "Overall sentiment:",
            summary.get(
                "overall_sentiment"
            ),
        )

        print(
            "Sentiment score:",
            summary.get(
                "sentiment_score"
            ),
        )

        # Assertions

        assert "sentiment_summary" in first_product

        assert (
            "overall_sentiment"
            in summary
        )

        assert (
            "sentiment_score"
            in summary
        )

        if first_product.get("reviews"):

            first_review = (
                first_product["reviews"][0]
            )

            assert "sentiment" in first_review

            assert (
                "sentiment_probabilities"
                in first_review
            )

            print(
                "\nFirst review sentiment:",
                first_review["sentiment"],
            )

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()