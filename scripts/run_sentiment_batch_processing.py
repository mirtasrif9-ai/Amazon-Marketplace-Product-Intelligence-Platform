import logging
import sys
import time
from pathlib import Path


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


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


# ============================================================
# PATH CONFIGURATION
# ============================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "sentiment"
    / "final_sentiment_model"
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_products.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "products_with_sentiment.json"
)

FAILED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "failed_sentiment_products.json"
)


# ============================================================
# PROCESSING CONFIGURATION
# ============================================================

REVIEW_BATCH_SIZE = 32

PROGRESS_INTERVAL = 50


# ============================================================
# LOGGING
# ============================================================

def configure_logging() -> None:
    """
    Configure application logging.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )


logger = logging.getLogger(__name__)


# ============================================================
# PIPELINE FACTORY
# ============================================================

def build_processor() -> ProductSentimentBatchProcessor:
    """
    Build and initialize the complete sentiment pipeline.
    """

    logger.info(
        "Initializing sentiment predictor..."
    )

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
        batch_size=REVIEW_BATCH_SIZE,
        progress_interval=PROGRESS_INTERVAL,
    )

    return processor


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    configure_logging()

    start_time = time.perf_counter()

    print()
    print("=" * 70)
    print("AMAZON PRODUCT SENTIMENT BATCH PROCESSING")
    print("=" * 70)

    logger.info(
        "Project root: %s",
        PROJECT_ROOT,
    )

    logger.info(
        "Input dataset: %s",
        INPUT_PATH,
    )

    logger.info(
        "Output file: %s",
        OUTPUT_PATH,
    )

    try:

        # --------------------------------------------------
        # Validate required paths
        # --------------------------------------------------

        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Model directory not found: "
                f"{MODEL_DIR}"
            )

        if not INPUT_PATH.exists():
            raise FileNotFoundError(
                f"Input dataset not found: "
                f"{INPUT_PATH}"
            )

        # --------------------------------------------------
        # Build processor
        # --------------------------------------------------

        processor = build_processor()

        # --------------------------------------------------
        # Process complete dataset
        # --------------------------------------------------

        logger.info(
            "Starting full dataset processing..."
        )

        batch_result = processor.process_dataset(
            input_path=INPUT_PATH,
        )

        # --------------------------------------------------
        # Save results
        # --------------------------------------------------

        logger.info(
            "Saving processing results..."
        )

        processor.save_results(
            batch_result=batch_result,
            output_path=OUTPUT_PATH,
            failed_output_path=FAILED_OUTPUT_PATH,
        )

        # --------------------------------------------------
        # Final summary
        # --------------------------------------------------

        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        print()
        print("=" * 70)
        print("PROCESSING COMPLETED")
        print("=" * 70)

        print(
            f"Total products: "
            f"{batch_result.total_products}"
        )

        print(
            f"Successfully processed: "
            f"{batch_result.successful_count}"
        )

        print(
            f"Failed: "
            f"{batch_result.failed_count}"
        )

        print(
            f"Elapsed time: "
            f"{elapsed_seconds:.2f} seconds"
        )

        print(
            f"\nOutput saved to:\n"
            f"{OUTPUT_PATH}"
        )

        if batch_result.failed_count > 0:

            print(
                f"\nFailed products saved to:\n"
                f"{FAILED_OUTPUT_PATH}"
            )

        print("=" * 70)

    except Exception as error:

        logger.exception(
            "Sentiment batch processing failed."
        )

        print()
        print("=" * 70)
        print("PROCESSING FAILED")
        print("=" * 70)
        print(f"Error: {error}")
        print("=" * 70)

        raise


if __name__ == "__main__":
    main()