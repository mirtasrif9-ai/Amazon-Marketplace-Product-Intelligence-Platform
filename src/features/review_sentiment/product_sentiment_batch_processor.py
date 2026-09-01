import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_collection.models.review import Review

from src.features.review_sentiment.models.batch_processing_result import (
    BatchProcessingResult,
)
from src.features.review_sentiment.product_review_sentiment_pipeline import (
    ProductReviewSentimentPipeline,
)


logger = logging.getLogger(__name__)


class ProductSentimentBatchProcessor:
    """
    Processes products through the review sentiment pipeline.

    Responsibilities:
    - Load cleaned product datasets
    - Parse serialized review data
    - Convert review dictionaries into Review objects
    - Run sentiment analysis
    - Build sentiment-enriched product records
    - Track processing failures
    - Report progress
    """

    def __init__(
        self,
        sentiment_pipeline: ProductReviewSentimentPipeline,
        batch_size: int = 32,
        progress_interval: int = 50,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if progress_interval <= 0:
            raise ValueError(
                "progress_interval must be greater than zero."
            )

        self.sentiment_pipeline = sentiment_pipeline
        self.batch_size = batch_size
        self.progress_interval = progress_interval

    def process_product(
        self,
        product: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process sentiment analysis for a single product record.
        """

        product_copy = product.copy()

        reviews = self._parse_reviews(
            product.get("reviews")
        )

        analysis = (
            self.sentiment_pipeline.analyze_reviews(
                reviews=reviews,
                batch_size=self.batch_size,
            )
        )

        product_copy["reviews"] = (
            self._build_enriched_reviews(
                reviews=reviews,
                sentiment_results=analysis.review_sentiments,
            )
        )

        product_copy["sentiment_summary"] = (
            self._build_summary_dict(
                analysis.summary
            )
        )

        return product_copy

    def process_dataset(
        self,
        input_path: str | Path,
        limit: int | None = None,
    ) -> BatchProcessingResult:
        """
        Load and process products from a cleaned CSV dataset.

        Failed products are recorded without stopping
        the entire batch process.
        """

        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}"
            )

        logger.info(
            "Loading dataset from: %s",
            input_path,
        )

        dataframe = pd.read_csv(input_path)

        if limit is not None:
            dataframe = dataframe.head(limit)

        total_products = len(dataframe)

        logger.info(
            "Starting sentiment processing for %d products.",
            total_products,
        )

        result = BatchProcessingResult(
            total_products=total_products
        )

        for position, (_, row) in enumerate(
            dataframe.iterrows(),
            start=1,
        ):

            product = self._row_to_product_dict(
                row
            )

            asin = product.get("asin", "UNKNOWN")

            try:
                enriched_product = (
                    self.process_product(product)
                )

                result.successful_products.append(
                    enriched_product
                )

            except Exception as error:

                failed_product = {
                    "product_number": product.get(
                        "product_number"
                    ),
                    "asin": asin,
                    "title": product.get("title"),
                    "error": str(error),
                }

                result.failed_products.append(
                    failed_product
                )

                logger.exception(
                    "Failed to process product "
                    "%d/%d (ASIN: %s)",
                    position,
                    total_products,
                    asin,
                )

            # Progress logging
            if (
                position % self.progress_interval == 0
                or position == total_products
            ):
                self._log_progress(
                    position=position,
                    total=total_products,
                    result=result,
                )

        self._log_final_summary(result)

        return result

    def _parse_reviews(
        self,
        reviews_data: Any,
    ) -> list[Review]:
        """
        Parse reviews from a serialized JSON string or list.
        """

        if reviews_data is None:
            return []

        if isinstance(reviews_data, str):

            if not reviews_data.strip():
                return []

            try:
                reviews_data = json.loads(
                    reviews_data
                )

            except json.JSONDecodeError as error:
                logger.warning(
                    "Failed to parse reviews JSON: %s",
                    error,
                )
                return []

        if not isinstance(reviews_data, list):
            logger.warning(
                "Reviews data is not a list."
            )
            return []

        reviews = []

        for review_data in reviews_data:

            if not isinstance(review_data, dict):
                continue

            try:
                review = Review(
                    reviewer_name=review_data.get(
                        "reviewer_name",
                        "",
                    ),
                    star_rating=review_data.get(
                        "star_rating",
                        0.0,
                    ),
                    review_title=review_data.get(
                        "review_title",
                        "",
                    ),
                    review_description=review_data.get(
                        "review_description",
                        "",
                    ),
                )

                reviews.append(review)

            except (TypeError, ValueError) as error:
                logger.warning(
                    "Skipping invalid review: %s",
                    error,
                )

        return reviews

    @staticmethod
    def _build_enriched_reviews(
        reviews: list[Review],
        sentiment_results: list,
    ) -> list[dict[str, Any]]:
        """
        Merge reviews with sentiment predictions.
        """

        enriched_reviews = []

        for review, result in zip(
            reviews,
            sentiment_results,
        ):
            enriched_reviews.append(
                {
                    "reviewer_name": review.reviewer_name,
                    "star_rating": review.star_rating,
                    "review_title": review.review_title,
                    "review_description": (
                        review.review_description
                    ),
                    "sentiment": result.sentiment,
                    "sentiment_probabilities": {
                        "negative": (
                            result.negative_probability
                        ),
                        "neutral": (
                            result.neutral_probability
                        ),
                        "positive": (
                            result.positive_probability
                        ),
                    },
                }
            )

        return enriched_reviews

    @staticmethod
    def _build_summary_dict(
        summary,
    ) -> dict[str, Any]:
        """
        Convert summary to a JSON-compatible dictionary.
        """

        return {
            "total_reviews": summary.total_reviews,
            "positive_count": summary.positive_count,
            "neutral_count": summary.neutral_count,
            "negative_count": summary.negative_count,
            "positive_percentage": (
                summary.positive_percentage
            ),
            "neutral_percentage": (
                summary.neutral_percentage
            ),
            "negative_percentage": (
                summary.negative_percentage
            ),
            "overall_sentiment": (
                summary.overall_sentiment
            ),
            "sentiment_score": summary.sentiment_score,
        }

    @staticmethod
    def _row_to_product_dict(
        row: pd.Series,
    ) -> dict[str, Any]:
        """
        Convert a Pandas Series into a clean Python dictionary.
        """

        product = {}

        for key, value in row.items():

            if pd.isna(value):
                product[key] = None
            else:
                product[key] = value

        return product

    def _log_progress(
        self,
        position: int,
        total: int,
        result: BatchProcessingResult,
    ) -> None:
        """
        Log current processing progress.
        """

        percentage = (position / total) * 100

        logger.info(
            "Progress: %d/%d (%.1f%%) | "
            "Success: %d | Failed: %d",
            position,
            total,
            percentage,
            result.successful_count,
            result.failed_count,
        )

    @staticmethod
    def _log_final_summary(
        result: BatchProcessingResult,
    ) -> None:
        """
        Log final batch processing summary.
        """

        logger.info("=" * 60)
        logger.info(
            "SENTIMENT BATCH PROCESSING COMPLETE"
        )
        logger.info("=" * 60)
        logger.info(
            "Total products: %d",
            result.total_products,
        )
        logger.info(
            "Successfully processed: %d",
            result.successful_count,
        )
        logger.info(
            "Failed: %d",
            result.failed_count,
        )
        logger.info("=" * 60)

    def save_results(
        self,
        batch_result: BatchProcessingResult,
        output_path: str | Path,
        failed_output_path: str | Path | None = None,
    ) -> None:
        """
        Save successful and failed sentiment processing results.

        Successful products are saved to output_path.
        Failed products are optionally saved separately.
        """

        output_path = Path(output_path)

        self._save_json(
            data=batch_result.successful_products,
            output_path=output_path,
        )

        logger.info(
            "Saved %d sentiment-enriched products to: %s",
            batch_result.successful_count,
            output_path,
        )

        if failed_output_path is not None:

            failed_output_path = Path(
                failed_output_path
            )

            self._save_json(
                data=batch_result.failed_products,
                output_path=failed_output_path,
            )

            logger.info(
                "Saved %d failed products to: %s",
                batch_result.failed_count,
                failed_output_path,
            )

    @staticmethod
    def _save_json(
        data: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """
        Safely save data as formatted JSON.

        Uses a temporary file and atomic replacement to reduce the
        risk of corrupting the final output file.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        try:
            with temp_path.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    data,
                    file,
                    indent=2,
                    ensure_ascii=False,
                    default=ProductSentimentBatchProcessor._json_serializer,
                )

            temp_path.replace(output_path)

        except Exception:

            if temp_path.exists():
                temp_path.unlink()

            raise


    @staticmethod
    def _json_serializer(
        value: Any,
    ) -> Any:
        """
        Convert unsupported Pandas/NumPy values into
        JSON-compatible Python values.
        """

        if hasattr(value, "item"):
            return value.item()

        raise TypeError(
            f"Object of type {type(value).__name__} "
            "is not JSON serializable."
        )