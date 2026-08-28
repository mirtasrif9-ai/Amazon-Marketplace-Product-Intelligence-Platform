from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data_collection.models.product import Product
from src.data_collection.models.product_reference import ProductReference


logger = logging.getLogger(__name__)


class JsonStorage:
    """Handle JSON storage for Amazon collection data."""

    def __init__(self, output_directory: str = "data/output") -> None:
        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_product_references(
        self,
        products: list[ProductReference],
        filename: str = "product_references.json",
    ) -> Path:
        """Save product references to JSON."""

        data = []

        for index, product in enumerate(products, start=1):
            data.append(
                {
                    "product_number": index,
                    "asin": product.asin,
                    "title": product.title,
                    "url": product.url,
                }
            )

        output_path = self.output_directory / filename

        self._write_json(data, output_path)

        logger.info(
            "Saved %d product references to %s",
            len(data),
            output_path,
        )

        return output_path

    def load_product_references(
        self,
        filename: str = "product_references.json",
    ) -> list[ProductReference]:
        """Load product references from JSON."""

        input_path = self.output_directory / filename

        if not input_path.exists():
            raise FileNotFoundError(
                f"Product reference file not found: {input_path}"
            )

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        products = []

        for item in data:
            products.append(
                ProductReference(
                    asin=item["asin"],
                    title=item["title"],
                    url=item["url"],
                )
            )

        logger.info(
            "Loaded %d product references from %s",
            len(products),
            input_path,
        )

        return products

    def save_products(
        self,
        products: list[Product],
        filename: str = "products.json",
    ) -> Path:
        """Save complete products to JSON."""

        data = []

        for index, product in enumerate(products, start=1):
            data.append(
                {
                    "product_number": index,
                    "asin": product.asin,
                    "title": product.title,
                    "description": product.description,
                    "brand": product.brand,
                    "price": product.price,
                    "image": product.image,
                    "review_count": product.review_count,
                    "average_rating": product.average_rating,
                    "video_url": product.video_url,
                    "reviews": [
                        {
                            "reviewer_name": review.reviewer_name,
                            "star_rating": review.star_rating,
                            "review_title": review.review_title,
                            "review_description": review.review_description,
                        }
                        for review in product.reviews
                    ],
                }
            )

        output_path = self.output_directory / filename

        self._write_json(data, output_path)

        logger.info(
            "Saved %d products to %s",
            len(data),
            output_path,
        )

        return output_path

    @staticmethod
    def _write_json(
        data: list[dict],
        output_path: Path,
    ) -> None:
        """Write data to a JSON file."""

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )