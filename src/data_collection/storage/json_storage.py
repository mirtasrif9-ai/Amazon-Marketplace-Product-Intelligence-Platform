from __future__ import annotations

import json
import logging
from pathlib import Path

from src.data_collection.models.product import Product
from src.data_collection.models.product_reference import ProductReference


logger = logging.getLogger(__name__)


class JsonStorage:
    """Handle JSON storage for Amazon collection data."""

    def __init__(
        self,
        output_directory: str = "data/output",
    ) -> None:
        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==========================================================
    # PRODUCT REFERENCES
    # ==========================================================

    def save_product_references(
        self,
        products: list[ProductReference],
        filename: str = "product_references.json",
    ) -> Path:
        """
        Replace the product reference JSON file.

        Normally the pipeline should use
        append_product_references() so that results are
        preserved after every keyword.
        """

        data = []

        for index, product in enumerate(
            products,
            start=1,
        ):
            data.append(
                {
                    "product_number": index,
                    "asin": product.asin,
                    "search_keyword": product.search_keyword,
                    "title": product.title,
                    "url": product.url,
                    "price": product.price,
                }
            )

        output_path = self.output_directory / filename

        self._write_json(
            data,
            output_path,
        )

        logger.info(
            "Saved %d product references to %s",
            len(data),
            output_path,
        )

        return output_path

    def append_product_references(
        self,
        products: list[ProductReference],
        filename: str = "product_references.json",
    ) -> tuple[int, int, Path]:
        """
        Append only new product references to JSON.

        Existing products are identified by ASIN and skipped.

        Returns:
            (
                new_products_count,
                total_products_count,
                output_path,
            )
        """

        output_path = self.output_directory / filename

        # ------------------------------------------------------
        # Load existing data
        # ------------------------------------------------------

        existing_data: list[dict] = []

        if output_path.exists():

            if output_path.stat().st_size == 0:
                logger.warning(
                    "Product reference JSON is empty. "
                    "Starting with an empty list: %s",
                    output_path,
                )

            else:
                try:
                    with output_path.open(
                        "r",
                        encoding="utf-8",
                    ) as file:
                        existing_data = json.load(file)

                except json.JSONDecodeError as exc:
                    logger.exception(
                        "Invalid JSON file: %s",
                        output_path,
                    )

                    raise ValueError(
                        f"Invalid JSON file: {output_path}"
                    ) from exc

                if not isinstance(existing_data, list):
                    raise ValueError(
                        "Product reference JSON must contain "
                        "a JSON list."
                    )

        # ------------------------------------------------------
        # Existing ASINs
        # ------------------------------------------------------

        existing_asins = {
            item["asin"]
            for item in existing_data
            if item.get("asin")
        }

        logger.info(
            "Existing product references: %d",
            len(existing_data),
        )

        # ------------------------------------------------------
        # Find new products
        # ------------------------------------------------------

        new_products: list[ProductReference] = []

        for product in products:

            if product.asin in existing_asins:
                logger.info(
                    "Skipping existing product reference: "
                    "asin=%s keyword=%s",
                    product.asin,
                    product.search_keyword,
                )
                continue

            new_products.append(product)

            # Prevent duplicates inside the current search.
            existing_asins.add(product.asin)

        # ------------------------------------------------------
        # Assign product numbers
        # ------------------------------------------------------

        starting_number = len(existing_data) + 1

        for offset, product in enumerate(
            new_products
        ):
            existing_data.append(
                {
                    "product_number": (
                        starting_number + offset
                    ),
                    "asin": product.asin,
                    "search_keyword": product.search_keyword,
                    "title": product.title,
                    "url": product.url,
                    "price": product.price,
                }
            )

        # ------------------------------------------------------
        # Save immediately
        # ------------------------------------------------------

        self._write_json(
            existing_data,
            output_path,
        )

        logger.info(
            "Product references updated: "
            "new=%d total=%d path=%s",
            len(new_products),
            len(existing_data),
            output_path,
        )

        return (
            len(new_products),
            len(existing_data),
            output_path,
        )

    def load_product_references(
        self,
        filename: str = "product_references.json",
    ) -> list[ProductReference]:
        """Load product references from JSON."""

        input_path = self.output_directory / filename

        if not input_path.exists():
            raise FileNotFoundError(
                f"Product reference file not found: "
                f"{input_path}"
            )

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                "Product reference JSON must contain "
                "a JSON list."
            )

        products: list[ProductReference] = []

        for item in data:
            products.append(
                ProductReference(
                    product_number=int(
                        item["product_number"]
                    ),
                    asin=item["asin"],
                    search_keyword=item["search_keyword"],
                    title=item["title"],
                    url=item["url"],
                    price=float(
                        item.get("price", 0.0)
                    ),
                )
            )

        logger.info(
            "Loaded %d product references from %s",
            len(products),
            input_path,
        )

        return products

    # ==========================================================
    # FULL PRODUCTS
    # ==========================================================

    def append_product(
        self,
        product: Product,
        filename: str = "products.json",
    ) -> tuple[bool, int, Path]:
        """
        Append one fully collected product immediately.

        If the product ASIN already exists in products.json,
        it will be skipped.

        Returns:
            (
                saved,
                total_products_count,
                output_path,
            )

        saved:
            True  -> product was newly saved
            False -> product already existed
        """

        output_path = self.output_directory / filename

        # ------------------------------------------------------
        # Load existing products
        # ------------------------------------------------------

        existing_data: list[dict] = []

        if output_path.exists():

            if output_path.stat().st_size == 0:
                logger.warning(
                    "Products JSON is empty. "
                    "Starting with an empty list: %s",
                    output_path,
                )

            else:
                try:
                    with output_path.open(
                        "r",
                        encoding="utf-8",
                    ) as file:
                        existing_data = json.load(file)

                except json.JSONDecodeError as exc:
                    logger.exception(
                        "Invalid products JSON file: %s",
                        output_path,
                    )

                    raise ValueError(
                        f"Invalid JSON file: {output_path}"
                    ) from exc

                if not isinstance(existing_data, list):
                    raise ValueError(
                        "Products JSON must contain "
                        "a JSON list."
                    )

        # ------------------------------------------------------
        # Check whether product already exists
        # ------------------------------------------------------

        existing_asins = {
            item["asin"]
            for item in existing_data
            if item.get("asin")
        }

        if product.asin in existing_asins:
            logger.info(
                "Product already exists in products.json. "
                "Skipping: asin=%s",
                product.asin,
            )

            return (
                False,
                len(existing_data),
                output_path,
            )

        # ------------------------------------------------------
        # Determine product number
        # ------------------------------------------------------

        product_number = product.product_number

        # ------------------------------------------------------
        # Convert Product model to JSON
        # ------------------------------------------------------

        product_data = {
            "product_number": product_number,

            # Information from product_references.json
            "asin": product.asin,
            "search_keyword": product.search_keyword,
            "title": product.title,
            "url": product.product_url,
            "price": product.price,

            # Information extracted by ProductCollector
            "description": product.description,
            "brand": product.brand,
            "image": product.image,
            "review_count": product.review_count,
            "average_rating": product.average_rating,
            "video_url": product.video_url,

            "reviews": [
                {
                    "reviewer_name": review.reviewer_name,
                    "star_rating": review.star_rating,
                    "review_title": review.review_title,
                    "review_description": (
                        review.review_description
                    ),
                }
                for review in product.reviews
            ],
        }

        # ------------------------------------------------------
        # Append product
        # ------------------------------------------------------

        existing_data.append(product_data)

        # ------------------------------------------------------
        # Save immediately
        # ------------------------------------------------------

        self._write_json(
            existing_data,
            output_path,
        )

        logger.info(
            "Full product saved immediately: "
            "product_number=%d asin=%s total=%d path=%s",
            product_number,
            product.asin,
            len(existing_data),
            output_path,
        )

        return (
            True,
            len(existing_data),
            output_path,
        )

    def load_existing_product_asins(
        self,
        filename: str = "products.json",
    ) -> set[str]:
        """
        Load ASINs already stored in products.json.

        Used by the pipeline to avoid re-scraping
        already collected products.
        """

        input_path = self.output_directory / filename

        if not input_path.exists():
            logger.info(
                "Products JSON does not exist yet: %s",
                input_path,
            )

            return set()

        if input_path.stat().st_size == 0:
            logger.info(
                "Products JSON is empty: %s",
                input_path,
            )

            return set()

        try:
            with input_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except json.JSONDecodeError as exc:
            logger.exception(
                "Invalid products JSON file: %s",
                input_path,
            )

            raise ValueError(
                f"Invalid JSON file: {input_path}"
            ) from exc

        if not isinstance(data, list):
            raise ValueError(
                "Products JSON must contain a JSON list."
            )

        asins = {
            item["asin"]
            for item in data
            if item.get("asin")
        }

        logger.info(
            "Existing fully collected products: %d",
            len(asins),
        )

        return asins

    def save_products(
        self,
        products: list[Product],
        filename: str = "products.json",
    ) -> Path:
        """
        Replace products.json with the supplied products.

        This method is retained for compatibility.

        For the main pipeline, prefer append_product()
        because it saves each product immediately.
        """

        data = []

        for index, product in enumerate(
            products,
            start=1,
        ):
            data.append(
                self._product_to_dict(
                    product,
                    product_number=index,
                )
            )

        output_path = (
            self.output_directory / filename
        )

        self._write_json(
            data,
            output_path,
        )

        logger.info(
            "Saved %d products to %s",
            len(data),
            output_path,
        )

        return output_path

    # ==========================================================
    # HELPERS
    # ==========================================================

    @staticmethod
    def _get_next_product_number(
        existing_data: list[dict],
    ) -> int:
        """Return the next sequential product number."""

        if not existing_data:
            return 1

        numbers = [
            int(item["product_number"])
            for item in existing_data
            if item.get("product_number") is not None
        ]

        if not numbers:
            return 1

        return max(numbers) + 1

    @staticmethod
    def _product_to_dict(
        product: Product,
        product_number: int,
    ) -> dict:
        """Convert a Product model into a JSON dictionary."""

        return {
            "product_number": product_number,

            # Reference information
            "asin": product.asin,
            "search_keyword": product.search_keyword,
            "title": product.title,
            "url": product.product_url,
            "price": product.price,

            # Extracted information
            "description": product.description,
            "brand": product.brand,
            "image": product.image,
            "review_count": product.review_count,
            "average_rating": product.average_rating,
            "video_url": product.video_url,

            "reviews": [
                {
                    "reviewer_name": review.reviewer_name,
                    "star_rating": review.star_rating,
                    "review_title": review.review_title,
                    "review_description": (
                        review.review_description
                    ),
                }
                for review in product.reviews
            ],
        }

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