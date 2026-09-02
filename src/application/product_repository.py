from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_collection.models.product import Product
from src.data_collection.models.review import Review


class ProductRepository:
    """
    Repository for loading and accessing cleaned Amazon product data.

    Converts rows from cleaned_products.csv into Product objects,
    including deserializing the reviews JSON field into Review objects.
    """

    def __init__(self, dataset_path: Path | str) -> None:
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {self.dataset_path}"
            )

        self._products = self._load_products()
        self._products_by_asin = {
            product.asin: product
            for product in self._products
        }

    def _load_products(self) -> list[Product]:
        """Load products from the processed CSV dataset."""

        df = pd.read_csv(self.dataset_path)

        products = []

        for _, row in df.iterrows():

            reviews = self._parse_reviews(
                row.get("reviews", "[]")
            )

            product = Product(
                product_number=self._safe_int(
                    row.get("product_number")
                ),
                search_keyword=self._safe_string(
                    row.get("search_keyword")
                ),
                asin=self._safe_string(
                    row.get("asin")
                ),
                title=self._safe_string(
                    row.get("title")
                ),
                product_url=self._safe_string(
                    row.get("product_url")
                ),
                price=self._safe_float(
                    row.get("price")
                ),
                description=self._safe_string(
                    row.get("description")
                ),
                brand=self._safe_string(
                    row.get("brand")
                ),
                image=self._safe_string(
                    row.get("image")
                ),
                review_count=self._safe_int(
                    row.get("review_count")
                ),
                average_rating=self._safe_float(
                    row.get("average_rating")
                ),
                video_url=self._safe_string(
                    row.get("video_url")
                ),
                reviews=reviews,
            )

            products.append(product)

        return products

    @staticmethod
    def _parse_reviews(
        reviews_value,
    ) -> list[Review]:
        """Deserialize the reviews JSON field."""

        if pd.isna(reviews_value):
            return []

        if isinstance(reviews_value, list):
            reviews_data = reviews_value
        else:
            try:
                reviews_data = json.loads(
                    str(reviews_value)
                )
            except (
                json.JSONDecodeError,
                TypeError,
            ):
                return []

        if not isinstance(reviews_data, list):
            return []

        reviews = []

        for review_data in reviews_data:

            if not isinstance(review_data, dict):
                continue

            review = Review(
                reviewer_name=ProductRepository._safe_string(
                    review_data.get("reviewer_name")
                ),
                star_rating=ProductRepository._safe_float(
                    review_data.get("star_rating")
                ),
                review_title=ProductRepository._safe_string(
                    review_data.get("review_title")
                ),
                review_description=ProductRepository._safe_string(
                    review_data.get(
                        "review_description"
                    )
                ),
            )

            reviews.append(review)

        return reviews

    def get_all_products(self) -> list[Product]:
        """Return all products."""

        return list(self._products)

    def get_product_by_asin(
        self,
        asin: str,
    ) -> Product:
        """Return a product by ASIN."""

        asin = str(asin).strip()

        if asin not in self._products_by_asin:
            raise ValueError(
                f"Product with ASIN '{asin}' was not found."
            )

        return self._products_by_asin[asin]

    def get_products_by_search_keyword(
        self,
        search_keyword: str,
    ) -> list[Product]:
        """
        Return products belonging to a search keyword.

        In the application, search_keyword acts as the
        product category/grouping.
        """

        search_keyword = (
            str(search_keyword)
            .strip()
            .lower()
        )

        return [
            product
            for product in self._products
            if str(product.search_keyword)
            .strip()
            .lower()
            == search_keyword
        ]

    def get_search_keywords(self) -> list[str]:
        """
        Return all unique search keywords.

        These act as category/group labels in the application.
        """

        return sorted(
            {
                product.search_keyword
                for product in self._products
                if product.search_keyword
            }
        )

    def __len__(self) -> int:
        """Return the number of products."""

        return len(self._products)

    @staticmethod
    def _safe_string(value) -> str:
        """Convert a value safely to string."""

        if pd.isna(value):
            return ""

        return str(value).strip()

    @staticmethod
    def _safe_int(value) -> int:
        """Convert a value safely to integer."""

        if pd.isna(value):
            return 0

        try:
            return int(float(value))
        except (
            ValueError,
            TypeError,
        ):
            return 0

    @staticmethod
    def _safe_float(value) -> float:
        """Convert a value safely to float."""

        if pd.isna(value):
            return 0.0

        try:
            return float(value)
        except (
            ValueError,
            TypeError,
        ):
            return 0.0