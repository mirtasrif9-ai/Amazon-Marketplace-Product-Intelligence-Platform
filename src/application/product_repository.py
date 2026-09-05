from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_collection.models.product import Product
from src.data_collection.models.review import Review


class ProductRepository:
    """Repository for loading and accessing Amazon product data.

    Supports:
    - CSV datasets such as cleaned_products.csv
    - Sentiment-enriched JSON datasets such as products_with_sentiment.json
    """

    def __init__(self, dataset_path: Path | str) -> None:
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        self._products = self._load_products()
        self._products_by_asin = {
            product.asin: product for product in self._products
        }

    # ---------------------------------------------------------
    # Dataset Loading
    # ---------------------------------------------------------

    def _load_products(self) -> list[Product]:
        """Load products based on dataset file format."""
        suffix = self.dataset_path.suffix.lower()

        if suffix == ".csv":
            return self._load_products_from_csv()

        if suffix == ".json":
            return self._load_products_from_json()

        raise ValueError(f"Unsupported dataset format: {self.dataset_path.suffix}")

    # ---------------------------------------------------------
    # CSV Loading
    # ---------------------------------------------------------

    def _load_products_from_csv(self) -> list[Product]:
        """Load products from a CSV dataset."""
        df = pd.read_csv(self.dataset_path)
        products = []

        for _, row in df.iterrows():
            reviews = self._parse_reviews(row.get("reviews", "[]"))

            product = Product(
                product_number=self._safe_int(row.get("product_number")),
                search_keyword=self._safe_string(row.get("search_keyword")),
                asin=self._safe_string(row.get("asin")),
                title=self._safe_string(row.get("title")),
                product_url=self._safe_string(row.get("product_url")),
                price=self._safe_float(row.get("price")),
                description=self._safe_string(row.get("description")),
                brand=self._safe_string(row.get("brand")),
                image=self._safe_string(row.get("image")),
                review_count=self._safe_int(row.get("review_count")),
                average_rating=self._safe_float(row.get("average_rating")),
                video_url=self._safe_string(row.get("video_url")),
                reviews=reviews,
            )

            products.append(product)

        return products

    # ---------------------------------------------------------
    # JSON Loading
    # ---------------------------------------------------------

    def _load_products_from_json(self) -> list[Product]:
        """Load sentiment-enriched products from JSON."""
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("JSON dataset must contain a list of products.")

        products = []

        for product_data in data:
            if not isinstance(product_data, dict):
                continue

            reviews = self._parse_reviews(product_data.get("reviews", []))
            sentiment_summary = product_data.get("sentiment_summary", {})

            if not isinstance(sentiment_summary, dict):
                sentiment_summary = {}

            product = Product(
                product_number=self._safe_int(product_data.get("product_number")),
                search_keyword=self._safe_string(product_data.get("search_keyword")),
                asin=self._safe_string(product_data.get("asin")),
                title=self._safe_string(product_data.get("title")),
                product_url=self._safe_string(product_data.get("url")),
                price=self._safe_float(product_data.get("price")),
                description=self._safe_string(product_data.get("description")),
                brand=self._safe_string(product_data.get("brand")),
                image=self._safe_string(product_data.get("image")),
                review_count=self._safe_int(product_data.get("review_count")),
                average_rating=self._safe_float(product_data.get("average_rating")),
                video_url=self._safe_string(product_data.get("video_url")),
                reviews=reviews,
                sentiment_summary=sentiment_summary,
            )

            products.append(product)

        return products

    # ---------------------------------------------------------
    # Review Parsing
    # ---------------------------------------------------------

    @staticmethod
    def _parse_reviews(reviews_value) -> list[Review]:
        """Deserialize reviews into Review objects.

        Supports both:
        - JSON strings from CSV
        - Python lists from JSON datasets
        """
        if reviews_value is None:
            return []

        if isinstance(reviews_value, list):
            reviews_data = reviews_value
        else:
            try:
                reviews_data = json.loads(str(reviews_value))
            except (json.JSONDecodeError, TypeError):
                return []

        if not isinstance(reviews_data, list):
            return []

        reviews = []

        for review_data in reviews_data:
            if not isinstance(review_data, dict):
                continue

            probabilities = review_data.get("sentiment_probabilities", {})
            if not isinstance(probabilities, dict):
                probabilities = {}

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
                    review_data.get("review_description")
                ),
                sentiment=ProductRepository._safe_optional_string(
                    review_data.get("sentiment")
                ),
                sentiment_probabilities={
                    str(label): ProductRepository._safe_float(probability)
                    for label, probability in probabilities.items()
                },
            )

            reviews.append(review)

        return reviews

    # ---------------------------------------------------------
    # Repository Access Methods
    # ---------------------------------------------------------

    def get_all_products(self) -> list[Product]:
        """Return all products."""
        return list(self._products)

    def get_product_by_asin(self, asin: str) -> Product:
        """Return a product by ASIN."""
        asin = str(asin).strip()

        if asin not in self._products_by_asin:
            raise ValueError(f"Product with ASIN '{asin}' was not found.")

        return self._products_by_asin[asin]

    def get_products_by_search_keyword(self, search_keyword: str) -> list[Product]:
        """Return products belonging to a search keyword."""
        search_keyword = str(search_keyword).strip().lower()

        return [
            product
            for product in self._products
            if str(product.search_keyword).strip().lower() == search_keyword
        ]

    def get_search_keywords(self) -> list[str]:
        """Return all unique search keywords."""
        return sorted(
            {
                product.search_keyword
                for product in self._products
                if product.search_keyword
            }
        )

    def __len__(self) -> int:
        """Return total number of products."""
        return len(self._products)

    # ---------------------------------------------------------
    # Safe Type Conversion Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _safe_string(value) -> str:
        """Convert value safely to string."""
        if value is None or pd.isna(value):
            return ""

        return str(value).strip()

    @staticmethod
    def _safe_optional_string(value) -> str | None:
        """Convert value to optional string."""
        if value is None or pd.isna(value):
            return None

        value = str(value).strip()
        return value or None

    @staticmethod
    def _safe_int(value) -> int:
        """Convert value safely to integer."""
        if value is None or pd.isna(value):
            return 0

        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _safe_float(value) -> float:
        """Convert value safely to float."""
        if value is None or pd.isna(value):
            return 0.0

        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0