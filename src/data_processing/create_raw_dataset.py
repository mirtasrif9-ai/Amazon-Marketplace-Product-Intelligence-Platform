from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


class RawDatasetCreator:
    """Convert collected Amazon products JSON into a master raw CSV dataset."""

    def __init__(
        self,
        input_path: str = "data/output/products.json",
        output_path: str = "data/raw/amazon_products_raw.csv",
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def create(self) -> Path:
        """
        Create the master raw CSV dataset.

        This method performs conversion only.
        No data cleaning or transformation is applied.
        """

        logger.info(
            "Starting raw dataset creation."
        )

        products = self._load_products()

        dataframe = self._create_dataframe(
            products
        )

        self._save_dataframe(
            dataframe
        )

        logger.info(
            "Raw dataset created successfully: "
            "rows=%d columns=%d path=%s",
            len(dataframe),
            len(dataframe.columns),
            self.output_path,
        )

        return self.output_path

    def _load_products(
        self,
    ) -> list[dict]:
        """Load products from the collected JSON file."""

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Products JSON file not found: "
                f"{self.input_path}"
            )

        if self.input_path.stat().st_size == 0:
            raise ValueError(
                f"Products JSON file is empty: "
                f"{self.input_path}"
            )

        try:
            with self.input_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                products = json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON file: "
                f"{self.input_path}"
            ) from exc

        if not isinstance(products, list):
            raise ValueError(
                "Products JSON must contain "
                "a list of products."
            )

        logger.info(
            "Loaded %d products from %s",
            len(products),
            self.input_path,
        )

        return products

    def _create_dataframe(
        self,
        products: list[dict],
    ) -> pd.DataFrame:
        """
        Convert products into a DataFrame.

        Reviews remain serialized as JSON strings so that
        each CSV row represents exactly one product.
        """

        rows: list[dict] = []

        for product in products:

            reviews = product.get(
                "reviews",
                [],
            )

            row = {
                "product_number": product.get(
                    "product_number"
                ),
                "asin": product.get(
                    "asin"
                ),
                "search_keyword": product.get(
                    "search_keyword"
                ),
                "title": product.get(
                    "title"
                ),
                "url": product.get(
                    "url"
                ),
                "price": product.get(
                    "price"
                ),
                "description": product.get(
                    "description"
                ),
                "brand": product.get(
                    "brand"
                ),
                "image": product.get(
                    "image"
                ),
                "review_count": product.get(
                    "review_count"
                ),
                "average_rating": product.get(
                    "average_rating"
                ),
                "video_url": product.get(
                    "video_url"
                ),

                # Keep all reviews inside one CSV field
                # as a JSON-formatted string.
                "reviews": json.dumps(
                    reviews,
                    ensure_ascii=False,
                ),
            }

            rows.append(row)

        dataframe = pd.DataFrame(rows)

        logger.info(
            "DataFrame created: rows=%d columns=%d",
            len(dataframe),
            len(dataframe.columns),
        )

        return dataframe

    def _save_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Save the master raw dataset as CSV."""

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            self.output_path,
            index=False,
            encoding="utf-8-sig",
        )

        logger.info(
            "Raw CSV saved successfully: %s",
            self.output_path,
        )


def main() -> None:
    """Create the master raw dataset."""

    creator = RawDatasetCreator()

    output_path = creator.create()

    print()
    print("Raw dataset created successfully.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()