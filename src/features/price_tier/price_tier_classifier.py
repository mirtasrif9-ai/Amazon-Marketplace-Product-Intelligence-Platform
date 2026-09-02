from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_collection.models.product import Product
from src.features.price_tier.models import PriceTierResult


class PriceTierClassifier:
    """Predict Amazon product price tiers.

    Uses the trained Feature D hybrid model:
    - TF-IDF text representation
    - numeric product metadata
    - categorical brand

    The product price itself is deliberately excluded from
    model inference because it was used to construct the
    training target.
    """

    CLASS_ORDER = [
        "budget",
        "mid_range",
        "premium",
    ]

    def __init__(
        self,
        model_path: Path | str,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Price tier model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

    def predict(
        self,
        product: Product,
    ) -> PriceTierResult:
        """Predict the price tier for a Product object."""

        asin = str(product.asin).strip()

        if not asin:
            raise ValueError("Product ASIN must be a non-empty string.")

        features = self._build_features(product)

        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        classes = self.model.classes_

        probability_map = {
            str(label): float(probability)
            for label, probability in zip(
                classes,
                probabilities,
            )
        }

        confidence = float(np.max(probabilities))

        return PriceTierResult(
            asin=asin,
            price_tier=str(prediction),
            confidence=confidence,
            probabilities=probability_map,
        )

    def predict_batch(
        self,
        products: list[Product],
    ) -> list[PriceTierResult]:
        """Predict price tiers for multiple products."""

        if not products:
            return []

        features = pd.concat(
            [self._build_features(product) for product in products],
            ignore_index=True,
        )

        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        classes = self.model.classes_

        results = []

        for index, product in enumerate(products):
            probability_map = {
                str(label): float(probability)
                for label, probability in zip(
                    classes,
                    probabilities[index],
                )
            }

            results.append(
                PriceTierResult(
                    asin=str(product.asin).strip(),
                    price_tier=str(predictions[index]),
                    confidence=float(np.max(probabilities[index])),
                    probabilities=probability_map,
                )
            )

        return results

    @staticmethod
    def _build_features(
        product: Product,
    ) -> pd.DataFrame:
        """Reconstruct the exact feature schema used during Feature D model training."""

        title = str(product.title) if product.title is not None else ""
        description = (
            str(product.description) if product.description is not None else ""
        )
        brand = str(product.brand) if product.brand is not None else ""

        product_text = title + " " + title + " " + description

        review_count = 0 if product.review_count is None else product.review_count
        average_rating = (
            0.0 if product.average_rating is None else product.average_rating
        )
        review_number_collected = len(product.reviews)

        return pd.DataFrame(
            [
                {
                    "product_text": product_text,
                    "brand": brand,
                    "review_count": review_count,
                    "average_rating": average_rating,
                    "review_number_collected": review_number_collected,
                    "title_length": len(title),
                    "description_length": len(description),
                }
            ]
        )