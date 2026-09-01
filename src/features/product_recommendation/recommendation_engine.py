from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.features.product_recommendation.models import RecommendationResult


class ProductRecommendationEngine:
    """
    Content-based similar product recommendation engine.

    Uses a pre-trained TF-IDF representation and cosine similarity
    to identify products similar to a given product ASIN.
    """

    def __init__(
        self,
        model_dir: Path | str,
    ):
        self.model_dir = Path(model_dir)

        self.vectorizer = None
        self.product_matrix = None
        self.product_metadata = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load persisted recommendation artifacts."""

        vectorizer_path = (
            self.model_dir / "tfidf_vectorizer.joblib"
        )
        matrix_path = (
            self.model_dir / "product_tfidf_matrix.joblib"
        )
        metadata_path = (
            self.model_dir / "product_metadata.csv"
        )

        required_files = [
            vectorizer_path,
            matrix_path,
            metadata_path,
        ]

        missing_files = [
            path for path in required_files
            if not path.exists()
        ]

        if missing_files:
            missing = ", ".join(
                str(path) for path in missing_files
            )
            raise FileNotFoundError(
                f"Missing recommendation artifacts: {missing}"
            )

        self.vectorizer = joblib.load(vectorizer_path)

        self.product_matrix = joblib.load(matrix_path)

        self.product_metadata = pd.read_csv(
            metadata_path
        )

        self._validate_artifacts()

    def _validate_artifacts(self) -> None:
        """Validate consistency between loaded artifacts."""

        if len(self.product_metadata) != self.product_matrix.shape[0]:
            raise ValueError(
                "Product metadata and TF-IDF matrix "
                "have inconsistent row counts."
            )

        if "asin" not in self.product_metadata.columns:
            raise ValueError(
                "Product metadata must contain an 'asin' column."
            )

        if self.product_metadata["asin"].duplicated().any():
            raise ValueError(
                "Product metadata contains duplicate ASINs."
            )

    def recommend(
        self,
        product_asin: str,
        top_k: int = 5,
    ) -> List[RecommendationResult]:
        """
        Return top-K products similar to the given ASIN.

        Ranking is deterministic:
        1. Similarity score descending
        2. ASIN ascending for ties
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        matches = self.product_metadata.index[
            self.product_metadata["asin"] == product_asin
        ].tolist()

        if not matches:
            raise ValueError(
                f"Product ASIN '{product_asin}' not found."
            )

        product_idx = matches[0]

        query_vector = self.product_matrix[
            product_idx
        ]

        similarity_scores = cosine_similarity(
            query_vector,
            self.product_matrix
        ).flatten()

        candidates = self.product_metadata.copy()

        candidates["similarity_score"] = (
            similarity_scores
        )

        candidates = candidates.drop(
            index=product_idx
        )

        candidates = candidates.sort_values(
            by=["similarity_score", "asin"],
            ascending=[False, True],
            kind="mergesort",
        )

        top_candidates = candidates.head(top_k)

        results = []

        for _, row in top_candidates.iterrows():

            results.append(
                RecommendationResult(
                    asin=str(row["asin"]),
                    title=str(row["title"]),
                    similarity_score=float(
                        row["similarity_score"]
                    ),
                    brand=self._safe_string(
                        row.get("brand")
                    ),
                    price=self._safe_float(
                        row.get("price")
                    ),
                    image=self._safe_string(
                        row.get("image")
                    ),
                    url=self._safe_string(
                        row.get("url")
                    ),
                    average_rating=self._safe_float(
                        row.get("average_rating")
                    ),
                    review_count=self._safe_int(
                        row.get("review_count")
                    ),
                )
            )

        return results

    @staticmethod
    def _safe_string(value):
        """Convert nullable values safely to string."""

        if pd.isna(value):
            return None

        return str(value)

    @staticmethod
    def _safe_float(value):
        """Convert nullable values safely to float."""

        if pd.isna(value):
            return None

        return float(value)

    @staticmethod
    def _safe_int(value):
        """Convert nullable values safely to int."""

        if pd.isna(value):
            return None

        return int(value)