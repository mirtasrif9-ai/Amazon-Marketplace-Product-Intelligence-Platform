import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class SentimentPredictor:
    """
    Reusable sentiment predictor for Amazon product reviews.

    Uses the final class-weighted DistilBERT model with
    configurable class-specific decision thresholds.
    """

    def __init__(
        self,
        model_dir: str | Path,
        max_length: int = 256,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = max_length

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(self.model_dir)
            .to(self.device)
        )

        self.model.eval()

        self.thresholds = self._load_json(
            "thresholds.json"
        )

        mappings = self._load_json(
            "label_mapping.json"
        )

        self.id2label = {
            int(key): value
            for key, value in mappings["id2label"].items()
        }

    def _load_json(self, filename: str) -> dict:
        """Load a JSON configuration file from the model directory."""

        file_path = self.model_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _predict_probabilities(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Generate class probabilities for one or more reviews."""

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        return (
            torch.softmax(
                outputs.logits,
                dim=1,
            )
            .cpu()
            .numpy()
        )

    def _apply_thresholds(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """
        Apply class-specific decision thresholds.

        Lower thresholds make a class relatively easier to select.
        Higher thresholds make a class relatively harder to select.
        """

        threshold_values = np.array([
            self.thresholds["negative"],
            self.thresholds["neutral"],
            self.thresholds["positive"],
        ])

        adjusted_scores = probabilities / threshold_values

        return np.argmax(
            adjusted_scores,
            axis=1,
        )

    def predict(self, text: str) -> dict:
        """Predict sentiment for a single review."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Review text must be a non-empty string."
            )

        probabilities = self._predict_probabilities(
            [text]
        )

        prediction_id = self._apply_thresholds(
            probabilities
        )[0]

        return {
            "sentiment": self.id2label[int(prediction_id)],
            "probabilities": {
                self.id2label[index]: round(
                    float(probabilities[0][index]),
                    4,
                )
                for index in range(probabilities.shape[1])
            },
        }

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[dict]:
        """Predict sentiment for multiple reviews efficiently."""

        if not texts:
            return []

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        results = []

        for start in range(
            0,
            len(texts),
            batch_size,
        ):
            batch = texts[start:start + batch_size]

            probabilities = self._predict_probabilities(
                batch
            )

            prediction_ids = self._apply_thresholds(
                probabilities
            )

            for probs, prediction_id in zip(
                probabilities,
                prediction_ids,
            ):
                results.append({
                    "sentiment": self.id2label[
                        int(prediction_id)
                    ],
                    "probabilities": {
                        self.id2label[index]: round(
                            float(probs[index]),
                            4,
                        )
                        for index in range(len(probs))
                    },
                })

        return results