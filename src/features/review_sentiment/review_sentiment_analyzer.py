from src.data_collection.models.review import Review

from src.features.review_sentiment.models.sentiment_result import (
    SentimentResult,
)
from src.features.review_sentiment.sentiment_predictor import (
    SentimentPredictor,
)


class ReviewSentimentAnalyzer:
    """
    Connects scraped Review objects with the SentimentPredictor.
    """

    def __init__(
        self,
        predictor: SentimentPredictor,
    ) -> None:
        self.predictor = predictor

    def analyze_review(
        self,
        review: Review,
    ) -> SentimentResult:
        """
        Analyze sentiment for a single review.
        """

        text = self._build_review_text(review)

        prediction = self.predictor.predict(text)

        probabilities = prediction["probabilities"]

        return SentimentResult(
            sentiment=prediction["sentiment"],
            negative_probability=probabilities["negative"],
            neutral_probability=probabilities["neutral"],
            positive_probability=probabilities["positive"],
        )

    @staticmethod
    def _build_review_text(
        review: Review,
    ) -> str:
        """
        Combine review title and description into
        one text input for sentiment analysis.
        """

        parts = []

        if review.review_title:
            parts.append(review.review_title.strip())

        if review.review_description:
            parts.append(
                review.review_description.strip()
            )

        return " ".join(parts)

    def analyze_reviews(
        self,
        reviews: list[Review],
        batch_size: int = 32,
    ) -> list[SentimentResult]:
        """
        Analyze sentiment for multiple reviews.
        """

        if not reviews:
            return []

        texts = [
            self._build_review_text(review)
            for review in reviews
        ]

        predictions = self.predictor.predict_batch(
            texts,
            batch_size=batch_size,
        )

        results = []

        for prediction in predictions:

            probabilities = prediction["probabilities"]

            results.append(
                SentimentResult(
                    sentiment=prediction["sentiment"],
                    negative_probability=probabilities[
                        "negative"
                    ],
                    neutral_probability=probabilities[
                        "neutral"
                    ],
                    positive_probability=probabilities[
                        "positive"
                    ],
                )
            )

        return results