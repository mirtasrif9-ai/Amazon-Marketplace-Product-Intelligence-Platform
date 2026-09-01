from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentResult:
    """
    Sentiment prediction result for a single review.
    """

    sentiment: str

    negative_probability: float
    neutral_probability: float
    positive_probability: float