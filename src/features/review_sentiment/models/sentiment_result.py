from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentResult:
    sentiment: str
    negative_probability: float
    neutral_probability: float
    positive_probability: float