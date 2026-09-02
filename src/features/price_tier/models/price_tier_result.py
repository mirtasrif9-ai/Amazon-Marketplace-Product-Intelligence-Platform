from dataclasses import dataclass


@dataclass
class PriceTierResult:
    """Result of price tier classification."""

    asin: str
    price_tier: str
    confidence: float
    probabilities: dict[str, float]