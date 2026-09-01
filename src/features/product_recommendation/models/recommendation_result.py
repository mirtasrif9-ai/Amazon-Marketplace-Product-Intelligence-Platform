from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RecommendationResult:
    """Represents a single product recommendation."""

    asin: str
    title: str
    similarity_score: float
    brand: Optional[str] = None
    price: Optional[float] = None
    image: Optional[str] = None
    url: Optional[str] = None
    average_rating: Optional[float] = None
    review_count: Optional[int] = None