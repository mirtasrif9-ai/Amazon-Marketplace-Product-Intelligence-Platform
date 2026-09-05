from dataclasses import dataclass, field


@dataclass
class Review:
    """Customer review with optional sentiment prediction."""

    reviewer_name: str
    star_rating: float
    review_title: str
    review_description: str

    # ---------------------------------------------------------
    # ML-generated sentiment fields
    # Optional because raw scraped reviews do not have these
    # values until sentiment inference is performed.
    # ---------------------------------------------------------

    sentiment: str | None = None
    sentiment_probabilities: dict[str, float] = field(default_factory=dict)