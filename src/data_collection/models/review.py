from dataclasses import dataclass
from datetime import date


@dataclass
class Review:
    reviewer_name: str
    review: str
    star_rating: float
    review_title: str
    description: str
    date: date