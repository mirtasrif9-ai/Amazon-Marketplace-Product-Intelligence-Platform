from dataclasses import dataclass, field

from src.data_collection.models.review import Review


@dataclass
class Product:
    asin: str
    title: str
    description: str
    brand: str
    price: float
    image: str
    review_count: int
    reviews: list[Review] = field(default_factory=list)