from dataclasses import dataclass, field

from src.data_collection.models.review import Review


@dataclass
class Product:
    """Complete Amazon product information."""

    # ---------------------------------------------------------
    # Information coming from product_references.json
    # ---------------------------------------------------------

    product_number: int
    search_keyword: str
    asin: str
    title: str
    product_url: str
    price: float

    # ---------------------------------------------------------
    # Information extracted from Amazon product page
    # ---------------------------------------------------------

    description: str
    brand: str
    image: str
    review_count: int
    average_rating: float
    video_url: str

    # ---------------------------------------------------------
    # Customer reviews
    # ---------------------------------------------------------

    reviews: list[Review] = field(default_factory=list)