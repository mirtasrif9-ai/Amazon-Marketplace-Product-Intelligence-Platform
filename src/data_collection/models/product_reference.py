from dataclasses import dataclass


@dataclass(frozen=True)
class ProductReference:
    """Reference information collected from Amazon search results."""

    product_number: int
    asin: str
    search_keyword: str
    title: str
    url: str
    price: float
"""Why frozen=True?A ProductReference represents an identified product.Once we've extracted:
ASIN
Title
URL
we don't want random parts of the application modifying it accidentally.It also makes the object immutable."""