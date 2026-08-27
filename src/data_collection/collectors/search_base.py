from abc import ABC, abstractmethod

from src.data_collection.models.product_reference import (
    ProductReference,
)


class BaseSearchCollector(ABC):
    """Interface for search-result collectors."""

    @abstractmethod
    def search(self, keyword: str) -> list[ProductReference]:
        """Search for a keyword and return product references."""
        raise NotImplementedError