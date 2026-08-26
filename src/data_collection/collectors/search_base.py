from abc import ABC, abstractmethod

from src.data_collection.models.product_reference import ProductReference


class BaseSearchCollector(ABC):
    """Interface for search-result collectors."""

    @abstractmethod
    def search(self, url: str) -> list[ProductReference]:
        """Extract product references from a search-results page."""
        raise NotImplementedError