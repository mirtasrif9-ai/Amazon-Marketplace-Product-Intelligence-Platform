from abc import ABC, abstractmethod

from src.data_collection.models.product import Product


class BaseProductCollector(ABC):
    """Interface for product collectors."""

    @abstractmethod
    def collect(self, url: str) -> Product:
        """Collect product information from a URL."""
        raise NotImplementedError