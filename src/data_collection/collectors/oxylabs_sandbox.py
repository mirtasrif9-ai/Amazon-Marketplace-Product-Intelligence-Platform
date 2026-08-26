import logging

import requests
from bs4 import BeautifulSoup

from src.common.config import REQUEST_TIMEOUT
from src.common.exceptions import RequestError, ParsingError
from src.data_collection.collectors.base import BaseProductCollector
from src.data_collection.models.product import Product


logger = logging.getLogger(__name__)


class OxylabsSandboxProductCollector(BaseProductCollector):
    """Collect product data from the Oxylabs sandbox."""

    def collect(self, url: str) -> Product:
        logger.info("Starting product collection: %s", url)

        try:
            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

        except requests.RequestException as exc:
            logger.exception(
                "Request failed for URL: %s",
                url,
            )
            raise RequestError(
                f"Failed to retrieve product page: {url}"
            ) from exc

        logger.info(
            "Product page retrieved successfully: status=%s",
            response.status_code,
        )

        try:
            soup = BeautifulSoup(
                response.text,
                "html.parser",
            )

        except Exception as exc:
            logger.exception(
                "Failed to parse product page: %s",
                url,
            )
            raise ParsingError(
                f"Failed to parse product page: {url}"
            ) from exc

        logger.info("Product page parsed successfully: %s", url)

        # Temporary implementation.
        # We will replace this with real extraction logic
        # after inspecting the sandbox HTML structure.

        return Product(
            asin="TEMPORARY",
            title=soup.title.get_text(strip=True)
            if soup.title
            else "Unknown",
            description="",
            brand="",
            price=0.0,
            image="",
            review_count=0,
            reviews=[],
        )