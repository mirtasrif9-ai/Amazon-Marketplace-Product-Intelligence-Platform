import logging
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.common.config import REQUEST_TIMEOUT
from src.common.exceptions import ParsingError, RequestError
from src.data_collection.collectors.search_base import (
    BaseSearchCollector,
)
from src.data_collection.models.product_reference import (
    ProductReference,
)


logger = logging.getLogger(__name__)


class OxylabsSandboxSearchCollector(BaseSearchCollector):
    """Collect product references from the Oxylabs sandbox."""

    def search(self, url: str) -> list[ProductReference]:
        logger.info("Starting search collection: %s", url)

        try:
            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

        except requests.RequestException as exc:
            logger.exception(
                "Search request failed: %s",
                url,
            )
            raise RequestError(
                f"Failed to retrieve search page: {url}"
            ) from exc

        logger.info(
            "Search page retrieved successfully: status=%s",
            response.status_code,
        )

        try:
            soup = BeautifulSoup(
                response.text,
                "html.parser",
            )

            product_cards = soup.select(".product-card")

        except Exception as exc:
            logger.exception(
                "Failed to parse search page: %s",
                url,
            )
            raise ParsingError(
                f"Failed to parse search page: {url}"
            ) from exc

        logger.info(
            "Found %d product cards",
            len(product_cards),
        )

        products: list[ProductReference] = []

        for card in product_cards:
            link = card.select_one("a.card-header")

            if link is None:
                logger.warning(
                    "Product card has no product link; skipping."
                )
                continue

            href = link.get("href")

            if not href:
                logger.warning(
                    "Product card has an empty href; skipping."
                )
                continue

            product_url = urljoin(url, href)

            identifier = href.rstrip("/").split("/")[-1]

            products.append(
                ProductReference(
                    identifier=identifier,
                    url=product_url,
                )
            )

        logger.info(
            "Collected %d product references",
            len(products),
        )

        return products