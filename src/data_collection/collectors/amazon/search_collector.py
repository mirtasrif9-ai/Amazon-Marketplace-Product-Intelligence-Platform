from __future__ import annotations

import logging
import time

from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from src.common.exceptions import (
    AmazonTemporaryError,
    BlockedPageError,
    ParsingError,
    RequestError,
)
from src.data_collection.collectors.search_base import (
    BaseSearchCollector,
)
from src.data_collection.models.product_reference import (
    ProductReference,
)


logger = logging.getLogger(__name__)


class AmazonSearchCollector(BaseSearchCollector):
    """Collect product references from Amazon search results."""

    BASE_URL = "https://www.amazon.com/"

    def __init__(self, page: Page) -> None:
        self.page = page

    def search(
        self,
        keyword: str,
    ) -> list[ProductReference]:
        """Search Amazon and collect products from the first page."""

        logger.info(
            "Starting Amazon search collection: keyword=%s",
            keyword,
        )

        search_url = self._build_search_url(keyword)

        logger.info(
            "Navigating to Amazon search page: %s",
            search_url,
        )

        try:
            self.page.goto(self.BASE_URL)
            time.sleep(5)
            response = self.page.goto(
                search_url,
                wait_until="domcontentloaded",
                timeout=30_000,
            )

        except PlaywrightTimeoutError as exc:
            logger.exception(
                "Amazon search request timed out: keyword=%s",
                keyword,
            )
            raise RequestError(
                f"Amazon search timed out for keyword: {keyword}"
            ) from exc

        except Exception as exc:
            logger.exception(
                "Amazon search request failed: keyword=%s",
                keyword,
            )
            raise RequestError(
                f"Amazon search failed for keyword: {keyword}"
            ) from exc

        status = response.status if response else None

        logger.info(
            "Amazon response status: %s",
            status,
        )

        logger.info(
            "Final page URL: %s",
            self.page.url,
        )

        title = self.page.title()

        logger.info(
            "Page title: %s",
            title,
        )

        self._validate_response(status, title)

        return self._extract_products()

    @staticmethod
    def _build_search_url(keyword: str) -> str:
        """Build the Amazon search URL."""

        encoded_keyword = quote_plus(keyword)

        return (
            f"{AmazonSearchCollector.BASE_URL}"
            f"/s?k={encoded_keyword}"
        )

    def _validate_response(
        self,
        status: int | None,
        title: str,
    ) -> None:
        """Validate the Amazon search response."""

        if status is None:
            raise RequestError(
                "Amazon returned no HTTP response."
            )

        if status in (403, 429):
            logger.warning(
                "Amazon blocked the request: status=%s",
                status,
            )

            raise BlockedPageError(
                f"Amazon blocked the request: HTTP {status}"
            )

        if status >= 500:
            logger.warning(
                "Amazon returned a temporary/server error: "
                "status=%s",
                status,
            )

            raise AmazonTemporaryError(
                f"Amazon returned HTTP {status}"
            )

        if "Sorry! Something went wrong" in title:
            logger.warning(
                "Amazon returned an error page."
            )

            raise AmazonTemporaryError(
                "Amazon returned an error page."
            )

    def _extract_products(self) -> list[ProductReference]:
        """Extract product references from the current search page."""

        products: list[ProductReference] = []

        product_cards = self.page.locator(
            'div[data-component-type="s-search-result"]'
        )

        count = product_cards.count()

        logger.info(
            "Amazon search result cards found: %d",
            count,
        )

        if count == 0:
            logger.warning(
                "No Amazon product cards found on search page."
            )
            return products

        for index in range(count):
            card = product_cards.nth(index)

            asin = card.get_attribute("data-asin")

            if not asin:
                logger.debug(
                    "Skipping product card without ASIN: index=%d",
                    index,
                )
                continue

            # -------------------------
            # Title
            # -------------------------

            title_locator = card.locator("h2")

            if title_locator.count() == 0:
                logger.warning(
                    "Skipping product without title: asin=%s",
                    asin,
                )
                continue

            title = title_locator.first.inner_text().strip()

            if not title:
                logger.warning(
                    "Skipping product with empty title: asin=%s",
                    asin,
                )
                continue

            # -------------------------
            # Product URL
            # -------------------------

            link_locator = card.locator(
                'a:has(h2)'
            )

            if link_locator.count() == 0:
                logger.warning(
                    "Skipping product without URL: asin=%s",
                    asin,
                )
                continue

            href = link_locator.first.get_attribute("href")

            if not href:
                logger.warning(
                    "Skipping product with empty URL: asin=%s",
                    asin,
                )
                continue

            product_url = self._build_product_url(href)

            # -------------------------
            # Product reference
            # -------------------------

            product = ProductReference(
                asin=asin,
                title=title,
                url=product_url,
            )

            products.append(product)

            logger.info(
                "Product reference collected: asin=%s",
                asin,
            )

        logger.info(
            "Amazon product references extracted: %d",
            len(products),
        )

        return products

    def _build_product_url(
        self,
        href: str,
    ) -> str:
        """Convert an Amazon search-result link to a product URL."""

        absolute_url = urljoin(self.BASE_URL, href)

        parsed_url = urlparse(absolute_url)

        # Amazon sponsored-product redirect
        if parsed_url.path.startswith("/sspa/click"):
            query_params = parse_qs(parsed_url.query)

            nested_url = query_params.get("url", [None])[0]

            if nested_url:
                return urljoin(self.BASE_URL, nested_url)

        return absolute_url