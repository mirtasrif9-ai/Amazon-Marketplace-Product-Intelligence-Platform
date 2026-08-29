from __future__ import annotations

import logging
import time
import re

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
            time.sleep(2)
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

        return self._extract_products(keyword)

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

    def _extract_products(
        self,
        keyword: str,
    ) -> list[ProductReference]:
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
                "No Amazon product cards found on search page: "
                "keyword=%s",
                keyword,
            )
            return products

        for index in range(count):
            card = product_cards.nth(index)

            # -------------------------
            # ASIN
            # -------------------------

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
                "a:has(h2)"
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
            # Price
            # -------------------------

            price = self._extract_price(card, asin)

            # -------------------------
            # Product reference
            # -------------------------
            product_number = 1
            product = ProductReference(
                product_number=product_number,
                asin=asin.strip(),
                search_keyword=keyword,
                title=title,
                url=product_url,
                price=price,
            )

            products.append(product)

            logger.info(
                "Product reference collected: "
                "keyword=%s asin=%s price=%.2f",
                keyword,
                asin,
                price,
            )

        logger.info(
            "Amazon product references extracted: "
            "keyword=%s count=%d",
            keyword,
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




    def _extract_price(
        self,
        card,
        asin: str,
    ) -> float:
        """Extract product price from a search result card."""

        # 1. Primary Attempt: Target explicit price locator
        try:
            price_locator = card.locator(".a-price .a-offscreen").first

            if price_locator.count() > 0:
                price_text = price_locator.inner_text().strip()
                cleaned_price = (
                    price_text
                    .replace("$", "")
                    .replace(",", "")
                    .strip()
                )
                price = float(cleaned_price)
                if price > 0.0:
                    return price
        except Exception as e:
            logger.debug("Primary price extraction failed: asin=%s, error=%s", asin, e)

        # 2. Fallback Attempt: Extract first numeric price pattern from card text
        try:
            card_text = card.inner_text()
            
            # Matches patterns like $25.71 or $1,250.00 and extracts just the numbers/decimal
            match = re.search(r'\$\s*([\d,]+(?:\.\d{2})?)', card_text)
            
            if match:
                fallback_price_str = match.group(1).replace(",", "")
                fallback_price = float(fallback_price_str)
                if fallback_price > 0.0:
                    logger.info("Extracted price via fallback regex: asin=%s, price=%s", asin, fallback_price)
                    return fallback_price
        except Exception as e:
            logger.debug("Fallback price extraction failed: asin=%s, error=%s", asin, e)

        # 3. Return 0.0 if both methods fail
        logger.warning("Could not extract price: asin=%s", asin)
        return 0.0