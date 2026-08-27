from __future__ import annotations

import logging
import time
import re

from urllib.parse import quote_plus

from playwright.sync_api import (
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

from src.common.exceptions import (
    AmazonTemporaryError,
    BlockedPageError,
    ParsingError,
    RequestError,
)
from src.data_collection.collectors.base import BaseProductCollector
from src.data_collection.models.product import Product
from src.data_collection.models.review import Review


logger = logging.getLogger(__name__)


class AmazonProductCollector(BaseProductCollector):
    """Collect product information from an Amazon product page."""

    BASE_URL = "https://www.amazon.com"

    def __init__(self, page: Page) -> None:
        self.page = page

    def collect(
        self,
        url: str,
        search_keyword: str | None = None,
    ) -> Product:
        """
        Collect product information from a single Amazon product URL.

        If search_keyword is provided, the same browser page first
        visits the Amazon search page and then navigates to the
        requested product.
        """

        logger.info(
            "Starting Amazon product collection: url=%s",
            url,
        )

        # ---------------------------------------------------------
        # Step 1: Optionally open Amazon search page first
        # ---------------------------------------------------------

        
        # ---------------------------------------------------------
        # Step 2: Open product page
        # ---------------------------------------------------------

        try:
            self.page.goto(self.BASE_URL)
            time.sleep(5)
            response = self.page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30_000,
            )

        except PlaywrightTimeoutError as exc:
            logger.exception(
                "Amazon product request timed out: url=%s",
                url,
            )

            raise RequestError(
                f"Amazon product request timed out: {url}"
            ) from exc

        except Exception as exc:
            logger.exception(
                "Amazon product request failed: url=%s",
                url,
            )

            raise RequestError(
                f"Amazon product request failed: {url}"
            ) from exc

        status = response.status if response else None

        logger.info(
            "Amazon product response status: %s",
            status,
        )

        logger.info(
            "Final product page URL: %s",
            self.page.url,
        )

        title = self.page.title()

        logger.info(
            "Product page title: %s",
            title,
        )

        # ---------------------------------------------------------
        # Step 3: Validate response
        # ---------------------------------------------------------

        self._validate_response(status, title)

        # ---------------------------------------------------------
        # Step 4: Make sure this is actually a product page
        # ---------------------------------------------------------

        self._validate_product_page()

        # ---------------------------------------------------------
        # Step 5: Extract product
        # ---------------------------------------------------------

        return self._extract_product()



    def _validate_response(
        self,
        status: int | None,
        title: str,
    ) -> None:
        """Validate an Amazon response."""

        if status is None:
            raise RequestError(
                "Amazon returned no HTTP response."
            )

        if status in (403, 429):
            logger.warning(
                "Amazon blocked product request: status=%s",
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

    def _validate_product_page(self) -> None:
        """Verify that Amazon actually returned a product page."""

        current_url = self.page.url

        # The requested page should normally be a product URL.
        if "/dp/" not in current_url:
            logger.warning(
                "Expected Amazon product URL but got: %s",
                current_url,
            )

            raise BlockedPageError(
                "Amazon did not return the requested product page."
            )

        title_locator = self.page.locator(
            "#productTitle"
        )

        if title_locator.count() == 0:
            logger.warning(
                "Amazon product page does not contain "
                "#productTitle."
            )

            raise BlockedPageError(
                "Amazon product page was not available."
            )

        logger.info(
            "Amazon product page validated successfully."
        )

    def _extract_product(self) -> Product:
        """Extract product information from the current page."""

        asin = self._extract_asin()
        title = self._extract_title()
        brand = self._extract_brand()
        price = self._extract_price()
        review_count = self._extract_review_count()
        average_rating = self._extract_average_rating()
        video_url = self._extract_video_url()
        reviews = self._extract_reviews()

        logger.info(
            "Product identity extracted: asin=%s",
            asin,
        )

        logger.info(
            "Product details extracted: "
            "brand=%s price=%s review_count=%s "
            "average_rating=%s video=%s reviews=%s",
            brand,
            price,
            review_count,
            average_rating,
            bool(video_url),
            len(reviews),
        )

        return Product(
            asin=asin,
            title=title,
            description="",
            brand=brand,
            price=price,
            image="",
            review_count=review_count,
            average_rating=average_rating,
            video_url=video_url,
            reviews=reviews,
        )

    def _extract_asin(self) -> str:
        """Extract ASIN from the product page."""

        asin = self.page.locator(
            "#ASIN"
        ).get_attribute("value")

        if asin:
            return asin.strip()

        asin = self.page.locator(
            '[name="ASIN"]'
        ).get_attribute("value")

        if asin:
            return asin.strip()

        raise ParsingError(
            "Could not extract ASIN from Amazon product page."
        )

    def _extract_title(self) -> str:
        """Extract product title."""

        title_locator = self.page.locator(
            "#productTitle"
        )

        if title_locator.count() == 0:
            raise ParsingError(
                "Could not find product title."
            )

        title = title_locator.first.inner_text().strip()

        if not title:
            raise ParsingError(
                "Product title is empty."
            )

        return title

    def _extract_brand(self) -> str:
        """Extract product brand."""

        selectors = [
            "#bylineInfo",
            "#brand",
            "a#bylineInfo",
        ]

        for selector in selectors:
            locator = self.page.locator(selector)

            if locator.count() == 0:
                continue

            brand = locator.first.inner_text().strip()

            if brand:
                # Amazon often returns values like:
                # "Visit the LISEN Store"
                brand = brand.replace("Visit the ", "")
                brand = brand.replace(" Store", "")
                return brand.strip()

        logger.warning("Could not extract product brand.")

        return ""

    def _extract_price(self) -> float:
        """Extract product price."""

        selectors = [
            "#corePriceDisplay_desktop_feature_div .a-offscreen",
            "#corePriceDisplay_mobile_feature_div .a-offscreen",
            ".a-price .a-offscreen",
            "#priceblock_ourprice",
            "#priceblock_dealprice",
        ]

        for selector in selectors:
            locator = self.page.locator(selector)

            if locator.count() == 0:
                continue

            price_text = locator.first.inner_text().strip()

            if not price_text:
                continue

            try:
                # Example:
                # "$15.99"
                # "$29.99"
                cleaned_price = (
                    price_text
                    .replace("$", "")
                    .replace(",", "")
                    .strip()
                )

                return float(cleaned_price)

            except ValueError:
                logger.warning(
                    "Could not parse product price: %s",
                    price_text,
                )

        logger.warning("Could not extract product price.")

        return 0.0

    def _extract_image(self) -> str:
        """Extract the main product image URL."""

        selectors = [
            "#landingImage",
            "#imgBlkFront",
            "#main-image",
        ]

        for selector in selectors:
            locator = self.page.locator(selector)

            if locator.count() == 0:
                continue

            image_url = locator.first.get_attribute("src")

            if image_url:
                return image_url.strip()

        logger.warning("Could not extract product image.")

        return ""

    def _extract_description(self) -> str:
        """Extract product description."""

        selectors = [
            "#productDescription",
            "#feature-bullets",
        ]

        for selector in selectors:
            locator = self.page.locator(selector)

            if locator.count() == 0:
                continue

            description = locator.first.inner_text().strip()

            if description:
                return description

        logger.warning("Could not extract product description.")

        return ""

    def _extract_review_count(self) -> int:
        """Extract total customer review count."""

        selectors = [
            "#acrCustomerReviewText",
            "#averageCustomerReviews .a-size-base",
        ]

        for selector in selectors:
            locator = self.page.locator(selector).first

            if locator.count() == 0:
                continue

            # Try aria-label first, fallback to inner text
            review_text = (
                locator.get_attribute("aria-label") or locator.inner_text() or ""
            ).strip()

            if not review_text:
                continue

            # Extract only the numeric digits, discarding parentheses, commas, and text
            digits_only = re.sub(r"[^\d]", "", review_text)

            if digits_only:
                try:
                    return int(digits_only)
                except ValueError:
                    pass

            logger.warning(
                "Could not parse review count from text: %s",
                review_text,
            )

        logger.warning("Could not extract review count.")
        return 0

    def _extract_average_rating(self) -> float:
        """Extract the product's average customer rating."""

        selectors = [
            "#acrPopover",
            "#averageCustomerReviews .a-icon-alt",
        ]

        for selector in selectors:
            locator = self.page.locator(selector)

            if locator.count() == 0:
                continue

            rating_text = (
                locator.first.get_attribute("title")
                or locator.first.inner_text()
            ).strip()

            if not rating_text:
                continue

            try:
                # Example:
                # "4.8 out of 5 stars"

                rating = float(rating_text.split()[0])

                if 0 <= rating <= 5:
                    return rating

            except (ValueError, IndexError):
                logger.warning(
                    "Could not parse average rating: %s",
                    rating_text,
                )

        logger.warning(
            "Could not extract average product rating."
        )

        return 0.0


    def _extract_video_url(self) -> str:
        """Extract the first available product video URL."""

        selectors = [
            "#videoBlock video",
            "#va-related-video video",
            "video",
        ]

        for selector in selectors:
            videos = self.page.locator(selector)

            count = videos.count()

            if count == 0:
                continue

            for index in range(count):
                video = videos.nth(index)

                video_url = (
                    video.get_attribute("src")
                    or video.get_attribute("data-video-url")
                    or video.get_attribute("data-src")
                )

                if video_url:
                    logger.info(
                        "Product video URL extracted."
                    )

                    return video_url.strip()

                # Sometimes the URL is stored in a <source>.
                source = video.locator("source")

                if source.count() > 0:
                    source_url = source.first.get_attribute("src")

                    if source_url:
                        logger.info(
                            "Product video URL extracted from source."
                        )

                        return source_url.strip()

        logger.info(
            "No product video found."
        )

        return ""


    def _extract_reviews(self) -> list[Review]:
        """Extract customer reviews from the current product page."""

        reviews: list[Review] = []

        review_cards = self.page.locator(
            'div[data-hook="review"]'
        )

        count = review_cards.count()

        logger.info(
            "Amazon reviews found on current page: %d",
            count,
        )

        # For initial testing, extract only the first 5 reviews.
        max_reviews = min(count, 5)

        for index in range(max_reviews):
            card = review_cards.nth(index)

            # Reviewer name
            reviewer_locator = card.locator(
                '[data-hook="genome-widget"] .a-profile-name'
            )

            reviewer_name = ""

            if reviewer_locator.count() > 0:
                reviewer_name = (
                    reviewer_locator.first.inner_text().strip()
                )

            # Star rating
            rating_locator = card.locator(
                '[data-hook="review-star-rating"] .a-icon-alt'
            )

            star_rating = 0.0

            if rating_locator.count() > 0:
                rating_text = (
                    rating_locator.first.inner_text().strip()
                )

                try:
                    star_rating = float(
                        rating_text.split()[0]
                    )
                except (ValueError, IndexError):
                    logger.warning(
                        "Could not parse review rating: index=%d",
                        index,
                    )

            # Review title
            title_locator = card.locator(
                '[data-hook="reviewTitle"]'
            )

            review_title = ""

            if title_locator.count() > 0:
                review_title = (
                    title_locator.first.inner_text().strip()
                )

            # Review description
            description_locator = card.locator(
                '[data-hook="reviewText"]'
            )

            review_description = ""

            if description_locator.count() > 0:
                review_description = (
                    description_locator.first.inner_text().strip()
                )

            review = Review(
                reviewer_name=reviewer_name,
                star_rating=star_rating,
                review_title=review_title,
                review_description=review_description,
            )

            reviews.append(review)

            logger.info(
                "Review extracted: reviewer=%s rating=%s",
                reviewer_name,
                star_rating,
            )

        logger.info(
            "Total reviews extracted: %d",
            len(reviews),
        )

        return reviews