from __future__ import annotations

import logging
import time
import re

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
from src.data_collection.models.product_reference import (
    ProductReference,
)


logger = logging.getLogger(__name__)


class AmazonProductCollector(BaseProductCollector):
    """Collect product information from an Amazon product page."""

    BASE_URL = "https://www.amazon.com"

    def __init__(self, page: Page) -> None:
        self.page = page

    def collect(
        self,
        reference: ProductReference,
    ) -> Product:
        """
        Collect detailed product information using a product reference.

        ASIN, title, price, and URL come from the search collector.
        Only additional product details are extracted from the product page.
        """

        logger.info(
            "Starting Amazon product collection: "
            "asin=%s keyword=%s url=%s",
            reference.asin,
            reference.search_keyword,
            reference.url,
        )

        try:
            self.page.goto(self.BASE_URL)

            time.sleep(1)

            response = self.page.goto(
                reference.url,
                wait_until="domcontentloaded",
                timeout=30_000,
            )

        except PlaywrightTimeoutError as exc:
            logger.exception(
                "Amazon product request timed out: asin=%s",
                reference.asin,
            )

            raise RequestError(
                f"Amazon product request timed out: {reference.url}"
            ) from exc

        except Exception as exc:
            logger.exception(
                "Amazon product request failed: asin=%s",
                reference.asin,
            )

            raise RequestError(
                f"Amazon product request failed: {reference.url}"
            ) from exc

        status = response.status if response else None

        logger.info(
            "Amazon product response status: asin=%s status=%s",
            reference.asin,
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

        self._validate_response(status, title)

        self._validate_product_page()

        return self._extract_product(reference)



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

    def _extract_product(
        self,
        reference: ProductReference,
    ) -> Product:
        """Extract additional product information."""

        description = self._extract_description()
        brand = self._extract_brand()
        image = self._extract_image()

        review_count = self._extract_review_count()
        average_rating = self._extract_average_rating()
        video_url = self.get_clean_video_url()
        reviews = self._extract_reviews()

        logger.info(
            "Product details extracted: "
            "product_number=%s asin=%s brand=%s "
            "review_count=%s average_rating=%s "
            "video=%s reviews=%s",
            reference.product_number,
            reference.asin,
            brand,
            review_count,
            average_rating,
            bool(video_url),
            len(reviews),
        )

        return Product(
            product_number=reference.product_number,
            search_keyword=reference.search_keyword,
            asin=reference.asin,
            title=reference.title,
            product_url=reference.url,
            price=reference.price,
            description=description,
            brand=brand,
            image=image,
            review_count=review_count,
            average_rating=average_rating,
            video_url=video_url,
            reviews=reviews,
        )


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
            "#a-unordered-list"
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
            """
            Extract real media URLs (.mp4 or .m3u8), bypassing dynamic 'blob:' references.
            Uses embedded JSON state parsing and dynamic DOM scrolling.
            """
            # Approach A: Parse embedded JSON configs inside <script> tags
            try:
                content = self.page.content()
                # Capture .mp4 or .m3u8 URLs while handling JSON-escaped slashes (\/)
                pattern = r'https?:\\?/\\?/[^"\'\s<>]+?\.(?:mp4|m3u8)(?:\?[^"\'\s<>]*)?'
                matches = re.findall(pattern, content)

                for raw_url in matches:
                    clean_url = raw_url.replace('\\/', '/')
                    if not clean_url.startswith("blob:"):
                        logger.info("Product video URL extracted via script parsing.")
                        return clean_url
            except Exception as exc:
                logger.warning("Script parsing for video URL failed: %s", exc)

            # Approach B: Scroll to video containers to lazy-load elements
            try:
                for video_container in ["#videoBlock", "#vse-video-container", "#video-block_feature_div"]:
                    locator = self.page.locator(video_container)
                    if locator.count() > 0:
                        locator.first.scroll_into_view_if_needed()
                        self.page.wait_for_timeout(1500)

                video_elements = self.page.locator("video")
                count = video_elements.count()

                for index in range(count):
                    video = video_elements.nth(index)
                    
                    # Check for direct source attributes
                    for attr in ["src", "data-video-url", "data-src"]:
                        video_url = video.get_attribute(attr)
                        if video_url and not video_url.startswith("blob:"):
                            logger.info("Product video URL extracted from DOM attribute: %s", attr)
                            return video_url.strip()

                    # Check nested <source> tags
                    source = video.locator("source")
                    if source.count() > 0:
                        source_url = source.first.get_attribute("src")
                        if source_url and not source_url.startswith("blob:"):
                            logger.info("Product video URL extracted from source tag.")
                            return source_url.strip()
            except Exception as exc:
                logger.warning("DOM inspection for video URL failed: %s", exc)

            logger.info("No valid non-blob product video found.")
            return "" 
        
    def get_clean_video_url(self) -> str:
            raw_text = self._extract_video_url()
            # Find all URLs starting with http/https, excluding quotes and HTML entities
            urls = re.findall(r'https?://[^\s"&]+', raw_text)
            
            # Returns the last URL if matches are found, otherwise returns the raw text or empty string
            return urls[-1] if urls else raw_text

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

        # For initial testing, extract only the first 8 reviews.
        max_reviews = min(count, 8)

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