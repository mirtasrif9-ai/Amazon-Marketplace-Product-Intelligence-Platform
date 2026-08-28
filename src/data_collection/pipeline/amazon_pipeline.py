from __future__ import annotations

import logging

from src.data_collection.collectors.amazon.product_collector import (
    AmazonProductCollector,
)
from src.data_collection.collectors.amazon.search_collector import (
    AmazonSearchCollector,
)
from src.data_collection.models.product import Product
from src.data_collection.models.product_reference import ProductReference

from src.data_collection.storage.json_storage import JsonStorage

logger = logging.getLogger(__name__)


class AmazonPipeline:
    """Orchestrate Amazon search and product collection."""

    def __init__(
        self,
        search_collector: AmazonSearchCollector,
        product_collector: AmazonProductCollector,
        storage: JsonStorage,
    ) -> None:
        self.search_collector = search_collector
        self.product_collector = product_collector
        self.storage = storage

    def collect_product_references(
        self,
        keywords: list[str],
    ) -> list[ProductReference]:
        """Search all keywords and collect unique product references."""

        all_products: list[ProductReference] = []

        for index, keyword in enumerate(keywords, start=1):
            logger.info(
                "Searching keyword %d/%d: %s",
                index,
                len(keywords),
                keyword,
            )

            try:
                products = self.search_collector.search(keyword)

                logger.info(
                    "Keyword '%s' returned %d products.",
                    keyword,
                    len(products),
                )

                all_products.extend(products)

            except Exception:
                logger.exception(
                    "Failed to search keyword: %s",
                    keyword,
                )

        logger.info(
            "Total product references collected before "
            "deduplication: %d",
            len(all_products),
        )

        unique_products = self._deduplicate_products(
            all_products
        )

        logger.info(
            "Unique products after deduplication: %d",
            len(unique_products),
        )

        return unique_products

    @staticmethod
    def _deduplicate_products(
        products: list[ProductReference],
    ) -> list[ProductReference]:
        """Remove duplicate products using ASIN."""

        unique_products: dict[str, ProductReference] = {}

        for product in products:
            if product.asin not in unique_products:
                unique_products[product.asin] = product

        return list(unique_products.values())

    def collect_products(
        self,
        product_references: list[ProductReference],
    ) -> list[Product]:
        """Collect and incrementally save full product information."""

        products: list[Product] = []

        total = len(product_references)

        for index, reference in enumerate(
            product_references,
            start=1,
        ):
            logger.info(
                "Collecting product %d/%d: ASIN=%s",
                index,
                total,
                reference.asin,
            )

            try:
                product = self.product_collector.collect(
                    reference.url
                )

                products.append(product)

                # Save immediately after successful collection.
                self.storage.save_products(products)

                logger.info(
                    "Product %d/%d saved successfully: ASIN=%s",
                    index,
                    total,
                    product.asin,
                )

            except Exception:
                logger.exception(
                    "Failed to collect product %d/%d: "
                    "ASIN=%s URL=%s",
                    index,
                    total,
                    reference.asin,
                    reference.url,
                )

        logger.info(
            "Full product collection completed: %d/%d",
            len(products),
            total,
        )

        return products

    def run(
        self,
        keywords: list[str],
    ) -> list[Product]:
        """Run the complete Amazon collection pipeline."""

        logger.info(
            "Starting Amazon pipeline with %d keywords.",
            len(keywords),
        )

        # ---------------------------------------------------------
        # Phase 1: Search all keywords
        # ---------------------------------------------------------

        product_references = self.collect_product_references(
            keywords
        )

        self.storage.save_product_references(
            product_references
        )

        logger.info(
            "Search phase completed. "
            "Unique product references: %d",
            len(product_references),
        )

        # ---------------------------------------------------------
        # Phase 2: Load references from JSON
        # ---------------------------------------------------------

        product_references = self.storage.load_product_references()

        logger.info(
            "Loaded product references for product collection: %d",
            len(product_references),
        )

        # ---------------------------------------------------------
        # Phase 3: Collect full product information
        # ---------------------------------------------------------

        products = self.collect_products(
            product_references
        )


        logger.info(
            "Amazon pipeline completed. "
            "Products collected: %d",
            len(products),
        )

        return products