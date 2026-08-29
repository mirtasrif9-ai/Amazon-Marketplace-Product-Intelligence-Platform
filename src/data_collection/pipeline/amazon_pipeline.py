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

    # =========================================================
    # PHASE 1
    # Search all keywords and save product references
    # =========================================================

    def collect_product_references(
        self,
        keywords: list[str],
    ) -> list[ProductReference]:
        """
        Search all keywords and incrementally save product
        references after each keyword.
        """

        all_products: list[ProductReference] = []

        total_keywords = len(keywords)

        for index, keyword in enumerate(
            keywords,
            start=1,
        ):
            logger.info(
                "========== KEYWORD %d/%d ==========",
                index,
                total_keywords,
            )

            logger.info(
                "Starting keyword search: keyword=%s",
                keyword,
            )

            try:
                products = self.search_collector.search(
                    keyword
                )

                logger.info(
                    "Keyword search completed: "
                    "keyword=%s products_found=%d",
                    keyword,
                    len(products),
                )

                # -------------------------------------------------
                # Immediately save products found for this keyword.
                # This protects already collected data if the
                # pipeline stops later.
                # -------------------------------------------------

                (
                    new_count,
                    total_count,
                    duplicate_count,
                ) = self.storage.append_product_references(
                    products
                )

                all_products.extend(products)

                logger.info(
                    "Keyword %d/%d completed: "
                    "keyword=%s found=%d new=%d "
                    " total_saved=%d",
                    index,
                    total_keywords,
                    keyword,
                    len(products),
                    new_count,
                    total_count,
                )

            except Exception:
                logger.exception(
                    "Failed to process keyword %d/%d: "
                    "keyword=%s",
                    index,
                    total_keywords,
                    keyword,
                )

                # Continue with the next keyword.
                continue

        logger.info(
            "All keyword searches completed."
        )

        return all_products

    # =========================================================
    # PHASE 2
    # Collect complete product information
    # =========================================================

    def collect_products(
        self,
        product_references: list[ProductReference],
    ) -> list[Product]:
        """Collect and incrementally save full product information."""

        products: list[Product] = []

        total = len(product_references)

        logger.info(
            "Starting full product collection: total=%d",
            total,
        )

        # ---------------------------------------------------------
        # Load ASINs already present in products.json
        # ---------------------------------------------------------

        existing_asins = (
            self.storage.load_existing_product_asins()
        )

        logger.info(
            "Already collected products: %d",
            len(existing_asins),
        )

        # ---------------------------------------------------------
        # Process product references one by one
        # ---------------------------------------------------------

        for index, reference in enumerate(
            product_references,
            start=1,
        ):

            logger.info(
                "========== PRODUCT %d/%d ==========",
                index,
                total,
            )

            logger.info(
                "Processing product: "
                "product_number=%d asin=%s keyword=%s",
                reference.product_number,
                reference.asin,
                reference.search_keyword,
            )

            # -----------------------------------------------------
            # Skip products already in products.json
            # -----------------------------------------------------

            if reference.asin in existing_asins:
                logger.info(
                    "Skipping already collected product: "
                    "product_number=%d asin=%s",
                    reference.product_number,
                    reference.asin,
                )
                continue

            # -----------------------------------------------------
            # Collect full product information
            # -----------------------------------------------------

            try:
                product = self.product_collector.collect(
                    reference
                )

                logger.info(
                    "Product information extracted: "
                    "product_number=%d asin=%s",
                    reference.product_number,
                    reference.asin,
                )

                # -------------------------------------------------
                # Immediately save the product
                # -------------------------------------------------

                saved, total_saved, _ = (
                    self.storage.append_product(
                        product
                    )
                )

                if saved:
                    products.append(product)

                    existing_asins.add(
                        reference.asin
                    )

                    logger.info(
                        "Product saved successfully: "
                        "product_number=%d asin=%s "
                        "total_saved=%d",
                        reference.product_number,
                        reference.asin,
                        total_saved,
                    )

                else:
                    logger.info(
                        "Product was already present. "
                        "Skipping save: asin=%s",
                        reference.asin,
                    )

            except Exception:
                logger.exception(
                    "Failed to collect product: "
                    "product_number=%d asin=%s url=%s",
                    reference.product_number,
                    reference.asin,
                    reference.url,
                )

                # Continue with the next product.
                continue

        logger.info(
            "Full product collection completed: "
            "newly_collected=%d total_references=%d",
            len(products),
            total,
        )

        return products

    # =========================================================
    # MAIN PIPELINE
    # =========================================================

    def run(
        self,
        keywords: list[str],
    ) -> list[Product]:
        """
        Run the complete Amazon collection pipeline.

        Phase 1:
            Search every keyword and save product references.

        Phase 2:
            Load all references and collect full product
            information one product at a time.
        """

        logger.info(
            "Starting Amazon pipeline with %d keywords.",
            len(keywords),
        )

        # =====================================================
        # PHASE 1
        # =====================================================

        logger.info(
            "========== PHASE 1: "
            "PRODUCT REFERENCE COLLECTION =========="
        )

        self.collect_product_references(
            keywords
        )

        # =====================================================
        # PHASE 2
        # Load product references from JSON.
        # =====================================================

        logger.info(
            "========== PHASE 2: "
            "LOADING PRODUCT REFERENCES =========="
        )

        try:
            product_references = (
                self.storage.load_product_references()
            )

        except FileNotFoundError:
            logger.error(
                "Product references file was not found. "
                "Cannot continue to product collection."
            )

            return []

        logger.info(
            "Total product references available: %d",
            len(product_references),
        )

        # =====================================================
        # PHASE 3
        # Collect complete product information.
        # =====================================================

        logger.info(
            "========== PHASE 3: "
            "FULL PRODUCT COLLECTION =========="
        )

        products = self.collect_products(
            product_references
        )

        # =====================================================
        # COMPLETED
        # =====================================================

        logger.info(
            "========== AMAZON PIPELINE COMPLETED =========="
        )

        logger.info(
            "New products collected in this run: %d",
            len(products),
        )

        return products