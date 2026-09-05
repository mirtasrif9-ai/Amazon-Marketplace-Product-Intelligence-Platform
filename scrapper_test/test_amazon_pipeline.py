from __future__ import annotations

import logging

from src.common.logger import setup_logging
from src.data_collection.browser.playwright_manager import (
    PlaywrightManager,
)
from src.data_collection.collectors.amazon.product_collector import (
    AmazonProductCollector,
)
from src.data_collection.collectors.amazon.search_collector import (
    AmazonSearchCollector,
)
from src.data_collection.keyword_reader import KeywordReader
from src.data_collection.pipeline.amazon_pipeline import AmazonPipeline
from src.data_collection.storage.json_storage import JsonStorage


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

setup_logging()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

KEYWORD_FILE = "data/input/test_keywords.xlsx"
OUTPUT_DIRECTORY = "data/output"


# ---------------------------------------------------------
# Start browser
# ---------------------------------------------------------

browser_manager = PlaywrightManager(
    headless=False
)


try:
    print("\n[1] Starting browser...")

    page = browser_manager.start()

    print("[2] Browser started.")

    # -----------------------------------------------------
    # Read keywords
    # -----------------------------------------------------

    keyword_reader = KeywordReader()

    keywords = keyword_reader.read(
        KEYWORD_FILE
    )

    print(
        f"[3] Loaded {len(keywords)} unique keywords."
    )

    # -----------------------------------------------------
    # Create collectors and storage
    # -----------------------------------------------------

    search_collector = AmazonSearchCollector(
        page
    )

    product_collector = AmazonProductCollector(
        page
    )

    storage = JsonStorage(
        output_directory=OUTPUT_DIRECTORY
    )

    pipeline = AmazonPipeline(
        search_collector=search_collector,
        product_collector=product_collector,
        storage=storage,
    )

    print("[4] Pipeline created.")

    # =====================================================
    # PHASE 1
    # Search all keywords
    # =====================================================

    print(
        f"\n[5] Starting Amazon search for "
        f"{len(keywords)} keywords..."
    )

    logger.info(
        "Starting product reference collection for %d keywords.",
        len(keywords),
    )

    pipeline.collect_product_references(
        keywords
    )

    print(
        "\n[6] Product reference collection completed."
    )

    print(
        "Product references have been saved to:"
    )

    print(
        f"    {OUTPUT_DIRECTORY}/product_references.json"
    )

    # -----------------------------------------------------
    # Show number of references saved
    # -----------------------------------------------------

    try:
        product_references = (
            storage.load_product_references()
        )

        print(
            f"\nTotal unique product references: "
            f"{len(product_references)}"
        )

    except FileNotFoundError:
        print(
            "\nNo product references file was found."
        )

        logger.error(
            "product_references.json was not found "
            "after reference collection."
        )

        product_references = []

    # =====================================================
    # ASK USER WHETHER TO RUN PRODUCT COLLECTOR
    # =====================================================

    if product_references:

        print(
            "\n---------------------------------------------"
        )

        choice = input(
            "\nDo you want to run the product collector "
            "now? (y/n): "
        ).strip().lower()

        print()

        if choice in ("y", "yes"):

            # =================================================
            # PHASE 2
            # Collect complete product information
            # =================================================

            print(
                "[7] Starting full product collection..."
            )

            logger.info(
                "User selected YES. "
                "Starting full product collection."
            )

            products = pipeline.collect_products(
                product_references
            )

            print(
                "\n[8] Product collection completed."
            )

            print(
                f"New products collected in this run: "
                f"{len(products)}"
            )

            print(
                "Complete product information has been "
                "saved incrementally to:"
            )

            print(
                f"    {OUTPUT_DIRECTORY}/products.json"
            )

        else:

            # =================================================
            # STOP AFTER REFERENCE COLLECTION
            # =================================================

            print(
                "[7] Product collector skipped."
            )

            print(
                "\nProduct reference collection is complete."
            )

            print(
                "The program will stop here."
            )

            logger.info(
                "User selected NO. "
                "Product collection skipped."
            )

    else:

        print(
            "\nNo product references are available."
        )

        logger.warning(
            "No product references available. "
            "Product collector will not run."
        )


except Exception as exc:

    print(
        "\n!!! ERROR OCCURRED !!!"
    )

    print(
        type(exc).__name__
    )

    print(
        str(exc)
    )

    logger.exception(
        "Amazon pipeline test failed."
    )


finally:

    input(
        "\nPress ENTER to close the browser..."
    )

    browser_manager.close()