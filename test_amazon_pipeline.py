import logging

from src.common.logger import setup_logging
from src.data_collection.browser.playwright_manager import (
    PlaywrightManager,
)
from src.data_collection.collectors.amazon.search_collector import (
    AmazonSearchCollector,
)
from src.data_collection.collectors.amazon.product_collector import (
    AmazonProductCollector,
)
from src.data_collection.pipeline.amazon_pipeline import (
    AmazonPipeline,
)
from src.data_collection.storage.json_storage import (
    JsonStorage,
)
from src.data_collection.keyword_reader import (
    KeywordReader,
)


setup_logging()

logger = logging.getLogger(__name__)

browser_manager = PlaywrightManager(headless=False)

try:
    print("\n[1] Starting browser...")

    page = browser_manager.start()

    print("[2] Browser started.")

    search_collector = AmazonSearchCollector(page)

    product_collector = AmazonProductCollector(page)

    storage = JsonStorage(
        output_directory="data/output"
    )

    keyword_reader = KeywordReader()

    keyword_file = "data/input/test_keywords.xlsx"

    keywords = keyword_reader.read(keyword_file)

    print(
        f"[3] Loaded {len(keywords)} unique keywords."
    )

    pipeline = AmazonPipeline(
        search_collector=search_collector,
        product_collector=product_collector,
        storage=storage,
    )

    print("[4] Pipeline created.")

    print(
        f"[5] Starting Amazon pipeline with "
        f"{len(keywords)} keywords..."
    )

    products = pipeline.run(keywords)

    print("\n[6] Pipeline completed.")

    print(
        f"Total full products collected: "
        f"{len(products)}"
    )

except Exception as exc:
    print("\n!!! ERROR OCCURRED !!!")
    print(type(exc).__name__)
    print(str(exc))

    logger.exception(
        "Amazon pipeline test failed."
    )

finally:
    input(
        "\nPress ENTER to close the browser..."
    )

    browser_manager.close()