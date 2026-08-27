import logging

from src.common.logger import setup_logging
from src.data_collection.browser.playwright_manager import PlaywrightManager
from src.data_collection.collectors.amazon.search_collector import AmazonSearchCollector


setup_logging()

logger = logging.getLogger(__name__)


browser_manager = PlaywrightManager(headless=False)

try:
    print("\n[1] Starting browser...")

    page = browser_manager.start()

    print("[2] Browser started.")
    print("[3] Opening Amazon...")

    collector = AmazonSearchCollector(page)

    print("[4] Collector created.")
    print("[5] Starting Amazon search...")

    products = collector.search("wireless headphones")

    print("[6] Search completed.")

    print("\nProducts:")

    for product in products:
        print(product)

except Exception as exc:
    print("\n!!! ERROR OCCURRED !!!")
    print(type(exc).__name__)
    print(str(exc))

    logger.exception("Amazon search test failed.")

finally:
    input("\nPress ENTER to close the browser...")
    browser_manager.close()