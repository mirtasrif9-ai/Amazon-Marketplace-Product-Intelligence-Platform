from src.common.logger import setup_logging
from src.data_collection.browser.playwright_manager import PlaywrightManager


setup_logging()

browser_manager = PlaywrightManager(headless=False)

try:
    page = browser_manager.start()

    url = "https://www.amazon.com/s?k=wireless+headphones"

    print("Opening Amazon...")
    response = page.goto(
        url,
        wait_until="domcontentloaded",
        timeout=30000,
    )

    print(f"HTTP status: {response.status if response else 'None'}")
    print(f"Final URL: {page.url}")
    print(f"Page title: {page.title()}")

    # Wait for Amazon product cards.
    page.wait_for_selector(
        'div[data-component-type="s-search-result"]',
        timeout=15000,
    )

    cards = page.locator(
        'div[data-component-type="s-search-result"]'
    )

    print(f"\nProduct cards found: {cards.count()}")

    if cards.count() > 0:
        first_card = cards.nth(0)

        print("\n========== FIRST CARD HTML ==========\n")

        print(
            first_card.evaluate(
                "(element) => element.outerHTML"
            )
        )

        print("\n========== END CARD HTML ==========\n")

    input("Press ENTER to close browser...")

finally:
    browser_manager.close()