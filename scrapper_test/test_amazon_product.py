from src.common.logger import setup_logging
from src.data_collection.browser.playwright_manager import (
    PlaywrightManager,
)
from src.data_collection.collectors.amazon.product_collector import (
    AmazonProductCollector,
)


setup_logging()

manager = PlaywrightManager(headless=False)

product_urls = [
    "https://www.amazon.com/dp/B00LH3DMUO",
    "https://www.amazon.com/dp/B0DJ8RRKVX",
    "https://www.amazon.com/dp/B0CG1LGWR6",
]

try:
    print("[1] Starting browser...")

    page = manager.start()

    print("[2] Browser started.")

    collector = AmazonProductCollector(page)

    for index, url in enumerate(product_urls, start=1):

        print()
        print("=" * 70)
        print(f"[3.{index}] Starting product collection...")
        print(f"URL: {url}")
        print("=" * 70)

        try:
            product = collector.collect(url=url)

            print(f"\n[{index}] Product collected.")

            print()
            print("Product:")
            print(f"ASIN: {product.asin}")
            print(f"Title: {product.title}")
            print(f"Brand: {product.brand}")
            print(f"Price: ${product.price:.2f}")
            print(f"Image: {product.image}")
            print(f"Review Count: {product.review_count}")
            print(f"Description: {product.description[:300]}")
            print(f"Average Rating: {product.average_rating}")
            print(
                f"Video URL: "
                f"{product.video_url or 'No video available'}"
            )

            print(f"\nReviews: {len(product.reviews)}")

            for review_index, review in enumerate(
                product.reviews,
                start=1,
            ):
                print(f"\nReview {review_index}")
                print(f"Reviewer: {review.reviewer_name}")
                print(f"Rating: {review.star_rating}")
                print(f"Title: {review.review_title}")
                print(
                    f"Description: "
                    f"{review.review_description}"
                )

        except Exception as exc:
            print(
                f"\n[ERROR] Failed to collect product "
                f"{index}: {url}"
            )
            print(f"Reason: {exc}")

            # Continue with the next product.
            continue

    input("\nPress ENTER to close the browser...")

finally:
    manager.close()