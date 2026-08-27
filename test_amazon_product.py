from src.common.logger import setup_logging
from src.data_collection.browser.playwright_manager import (
    PlaywrightManager,
)
from src.data_collection.collectors.amazon.product_collector import (
    AmazonProductCollector,
)


setup_logging()

manager = PlaywrightManager(headless=False)

try:
    print("[1] Starting browser...")

    page = manager.start()

    print("[2] Browser started.")

    collector = AmazonProductCollector(page)

    print("[3] Starting product collection...")

    product = collector.collect(
        # url="https://www.amazon.com/dp/B0DJ8RRKVX",
        url="https://www.amazon.com/dp/B0HDN7ZZWT"
    )

    print("[4] Product collected.")

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
    print(f"Video URL: {product.video_url or 'No video available'}")
    print("\nReviews:")

    for index, review in enumerate(product.reviews, start=1):
        print(f"\nReview {index}")
        print(f"Reviewer: {review.reviewer_name}")
        print(f"Rating: {review.star_rating}")
        print(f"Title: {review.review_title}")
        print(f"Description: {review.review_description}")

    input("\nPress ENTER to close the browser...")

finally:
    manager.close()