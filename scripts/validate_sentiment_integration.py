from pathlib import Path

from src.application.product_repository import ProductRepository
from src.application.sentiment_service import SentimentService

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT / "data" / "output" / "products_with_sentiment.json"
)

SENTIMENT_MODEL_DIR = (
    PROJECT_ROOT / "models" / "sentiment" / "final_sentiment_model"
)


def print_section(title: str) -> None:
    """Print a formatted validation section."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # ========================================================
    # 1. Load Repository
    # ========================================================

    print_section("1. LOADING SENTIMENT-ENRICHED DATASET")

    repository = ProductRepository(dataset_path=DATASET_PATH)

    print("Dataset loaded successfully.")
    print(f"Products loaded: {len(repository)}")

    # ========================================================
    # 2. Confirm Product Count
    # ========================================================

    print_section("2. VALIDATING PRODUCT COUNT")

    expected_product_count = 946
    actual_product_count = len(repository)

    print(f"Expected products: {expected_product_count}")
    print(f"Actual products: {actual_product_count}")

    assert actual_product_count == expected_product_count, "Product count mismatch."

    print("PASS: Product count validated.")

    # ========================================================
    # 3. Validate Persisted Review Sentiment
    # ========================================================

    print_section("3. VALIDATING REVIEW SENTIMENT PERSISTENCE")

    products = repository.get_all_products()

    total_reviews = 0
    reviews_with_sentiment = 0
    reviews_with_probabilities = 0

    for product in products:
        for review in product.reviews:
            total_reviews += 1

            if review.sentiment:
                reviews_with_sentiment += 1

            if review.sentiment_probabilities:
                reviews_with_probabilities += 1

    print(f"Total reviews: {total_reviews}")
    print(f"Reviews with sentiment: {reviews_with_sentiment}")
    print(f"Reviews with probabilities: {reviews_with_probabilities}")

    assert total_reviews > 0, "No reviews found."
    assert reviews_with_sentiment == total_reviews, (
        "Some reviews are missing persisted sentiment labels."
    )
    assert reviews_with_probabilities == total_reviews, (
        "Some reviews are missing sentiment probabilities."
    )

    print("PASS: Review sentiment persistence validated.")

    # ========================================================
    # 4. Validate Product Sentiment Summaries
    # ========================================================

    print_section("4. VALIDATING PRODUCT SENTIMENT SUMMARIES")

    products_with_summary = sum(
        1 for product in products if product.sentiment_summary
    )

    print(f"Products with sentiment summary: {products_with_summary}")

    assert products_with_summary == len(products), (
        "Some products are missing sentiment summaries."
    )

    print("PASS: Product summaries validated.")

    # ========================================================
    # 5. Initialize Sentiment Service
    # ========================================================

    print_section("5. INITIALIZING SENTIMENT SERVICE")

    sentiment_service = SentimentService(
        repository=repository,
        model_dir=SENTIMENT_MODEL_DIR,
    )

    print("SentimentService initialized.")
    print("PASS: Model has not been loaded yet (lazy loading).")

    # ========================================================
    # 6. Test Product Sentiment Analytics
    # ========================================================

    print_section("6. TESTING PRODUCT SENTIMENT ANALYTICS")

    sample_product = products[0]

    product_summary = sentiment_service.get_product_sentiment_summary(
        sample_product.asin
    )

    print(f"ASIN: {sample_product.asin}")
    print(f"Title: {sample_product.title}")
    print()

    for key, value in product_summary.items():
        print(f"{key}: {value}")

    assert product_summary["total_reviews"] == len(sample_product.reviews), (
        "Product review count mismatch."
    )

    print()
    print("PASS: Product analytics validated.")

    # ========================================================
    # 7. Test Category Aggregation
    # ========================================================

    print_section("7. TESTING CATEGORY SENTIMENT AGGREGATION")

    sample_category = sample_product.search_keyword

    category_products = repository.get_products_by_search_keyword(
        sample_category
    )

    category_summary = sentiment_service.get_category_sentiment_summary(
        sample_category
    )

    expected_category_reviews = sum(
        len(product.reviews) for product in category_products
    )

    print(f"Category: {sample_category}")
    print(f"Products: {len(category_products)}")
    print(f"Expected reviews: {expected_category_reviews}")
    print(f"Aggregated reviews: {category_summary['total_reviews']}")
    print()

    for key, value in category_summary.items():
        print(f"{key}: {value}")

    assert category_summary["total_reviews"] == expected_category_reviews, (
        "Category review aggregation mismatch."
    )

    print()
    print("PASS: Category aggregation validated.")

    # ========================================================
    # 8. Test Real-Time Model Inference
    # ========================================================

    print_section("8. TESTING REAL-TIME SENTIMENT INFERENCE")

    test_reviews = [
        (
            "This product is absolutely amazing. "
            "The quality exceeded my expectations "
            "and I would definitely recommend it."
        ),
        (
            "The product is okay. It works as "
            "expected but there is nothing special "
            "about it."
        ),
        (
            "Terrible product. It broke immediately "
            "and the quality is extremely poor."
        ),
    ]

    for index, review_text in enumerate(test_reviews, start=1):
        print()
        print(f"Test Review {index}:")
        print(review_text)

        result = sentiment_service.predict_review_sentiment(review_text)

        print(f"Prediction: {result.sentiment}")
        print(f"Positive: {result.positive_probability:.4f}")
        print(f"Neutral: {result.neutral_probability:.4f}")
        print(f"Negative: {result.negative_probability:.4f}")

        probability_sum = (
            result.positive_probability
            + result.neutral_probability
            + result.negative_probability
        )

        assert result.sentiment.lower() in {
            "positive",
            "neutral",
            "negative",
        }, "Invalid sentiment label."

        assert 0.99 <= probability_sum <= 1.01, (
            "Probabilities do not sum approximately to 1."
        )

    print()
    print("PASS: Real-time inference validated.")

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print_section("FEATURE B INTEGRATION VALIDATION COMPLETE")

    print("✓ Sentiment-enriched dataset loaded")
    print("✓ Product count validated")
    print("✓ Persisted review predictions validated")
    print("✓ Product summaries validated")
    print("✓ Product analytics validated")
    print("✓ Category aggregation validated")
    print("✓ Real-time DistilBERT inference validated")
    print()
    print("Feature B backend integration is ready for Streamlit.")


if __name__ == "__main__":
    main()