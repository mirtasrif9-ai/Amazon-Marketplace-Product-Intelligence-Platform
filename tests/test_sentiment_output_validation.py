import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "products_with_sentiment.json"
)


EXPECTED_PRODUCT_COUNT = 946

VALID_SENTIMENTS = {
    "positive",
    "neutral",
    "negative",
}


def load_products() -> list[dict]:
    """
    Load sentiment-enriched products JSON.
    """

    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Output file not found: {OUTPUT_PATH}"
        )

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_product_structure(
    product: dict,
) -> list[str]:
    """
    Validate required product sentiment fields.
    """

    errors = []

    required_fields = [
        "product_number",
        "asin",
        "title",
        "reviews",
        "sentiment_summary",
    ]

    for field in required_fields:

        if field not in product:
            errors.append(
                f"Missing product field: {field}"
            )

    return errors


def validate_sentiment_summary(
    product: dict,
) -> list[str]:
    """
    Validate product-level sentiment summary.
    """

    errors = []

    summary = product.get(
        "sentiment_summary"
    )

    if not isinstance(summary, dict):
        return [
            "sentiment_summary is missing "
            "or not a dictionary."
        ]

    required_fields = [
        "total_reviews",
        "positive_count",
        "neutral_count",
        "negative_count",
        "positive_percentage",
        "neutral_percentage",
        "negative_percentage",
        "overall_sentiment",
        "sentiment_score",
    ]

    for field in required_fields:

        if field not in summary:
            errors.append(
                f"Missing summary field: {field}"
            )

    if errors:
        return errors

    total_reviews = summary["total_reviews"]

    sentiment_count_sum = (
        summary["positive_count"]
        + summary["neutral_count"]
        + summary["negative_count"]
    )

    if sentiment_count_sum != total_reviews:
        errors.append(
            "Sentiment counts do not match "
            f"total_reviews "
            f"({sentiment_count_sum} != "
            f"{total_reviews})"
        )

    percentage_sum = (
        summary["positive_percentage"]
        + summary["neutral_percentage"]
        + summary["negative_percentage"]
    )

    if total_reviews > 0:

        if abs(percentage_sum - 100.0) > 0.01:
            errors.append(
                "Sentiment percentages do not "
                f"sum to 100 ({percentage_sum})"
            )

    overall_sentiment = (
        summary["overall_sentiment"]
    )

    if overall_sentiment not in VALID_SENTIMENTS:
        errors.append(
            f"Invalid overall sentiment: "
            f"{overall_sentiment}"
        )

    sentiment_score = (
        summary["sentiment_score"]
    )

    if not isinstance(
        sentiment_score,
        (int, float),
    ):
        errors.append(
            "sentiment_score is not numeric"
        )

    elif not -1.0 <= sentiment_score <= 1.0:
        errors.append(
            f"sentiment_score out of range: "
            f"{sentiment_score}"
        )

    return errors


def validate_reviews(
    product: dict,
) -> list[str]:
    """
    Validate review-level sentiment data.
    """

    errors = []

    reviews = product.get(
        "reviews",
        [],
    )

    if not isinstance(reviews, list):
        return [
            "Reviews field is not a list."
        ]

    for index, review in enumerate(reviews):

        if "sentiment" not in review:

            errors.append(
                f"Review {index} missing sentiment"
            )

            continue

        sentiment = review["sentiment"]

        if sentiment not in VALID_SENTIMENTS:

            errors.append(
                f"Review {index} has invalid "
                f"sentiment: {sentiment}"
            )

        probabilities = review.get(
            "sentiment_probabilities"
        )

        if not isinstance(
            probabilities,
            dict,
        ):
            errors.append(
                f"Review {index} missing "
                "sentiment probabilities"
            )

            continue

        required_probabilities = [
            "negative",
            "neutral",
            "positive",
        ]

        for label in required_probabilities:

            if label not in probabilities:
                errors.append(
                    f"Review {index} missing "
                    f"{label} probability"
                )

    return errors


def main():

    print("=" * 70)
    print("SENTIMENT OUTPUT VALIDATION")
    print("=" * 70)

    products = load_products()

    print(
        f"\nProducts loaded: "
        f"{len(products)}"
    )

    # --------------------------------------------------
    # Product count validation
    # --------------------------------------------------

    assert len(products) == EXPECTED_PRODUCT_COUNT, (
        f"Expected {EXPECTED_PRODUCT_COUNT} products, "
        f"but found {len(products)}"
    )

    print(
        "✓ Product count validation passed"
    )

    # --------------------------------------------------
    # Validation counters
    # --------------------------------------------------

    total_reviews = 0
    total_errors = 0

    invalid_products = []

    # --------------------------------------------------
    # Validate every product
    # --------------------------------------------------

    for index, product in enumerate(
        products,
        start=1,
    ):

        product_errors = []

        product_errors.extend(
            validate_product_structure(product)
        )

        product_errors.extend(
            validate_sentiment_summary(product)
        )

        product_errors.extend(
            validate_reviews(product)
        )

        total_reviews += len(
            product.get("reviews", [])
        )

        if product_errors:

            total_errors += len(
                product_errors
            )

            invalid_products.append(
                {
                    "index": index,
                    "asin": product.get("asin"),
                    "errors": product_errors,
                }
            )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(f"Total products: {len(products)}")
    print(f"Total reviews: {total_reviews}")

    print(
        f"Products with validation errors: "
        f"{len(invalid_products)}"
    )

    print(
        f"Total validation errors: "
        f"{total_errors}"
    )

    # --------------------------------------------------
    # Display invalid products
    # --------------------------------------------------

    if invalid_products:

        print("\nINVALID PRODUCTS")
        print("-" * 70)

        for invalid_product in invalid_products[:10]:

            print(
                f"\nASIN: "
                f"{invalid_product['asin']}"
            )

            for error in invalid_product["errors"]:

                print(f"  - {error}")

    # --------------------------------------------------
    # Final assertion
    # --------------------------------------------------

    assert not invalid_products, (
        f"Validation failed for "
        f"{len(invalid_products)} products."
    )

    print("\n" + "=" * 70)
    print("ALL VALIDATIONS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()