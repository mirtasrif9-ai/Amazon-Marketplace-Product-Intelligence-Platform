from pathlib import Path

from src.features.review_sentiment.analysis.sentiment_dataset_analyzer import (
    SentimentDatasetAnalyzer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "products_with_sentiment.json"
)


def print_distribution(
    distribution: dict,
) -> None:

    print("\nPRODUCT SENTIMENT DISTRIBUTION")
    print("-" * 70)

    for sentiment, values in distribution.items():

        print(
            f"{sentiment.capitalize():<10} "
            f"{values['count']:>5} products "
            f"({values['percentage']:.2f}%)"
        )


def print_score_statistics(
    statistics: dict,
) -> None:

    print("\nSENTIMENT SCORE STATISTICS")
    print("-" * 70)

    for key, value in statistics.items():

        if isinstance(value, float):
            print(
                f"{key.capitalize():<10}: "
                f"{value:.4f}"
            )
        else:
            print(
                f"{key.capitalize():<10}: "
                f"{value}"
            )


def print_products(
    title: str,
    products: list[dict],
) -> None:

    print(f"\n{title}")
    print("-" * 100)

    for index, product in enumerate(
        products,
        start=1,
    ):

        product_title = (
            product["title"] or ""
        )

        if len(product_title) > 65:
            product_title = (
                product_title[:62] + "..."
            )

        print(
            f"{index}. "
            f"Score: "
            f"{product['sentiment_score']:+.3f} | "
            f"Rating: "
            f"{product['average_rating']} | "
            f"{product_title}"
        )

        print(
            f"   ASIN: {product['asin']} | "
            f"Keyword: {product['search_keyword']}"
        )

        print(
            f"   Positive: "
            f"{product['positive_percentage']:.1f}% | "
            f"Neutral: "
            f"{product['neutral_percentage']:.1f}% | "
            f"Negative: "
            f"{product['negative_percentage']:.1f}%"
        )


def main():

    print("=" * 100)
    print("PRODUCT-LEVEL SENTIMENT ANALYSIS")
    print("=" * 100)

    analyzer = SentimentDatasetAnalyzer(
        input_path=INPUT_PATH
    )

    # --------------------------------------------------
    # Distribution
    # --------------------------------------------------

    distribution = (
        analyzer.get_product_sentiment_distribution()
    )

    print_distribution(distribution)

    # --------------------------------------------------
    # Score statistics
    # --------------------------------------------------

    statistics = (
        analyzer.get_sentiment_score_statistics()
    )

    print_score_statistics(statistics)

    # --------------------------------------------------
    # Most positive
    # --------------------------------------------------

    most_positive = (
        analyzer.get_top_products_by_sentiment(
            top_n=10,
            ascending=False,
        )
    )

    print_products(
        "TOP 10 MOST POSITIVE PRODUCTS",
        most_positive,
    )

    # --------------------------------------------------
    # Most negative
    # --------------------------------------------------

    most_negative = (
        analyzer.get_top_products_by_sentiment(
            top_n=10,
            ascending=True,
        )
    )

    print_products(
        "TOP 10 MOST NEGATIVE PRODUCTS",
        most_negative,
    )

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()