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


def main():

    print("=" * 80)
    print("REVIEW-LEVEL SENTIMENT ANALYSIS")
    print("=" * 80)

    analyzer = SentimentDatasetAnalyzer(
        input_path=INPUT_PATH
    )

    # --------------------------------------------------
    # Review statistics
    # --------------------------------------------------

    review_statistics = (
        analyzer.get_review_count_statistics()
    )

    print("\nREVIEW COLLECTION STATISTICS")
    print("-" * 70)

    print(
        f"Products: "
        f"{review_statistics['products']}"
    )

    print(
        f"Total reviews: "
        f"{review_statistics['total_reviews']}"
    )

    print(
        f"Minimum reviews/product: "
        f"{review_statistics['minimum_reviews_per_product']}"
    )

    print(
        f"Maximum reviews/product: "
        f"{review_statistics['maximum_reviews_per_product']}"
    )

    print(
        f"Mean reviews/product: "
        f"{review_statistics['mean_reviews_per_product']:.2f}"
    )

    print(
        f"Median reviews/product: "
        f"{review_statistics['median_reviews_per_product']:.2f}"
    )

    # --------------------------------------------------
    # Sentiment distribution
    # --------------------------------------------------

    distribution = (
        analyzer.get_review_sentiment_distribution()
    )

    print("\nREVIEW SENTIMENT DISTRIBUTION")
    print("-" * 70)

    total_reviews = sum(
        values["count"]
        for values in distribution.values()
    )

    print(
        f"Total analyzed reviews: "
        f"{total_reviews}"
    )

    for sentiment in [
        "positive",
        "neutral",
        "negative",
    ]:

        values = distribution[sentiment]

        print(
            f"{sentiment.capitalize():<10} "
            f"{values['count']:>6} reviews "
            f"({values['percentage']:.2f}%)"
        )

    # --------------------------------------------------
    # Final check
    # --------------------------------------------------

    percentage_sum = sum(
        values["percentage"]
        for values in distribution.values()
    )

    print(
        f"\nPercentage total: "
        f"{percentage_sum:.2f}%"
    )

    print("\n" + "=" * 80)
    print("REVIEW-LEVEL ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()