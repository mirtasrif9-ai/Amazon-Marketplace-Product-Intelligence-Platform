from pathlib import Path


from src.features.review_sentiment.sentiment_predictor import (
    SentimentPredictor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "sentiment"
    / "final_sentiment_model"
)


def main():
    print("=" * 70)
    print("SENTIMENT PREDICTOR STANDALONE TEST")
    print("=" * 70)

    predictor = SentimentPredictor(
        model_dir=MODEL_DIR
    )

    print(f"\nDevice: {predictor.device}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Thresholds: {predictor.thresholds}")

    reviews = [
        "Absolutely amazing product. The quality is excellent and I love it!",

        "The product is okay. It works as expected but nothing special.",

        "Terrible quality. It stopped working after two days. Very disappointed.",

        "Good product for the price. It does the job, although there are a few minor issues.",

        "Excellent quality and very easy to use. Highly recommended!",
    ]

    results = predictor.predict_batch(reviews)

    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)

    for index, (review, result) in enumerate(
        zip(reviews, results),
        start=1,
    ):
        print(f"\nReview {index}:")
        print(f"Text: {review}")
        print(f"Sentiment: {result['sentiment']}")
        print("Probabilities:")

        for label, probability in result[
            "probabilities"
        ].items():
            print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()