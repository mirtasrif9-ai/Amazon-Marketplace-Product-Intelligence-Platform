from pathlib import Path

from src.data_collection.models.product import Product
from src.features.price_tier.price_tier_classifier import PriceTierClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "price_tier" / "price_tier_model.joblib"
)


def create_test_product() -> Product:
    """Create a representative Product for inference testing."""

    return Product(
        product_number=1,
        search_keyword="test",
        asin="TESTASIN123",
        title="Wireless Bluetooth Headphones",
        product_url="https://example.com/product",
        price=49.99,
        description=(
            "Wireless headphones with Bluetooth connectivity, "
            "comfortable ear cushions, and long battery life."
        ),
        brand="TestBrand",
        image="https://example.com/image.jpg",
        review_count=1250,
        average_rating=4.3,
        video_url="",
        reviews=[],
    )


def test_price_tier_model_artifact_exists():
    assert MODEL_PATH.exists(), f"Missing model artifact: {MODEL_PATH}"


def test_price_tier_prediction():
    classifier = PriceTierClassifier(model_path=MODEL_PATH)

    product = create_test_product()

    result = classifier.predict(product)

    assert result.asin == "TESTASIN123"

    assert result.price_tier in {
        "budget",
        "mid_range",
        "premium",
    }

    assert 0.0 <= result.confidence <= 1.0

    assert set(result.probabilities.keys()) == {
        "budget",
        "mid_range",
        "premium",
    }

    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6


def test_price_is_not_used_as_model_input():
    classifier = PriceTierClassifier(model_path=MODEL_PATH)

    product = create_test_product()

    first_result = classifier.predict(product)

    product.price = 999999.99

    second_result = classifier.predict(product)

    assert first_result.price_tier == second_result.price_tier

    assert first_result.probabilities == second_result.probabilities