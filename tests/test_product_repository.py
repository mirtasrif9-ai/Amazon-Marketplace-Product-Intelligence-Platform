from pathlib import Path

import pytest

from src.application.product_repository import ProductRepository
from src.data_collection.models.product import Product
from src.data_collection.models.review import Review


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_products.csv"
)


@pytest.fixture(scope="module")
def repository():
    """Create a shared ProductRepository instance."""

    return ProductRepository(DATASET_PATH)


def test_repository_loads_dataset(repository):

    assert len(repository) == 946


def test_get_all_products(repository):

    products = repository.get_all_products()

    assert len(products) == 946

    assert all(
        isinstance(product, Product)
        for product in products
    )


def test_asin_lookup(repository):

    products = repository.get_all_products()

    sample_product = products[0]

    found_product = repository.get_product_by_asin(
        sample_product.asin
    )

    assert isinstance(found_product, Product)

    assert (
        found_product.asin
        == sample_product.asin
    )


def test_invalid_asin_raises_error(repository):

    with pytest.raises(
        ValueError,
        match="was not found",
    ):
        repository.get_product_by_asin(
            "INVALID_ASIN_12345"
        )


def test_search_keywords(repository):

    keywords = repository.get_search_keywords()

    assert isinstance(keywords, list)

    assert len(keywords) > 0

    assert all(
        isinstance(keyword, str)
        for keyword in keywords
    )


def test_products_by_search_keyword(repository):

    keywords = repository.get_search_keywords()

    sample_keyword = keywords[0]

    products = (
        repository.get_products_by_search_keyword(
            sample_keyword
        )
    )

    assert len(products) > 0

    assert all(
        product.search_keyword.lower()
        == sample_keyword.lower()
        for product in products
    )


def test_reviews_are_deserialized(repository):

    products = repository.get_all_products()

    products_with_reviews = [
        product
        for product in products
        if product.reviews
    ]

    assert len(products_with_reviews) > 0, (
        "No products contain deserialized reviews."
    )

    sample_product = products_with_reviews[0]

    assert all(
        isinstance(review, Review)
        for review in sample_product.reviews
    )


def test_product_has_required_application_fields(repository):

    product = repository.get_all_products()[0]

    assert product.asin
    assert product.title
    assert product.search_keyword