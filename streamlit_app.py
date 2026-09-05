from pathlib import Path

import streamlit as st

from src.application.product_repository import ProductRepository
from src.application.recommendation_service import (
    RecommendationService,
)
from src.application.sentiment_service import SentimentService
from src.features.price_tier.price_tier_classifier import (
    PriceTierClassifier,
)
from src.features.thumbnail_grouping.thumbnail_grouping_service import (
    ThumbnailGroupingService,
)

from ui.pages.product_explorer import (
    render_product_explorer,
)

from ui.pages.recommendations import (
    render_recommendations,
)

from ui.pages.sentiment import render_sentiment_page

from ui.pages.thumbnail_groups import (
    render_thumbnail_groups,
)


from ui.pages.price_tier import (
    render_price_tier_page,
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "products_with_sentiment.json"
)

RECOMMENDATION_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
)

SENTIMENT_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "sentiment"
    / "final_sentiment_model"
)

PRICE_TIER_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "price_tier"
    / "price_tier_model.joblib"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Amazon Marketplace Product Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BACKEND LOADERS
# ============================================================

@st.cache_resource
def load_repository() -> ProductRepository:
    """Load and cache the central product repository."""

    return ProductRepository(
        dataset_path=DATASET_PATH
    )


@st.cache_resource
def load_recommendation_service(
    _repository: ProductRepository,
) -> RecommendationService:
    """Load and cache Feature A."""

    return RecommendationService(
        repository=_repository,
        model_dir=RECOMMENDATION_MODEL_DIR,
    )


@st.cache_resource
def load_sentiment_service(
    _repository: ProductRepository,
) -> SentimentService:
    """Load and cache Feature B."""

    return SentimentService(
        repository=_repository,
        model_dir=SENTIMENT_MODEL_DIR,
    )


@st.cache_resource
def load_thumbnail_service() -> ThumbnailGroupingService:
    """Load and cache Feature C."""

    return ThumbnailGroupingService(
        project_root=PROJECT_ROOT
    )


@st.cache_resource
def load_price_tier_classifier() -> PriceTierClassifier:
    """Load and cache Feature D."""

    return PriceTierClassifier(
        model_path=PRICE_TIER_MODEL_PATH
    )





# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard(
    repository: ProductRepository,
):
    """Render application overview."""

    products = repository.get_all_products()
    categories = repository.get_search_keywords()

    st.title(
        "🛒 Amazon Marketplace Product Intelligence Platform"
    )

    st.markdown(
        """
        A centralized application for exploring collected Amazon
        marketplace products and running four analytical features.
        """
    )

    st.divider()

    total_reviews = sum(
        len(product.reviews)
        for product in products
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Products",
        len(products),
    )

    col2.metric(
        "Search Keywords",
        len(categories),
    )

    col3.metric(
        "Collected Reviews",
        total_reviews,
    )

    st.divider()

    st.subheader("Available Features")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            **Feature A — Similar Product Recommendation**

            Find visually and textually similar products using a
            content-based recommendation model.
            """
        )

        st.info(
            """
            **Feature B — Review Sentiment Analysis**

            Classify reviews as Positive, Neutral, or Negative and
            generate product and category-level summaries.
            """
        )

    with col2:

        st.info(
            """
            **Feature C — Thumbnail Grouping**

            Explore product images grouped automatically based on
            visual similarity without using Amazon category labels.
            """
        )

        st.info(
            """
            **Feature D — Price Tier Classification**

            Predict Budget, Mid-range, or Premium tiers using product
            attributes while deliberately excluding product price from
            model input.
            """
        )



# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Run the Streamlit application."""

    try:

        repository = load_repository()

    except Exception as error:

        st.error(
            "Failed to load product data."
        )

        st.exception(error)

        st.stop()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.title("🛒 Navigation")

    page = st.sidebar.radio(
        "Go to",
        [
            "Dashboard",
            "Product Explorer",
            "Feature A — Recommendations",
            "Feature B — Sentiment",
            "Feature C — Thumbnail Groups",
            "Feature D — Price Tier",
        ],
    )

    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------

    if page == "Dashboard":

        render_dashboard(repository)
        
    elif page == "Product Explorer":
            try:
                recommendation_service = load_recommendation_service(
                    repository
                )

                sentiment_service = load_sentiment_service(
                    repository
                )

                thumbnail_service = load_thumbnail_service()

                price_tier_classifier = (
                    load_price_tier_classifier()
                )

                render_product_explorer(
                    repository=repository,
                    recommendation_service=recommendation_service,
                    sentiment_service=sentiment_service,
                    thumbnail_service=thumbnail_service,
                    price_tier_classifier=price_tier_classifier,
                )

            except Exception as error:
                st.error("Failed to load Product Explorer.")
                st.exception(error)

    elif page == "Feature A — Recommendations":

        try:

            recommendation_service = (
                load_recommendation_service(
                    repository
                )
            )

            render_recommendations(
                repository=repository,
                recommendation_service=(
                    recommendation_service
                ),
            )

        except Exception as error:

            st.error(
                "Failed to load recommendation feature."
            )

            st.exception(error)

    elif page == "Feature B — Sentiment":

        try:

            sentiment_service = (
                load_sentiment_service(
                    repository
                )
            )

            render_sentiment_page(
                repository=repository,
                sentiment_service=(
                    sentiment_service
                ),
            )

        except Exception as error:

            st.error(
                "Failed to load sentiment feature."
            )

            st.exception(error)

    elif page == "Feature C — Thumbnail Groups":

        try:

            thumbnail_service = (
                load_thumbnail_service()
            )

            render_thumbnail_groups(
                thumbnail_service=thumbnail_service,
            )

        except Exception as error:

            st.error(
                "Failed to load thumbnail grouping feature."
            )

            st.exception(error)

    elif page == "Feature D — Price Tier":
        try:
            price_tier_classifier = load_price_tier_classifier()

            render_price_tier_page(
                repository=repository,
                price_tier_classifier=price_tier_classifier,
            )

        except Exception as error:
            st.error("Failed to load price tier classification feature.")
            st.exception(error)


if __name__ == "__main__":
    main()