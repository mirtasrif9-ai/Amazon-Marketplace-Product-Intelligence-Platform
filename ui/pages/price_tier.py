from typing import Optional

import streamlit as st

from src.application.product_repository import ProductRepository
from src.data_collection.models.product import Product
from src.features.price_tier.price_tier_classifier import PriceTierClassifier


def render_price_tier_page(
    repository: ProductRepository,
    price_tier_classifier: PriceTierClassifier,
):
    """Render the Feature D price tier classification interface."""

    st.title("💰 Feature D — Price Tier Classification")

    st.markdown(
        """
        Predict whether an Amazon product belongs to the
        **Budget**, **Mid-range**, or **Premium** tier using product
        metadata and textual information.

        **Note:** Product price is deliberately excluded from model inference.
        """
    )

    st.divider()

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    st.subheader("🤖 Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            "Hybrid Logistic Regression",
        )

    with col2:
        st.metric(
            "Input Types",
            "Text + Numeric + Brand",
        )

    with col3:
        st.metric(
            "Classes",
            "3",
        )

    st.caption("Classes: Budget • Mid-range • Premium")

    st.divider()

    # ============================================================
    # PRODUCT SELECTION
    # ============================================================

    st.subheader("🔎 Select a Product")

    products = repository.get_all_products()

    if not products:
        st.warning("No products are available in the repository.")
        return

    product_lookup = {
        str(product.asin).strip(): product for product in products
    }

    asin_options = list(product_lookup.keys())

    selected_asin = st.selectbox(
        "Select product ASIN",
        asin_options,
    )

    if not selected_asin:
        return

    product: Product = product_lookup[selected_asin]

    st.divider()

    # ============================================================
    # PRODUCT INFORMATION
    # ============================================================

    st.subheader("📦 Product Information")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**Title:** {product.title}")
        st.markdown(f"**Brand:** {product.brand or 'N/A'}")
        st.markdown(f"**ASIN:** `{product.asin}`")

    with col2:
        if product.image:
            try:
                st.image(
                    product.image,
                    caption="Product thumbnail",
                    use_container_width=True,
                )

            except Exception:
                st.caption("Product image unavailable.")

    # ------------------------------------------------------------
    # PRODUCT METADATA
    # ------------------------------------------------------------

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            "Price",
            (
                f"${product.price:,.2f}"
                if product.price is not None
                else "N/A"
            ),
        )

    with metric_col2:
        st.metric(
            "Reviews",
            (
                product.review_count
                if product.review_count is not None
                else 0
            ),
        )

    with metric_col3:
        st.metric(
            "Rating",
            (
                f"{product.average_rating:.1f}"
                if product.average_rating is not None
                else "N/A"
            ),
        )

    with metric_col4:
        st.metric(
            "Collected Reviews",
            len(product.reviews),
        )

    if product.description:
        with st.expander("📄 Product Description"):
            st.write(product.description)

    st.divider()

    # ============================================================
    # PREDICTION
    # ============================================================

    st.subheader("🎯 Price Tier Prediction")

    if st.button(
        "Predict Price Tier",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Running price tier classification..."):
                result = price_tier_classifier.predict(product)

            st.success("Price tier prediction completed.")

            # ----------------------------------------------------
            # MAIN RESULT
            # ----------------------------------------------------

            result_col1, result_col2 = st.columns([1, 2])

            with result_col1:
                tier_labels = {
                    "budget": "💵 Budget",
                    "mid_range": "⚖️ Mid-range",
                    "premium": "💎 Premium",
                }

                tier_display = tier_labels.get(
                    result.price_tier,
                    result.price_tier,
                )

                st.metric(
                    "Predicted Tier",
                    tier_display,
                )

            with result_col2:
                st.metric(
                    "Confidence",
                    f"{result.confidence:.2%}",
                )

                st.progress(result.confidence)

            # ----------------------------------------------------
            # PROBABILITY DISTRIBUTION
            # ----------------------------------------------------

            st.markdown("#### 📊 Prediction Probabilities")

            probability_columns = st.columns(3)

            display_order = [
                "budget",
                "mid_range",
                "premium",
            ]

            display_labels = {
                "budget": "💵 Budget",
                "mid_range": "⚖️ Mid-range",
                "premium": "💎 Premium",
            }

            for column, tier in zip(
                probability_columns,
                display_order,
            ):
                probability = result.probabilities.get(
                    tier,
                    0.0,
                )

                with column:
                    st.metric(
                        display_labels[tier],
                        f"{probability:.2%}",
                    )

                    st.progress(probability)

            # ----------------------------------------------------
            # IMPORTANT MODEL NOTE
            # ----------------------------------------------------

            st.info(
                """
                **How this prediction works**

                The classifier uses product title, description, brand,
                review count, average rating, collected review count,
                title length, and description length.

                The actual product price is **not used as an input
                feature** during inference.
                """
            )

        except Exception as error:
            st.error("Failed to predict the product price tier.")
            st.exception(error)