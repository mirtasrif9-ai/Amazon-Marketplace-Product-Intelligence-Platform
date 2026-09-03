import streamlit as st

from ui.components.helpers import format_price


def render_recommendations(
    repository,
    recommendation_service,
):
    """
    Render Feature A — Similar Product Recommendations.
    """

    st.title("🔍 Feature A — Similar Product Recommendation")

    st.markdown(
        """
        Select a product to find the most similar products using the
        trained content-based recommendation model.
        """
    )

    products = repository.get_all_products()

    # ========================================================
    # PRODUCT SELECTION
    # ========================================================

    def format_product(product):
        title = product.title or "Untitled Product"

        if len(title) > 80:
            title = title[:80] + "..."

        return f"{product.asin} | {title}"

    selected_product = st.selectbox(
        "Select a Product",
        options=products,
        format_func=format_product,
        key="recommendation_product_select",
    )

    top_k = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5,
    )

    if not st.button(
        "Find Similar Products",
        type="primary",
        use_container_width=True,
    ):
        return

    # ========================================================
    # RUN RECOMMENDATION ENGINE
    # ========================================================

    try:

        with st.spinner(
            "Finding similar products..."
        ):

            recommendations = (
                recommendation_service
                .get_recommendations(
                    asin=selected_product.asin,
                    top_k=top_k,
                )
            )

    except Exception as error:

        st.error(
            "Unable to generate recommendations."
        )

        st.exception(error)

        return

    # ========================================================
    # SELECTED PRODUCT
    # ========================================================

    st.divider()

    st.subheader("Selected Product")

    source_col1, source_col2 = st.columns(
        [1, 3]
    )

    with source_col1:

        if selected_product.image:

            st.image(
                selected_product.image,
                use_container_width=True,
            )

    with source_col2:

        st.write(
            f"**{selected_product.title}**"
        )

        st.caption(
            f"ASIN: {selected_product.asin}"
        )

        st.write(
            f"**Brand:** "
            f"{selected_product.brand or 'N/A'}"
        )

        st.write(
            f"**Price:** "
            f"{format_price(selected_product.price)}"
        )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.divider()

    st.subheader(
        f"Top {len(recommendations)} Similar Products"
    )

    if not recommendations:

        st.info(
            "No similar products were found."
        )

        return

    for rank, recommendation in enumerate(
        recommendations,
        start=1,
    ):

        with st.container(border=True):

            col1, col2 = st.columns(
                [1, 3]
            )

            with col1:

                if recommendation.image:

                    try:

                        st.image(
                            recommendation.image,
                            use_container_width=True,
                        )

                    except Exception:

                        st.caption(
                            "Image unavailable"
                        )

                else:

                    st.caption(
                        "No image available"
                    )

            with col2:

                st.markdown(
                    f"### #{rank} "
                    f"{recommendation.title}"
                )

                st.caption(
                    f"ASIN: "
                    f"{recommendation.asin}"
                )

                metric1, metric2 = st.columns(2)

                metric1.metric(
                    "Similarity",
                    f"{recommendation.similarity_score:.1%}",
                )

                metric2.metric(
                    "Price",
                    format_price(
                        recommendation.price
                    ),
                )

                st.write(
                    f"**Brand:** "
                    f"{recommendation.brand or 'N/A'}"
                )

                if (
                    recommendation.average_rating
                    is not None
                ):

                    st.write(
                        f"⭐ "
                        f"{recommendation.average_rating:.1f}"
                    )

                # ------------------------------------------------
                # SIMPLE RELEVANCE EXPLANATION
                # ------------------------------------------------

                similarity = (
                    recommendation.similarity_score
                )

                if similarity >= 0.70:

                    reason = (
                        "Highly similar based on product "
                        "textual attributes."
                    )

                elif similarity >= 0.40:

                    reason = (
                        "Moderately similar based on product "
                        "content and attributes."
                    )

                else:

                    reason = (
                        "Related product identified by the "
                        "content-based similarity model."
                    )

                st.caption(
                    f"Why recommended: {reason}"
                )

                if recommendation.url:

                    st.link_button(
                        "Open Product",
                        recommendation.url,
                    )