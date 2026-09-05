import difflib
import math

import random
import pandas as pd
import streamlit as st

from ui.components.helpers import (
    format_price,
    get_product_search_text,
)


PRODUCTS_PER_PAGE = 20
PRODUCTS_PER_ROW = 4


# ============================================================
# MAIN PRODUCT EXPLORER
# ============================================================

def render_product_explorer(
    repository,
    recommendation_service=None,
    sentiment_service=None,
    thumbnail_service=None,
    price_tier_classifier=None,
):
    """Render product browsing and filtering interface.

    The page supports:
        - Product filtering
        - Product search
        - Pagination
        - Product detail navigation
        - Feature A recommendations
        - Feature B sentiment analysis
        - Feature C thumbnail grouping
        - Feature D price-tier classification
    """
    # --------------------------------------------------------
    # PRODUCT DETAIL VIEW
    # --------------------------------------------------------
    selected_asin = st.session_state.get("selected_product_asin")

    if selected_asin:
        product = repository.get_product_by_asin(selected_asin)

        if product is None:
            st.error("Selected product could not be found.")
            st.session_state.pop("selected_product_asin", None)
            st.rerun()

        render_product_detail(
            product=product,
            repository=repository,
            recommendation_service=recommendation_service,
            sentiment_service=sentiment_service,
            thumbnail_service=thumbnail_service,
            price_tier_classifier=price_tier_classifier,
        )
        return

    # --------------------------------------------------------
    # PRODUCT EXPLORER
    # --------------------------------------------------------

    st.title("📦 Product Explorer")

    if "shuffled_products" not in st.session_state:
        products = list(repository.get_all_products())
        random.shuffle(products)
        st.session_state["shuffled_products"] = products
    else:
        products = st.session_state["shuffled_products"]

    categories = repository.get_search_keywords()

    # ========================================================
    # FILTERS
    # ========================================================
    col1, col2 = st.columns([1, 2])

    with col1:
        selected_category = st.selectbox(
            "Filter by Search Keyword",
            options=["All"] + categories,
        )

    with col2:
        search_query = st.text_input(
            "Search Products",
            placeholder="Try product name, brand, ASIN, or keyword...",
        )

    # ========================================================
    # CATEGORY FILTER
    # ========================================================
    filtered_products = products

    if selected_category != "All":
        filtered_products = [
            product
            for product in filtered_products
            if product.search_keyword == selected_category
        ]

    # ========================================================
    # FLEXIBLE SEARCH
    # ========================================================
    if search_query:
        query = search_query.lower().strip()

        if query:
            exact_matches = [
                product
                for product in filtered_products
                if query in get_product_search_text(product)
            ]

            if exact_matches:
                filtered_products = exact_matches
            else:
                # Fuzzy token matching
                fuzzy_matches = []

                for product in filtered_products:
                    fields = [
                        product.title or "",
                        product.brand or "",
                        product.search_keyword or "",
                    ]

                    best_score = 0.0

                    for field in fields:
                        words = field.lower().split()

                        for word in words:
                            score = difflib.SequenceMatcher(
                                None,
                                query,
                                word,
                            ).ratio()
                            best_score = max(best_score, score)

                    if best_score >= 0.60:
                        fuzzy_matches.append((product, best_score))

                fuzzy_matches.sort(
                    key=lambda item: item[1],
                    reverse=True,
                )

                filtered_products = [
                    product for product, _score in fuzzy_matches
                ]

    # ========================================================
    # RESET PAGE WHEN FILTER CHANGES
    # ========================================================
    filter_signature = f"{selected_category}|{search_query}"

    if (
        st.session_state.get("product_filter_signature")
        != filter_signature
    ):
        st.session_state["product_filter_signature"] = filter_signature
        st.session_state["product_page"] = 1

    # ========================================================
    # RESULT COUNT
    # ========================================================
    total_products = len(filtered_products)

    st.caption(f"Showing {total_products} matching products")

    if total_products == 0:
        st.warning("No products match your search.")
        st.info("Try a shorter keyword or check the spelling.")
        return

    # ========================================================
    # PAGINATION
    # ========================================================
    total_pages = math.ceil(total_products / PRODUCTS_PER_PAGE)

    current_page = st.session_state.get("product_page", 1)
    current_page = max(1, min(current_page, total_pages))

    start_index = (current_page - 1) * PRODUCTS_PER_PAGE
    end_index = min(start_index + PRODUCTS_PER_PAGE, total_products)

    page_products = filtered_products[start_index:end_index]

    st.write(
        f"Page **{current_page}** of **{total_pages}** "
        f"· Products **{start_index + 1}–{end_index}**"
    )

    st.divider()

    # ========================================================
    # PRODUCT CARD GRID
    # ========================================================
    for row_start in range(0, len(page_products), PRODUCTS_PER_ROW):
        row_products = page_products[
            row_start : row_start + PRODUCTS_PER_ROW
        ]
        columns = st.columns(PRODUCTS_PER_ROW)

        for column, product in zip(columns, row_products):
            with column:
                render_product_card(product)

    # ========================================================
    # PAGINATION CONTROLS
    # ========================================================
    st.divider()

    previous_col, page_col, next_col = st.columns([1, 3, 1])

    with previous_col:
        if st.button(
            "← Previous",
            disabled=current_page <= 1,
            use_container_width=True,
        ):
            st.session_state["product_page"] = current_page - 1
            st.rerun()

    with page_col:
        st.markdown(
            f"<div style='text-align:center'>"
            f"Page <b>{current_page}</b> of <b>{total_pages}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with next_col:
        if st.button(
            "Next →",
            disabled=current_page >= total_pages,
            use_container_width=True,
        ):
            st.session_state["product_page"] = current_page + 1
            st.rerun()


# ============================================================
# PRODUCT CARD
# ============================================================

def render_product_card(product):
    """Render one product card."""
    st.markdown(
        """
        <style>
        .product-image {
            width: 100%;
            height: 220px;
            object-fit: contain;
            border-radius: 14px;
            background-color: #f5f5f5;
            padding: 8px;
        }

        .product-card {
            min-height: 420px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------
        if product.image:
            st.markdown(
                f"""
                <div style="
                    width:100%;
                    height:220px;
                    border-radius:14px;
                    overflow:hidden;
                    background:#f5f5f5;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                ">
                    <img
                        src="{product.image}"
                        style="
                            width:100%;
                            height:220px;
                            object-fit:contain;
                        "
                    >
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="
                    width:100%;
                    height:220px;
                    border-radius:14px;
                    background:#f5f5f5;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                ">
                    No Image
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------
        title = product.title or "Untitled Product"

        if len(title) > 65:
            title = title[:65] + "..."

        st.markdown(f"**{title}**")

        # ----------------------------------------------------
        # ASIN / BRAND
        # ----------------------------------------------------
        st.caption(f"ASIN: {product.asin}")
        st.write(f"**{product.brand or 'N/A'}**")

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------
        metric1, metric2 = st.columns(2)

        metric1.metric("Price", format_price(product.price))
        metric2.metric("Rating", f"{product.average_rating:.1f} ⭐")

        # ----------------------------------------------------
        # VIEW DETAILS
        # ----------------------------------------------------
        if st.button(
            "View Details",
            key=f"view_{product.asin}",
            use_container_width=True,
        ):
            st.session_state["selected_product_asin"] = product.asin
            st.rerun()


# ============================================================
# PRODUCT DETAIL PAGE
# ============================================================

def render_product_detail(
    product,
    repository,
    recommendation_service=None,
    sentiment_service=None,
    thumbnail_service=None,
    price_tier_classifier=None,
):
    """Render complete product intelligence detail page."""
    # --------------------------------------------------------
    # BACK BUTTON
    # --------------------------------------------------------
    if st.button("← Back to Product Explorer", type="secondary"):
        st.session_state.pop("selected_product_asin", None)
        st.rerun()

    st.title("🛍️ Product Details")

    # ========================================================
    # PRODUCT OVERVIEW
    # ========================================================
    image_col, info_col = st.columns([1, 2])

    with image_col:
        if product.image:
            st.image(product.image, use_container_width=True)
        else:
            st.info("No product image available.")

    with info_col:
        st.header(product.title or "Untitled Product")
        st.write(f"**Brand:** {product.brand or 'N/A'}")
        st.write(f"**ASIN:** {product.asin}")
        st.write(f"**Search Keyword:** {product.search_keyword or 'N/A'}")
        st.write(f"**Price:** {format_price(product.price)}")
        st.write(f"**Amazon Rating:** {product.average_rating:.1f} ⭐")
        st.write(f"**Review Count:** {product.review_count:,}")
        st.write(f"**Collected Reviews:** {len(product.reviews):,}")

        if product.product_url:

            st.link_button(
                "Open Product on Amazon",
                product.product_url,
            )

    st.divider()

    # ========================================================
    # PRODUCT DESCRIPTION
    # ========================================================
    st.subheader("📝 Product Description")

    description = (
        product.description
        or product.description_imputed
        or "No product description available."
    )

    with st.expander("View Full Description", expanded=True):
        st.write(description)

    st.divider()

    # ========================================================
    # PRODUCT INTELLIGENCE
    # ========================================================
    st.subheader("🧠 Product Intelligence")

    # --------------------------------------------------------
    # Feature B — Sentiment
    # --------------------------------------------------------
    sentiment_result = None

    if sentiment_service is not None:
        try:
            sentiment_result = (
                sentiment_service.get_product_sentiment_summary(
                    product.asin
                )
            )
        except Exception as error:
            st.warning(f"Sentiment analysis unavailable: {error}")

    # --------------------------------------------------------
    # Feature C — Visual Group
    # --------------------------------------------------------
    thumbnail_result = None

    if thumbnail_service is not None:
        try:
            thumbnail_result = thumbnail_service.get_product_group(
                product.asin
            )
        except Exception as error:
            st.warning(f"Visual grouping unavailable: {error}")

    # --------------------------------------------------------
    # Feature D — Price Tier
    # --------------------------------------------------------
    price_tier_result = None

    if price_tier_classifier is not None:
        try:
            price_tier_result = price_tier_classifier.predict(product)
        except Exception as error:
            st.warning(f"Price-tier prediction unavailable: {error}")

    intelligence_columns = st.columns(3)

    # --------------------------------------------------------
    # Sentiment Card
    # --------------------------------------------------------
    with intelligence_columns[0]:
        st.markdown("### 🧠 Sentiment")

        if sentiment_result:
            overall_sentiment = sentiment_result.get("overall_sentiment")
            sentiment_score = sentiment_result.get("sentiment_score")

            if overall_sentiment:
                st.metric(
                    "Overall Sentiment",
                    str(overall_sentiment).title(),
                )

            if sentiment_score is not None:
                st.metric(
                    "Sentiment Score",
                    f"{float(sentiment_score):.2f}",
                )
        else:
            st.info("Sentiment information unavailable.")

    # --------------------------------------------------------
    # Price Tier Card
    # --------------------------------------------------------
    with intelligence_columns[1]:
        st.markdown("### 💰 Price Tier")

        if price_tier_result:
            st.metric(
                "Predicted Tier",
                str(price_tier_result.price_tier)
                .replace("_", " ")
                .title(),
            )
            st.metric(
                "Confidence",
                f"{price_tier_result.confidence:.1%}",
            )
        else:
            st.info("Price-tier prediction unavailable.")

    # --------------------------------------------------------
    # Visual Group Card
    # --------------------------------------------------------
    with intelligence_columns[2]:
        st.markdown("### 🖼️ Visual Group")

        if thumbnail_result:
            st.metric("Visual Group", thumbnail_result.visual_group)
            st.metric("Group Size", f"{thumbnail_result.group_size:,}")
        else:
            st.info("No visual group assigned.")

    # ========================================================
    # FEATURE B — SENTIMENT ANALYSIS
    # ========================================================
    if sentiment_result:
        st.divider()

        st.subheader("💬 Review Sentiment Analysis")

        total_reviews = sentiment_result.get("total_reviews", 0)
        positive = sentiment_result.get("positive", {})
        neutral = sentiment_result.get("neutral", {})
        negative = sentiment_result.get("negative", {})

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric("Total Reviews", f"{total_reviews:,}")

        with metric2:
            st.metric(
                "Positive",
                f"{positive.get('count', 0):,}",
                f"{float(positive.get('percentage', 0)):.1f}%",
            )

        with metric3:
            st.metric(
                "Neutral",
                f"{neutral.get('count', 0):,}",
                f"{float(neutral.get('percentage', 0)):.1f}%",
            )

        with metric4:
            st.metric(
                "Negative",
                f"{negative.get('count', 0):,}",
                f"{float(negative.get('percentage', 0)):.1f}%",
            )

        chart_data = pd.DataFrame(
            {
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Reviews": [
                    positive.get("count", 0),
                    neutral.get("count", 0),
                    negative.get("count", 0),
                ],
            }
        )

        st.bar_chart(
            chart_data,
            x="Sentiment",
            y="Reviews",
        )

        # ----------------------------------------------------
        # Review-Level Predictions
        # ----------------------------------------------------
        reviews = sentiment_service.get_product_review_sentiments(
            product.asin
        )

        if reviews:
            st.subheader("Review-Level Predictions")

            for index, review in enumerate(reviews, start=1):
                sentiment = str(
                    review.get("sentiment", "unknown")
                ).title()

                rating = review.get("star_rating")
                title = review.get("review_title", "")
                description = review.get("review_description", "")

                positive_probability = review.get(
                    "positive_probability", 0.0
                )
                neutral_probability = review.get(
                    "neutral_probability", 0.0
                )
                negative_probability = review.get(
                    "negative_probability", 0.0
                )

                with st.expander(f"Review {index} · {sentiment}"):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        if rating is not None:
                            st.metric("Rating", f"{rating} ⭐")

                    with col2:
                        st.metric(
                            "Positive",
                            f"{positive_probability:.1%}",
                        )

                    with col3:
                        st.metric(
                            "Neutral",
                            f"{neutral_probability:.1%}",
                        )

                    with col4:
                        st.metric(
                            "Negative",
                            f"{negative_probability:.1%}",
                        )

                    if title:
                        st.markdown(f"**{title}**")

                    if description:
                        st.write(description)

    # ========================================================
    # FEATURE D — PRICE TIER DETAILS
    # ========================================================
    if price_tier_result:
        st.divider()

        st.subheader("💰 Price Tier Classification")

        st.caption(
            "Feature D predicts the price tier from product "
            "attributes without using the actual product price "
            "as a model input."
        )

        probability_data = pd.DataFrame(
            {
                "Price Tier": [
                    str(label).replace("_", " ").title()
                    for label in price_tier_result.probabilities.keys()
                ],
                "Probability": [
                    float(probability)
                    for probability in price_tier_result.probabilities.values()
                ],
            }
        )

        st.bar_chart(
            probability_data,
            x="Price Tier",
            y="Probability",
        )



    # ========================================================
    # FEATURE A — SIMILAR PRODUCTS
    # ========================================================
    if recommendation_service is not None:
        st.divider()

        st.subheader("🤝 Similar Products")

        try:
            recommendations = (
                recommendation_service.get_recommendations(
                    asin=product.asin,
                    top_k=5,
                )
            )

            if not recommendations:
                st.info("No similar products were found.")
            else:
                recommendation_columns = st.columns(5)

                for index, recommendation in enumerate(
                    recommendations
                ):
                    recommended_asin = getattr(
                        recommendation,
                        "asin",
                        None,
                    )

                    if recommended_asin is None:
                        recommended_asin = getattr(
                            recommendation,
                            "product_asin",
                            None,
                        )

                    if recommended_asin is None:
                        continue

                    try:
                        recommended_product = (
                            repository.get_product_by_asin(
                                recommended_asin
                            )
                        )
                    except Exception:
                        recommended_product = None

                    if recommended_product is None:
                        continue

                    with recommendation_columns[index % 5]:
                        if recommended_product.image:
                            st.image(
                                recommended_product.image,
                                use_container_width=True,
                            )

                        title = (
                            recommended_product.title
                            or "Untitled Product"
                        )

                        if len(title) > 55:
                            title = title[:55] + "..."

                        st.markdown(f"**{title}**")
                        st.caption(f"ASIN: {recommended_product.asin}")
                        st.write(
                            format_price(recommended_product.price)
                        )

                        similarity_score = getattr(
                            recommendation,
                            "similarity_score",
                            None,
                        )

                        if similarity_score is None:
                            similarity_score = getattr(
                                recommendation,
                                "score",
                                None,
                            )

                        if similarity_score is not None:
                            st.caption(
                                "Similarity: "
                                f"{float(similarity_score):.3f}"
                            )

        except Exception as error:
            st.warning(
                "Similar-product recommendations "
                f"are unavailable: {error}"
            )
    # ========================================================
    # FEATURE C — VISUAL GROUP
    # ========================================================
    if thumbnail_result:
        st.divider()

        st.subheader("🖼️ Visual Product Group")

        st.write(
            f"Products visually similar to this product "
            f"are assigned to **{thumbnail_result.visual_group}**."
        )

        group_images = thumbnail_service.get_group_images(
            thumbnail_result.visual_group
        )

        # Exclude the currently selected product.
        group_images = [
            item
            for item in group_images
            if str(item["asin"]) != str(product.asin)
        ]

        if group_images:
            display_images = group_images[:12]
            image_columns = st.columns(4)

            for index, item in enumerate(display_images):
                with image_columns[index % 4]:
                    image_path = item["image_path"]

                    st.image(
                        str(image_path),
                        use_container_width=True,
                    )
                    st.caption(f"ASIN: {item['asin']}")
        else:
            st.info(
                "No additional product images are available "
                "for this visual group."
            )


    # ========================================================
    # ADDITIONAL PRODUCT INFORMATION
    # ========================================================
    st.divider()

    st.subheader("ℹ️ Additional Product Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write(f"**Product Number:** {product.product_number}")
        st.write(f"**Review Count:** {product.review_count:,}")


    with info_col2:
        st.write(
            f"**Search Keyword:** {product.search_keyword or 'N/A'}"
        )
        st.write(f"**Average Rating:** {product.average_rating:.1f} ⭐")


        if product.video_url:
            st.link_button("🎥 Product Video", product.video_url)

    # ========================================================
    # AMAZON LINK
    # ========================================================
    if product.product_url:

        st.divider()

        st.link_button(
            "🛒 Open This Product on Amazon",
            product.product_url,
            type="primary",
        )