import streamlit as st


def _safe_percentage(value) -> float:
    """Safely convert a percentage value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _get_summary_value(summary, field_name, default=0):
    """
    Read a summary field from either a model object or dictionary.

    This keeps the UI flexible without changing SentimentService.
    """
    if isinstance(summary, dict):
        return summary.get(field_name, default)

    return getattr(summary, field_name, default)


def _render_sentiment_metrics(summary):
    """Render positive, neutral and negative sentiment metrics."""

    positive_count = _get_summary_value(summary, "positive_count")
    neutral_count = _get_summary_value(summary, "neutral_count")
    negative_count = _get_summary_value(summary, "negative_count")

    positive_percentage = _get_summary_value(
        summary,
        "positive_percentage",
    )
    neutral_percentage = _get_summary_value(
        summary,
        "neutral_percentage",
    )
    negative_percentage = _get_summary_value(
        summary,
        "negative_percentage",
    )

    total_reviews = (
        positive_count
        + neutral_count
        + negative_count
    )

    st.subheader("Sentiment Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Reviews",
            f"{total_reviews:,}",
        )

    with col2:
        st.metric(
            "Positive",
            f"{positive_count:,}",
            f"{_safe_percentage(positive_percentage):.1f}%",
        )

    with col3:
        st.metric(
            "Neutral",
            f"{neutral_count:,}",
            f"{_safe_percentage(neutral_percentage):.1f}%",
        )

    with col4:
        st.metric(
            "Negative",
            f"{negative_count:,}",
            f"{_safe_percentage(negative_percentage):.1f}%",
        )


def _render_sentiment_chart(summary):
    """Render a sentiment distribution chart."""

    positive_count = _get_summary_value(summary, "positive_count")
    neutral_count = _get_summary_value(summary, "neutral_count")
    negative_count = _get_summary_value(summary, "negative_count")

    chart_data = {
        "Sentiment": [
            "Positive",
            "Neutral",
            "Negative",
        ],
        "Reviews": [
            positive_count,
            neutral_count,
            negative_count,
        ],
    }

    st.subheader("Sentiment Distribution")

    st.bar_chart(
        chart_data,
        x="Sentiment",
        y="Reviews",
    )


def _render_review_list(reviews):
    """Render individual review sentiment results."""

    st.subheader("Review-Level Sentiment")

    if not reviews:
        st.info("No reviews available for this product.")
        return

    for index, review in enumerate(reviews, start=1):

        if isinstance(review, dict):
            sentiment = review.get("sentiment", "Unknown")
            confidence = review.get("confidence")
            title = review.get("review_title", "")
            description = review.get("review_description", "")
            rating = review.get("star_rating")
        else:
            sentiment = getattr(review, "sentiment", "Unknown")
            confidence = getattr(review, "confidence", None)
            title = getattr(review, "review_title", "")
            description = getattr(
                review,
                "review_description",
                "",
            )
            rating = getattr(review, "star_rating", None)

        sentiment = str(sentiment).title()

        with st.expander(
            f"Review {index} · {sentiment}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                if rating:
                    st.write(f"**Rating:** {rating} ⭐")

            with col2:
                if confidence is not None:
                    try:
                        st.write(
                            f"**Confidence:** "
                            f"{float(confidence):.1%}"
                        )
                    except (TypeError, ValueError):
                        pass

            if title:
                st.write(f"**{title}**")

            if description:
                st.write(description)


def render_sentiment_page(repository, sentiment_service):
    """
    Render Feature B — Review Sentiment page.

    Uses the existing SentimentService for all sentiment analysis.
    """

    st.title("💬 Review Sentiment")

    st.write(
        "Analyze customer review sentiment at the "
        "product and category level."
    )

    mode = st.radio(
        "Analysis Level",
        ["Product", "Category"],
        horizontal=True,
    )

    st.divider()

    if mode == "Product":
        _render_product_sentiment(
            repository,
            sentiment_service,
        )
    else:
        _render_category_sentiment(
            repository,
            sentiment_service,
        )


def _render_product_sentiment(
    repository,
    sentiment_service,
):
    """Render product-level sentiment analysis."""

    products = repository.get_all_products()

    if not products:
        st.warning("No products are available.")
        return

    product_options = {
        f"{product.asin} | {product.title[:80] if product.title else 'Untitled'}":
        product.asin
        for product in products
    }

    selected_label = st.selectbox(
        "Select Product",
        options=list(product_options.keys()),
    )

    selected_asin = product_options[selected_label]

    product = repository.get_product_by_asin(
        selected_asin
    )

    st.markdown(f"### {product.title}")

    meta_col1, meta_col2, meta_col3 = st.columns(3)

    with meta_col1:
        st.write(
            f"**Brand:** {product.brand or 'N/A'}"
        )

    with meta_col2:
        st.write(
            f"**Rating:** "
            f"{product.average_rating:.1f} ⭐"
        )

    with meta_col3:
        st.write(
            f"**Reviews:** "
            f"{product.review_count:,}"
        )

    with st.spinner("Analyzing review sentiment..."):
        summary = (
            sentiment_service
            .get_product_sentiment_summary(
                selected_asin
            )
        )

    st.divider()

    _render_sentiment_metrics(summary)

    st.divider()

    _render_sentiment_chart(summary)

    try:
        reviews = (
            sentiment_service
            .get_product_review_sentiments(
                selected_asin
            )
        )
        _render_review_list(reviews)
    except Exception:
        st.info(
            "Individual review sentiment details "
            "are not available."
        )


def _render_category_sentiment(
    repository,
    sentiment_service,
):
    """Render category-level sentiment analysis."""

    categories = repository.get_search_keywords()

    if not categories:
        st.warning("No categories are available.")
        return

    selected_category = st.selectbox(
        "Select Category",
        options=categories,
    )

    products = repository.get_products_by_search_keyword(
        selected_category
    )

    st.caption(
        f"{len(products):,} products in this category"
    )

    with st.spinner("Analyzing category sentiment..."):
        summary = (
            sentiment_service
            .get_category_sentiment_summary(
                selected_category
            )
        )

    st.divider()

    _render_sentiment_metrics(summary)

    st.divider()

    _render_sentiment_chart(summary)

    st.divider()

    st.subheader("Category Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Products",
            f"{len(products):,}",
        )

    with col2:
        total_reviews = sum(
            product.review_count
            for product in products
        )

        st.metric(
            "Collected Reviews",
            f"{total_reviews:,}",
        )