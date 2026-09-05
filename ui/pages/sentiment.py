from collections import Counter

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _safe_percentage(value) -> float:
    """Safely convert a percentage value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _get_summary_value(
    summary,
    field_name,
    default=0,
):
    """Safely read a summary dictionary value.

    Supports the current nested SentimentService response:

        {
            "positive": {
                "count": ...,
                "percentage": ...
            },
            ...
        }

    Also supports the older flattened structure for compatibility.
    """
    if not isinstance(summary, dict):
        return default

    # Direct field exists.
    if field_name in summary:
        return summary.get(field_name, default)

    # Sentiment count fields.
    sentiment_count_fields = {
        "positive_count": ("positive", "count"),
        "neutral_count": ("neutral", "count"),
        "negative_count": ("negative", "count"),
    }

    if field_name in sentiment_count_fields:
        sentiment, nested_field = sentiment_count_fields[field_name]
        sentiment_data = summary.get(sentiment, {})

        if isinstance(sentiment_data, dict):
            return sentiment_data.get(nested_field, default)

    # Sentiment percentage fields.
    sentiment_percentage_fields = {
        "positive_percentage": ("positive", "percentage"),
        "neutral_percentage": ("neutral", "percentage"),
        "negative_percentage": ("negative", "percentage"),
    }

    if field_name in sentiment_percentage_fields:
        sentiment, nested_field = sentiment_percentage_fields[field_name]
        sentiment_data = summary.get(sentiment, {})

        if isinstance(sentiment_data, dict):
            return sentiment_data.get(nested_field, default)

    return default


# ---------------------------------------------------------
# Shared Sentiment Components
# ---------------------------------------------------------

def _render_sentiment_metrics(summary):
    """Render sentiment metrics."""
    total_reviews = _get_summary_value(summary, "total_reviews", 0)

    positive_count = _get_summary_value(summary, "positive_count", 0)
    neutral_count = _get_summary_value(summary, "neutral_count", 0)
    negative_count = _get_summary_value(summary, "negative_count", 0)

    positive_percentage = _safe_percentage(
        _get_summary_value(summary, "positive_percentage", 0.0)
    )

    neutral_percentage = _safe_percentage(
        _get_summary_value(summary, "neutral_percentage", 0.0)
    )

    negative_percentage = _safe_percentage(
        _get_summary_value(summary, "negative_percentage", 0.0)
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Reviews", f"{total_reviews:,}")

    with col2:
        st.metric(
            "Positive",
            f"{positive_count:,}",
            f"{positive_percentage:.1f}%",
        )

    with col3:
        st.metric(
            "Neutral",
            f"{neutral_count:,}",
            f"{neutral_percentage:.1f}%",
        )

    with col4:
        st.metric(
            "Negative",
            f"{negative_count:,}",
            f"{negative_percentage:.1f}%",
        )


def _render_sentiment_chart(summary):
    """Render sentiment distribution chart."""
    chart_data = pd.DataFrame(
        {
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Reviews": [
                _get_summary_value(summary, "positive_count", 0),
                _get_summary_value(summary, "neutral_count", 0),
                _get_summary_value(summary, "negative_count", 0),
            ],
        }
    )

    st.bar_chart(
        chart_data,
        x="Sentiment",
        y="Reviews",
    )


# ---------------------------------------------------------
# Dataset Overview
# ---------------------------------------------------------

def _render_dataset_overview(repository):
    """Render dataset-level sentiment analytics."""
    products = repository.get_all_products()

    all_reviews = []
    for product in products:
        all_reviews.extend(product.reviews)

    counts = Counter(
        str(review.sentiment).lower()
        for review in all_reviews
        if review.sentiment
    )

    total_reviews = len(all_reviews)

    positive_count = counts.get("positive", 0)
    neutral_count = counts.get("neutral", 0)
    negative_count = counts.get("negative", 0)

    summary = {
        "total_reviews": total_reviews,
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count,
        "positive_percentage": (
            positive_count / total_reviews * 100 if total_reviews else 0
        ),
        "neutral_percentage": (
            neutral_count / total_reviews * 100 if total_reviews else 0
        ),
        "negative_percentage": (
            negative_count / total_reviews * 100 if total_reviews else 0
        ),
    }

    st.subheader("Dataset Sentiment Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Products", f"{len(products):,}")

    with col2:
        st.metric(
            "Search Categories",
            f"{len(repository.get_search_keywords()):,}",
        )

    st.divider()

    _render_sentiment_metrics(summary)

    st.divider()

    st.subheader("Review Sentiment Distribution")

    _render_sentiment_chart(summary)


# ---------------------------------------------------------
# Product Explorer
# ---------------------------------------------------------

def _render_product_sentiment(repository, sentiment_service):
    """Render product-level sentiment analytics."""
    products = repository.get_all_products()

    if not products:
        st.warning("No products are available.")
        return

    product_options = {
        f"{product.asin} | {product.title[:70]}": product.asin
        for product in products
    }

    selected_label = st.selectbox(
        "Select Product",
        options=list(product_options.keys()),
    )

    selected_asin = product_options[selected_label]
    product = repository.get_product_by_asin(selected_asin)

    # Product information
    info_col1, info_col2 = st.columns([1, 3])

    with info_col1:
        if product.image:
            st.image(product.image, use_container_width=True)

    with info_col2:
        st.subheader(product.title)
        st.write(f"**Brand:** {product.brand or 'N/A'}")
        st.write(f"**ASIN:** {product.asin}")
        st.write(f"**Price:** ${product.price:.2f}")
        st.write(f"**Amazon Rating:** {product.average_rating:.1f} ⭐")

    summary = sentiment_service.get_product_sentiment_summary(selected_asin)

    st.divider()

    st.subheader("Product Sentiment Summary")

    _render_sentiment_metrics(summary)

    overall_sentiment = summary.get("overall_sentiment")
    sentiment_score = summary.get("sentiment_score")

    if overall_sentiment:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Overall Sentiment", str(overall_sentiment).title())

        with col2:
            if sentiment_score is not None:
                st.metric("Sentiment Score", f"{float(sentiment_score):.2f}")

    st.divider()

    st.subheader("Sentiment Distribution")

    _render_sentiment_chart(summary)

    st.divider()

    _render_product_reviews(sentiment_service, selected_asin)


# ---------------------------------------------------------
# Product Reviews
# ---------------------------------------------------------

def _render_product_reviews(sentiment_service, asin):
    """Render persisted review-level predictions."""
    reviews = sentiment_service.get_product_review_sentiments(asin)

    st.subheader("Review-Level Predictions")

    if not reviews:
        st.info("No reviews available.")
        return

    for index, review in enumerate(reviews, start=1):
        sentiment = str(review.get("sentiment", "unknown")).title()
        rating = review.get("star_rating")
        title = review.get("review_title", "")
        description = review.get("review_description", "")

        positive_probability = review.get("positive_probability", 0.0)
        neutral_probability = review.get("neutral_probability", 0.0)
        negative_probability = review.get("negative_probability", 0.0)

        with st.expander(f"Review {index} · {sentiment}"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if rating is not None:
                    st.metric("Rating", f"{rating} ⭐")

            with col2:
                st.metric("Positive", f"{positive_probability:.1%}")

            with col3:
                st.metric("Neutral", f"{neutral_probability:.1%}")

            with col4:
                st.metric("Negative", f"{negative_probability:.1%}")

            if title:
                st.markdown(f"**{title}**")

            if description:
                st.write(description)


# ---------------------------------------------------------
# Category Analysis
# ---------------------------------------------------------

def _render_category_sentiment(repository, sentiment_service):
    """Render category-level sentiment analytics."""
    categories = repository.get_search_keywords()

    if not categories:
        st.warning("No categories available.")
        return

    selected_category = st.selectbox(
        "Select Search Category",
        options=categories,
    )

    products = repository.get_products_by_search_keyword(selected_category)
    summary = sentiment_service.get_category_sentiment_summary(
        selected_category
    )

    st.caption(f"{len(products):,} products in this category")

    st.divider()

    st.subheader("Category Sentiment Summary")

    _render_sentiment_metrics(summary)

    st.divider()

    st.subheader("Sentiment Distribution")

    _render_sentiment_chart(summary)


# ---------------------------------------------------------
# Live Prediction
# ---------------------------------------------------------

def _render_live_prediction(sentiment_service):
    """Render real-time DistilBERT sentiment inference."""
    st.subheader("Real-Time Review Sentiment Prediction")

    st.write(
        "Enter a new customer review below. "
        "The saved DistilBERT model will analyze "
        "the sentiment in real time."
    )

    review_text = st.text_area(
        "Customer Review",
        height=180,
        placeholder=(
            "Example: The product quality is "
            "excellent and exceeded my expectations."
        ),
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not review_text.strip():
            st.warning("Please enter a review first.")
            return

        try:
            with st.spinner("Running sentiment model..."):
                result = sentiment_service.predict_review_sentiment(
                    review_text
                )

            st.success("Prediction completed.")

            st.subheader(
                f"Predicted Sentiment: {result.sentiment.title()}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Positive", f"{result.positive_probability:.2%}")

            with col2:
                st.metric("Neutral", f"{result.neutral_probability:.2%}")

            with col3:
                st.metric("Negative", f"{result.negative_probability:.2%}")

        except Exception as error:
            st.error(f"Sentiment prediction failed: {error}")


# ---------------------------------------------------------
# Main Page
# ---------------------------------------------------------

def render_sentiment_page(repository, sentiment_service):
    """Render Feature B — Review Sentiment Analysis.

    Architecture:
    - Precomputed predictions for analytics
    - Real-time inference for new reviews
    """
    st.title("💬 Review Sentiment Analysis")

    st.write(
        "Explore precomputed sentiment analytics "
        "from Amazon customer reviews and run "
        "real-time predictions using the trained "
        "DistilBERT model."
    )

    overview_tab, product_tab, category_tab, live_tab = st.tabs(
        [
            "📊 Dataset Overview",
            "🔍 Product Explorer",
            "📂 Category Analysis",
            "⚡ Live Prediction",
        ]
    )

    with overview_tab:
        _render_dataset_overview(repository)

    with product_tab:
        _render_product_sentiment(repository, sentiment_service)

    with category_tab:
        _render_category_sentiment(repository, sentiment_service)

    with live_tab:
        _render_live_prediction(sentiment_service)