import difflib
import math

import streamlit as st

from ui.components.helpers import (
    format_price,
    get_product_search_text,
)


PRODUCTS_PER_PAGE = 20
PRODUCTS_PER_ROW = 4


def render_product_explorer(
    repository,
):
    """
    Render product browsing and filtering interface.

    Displays products in a paginated card grid.
    """

    st.title("📦 Product Explorer")

    products = repository.get_all_products()
    categories = repository.get_search_keywords()

    # ========================================================
    # FILTERS
    # ========================================================

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        selected_category = st.selectbox(
            "Filter by Search Keyword",
            options=["All"] + categories,
        )

    with col2:

        search_query = st.text_input(
            "Search Products",
            placeholder=(
                "Try product name, brand, "
                "ASIN, or keyword..."
            ),
        )

    # ========================================================
    # CATEGORY FILTER
    # ========================================================

    filtered_products = products

    if selected_category != "All":

        filtered_products = [
            product
            for product in filtered_products
            if product.search_keyword
            == selected_category
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
                if query in get_product_search_text(
                    product
                )
            ]

            if exact_matches:

                filtered_products = exact_matches

            else:

                # --------------------------------------------
                # Fuzzy token matching
                # --------------------------------------------

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

                            best_score = max(
                                best_score,
                                score,
                            )

                    if best_score >= 0.60:

                        fuzzy_matches.append(
                            (
                                product,
                                best_score,
                            )
                        )

                fuzzy_matches.sort(
                    key=lambda item: item[1],
                    reverse=True,
                )

                filtered_products = [
                    product
                    for product, _score
                    in fuzzy_matches
                ]

    # ========================================================
    # RESET PAGE WHEN FILTER CHANGES
    # ========================================================

    filter_signature = (
        f"{selected_category}|{search_query}"
    )

    if (
        st.session_state.get(
            "product_filter_signature"
        )
        != filter_signature
    ):

        st.session_state[
            "product_filter_signature"
        ] = filter_signature

        st.session_state[
            "product_page"
        ] = 1

    # ========================================================
    # RESULT COUNT
    # ========================================================

    total_products = len(filtered_products)

    st.caption(
        f"Showing {total_products} matching products"
    )

    if total_products == 0:

        st.warning(
            "No products match your search."
        )

        st.info(
            "Try a shorter keyword or check the spelling."
        )

        return

    # ========================================================
    # PAGINATION
    # ========================================================

    total_pages = math.ceil(
        total_products / PRODUCTS_PER_PAGE
    )

    current_page = st.session_state.get(
        "product_page",
        1,
    )

    current_page = max(
        1,
        min(current_page, total_pages),
    )

    start_index = (
        current_page - 1
    ) * PRODUCTS_PER_PAGE

    end_index = min(
        start_index + PRODUCTS_PER_PAGE,
        total_products,
    )

    page_products = filtered_products[
        start_index:end_index
    ]

    st.write(
        f"Page **{current_page}** of **{total_pages}** "
        f"· Products **{start_index + 1}–{end_index}**"
    )

    st.divider()

    # ========================================================
    # PRODUCT CARD GRID
    # ========================================================

    for row_start in range(
        0,
        len(page_products),
        PRODUCTS_PER_ROW,
    ):

        row_products = page_products[
            row_start:
            row_start + PRODUCTS_PER_ROW
        ]

        columns = st.columns(
            PRODUCTS_PER_ROW
        )

        for column, product in zip(
            columns,
            row_products,
        ):

            with column:

                render_product_card(
                    product
                )

    # ========================================================
    # PAGINATION CONTROLS
    # ========================================================

    st.divider()

    previous_col, page_col, next_col = st.columns(
        [1, 3, 1]
    )

    with previous_col:

        if st.button(
            "← Previous",
            disabled=current_page <= 1,
            use_container_width=True,
        ):

            st.session_state[
                "product_page"
            ] = current_page - 1

            st.rerun()

    with page_col:

        st.markdown(
            f"<div style='text-align:center'>"
            f"Page <b>{current_page}</b> "
            f"of <b>{total_pages}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with next_col:

        if st.button(
            "Next →",
            disabled=current_page >= total_pages,
            use_container_width=True,
        ):

            st.session_state[
                "product_page"
            ] = current_page + 1

            st.rerun()


# ============================================================
# PRODUCT CARD
# ============================================================

def render_product_card(
    product,
):
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

    with st.container(
        border=True,
    ):

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

        title = (
            product.title
            or "Untitled Product"
        )

        if len(title) > 65:

            title = title[:65] + "..."

        st.markdown(
            f"**{title}**"
        )

        # ----------------------------------------------------
        # ASIN / BRAND
        # ----------------------------------------------------

        st.caption(
            f"ASIN: {product.asin}"
        )

        st.write(
            f"**{product.brand or 'N/A'}**"
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "Price",
            format_price(
                product.price
            ),
        )

        metric2.metric(
            "Rating",
            f"{product.average_rating:.1f} ⭐",
        )

        # ----------------------------------------------------
        # VIEW DETAILS
        # ----------------------------------------------------

        st.button(
            "View Details",
            key=f"view_{product.asin}",
            use_container_width=True,
        )