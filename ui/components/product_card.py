import streamlit as st

from ui.components.helpers import format_price


def render_product_card(
    product,
    key_prefix: str = "",
) -> bool:
    """
    Render a compact product card.

    Returns True when the user clicks View Details.
    """

    with st.container(border=True):

        image_col, content_col = st.columns(
            [1, 3]
        )

        with image_col:

            if product.image:

                try:
                    st.image(
                        product.image,
                        use_container_width=True,
                    )

                except Exception:
                    st.caption("Image unavailable")

            else:
                st.caption("No image")

        with content_col:

            title = product.title or "Untitled Product"

            if len(title) > 100:
                title = title[:100] + "..."

            st.markdown(
                f"**{title}**"
            )

            st.caption(
                f"ASIN: {product.asin}"
            )

            st.write(
                f"**Brand:** {product.brand or 'N/A'}"
            )

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Price",
                format_price(product.price),
            )

            metric2.metric(
                "Rating",
                f"{product.average_rating:.1f} ⭐",
            )

            metric3.metric(
                "Reviews",
                product.review_count,
            )

            view_clicked = st.button(
                "View Details",
                key=(
                    f"{key_prefix}_"
                    f"details_{product.asin}"
                ),
            )

    return view_clicked