def get_product_label(product) -> str:
    """Create a readable product label."""

    title = product.title or "Untitled Product"

    if len(title) > 70:
        title = title[:70] + "..."

    return f"{product.asin} | {title}"


def format_price(price) -> str:
    """Format product price safely."""

    if price is None or price == 0:
        return "N/A"

    return f"${float(price):,.2f}"


def get_product_search_text(product) -> str:
    """Combine searchable product fields."""

    fields = [
        product.title,
        product.brand,
        product.asin,
        product.search_keyword,
        product.description,
    ]

    return " ".join(
        str(field).lower()
        for field in fields
        if field
    )