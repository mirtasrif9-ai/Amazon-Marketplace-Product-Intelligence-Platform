from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.features.thumbnail_grouping.thumbnail_grouping_service import (
    ThumbnailGroupingService,
)


def render_thumbnail_groups(
    thumbnail_service: ThumbnailGroupingService,
):
    """Render the Feature C thumbnail grouping interface."""

    st.title("🖼️ Feature C — Thumbnail Grouping")

    st.markdown(
        """
        Explore product thumbnails grouped by visual similarity using
        **CLIP ViT-B/32 → PCA → K-Means**.

        You can either explore an existing visual group or upload a new
        product image and predict which visual group it belongs to.
        """
    )

    st.divider()

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    try:
        model_info = thumbnail_service.get_model_information()

        st.subheader("🔬 Model Information")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "CLIP Model",
            model_info.get("clip_model", "ViT-B/32"),
        )

        col2.metric(
            "Embedding Dimensions",
            model_info.get("embedding_dimension", 512),
        )

        col3.metric(
            "Normalization",
            model_info.get("normalization", "L2"),
        )

        col4.metric(
            "Visual Groups",
            model_info.get("number_of_clusters", 0),
        )

    except Exception as error:
        st.warning("Unable to load model information.")
        st.exception(error)

    st.divider()

    # ============================================================
    # SECTION 1 — EXISTING VISUAL GROUPS
    # ============================================================

    st.subheader("📦 Explore Existing Visual Groups")

    try:
        groups = thumbnail_service.get_all_groups()

        if groups is None or (isinstance(groups, (list, pd.DataFrame)) and len(groups) == 0):
            st.warning("No visual groups are available.")

        else:
            # Convert possible group representations into names.
            group_names = []

            # Handle DataFrame output from get_all_groups()
            if isinstance(groups, pd.DataFrame):
                group_list = groups.to_dict("records")
            else:
                group_list = groups

            for group in group_list:
                if isinstance(group, str):
                    group_names.append(group)

                elif isinstance(group, dict):
                    visual_group = group.get("visual_group")
                    if visual_group:
                        group_names.append(visual_group)

                else:
                    visual_group = getattr(group, "visual_group", None)
                    if visual_group:
                        group_names.append(visual_group)

            group_names = sorted(
                set(group_names),
                key=lambda value: (
                    int(value.rsplit("_", 1)[-1])
                    if value.rsplit("_", 1)[-1].isdigit()
                    else value
                ),
            )

            selected_group = st.selectbox(
                "Select a visual group",
                group_names,
            )

            if selected_group:
                try:
                    group_size = thumbnail_service.get_group_size(selected_group)
                    st.metric("Products in Group", group_size)

                except Exception:
                    group_size = None

                st.markdown(f"### 👁️ {selected_group}")

                # ------------------------------------------------
                # PRODUCTS
                # ------------------------------------------------

                try:
                    products = thumbnail_service.get_group_products(selected_group)

                    if products is not None and not (isinstance(products, pd.DataFrame) and products.empty):
                        # Extract ASINs whether products is a DataFrame, Series, or list
                        if isinstance(products, pd.DataFrame):
                            asin_list = products["asin"].tolist()
                        elif isinstance(products, pd.Series):
                            asin_list = products.tolist()
                        else:
                            asin_list = products

                        st.write(f"**Products:** {len(asin_list)}")

                        product_columns = st.columns(5)

                        for index, asin in enumerate(asin_list):
                            with product_columns[index % 5]:
                                st.code(str(asin), language=None)

                except Exception as error:
                    st.warning("Unable to load products for this group.")
                    st.exception(error)

                # ------------------------------------------------
                # IMAGES
                # ------------------------------------------------

                st.markdown("#### 🖼️ Group Thumbnails")

                try:
                    group_images = thumbnail_service.get_group_images(selected_group)

                    if not group_images:
                        st.info("No images are available for this group.")

                    else:
                        image_columns = st.columns(5)

                        for index, image_info in enumerate(group_images):
                            with image_columns[index % 5]:
                                # Support both direct path strings/Path objects and dicts
                                if isinstance(image_info, dict):
                                    path = Path(image_info["image_path"])
                                    asin_label = image_info.get("asin", path.stem)
                                else:
                                    path = Path(image_info)
                                    asin_label = path.stem

                                if path.exists():
                                    st.image(
                                        str(path),
                                        use_container_width=True,
                                    )
                                    st.caption(asin_label)

                                else:
                                    st.warning(f"Image not found: {path.name}")

                except Exception as error:
                    st.error("Unable to load group images.")
                    st.exception(error)

    except Exception as error:
        st.error("Failed to load visual groups.")
        st.exception(error)

    st.divider()

    # ============================================================
    # SECTION 2 — REAL-TIME IMAGE GROUPING
    # ============================================================

    st.subheader("🔍 Predict Visual Group for a New Image")

    st.markdown(
        """
        Upload a product thumbnail to run real-time inference:

        **Image → CLIP ViT-B/32 → L2 Normalization → PCA → K-Means**
        """
    )

    uploaded_file = st.file_uploader(
        "Upload a product thumbnail",
        type=["jpg", "jpeg", "png", "webp"],
        key="thumbnail_group_upload",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.markdown("#### Uploaded Image")

            preview_col, result_col = st.columns([1, 1])

            with preview_col:
                st.image(
                    image,
                    caption="Uploaded product thumbnail",
                    use_container_width=True,
                )

            with result_col:
                with st.spinner("Running CLIP + PCA + K-Means inference..."):
                    result = thumbnail_service.predict_image_group(image)

                st.success("Visual group predicted successfully.")

                st.metric(
                    "Predicted Visual Group",
                    result.visual_group,
                )

                st.metric(
                    "Cluster ID",
                    result.cluster_id,
                )

                st.metric(
                    "Existing Products in Group",
                    result.group_size,
                )

                st.caption(
                    "The uploaded image is treated as a new image "
                    "and is not added to the dataset."
                )

                # ------------------------------------------------
                # SHOW OTHER PRODUCTS FROM PREDICTED GROUP
                # ------------------------------------------------

                st.markdown("#### Similar Visual Group")

                try:
                    predicted_images = thumbnail_service.get_group_images(
                        result.visual_group
                    )

                    if predicted_images:
                        image_columns = st.columns(5)

                        for index, image_info in enumerate(predicted_images):
                            with image_columns[index % 5]:
                                if isinstance(image_info, dict):
                                    path = Path(image_info["image_path"])
                                    asin_label = image_info.get("asin", path.stem)
                                else:
                                    path = Path(image_info)
                                    asin_label = path.stem

                                if path.exists():
                                    st.image(
                                        str(path),
                                        use_container_width=True,
                                    )
                                    st.caption(asin_label)

                except Exception as error:
                    st.warning(
                        "Predicted group was found, "
                        "but existing group images could not be displayed."
                    )
                    st.exception(error)

        except Exception as error:
            st.error("Failed to process the uploaded image.")
            st.exception(error)