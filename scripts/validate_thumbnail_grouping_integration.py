from pathlib import Path

from src.features.thumbnail_grouping.thumbnail_grouping_service import (
    ThumbnailGroupingService,
)

# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# HELPERS
# ============================================================================


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)

    print(f"✓ {message}")


# ============================================================================
# VALIDATION
# ============================================================================


def main() -> None:
    print_section("FEATURE C — THUMBNAIL GROUPING INTEGRATION VALIDATION")

    print(f"Project root: {PROJECT_ROOT}")

    # ------------------------------------------------------------------------
    # 1. Initialize service
    # ------------------------------------------------------------------------

    print_section("1. INITIALIZE THUMBNAIL GROUPING SERVICE")

    service = ThumbnailGroupingService(project_root=PROJECT_ROOT)

    check(
        service.assignments_path.exists(),
        "Thumbnail group assignments file exists",
    )

    check(
        service.manifest_path.exists(),
        "Image manifest exists",
    )

    check(
        service.model_dir.exists(),
        "Feature C model directory exists",
    )

    print(f"Assignments: {service.assignments_path}")
    print(f"Manifest:    {service.manifest_path}")
    print(f"Model dir:   {service.model_dir}")

    # ------------------------------------------------------------------------
    # 2. Validate loaded data
    # ------------------------------------------------------------------------

    print_section("2. VALIDATE PRECOMPUTED DATA")

    check(
        not service.assignments.empty,
        "Thumbnail group assignments loaded",
    )

    check(
        not service.manifest.empty,
        "Image manifest loaded",
    )

    check(
        "asin" in service.assignments.columns,
        "Assignments contain 'asin' column",
    )

    check(
        "cluster_id" in service.assignments.columns,
        "Assignments contain 'cluster_id' column",
    )

    check(
        "visual_group" in service.assignments.columns,
        "Assignments contain 'visual_group' column",
    )

    check(
        "filename" in service.assignments.columns,
        "Assignments contain 'filename' column",
    )

    print(f"Assignment rows: {len(service.assignments)}")
    print(f"Manifest rows:   {len(service.manifest)}")

    # ------------------------------------------------------------------------
    # 3. Validate group summary
    # ------------------------------------------------------------------------

    print_section("3. VALIDATE VISUAL GROUPS")

    groups = service.get_all_groups()

    check(
        not groups.empty,
        "Visual group summary is not empty",
    )

    check(
        len(groups) > 0,
        "At least one visual group exists",
    )

    check(
        "cluster_id" in groups.columns,
        "Group summary contains cluster_id",
    )

    check(
        "visual_group" in groups.columns,
        "Group summary contains visual_group",
    )

    check(
        "group_size" in groups.columns,
        "Group summary contains group_size",
    )

    print(f"Number of visual groups: {len(groups)}")
    print(f"Number of clusters:      {groups['cluster_id'].nunique()}")

    # ------------------------------------------------------------------------
    # 4. Validate ASIN → group
    # ------------------------------------------------------------------------

    print_section("4. VALIDATE ASIN → VISUAL GROUP")

    test_asin = str(service.assignments.iloc[0]["asin"])

    result = service.get_product_group(test_asin)

    check(
        result is not None,
        "ASIN resolves to a visual group",
    )

    check(
        result.asin == test_asin,
        "Returned ASIN matches requested ASIN",
    )

    check(
        isinstance(result.cluster_id, int),
        "Cluster ID is an integer",
    )

    check(
        result.visual_group.startswith("visual_group_"),
        "Visual group uses expected naming convention",
    )

    check(
        result.group_size > 0,
        "Visual group has at least one product",
    )

    print()
    print(f"Test ASIN:      {result.asin}")
    print(f"Cluster ID:     {result.cluster_id}")
    print(f"Visual group:   {result.visual_group}")
    print(f"Group size:     {result.group_size}")

    # ------------------------------------------------------------------------
    # 5. Validate group → products
    # ------------------------------------------------------------------------

    print_section("5. VALIDATE VISUAL GROUP → PRODUCTS")

    group_products = service.get_group_products(result.visual_group)

    check(
        not group_products.empty,
        "Visual group returns products",
    )

    check(
        len(group_products) == result.group_size,
        "Returned product count matches group size",
    )

    print(f"Products in {result.visual_group}: {len(group_products)}")

    # ------------------------------------------------------------------------
    # 6. Validate group → images
    # ------------------------------------------------------------------------

    print_section("6. VALIDATE VISUAL GROUP → IMAGES")

    group_images = service.get_group_images(result.visual_group)

    check(
        len(group_images) > 0,
        "Visual group contains available images",
    )

    for image_info in group_images[:3]:
        check(
            image_info["image_path"].exists(),
            f"Image exists: {image_info['filename']}",
        )

    print(f"Available images in group: {len(group_images)}")

    # ------------------------------------------------------------------------
    # 7. Validate saved model paths
    # ------------------------------------------------------------------------

    print_section("7. VALIDATE SAVED MODEL ARTIFACTS")

    check(
        service.pca_model_path.exists(),
        "Saved PCA model exists",
    )

    check(
        service.kmeans_model_path.exists(),
        "Saved K-Means model exists",
    )

    print(f"PCA:     {service.pca_model_path}")
    print(f"K-Means: {service.kmeans_model_path}")

    # ------------------------------------------------------------------------
    # 8. Load saved PCA + K-Means
    # ------------------------------------------------------------------------

    print_section("8. LOAD SAVED PCA + K-MEANS MODELS")

    service._load_clustering_models()

    check(
        service._pca_model is not None,
        "PCA model loaded successfully",
    )

    check(
        service._kmeans_model is not None,
        "K-Means model loaded successfully",
    )

    cluster_count = service._kmeans_model.n_clusters

    check(
        cluster_count > 0,
        "K-Means reports a valid cluster count",
    )

    print(f"K-Means clusters: {cluster_count}")

    # ------------------------------------------------------------------------
    # 9. Validate model information
    # ------------------------------------------------------------------------

    print_section("9. VALIDATE MODEL INFORMATION")

    model_info = service.get_model_information()

    check(
        model_info["clip_model"] == "ViT-B/32",
        "CLIP model is ViT-B/32",
    )

    check(
        model_info["embedding_dimension"] == 512,
        "CLIP embedding dimension is 512",
    )

    check(
        model_info["normalization"] == "L2",
        "Normalization is L2",
    )

    check(
        model_info["number_of_clusters"] == cluster_count,
        "Reported cluster count matches K-Means",
    )

    print()
    for key, value in model_info.items():
        print(f"{key}: {value}")

    # ------------------------------------------------------------------------
    # 10. Validate CLIP + real-time inference
    # ------------------------------------------------------------------------

    print_section("10. VALIDATE REAL-TIME IMAGE GROUPING")

    test_image_info = group_images[0]
    test_image_path = Path(test_image_info["image_path"])

    check(
        test_image_path.exists(),
        "Real-time inference test image exists",
    )

    print(f"Test image: {test_image_path}")

    realtime_result = service.predict_image_group_from_file(test_image_path)

    check(
        realtime_result is not None,
        "Real-time prediction returned a result",
    )

    check(
        realtime_result.asin == "new_image",
        "Real-time prediction identifies input as new_image",
    )

    check(
        isinstance(realtime_result.cluster_id, int),
        "Predicted cluster ID is an integer",
    )

    check(
        realtime_result.visual_group.startswith("visual_group_"),
        "Predicted visual group uses expected naming convention",
    )

    check(
        0 <= realtime_result.cluster_id < cluster_count,
        "Predicted cluster ID is within valid K-Means range",
    )

    print()
    print(f"Predicted cluster:   {realtime_result.cluster_id}")
    print(f"Predicted group:     {realtime_result.visual_group}")
    print(f"Existing group size: {realtime_result.group_size}")

    # ------------------------------------------------------------------------
    # 11. Final validation
    # ------------------------------------------------------------------------

    print_section("VALIDATION COMPLETE")

    print("✓ Feature C integration validation PASSED")
    print()
    print("Precomputed grouping: PASS")
    print("Saved PCA/K-Means loading: PASS")
    print("CLIP ViT-B/32 inference: PASS")
    print("Real-time image grouping: PASS")
    print()
    print("Feature C backend is ready for Streamlit integration.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()