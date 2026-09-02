# ============================================================
# TEST — Feature C Thumbnail Grouping
# ============================================================

from pathlib import Path

from src.features.thumbnail_grouping.thumbnail_grouping_service import (
    ThumbnailGroupingService,
)


def test_feature_c_thumbnail_grouping():
    """
    Verify that Feature C artifacts are correctly integrated.

    Expected:
    - 946 products
    - 25 visual groups
    - every product has a visual-group assignment
    - every product image exists locally
    """

    project_root = Path(__file__).resolve().parents[1]

    service = ThumbnailGroupingService(
        project_root=project_root
    )

    # --------------------------------------------------------
    # 1. Verify product count
    # --------------------------------------------------------

    assignments = service.assignments

    assert len(assignments) == 946, (
        f"Expected 946 products, "
        f"found {len(assignments)}"
    )

    assert assignments["asin"].nunique() == 946, (
        "ASIN count does not equal 946"
    )

    # --------------------------------------------------------
    # 2. Verify visual-group count
    # --------------------------------------------------------

    groups = service.get_all_groups()

    assert len(groups) == 25, (
        f"Expected 25 visual groups, "
        f"found {len(groups)}"
    )

    # --------------------------------------------------------
    # 3. Verify every product has a group
    # --------------------------------------------------------

    assert assignments["visual_group"].notna().all(), (
        "Some products do not have a visual group"
    )

    assert assignments["cluster_id"].notna().all(), (
        "Some products do not have a cluster ID"
    )

    # --------------------------------------------------------
    # 4. Verify every image exists
    # --------------------------------------------------------

    missing_images = []

    for filename in assignments["filename"]:

        image_path = service.image_dir / filename

        if not image_path.exists():
            missing_images.append(str(image_path))

    assert len(missing_images) == 0, (
        f"{len(missing_images)} image files are missing.\n"
        f"Examples:\n{missing_images[:10]}"
    )

    # --------------------------------------------------------
    # 5. Verify group sizes
    # --------------------------------------------------------

    calculated_total = int(
        groups["group_size"].sum()
    )

    assert calculated_total == 946, (
        f"Group sizes total {calculated_total}, "
        f"expected 946"
    )

    # --------------------------------------------------------
    # 6. Verify individual ASIN lookup
    # --------------------------------------------------------

    sample_asin = assignments.iloc[0]["asin"]

    result = service.get_product_group(sample_asin)

    assert result is not None, (
        f"Could not find visual group for ASIN {sample_asin}"
    )

    assert result.asin == sample_asin
    assert result.visual_group.startswith("visual_group_")
    assert result.group_size > 0

    # --------------------------------------------------------
    # 7. Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FEATURE C — INTEGRATION VERIFICATION")
    print("=" * 70)

    print(f"Products             : {len(assignments)}")
    print(f"Unique ASINs         : {assignments['asin'].nunique()}")
    print(f"Visual groups        : {len(groups)}")
    print(f"Assigned products    : {assignments['visual_group'].notna().sum()}")
    print(f"Missing images       : {len(missing_images)}")
    print(f"Group size total     : {calculated_total}")

    print()
    print("Group size range:")
    print(
        f"  Minimum            : {groups['group_size'].min()}"
    )
    print(
        f"  Maximum            : {groups['group_size'].max()}"
    )
    print(
        f"  Median             : {groups['group_size'].median()}"
    )

    print()
    print(f"Sample ASIN          : {sample_asin}")
    print(f"Sample visual group  : {result.visual_group}")
    print(f"Sample group size    : {result.group_size}")

    print()
    print("=" * 70)
    print("✓ FEATURE C VERIFICATION PASSED")
    print("=" * 70)