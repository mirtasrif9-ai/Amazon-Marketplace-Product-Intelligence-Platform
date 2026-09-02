from pathlib import Path

import pandas as pd

from .models.thumbnail_group_result import ThumbnailGroupResult


class ThumbnailGroupingService:
    """
    Service for retrieving Feature C visual-group assignments.

    The final visual groups were generated offline using:

        CLIP ViT-B/32
            ↓
        L2 normalization
            ↓
        PCA (95% variance)
            ↓
        K-Means (K=25)

    The Streamlit application only loads the final assignments
    and does not need to rerun clustering.
    """

    def __init__(self, project_root: Path | None = None):

        if project_root is None:
            project_root = Path(__file__).resolve().parents[3]

        self.project_root = Path(project_root)

        self.assignments_path = (
            self.project_root
            / "data"
            / "processed"
            / "feature_c"
            / "outputs"
            / "thumbnail_group_assignments.csv"
        )

        self.manifest_path = (
            self.project_root
            / "data"
            / "processed"
            / "feature_c"
            / "manifests"
            / "image_manifest_final.csv"
        )

        self.image_dir = (
            self.project_root
            / "data"
            / "images"
        )

        self._load_data()

    def _load_data(self) -> None:
        """Load Feature C assignments and image manifest."""

        if not self.assignments_path.exists():
            raise FileNotFoundError(
                f"Feature C assignments not found: "
                f"{self.assignments_path}"
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Feature C manifest not found: "
                f"{self.manifest_path}"
            )

        self.assignments = pd.read_csv(
            self.assignments_path,
            dtype={"asin": str}
        )

        self.manifest = pd.read_csv(
            self.manifest_path,
            dtype={"asin": str}
        )

        self.assignments["asin"] = (
            self.assignments["asin"]
            .str.strip()
        )

        self.manifest["asin"] = (
            self.manifest["asin"]
            .str.strip()
        )

    def get_product_group(
        self,
        asin: str
    ) -> ThumbnailGroupResult | None:
        """
        Return the visual group assigned to an ASIN.
        """

        asin = str(asin).strip()

        matches = self.assignments[
            self.assignments["asin"] == asin
        ]

        if matches.empty:
            return None

        row = matches.iloc[0]

        group_size = int(
            (
                self.assignments["visual_group"]
                == row["visual_group"]
            ).sum()
        )

        return ThumbnailGroupResult(
            asin=asin,
            cluster_id=int(row["cluster_id"]),
            visual_group=str(row["visual_group"]),
            group_size=group_size,
        )

    def get_group_products(
        self,
        visual_group: str
    ) -> pd.DataFrame:
        """
        Return all products belonging to a visual group.
        """

        return self.assignments[
            self.assignments["visual_group"] == visual_group
        ].copy()

    def get_group_images(
        self,
        visual_group: str
    ) -> list[dict]:
        """
        Return image information for a visual group.
        """

        group = self.get_group_products(visual_group)

        results = []

        for _, row in group.iterrows():

            image_path = self.image_dir / row["filename"]

            if image_path.exists():
                results.append(
                    {
                        "asin": row["asin"],
                        "filename": row["filename"],
                        "image_path": image_path,
                        "visual_group": row["visual_group"],
                    }
                )

        return results

    def get_group_size(
        self,
        visual_group: str
    ) -> int:
        """Return the number of products in a visual group."""

        return int(
            (
                self.assignments["visual_group"]
                == visual_group
            ).sum()
        )

    def get_all_groups(self) -> pd.DataFrame:
        """Return summary information for all visual groups."""

        return (
            self.assignments
            .groupby(
                ["cluster_id", "visual_group"],
                as_index=False
            )
            .size()
            .rename(columns={"size": "group_size"})
            .sort_values("cluster_id")
            .reset_index(drop=True)
        )