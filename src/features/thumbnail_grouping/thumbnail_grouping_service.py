from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from .models.thumbnail_group_result import ThumbnailGroupResult


class ThumbnailGroupingService:
    """Service for Feature C visual thumbnail grouping.

    Existing products:
        Uses precomputed visual-group assignments.

    New images:
        CLIP ViT-B/32
            ↓
        L2 normalization
            ↓
        Saved PCA
            ↓
        Saved K-Means
            ↓
        Visual group prediction

    The clustering model itself is not retrained during inference.
    """

    def __init__(self, project_root: Path | None = None):
        if project_root is None:
            project_root = Path(__file__).resolve().parents[3]

        self.project_root = Path(project_root)

        # ------------------------------------------------------------------
        # Data paths
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Model paths
        # ------------------------------------------------------------------

        self.model_dir = (
            self.project_root
            / "models"
            / "thumbnail_grouping"
        )

        self.pca_model_path = (
            self.model_dir
            / "clip_vit_b32_pca.joblib"
        )

        self.kmeans_model_path = (
            self.model_dir
            / "kmeans_clip_vit_b32_thumbnail_grouping.joblib"
        )

        # ------------------------------------------------------------------
        # Lazy-loaded inference objects
        # ------------------------------------------------------------------

        self._clip_model = None
        self._clip_preprocess = None
        self._clip_device = None

        self._pca_model = None
        self._kmeans_model = None

        # ------------------------------------------------------------------
        # Load precomputed data immediately
        # ------------------------------------------------------------------

        self._load_data()

    # ======================================================================
    # PRECOMPUTED DATA
    # ======================================================================

    def _load_data(self) -> None:
        """Load Feature C assignments and image manifest."""

        if not self.assignments_path.exists():
            raise FileNotFoundError(
                f"Feature C assignments not found: {self.assignments_path}"
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Feature C manifest not found: {self.manifest_path}"
            )

        self.assignments = pd.read_csv(
            self.assignments_path,
            dtype={"asin": str},
        )

        self.manifest = pd.read_csv(
            self.manifest_path,
            dtype={"asin": str},
        )

        self.assignments["asin"] = (
            self.assignments["asin"].astype(str).str.strip()
        )

        self.manifest["asin"] = self.manifest["asin"].astype(str).str.strip()

    # ======================================================================
    # PRECOMPUTED PRODUCT GROUPING
    # ======================================================================

    def get_product_group(
        self,
        asin: str,
    ) -> ThumbnailGroupResult | None:
        """Return the precomputed visual group assigned to an ASIN."""

        asin = str(asin).strip()

        matches = self.assignments[self.assignments["asin"] == asin]

        if matches.empty:
            return None

        row = matches.iloc[0]

        group_size = int(
            (self.assignments["visual_group"] == row["visual_group"]).sum()
        )

        return ThumbnailGroupResult(
            asin=asin,
            cluster_id=int(row["cluster_id"]),
            visual_group=str(row["visual_group"]),
            group_size=group_size,
        )

    def get_group_products(
        self,
        visual_group: str,
    ) -> pd.DataFrame:
        """Return all products belonging to a visual group."""

        return self.assignments[
            self.assignments["visual_group"] == visual_group
        ].copy()

    def get_group_images(
        self,
        visual_group: str,
    ) -> list[dict]:
        """Return available image information for a visual group."""

        group = self.get_group_products(visual_group)

        results = []

        for _, row in group.iterrows():
            image_path = self.image_dir / str(row["filename"])

            if image_path.exists():
                results.append(
                    {
                        "asin": str(row["asin"]),
                        "filename": str(row["filename"]),
                        "image_path": image_path,
                        "visual_group": str(row["visual_group"]),
                    }
                )

        return results

    def get_group_size(
        self,
        visual_group: str,
    ) -> int:
        """Return the number of products in a visual group."""

        return int(
            (self.assignments["visual_group"] == visual_group).sum()
        )

    def get_all_groups(self) -> pd.DataFrame:
        """Return summary information for all visual groups."""

        return (
            self.assignments.groupby(
                ["cluster_id", "visual_group"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "group_size"})
            .sort_values("cluster_id")
            .reset_index(drop=True)
        )

    # ======================================================================
    # MODEL LOADING
    # ======================================================================

    def _load_clustering_models(self) -> None:
        """Load the saved PCA and K-Means models.

        These are loaded lazily because the Streamlit application does not need
        them when only browsing precomputed groups.
        """

        if self._pca_model is None:
            if not self.pca_model_path.exists():
                raise FileNotFoundError(
                    f"Feature C PCA model not found: {self.pca_model_path}"
                )

            self._pca_model = joblib.load(self.pca_model_path)

        if self._kmeans_model is None:
            if not self.kmeans_model_path.exists():
                raise FileNotFoundError(
                    f"Feature C K-Means model not found: {self.kmeans_model_path}"
                )

            self._kmeans_model = joblib.load(self.kmeans_model_path)

    def _load_clip_model(self) -> None:
        """Lazily load OpenAI CLIP ViT-B/32.

        The notebook used: clip.load("ViT-B/32", device=DEVICE, jit=False)

        The same model and official preprocessing pipeline are used here.
        """

        if self._clip_model is not None:
            return

        try:
            import clip
            import torch

        except ImportError as error:
            raise ImportError(
                "Feature C real-time inference requires the 'clip' and 'torch' packages."
            ) from error

        self._clip_device = "cuda" if torch.cuda.is_available() else "cpu"

        self._clip_model, self._clip_preprocess = clip.load(
            "ViT-B/32",
            device=self._clip_device,
            jit=False,
        )

        self._clip_model.eval()

        for parameter in self._clip_model.parameters():
            parameter.requires_grad = False

    # ======================================================================
    # IMAGE EMBEDDING
    # ======================================================================

    def _extract_clip_embedding(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        """Extract one CLIP ViT-B/32 image embedding.

        Preprocessing follows the Feature C model-development notebook.
        """

        import torch

        self._load_clip_model()

        image = image.convert("RGB")

        processed_image = self._clip_preprocess(image).unsqueeze(0)

        processed_image = processed_image.to(self._clip_device)

        with torch.no_grad():
            embedding = self._clip_model.encode_image(processed_image)

        embedding = embedding.detach().cpu().numpy().astype(np.float32)

        return embedding

    # ======================================================================
    # REAL-TIME IMAGE GROUPING
    # ======================================================================

    def predict_image_group(
        self,
        image: Image.Image,
    ) -> ThumbnailGroupResult:
        """Predict the visual group for a new image.

        Pipeline:
            Image ↓ CLIP ViT-B/32 ↓ L2 normalization ↓ Saved PCA ↓ Saved K-Means
            ↓ cluster_id ↓ visual_group
        """

        if image is None:
            raise ValueError("Image cannot be None.")

        self._load_clustering_models()

        embedding = self._extract_clip_embedding(image)

        # --------------------------------------------------------------
        # L2 normalization
        # --------------------------------------------------------------

        norm = np.linalg.norm(
            embedding,
            axis=1,
            keepdims=True,
        )

        if np.any(norm == 0):
            raise ValueError("CLIP embedding has zero L2 norm.")

        normalized_embedding = (embedding / norm).astype(np.float32)

        # --------------------------------------------------------------
        # PCA
        # --------------------------------------------------------------

        pca_embedding = self._pca_model.transform(normalized_embedding)

        pca_embedding = np.asarray(
            pca_embedding,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # K-Means prediction
        # --------------------------------------------------------------

        cluster_id = int(self._kmeans_model.predict(pca_embedding)[0])

        visual_group = f"visual_group_{cluster_id + 1}"

        # --------------------------------------------------------------
        # Existing group size
        # --------------------------------------------------------------

        group_size = self.get_group_size(visual_group)

        return ThumbnailGroupResult(
            asin="new_image",
            cluster_id=cluster_id,
            visual_group=visual_group,
            group_size=group_size,
        )

    def predict_image_group_from_file(
        self,
        image_path: str | Path,
    ) -> ThumbnailGroupResult:
        """Predict the visual group for an image file."""

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            with Image.open(image_path) as image:
                return self.predict_image_group(image.copy())

        except Exception as error:
            if isinstance(
                error,
                (
                    FileNotFoundError,
                    ValueError,
                ),
            ):
                raise

            raise ValueError(f"Unable to process image: {image_path}") from error

    # ======================================================================
    # INFORMATION / DIAGNOSTICS
    # ======================================================================

    def get_model_information(self) -> dict:
        """Return information about Feature C inference artifacts."""

        return {
            "clip_model": "ViT-B/32",
            "embedding_dimension": 512,
            "normalization": "L2",
            "pca_model": self.pca_model_path.name,
            "kmeans_model": self.kmeans_model_path.name,
            "number_of_clusters": int(self._get_kmeans_cluster_count()),
        }

    def _get_kmeans_cluster_count(self) -> int:
        """Return number of clusters without loading CLIP."""

        self._load_clustering_models()

        return int(self._kmeans_model.n_clusters)