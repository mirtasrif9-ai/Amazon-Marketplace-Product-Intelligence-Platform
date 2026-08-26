import logging
from pathlib import Path

import pandas as pd

from src.common.exceptions import CollectionError


logger = logging.getLogger(__name__)


class KeywordReader:
    """Read and validate search keywords from an Excel file."""

    def read(self, file_path: str | Path) -> list[str]:
        """Read search keywords from an Excel file."""

        path = Path(file_path)

        logger.info("Reading keyword file: %s", path)

        if not path.exists():
            logger.error("Keyword file does not exist: %s", path)
            raise CollectionError(f"Keyword file not found: {path}")

        try:
            dataframe = pd.read_excel(path)

        except Exception as exc:
            logger.exception("Failed to read keyword file: %s", path)
            raise CollectionError(
                f"Failed to read keyword file: {path}"
            ) from exc

        if "Search Keywords" not in dataframe.columns:
            logger.error("Required 'Search Keywords' column is missing")
            raise CollectionError(
                "Excel file must contain a 'Search Keywords' column."
            )

        keywords = (
            dataframe["Search Keywords"]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
        )

        keywords = keywords[keywords != ""].drop_duplicates().tolist()

        if not keywords:
            logger.error("No valid keywords found in: %s", path)
            raise CollectionError(
                "The keyword file contains no valid keywords."
            )

        logger.info(
            "Successfully loaded %d unique keywords",
            len(keywords),
        )

        return keywords