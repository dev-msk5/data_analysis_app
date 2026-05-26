"""Data loading utilities.

This module provides `DataLoader`, a thin I/O helper whose single
responsibility is reading tabular files (CSV, Excel, JSON) into
``pandas.DataFrame`` objects. It intentionally exposes only a minimal
set of convenience helpers — primarily ``preview_data`` — so that
higher-level responsibilities (dataset metadata, summaries, and
transformation pipelines) remain owned by application-level code such
as :mod:`datalab.core.dataset_controller`.

Note: ``file_path`` parameters accept either a filesystem path (string
or ``pathlib.Path``) or a file-like object (for example, a Streamlit
upload). When a file-like object is provided it must expose a ``read``
attribute and may include a ``name`` attribute used to infer file
type.
"""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """Thin loader for tabular datasets.

    Responsibilities:
    - Detect file type from suffix or uploaded file metadata and load
      it into a ``pandas.DataFrame``.
    - Provide a tiny set of convenience helpers closely related to
      loading (for example, a quick preview). Application-facing
      dataset summaries belong to the controller layer and are not
      reproduced here to avoid duplication.
    """

    def __init__(self):
        """Create a DataLoader instance.

        The class does not hold state; instances are provided for
        discoverability and potential future extension.
        """
        pass

    def load_data(self, file_path: str | Path | object, np: bool = False) -> pd.DataFrame:
        """Load a dataset from ``file_path`` into a DataFrame.

        Supported formats are inferred from the file suffix:
        - ``.csv`` -> :func:`pandas.read_csv`
        - ``.xls``, ``.xlsx`` -> :func:`pandas.read_excel`
        - ``.json`` -> :func:`pandas.read_json`

                ``file_path`` may be a filesystem path (string or ``Path``) or a
                file-like object (for example, a Streamlit upload). File-like
                objects should expose a ``read`` attribute and may include a
                ``name`` attribute used to infer the file type.

                Parameters
                - file_path (str | pathlib.Path | file-like): Path or file-like
                    object to load. For uploads, the object should expose a
                    ``read`` attribute and may include ``name`` to infer suffix.
                - np (bool): Reserved for future use (kept for API compatibility).

                Returns
                - pandas.DataFrame: The loaded dataset.

                Raises
                - FileNotFoundError: If ``file_path`` does not exist (filesystem
                    path case).
                - ValueError: If the file suffix is not a supported/recognized
                    type.
                """
        if hasattr(file_path, "read"):
            suffix = Path(getattr(file_path, "name", "")).suffix.lower()
            if suffix == ".csv":
                logger.info("Loading file-like object: %s",
                            getattr(file_path, "name", "<uploaded>"))
                return pd.read_csv(file_path)
            if suffix in (".xls", ".xlsx"):
                logger.info("Loading file-like object: %s",
                            getattr(file_path, "name", "<uploaded>"))
                try:
                    engine = "xlrd" if suffix == ".xls" else "openpyxl"
                    return pd.read_excel(file_path, engine=engine)
                except ImportError as exc:
                    required = "xlrd" if suffix == ".xls" else "openpyxl"
                    raise ImportError(
                        f"Missing optional dependency '{required}'. Install it to read {suffix} files.") from exc
            if suffix == ".json":
                logger.info("Loading file-like object: %s",
                            getattr(file_path, "name", "<uploaded>"))
                return pd.read_json(file_path)
            raise ValueError(f"Unsupported file type: {suffix}")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            logger.info("Loading file: %s", file_path)
            return pd.read_csv(file_path)
        if suffix in (".xls", ".xlsx"):
            logger.info("Loading file: %s", file_path)
            try:
                if suffix == ".xls":
                    return pd.read_excel(file_path, engine="xlrd")
                return pd.read_excel(file_path, engine="openpyxl")
            except ImportError as exc:
                required = "xlrd" if suffix == ".xls" else "openpyxl"
                raise ImportError(
                    f"Missing optional dependency '{required}'. Install it to read {suffix} files.") from exc
        if suffix == ".json":
            logger.info("Loading file: %s", file_path)
            return pd.read_json(file_path)
        raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def preview_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Return the first ``n`` rows of ``df``.

        Convenience wrapper around :meth:`pandas.DataFrame.head` used by
        quick inspections in the UI.

        Parameters
        - df (pandas.DataFrame): DataFrame to preview.
        - n (int): Number of rows to return. Defaults to 5.

        Returns
        - pandas.DataFrame: The first ``n`` rows of ``df``.
        """
        return df.head(n)
