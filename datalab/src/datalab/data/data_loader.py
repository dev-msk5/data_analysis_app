"""Data loading utilities.

This module provides a small `DataLoader` helper that can read common
tabular formats (CSV, Excel, JSON) into a pandas.DataFrame and a couple
of convenience helpers for previewing and summarizing datasets.

Note: `file_path` parameters accept either a string path or a
`pathlib.Path`-like object.
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """Simple dataset loader and lightweight helpers.

    This class contains convenience methods for loading data from disk
    into pandas.DataFrame objects and for getting quick previews and
    summary information about a loaded DataFrame.
    """

    def __init__(self):
        """Create a DataLoader instance.

        The class does not hold state; instances are provided for
        discoverability and potential future extension.
        """
        pass

    def load_data(self, file_path: str | Path, np: bool = False) -> pd.DataFrame:
        """Load a dataset from `file_path` into a DataFrame.

        Supported formats are inferred from the file suffix:
        - `.csv` -> :func:`pandas.read_csv`
        - `.xls`, `.xlsx` -> :func:`pandas.read_excel`
        - `.json` -> :func:`pandas.read_json`

        Parameters
        - file_path (str | pathlib.Path): Path to the dataset file.
        - np (bool): Reserved for future use (kept for API compatibility).

        Returns
        - pandas.DataFrame: The loaded dataset.

        Raises
        - FileNotFoundError: If `file_path` does not exist.
        - ValueError: If the file suffix is not a supported/recognized type.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            print(f"Loading file: {file_path}")
            return pd.read_csv(file_path)
        elif suffix in (".xls", ".xlsx"):
            print(f"Loading file: {file_path}")
            return pd.read_excel(file_path)
        elif suffix == ".json":
            print(f"Loading file: {file_path}")
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def preview_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Return the first `n` rows of `df`.

        This is a convenience wrapper around :meth:`pandas.DataFrame.head`.

        Parameters
        - df (pandas.DataFrame): DataFrame to preview.
        - n (int): Number of rows to return. Defaults to 5.

        Returns
        - pandas.DataFrame: The first `n` rows of `df`.
        """
        return df.head(n)

    @staticmethod
    def dataset_info(df: pd.DataFrame) -> dict:
        """Return a small summary about `df`.

        The returned dictionary contains the shape, column index, and a
        mapping of missing value counts per column.

        Parameters
        - df (pandas.DataFrame): DataFrame to analyze.

        Returns
        - dict: Summary information with keys: ``shape``, ``columns``,
          and ``missing values``.
        """
        return {
            "shape": df.shape,
            "columns": df.columns,
            "missing values": df.isnull().sum().to_dict(),
        }
