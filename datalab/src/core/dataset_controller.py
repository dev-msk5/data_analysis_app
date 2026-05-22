from __future__ import annotations

import pandas as pd

from data.data_loader import DataLoader
from data.pipeline import Pipeline

# example
# from datalab.core.dataset_controller import DatasetController
# from datalab.data.data_cleaner import fill_missing, normalize_column

# controller = DatasetController()

# controller.load("data.csv")

# controller.add_step("fill_missing", fill_missing, value=0)
# controller.add_step("normalize_price", normalize_column, column="price")

# controller.apply_pipeline()

# print(controller.preview())


class DatasetController:
    """Manage the active dataset, preview state, and reusable pipelines.

    The controller keeps both the original dataset and the current view
    so the UI can load a file, apply transformations, and reset back to
    the source data without losing reproducibility.
    """

    def __init__(self):
        """Create an empty controller with no loaded dataset.

        Returns
        - None: The controller is initialized in memory.
        """
        self.original_df: pd.DataFrame | None = None
        self.current_df: pd.DataFrame | None = None
        self.source_name: str | None = None
        self.pipeline: Pipeline = Pipeline()

    def load(self, file_path: str):
        """Load a dataset from disk and make it the active dataset.

        Parameters
        - file_path (str): Path to the dataset file.

        Returns
        - None: The loaded DataFrame is stored on the controller.

        Raises
        - FileNotFoundError: Raised when ``file_path`` does not exist.
        - ValueError: Raised when the file type is not supported by the
          underlying loader.
        - ImportError: Raised when an optional Excel dependency is missing.
        """
        loader = DataLoader()
        df = loader.load_data(file_path)

        self.original_df = df.copy()
        self.current_df = df
        self.source_name = str(file_path)

    def load_dataframe(self, df: pd.DataFrame, source_name: str | None = None):
        """Load an in-memory DataFrame as the active dataset.

        Parameters
        - df (pandas.DataFrame): DataFrame to store as the active dataset.
        - source_name (str | None): Optional display name for the data
          source shown in the UI.

        Returns
        - None: The controller stores a copy of ``df`` internally.
        """
        self.original_df = df.copy()
        self.current_df = df.copy()
        self.source_name = source_name

    def get_data(self) -> pd.DataFrame:
        """Return the current active dataset.

        Returns
        - pandas.DataFrame: The current dataset view.

        Raises
        - ValueError: Raised when no dataset has been loaded.
        """
        if self.current_df is None:
            raise ValueError("No dataset found")
        return self.current_df

    def info(self) -> dict:
        """Return a compact summary of the active dataset.

        The summary includes the shape, column names, data types, and
        missing-value counts for each column.

        Returns
        - dict: A dictionary containing dataset metadata.
        """
        df = self.get_data()
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
        }

    def preview(self, n: int = 5) -> pd.DataFrame:
        """Return the first ``n`` rows of the active dataset.

        Parameters
        - n (int): Number of rows to return. Defaults to 5.

        Returns
        - pandas.DataFrame: A preview slice of the current dataset.
        """
        return self.get_data().head(n)

    def apply_pipeline(self):
        """Apply the configured transformation pipeline to the original data.

        Returns
        - None: The transformed DataFrame replaces the current dataset.

        Raises
        - ValueError: Raised when no original dataset has been loaded.
        - RuntimeError: Raised when a pipeline step fails.
        """
        if self.original_df is None:
            raise ValueError("No dataset is loaded")

        self.current_df = self.pipeline.run(self.original_df.copy())

    def reset(self):
        """Restore the current dataset back to the original loaded data.

        Returns
        - None: The active dataset is replaced in memory.
        """
        if self.original_df is None:
            return
        self.current_df = self.original_df.copy()

    def add_step(self, name: str, func, **params):
        """Add a new transformation step to the pipeline.

        Parameters
        - name (str): Human-readable step name.
        - func (callable): Transformation function applied to the data.
        - **params: Keyword arguments forwarded to ``func`` when the step
          is executed.

        Returns
        - None: The step is registered on the internal pipeline.
        """
        self.pipeline.add_step(name, func, **params)
