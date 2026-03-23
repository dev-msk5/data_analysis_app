from __future__ import annotations
import datalab.data.data_loader as dl
import datalab.data.pipeline as pl
import pandas as pd

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
    def __init__(self):
        self.original_df: pd.DataFrame | None = None
        self.current_df: pd.DataFrame | None = None
        self.pipeline: pl.Pipeline = pl.Pipeline()

    def load(self, file_path: str):
        loader = dl.DataLoader()
        df = loader.load_data(file_path)

        self.original_df = df.copy()
        self.current_df = df

    def get_data(self) -> pd.DataFrame:
        if self.current_df is None:
            raise ValueError("No dataset found")
        return self.current_df

    def info(self) -> dict:
        df = self.get_data()
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
        }

    def preview(self, n: int = 5) -> pd.DataFrame:
        return self.get_data().head(n)

    def apply_pipeline(self):
        if self.original_df is None:
            raise ValueError("No dataset is loaded")

        self.current_df = self.pipeline.run(self.original_df.copy())

    def reset(self):
        if self.original_df is None:
            return
        self.current_df = self.original_df.copy()

    def add_step(self, name: str, func, **params):
        self.pipeline.add_step(name, func, **params)
