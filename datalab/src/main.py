from pathlib import Path

import streamlit as st

from core.dataset_controller import DatasetController
from data.data_loader import DataLoader
from gui.session_helpers import get_source_name, get_controller
from gui.pages.main_window import run_app as home_page
from gui.pages.ML_Lab import ml_page
from gui.pages.data_visualization import data_visualization_page
from gui.pages.data_cleaning import data_cleaning_page


SAMPLE_DATASET = Path(__file__).resolve().parent / "testy.csv"


def _get_controller():
    return get_controller()


def _load_uploaded_dataset(uploaded_file):
    loader = DataLoader()
    return loader.load_data(uploaded_file)


def _load_sample_dataset(controller: DatasetController) -> None:
    controller.load(SAMPLE_DATASET)
    st.session_state.dataset_label = SAMPLE_DATASET.name


def _render_global_dataset_sidebar(controller: DatasetController) -> None:
    with st.sidebar:
        st.header("Dataset")

        st.subheader("Current dataset")
        source_name = get_source_name(default_name="No dataset loaded")
        st.caption(source_name)

        uploaded_file = st.file_uploader(
            "Load or replace the dataset",
            type=["csv", "xlsx", "xls", "json"],
        )

        if uploaded_file is not None:
            try:
                dataframe = _load_uploaded_dataset(uploaded_file)
                controller.load_dataframe(dataframe, uploaded_file.name)
                st.session_state.dataset_label = uploaded_file.name
                st.success(f"Loaded {uploaded_file.name}")
            except Exception as exc:
                st.error(f"Could not load {uploaded_file.name}: {exc}")

        if st.button("Reload sample dataset"):
            _load_sample_dataset(controller)
            st.success(f"Loaded {SAMPLE_DATASET.name}")

        if st.button("Reset current dataset"):
            controller.reset()


def main():
    st.set_page_config(page_title="DataLab", page_icon="📊", layout="wide")

    controller = _get_controller()
    if controller.original_df is None and SAMPLE_DATASET.exists():
        _load_sample_dataset(controller)

    _render_global_dataset_sidebar(controller)

    navigation = st.navigation(
        [
            st.Page(home_page, title="Home", icon="🏠", default=True),
            st.Page(data_visualization_page,
                    title="Data Visualization", icon="📈"),
            st.Page(data_cleaning_page, title="Data Cleaning", icon="🧹"),
            st.Page(ml_page, title="ML Lab", icon="🤖"),
        ]
    )
    navigation.run()


if __name__ == "__main__":
    main()
