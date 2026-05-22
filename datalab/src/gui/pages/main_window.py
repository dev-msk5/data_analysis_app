"""Streamlit entrypoint for the DataLab home page."""

import streamlit as st

from core.dataset_controller import DatasetController


def _get_controller() -> DatasetController:
    if "dataset_controller" not in st.session_state:
        st.session_state.dataset_controller = DatasetController()
    return st.session_state.dataset_controller


def _render_dataset_summary(controller: DatasetController) -> None:
    dataframe = controller.get_data()
    summary = controller.info()

    source_name = controller.source_name or st.session_state.get(
        "dataset_label", "Unknown dataset")
    total_missing = int(sum(summary["missing"].values()))

    st.subheader("Current dataset")
    st.caption(source_name)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{summary['shape'][0]:,}")
    metric_columns[1].metric("Columns", f"{summary['shape'][1]:,}")
    metric_columns[2].metric("Missing values", f"{total_missing:,}")
    metric_columns[3].metric("Named fields", f"{len(summary['columns']):,}")

    st.markdown("#### Preview")
    st.dataframe(controller.preview(8), use_container_width=True)

    with st.expander("Dataset details", expanded=False):
        st.write("Columns")
        st.write(summary["columns"])
        st.write("Missing values by column")
        st.json(summary["missing"])
        st.write("Data types")
        st.json(summary["dtypes"])


def run_app() -> None:
    st.title("DataLab")
    st.write(
        "A lightweight analytics workspace for inspecting the active dataset. "
        "Use the sidebar to load, replace, or reset data from any page."
    )

    controller = _get_controller()

    if controller.original_df is None:
        st.info("Load a dataset to see the summary and preview.")
        return

    _render_dataset_summary(controller)


if __name__ == "__main__":
    run_app()
