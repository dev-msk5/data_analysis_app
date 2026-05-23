"""Helpers for shared Streamlit session state access."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.dataset_controller import DatasetController


def get_controller() -> DatasetController:
    """Return the shared dataset controller from Streamlit session state."""
    if "dataset_controller" not in st.session_state:
        st.session_state.dataset_controller = DatasetController()
    return st.session_state.dataset_controller


def get_active_dataframe() -> pd.DataFrame | None:
    """Return the active DataFrame if available, otherwise None."""
    controller = get_controller()
    if controller.original_df is None:
        return None
    try:
        return controller.get_data()
    except Exception:
        return None


def get_source_name(default_name: str = "Unknown dataset") -> str:
    """Return a display-friendly name for the active dataset source."""
    controller = get_controller()
    return controller.source_name or st.session_state.get("dataset_label", default_name)
