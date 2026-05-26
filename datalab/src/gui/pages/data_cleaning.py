import streamlit as st


def data_cleaning_page() -> None:
    st.title("Data Cleaning")
    st.write("Data Cleaning")

    menu_cols = st.columns([2, 3])

    with menu_cols[0].container(border=True):
        st.write("Data Handling")

        nan_handling = st.selectbox("NaN handling", [
            "Remove rows",
            "Remove columns",
            "Replace with mean",
            "Replace with median",
            "Interpolate",
        ])
        if nan_handling == "Remove rows":
            st.multiselect("Which row to remove", ["1st", "2nd", "3rd"])
        out_handling = st.selectbox("Outlier handling", [
            "Remove",
            "Cap (winsorize)",
            "Replace with min/max",
            "IQR filter",
            "Z-score clamp",
        ])
        dtype_change = st.selectbox("Change the datatype", [
            "To numeric",
            "To categorical",
            "To datetime",
            "To string",
            "Manual mapping",
        ])
        scaling_handling = st.selectbox("Scaling", [
            "None",
            "Standard (z-score)",
            "Min-Max (0-1)",
            "Robust (median/IQR)",
            "MaxAbs",
        ])
        feature_handling = st.selectbox("Feature engineering", [
            "One-hot encoding",
            "Create interaction terms",
            "Polynomial features",
            "Bin numeric features",
            "Extract date parts",
            "Drop correlated features",
        ])
        duplicate_handling = st.selectbox("Duplicates handling", [
            "Remove duplicates",
            "Mark duplicates",
            "Keep first occurrence",
            "Keep last occurrence",
        ])

        st.button("Transform data")
        st.button("Export your data")

    with menu_cols[1].container(border=True):
        st.write("col1")
