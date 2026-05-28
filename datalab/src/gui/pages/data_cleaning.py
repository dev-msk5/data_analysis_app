import streamlit as st


def data_cleaning_page() -> None:
    st.title("Data Cleaning")
    st.write("Data Cleaning")

    menu_cols = st.columns([2, 3])

    with menu_cols[0].container(border=True):
        st.write("Data Handling")
        rows_or_columns = st.selectbox(
            "Do you want to clean rows or columns?",
            ["Rows", "Columns"],
        )

        duplicate_handling = st.selectbox(
            "Duplicates handling",
            [
                "Remove duplicates",
                "Mark duplicates",
                "Keep first occurrence",
                "Keep last occurrence",
            ],
        )

        if rows_or_columns == "Columns":
            nan_handling = st.selectbox(
                "NaN handling",
                [
                    "Replace with mean",
                    "Replace with median",
                    "Interpolate",
                    "Fill with constant",
                    "Remove columns",
                ],
            )

            dtype_change = st.selectbox("Change the datatype", [
                "To numeric",
                "To categorical",
                "To datetime",
                "To string",
                "Manual mapping",
            ])

            out_handling = st.selectbox("Outlier handling", [
                "Remove",
                "Cap (winsorize)",
                "Replace with min/max",
                "IQR filter",
                "Z-score clamp",
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

        else:  # Rows
            nan_handling = st.selectbox(
                "NaN handling",
                [
                    "Remove rows",
                    "Interpolate",
                    "Fill with constant",
                    "Remove rows by condition",
                ],
            )

            row_operation = st.selectbox(
                "Row operation",
                [
                    "Filter by condition",
                    "Remove by index",
                    "Keep first N rows",
                    "Random sample",
                    "Manual selection",
                ],
            )

            if row_operation == "Keep first N rows":
                st.number_input("N", min_value=1, value=100, step=1)

            if row_operation == "Random sample":
                st.slider("Sample fraction", min_value=0.01,
                          max_value=1.0, value=0.1, step=0.01)

        st.button("Transform data")
        st.button("Export your data")

    with menu_cols[1].container(border=True):
        st.write("col1")
