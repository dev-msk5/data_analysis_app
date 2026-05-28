from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

from gui.session_helpers import get_active_dataframe, get_controller

nan_handling_dict = {
    "Replace with mean": "",
    "Replace with median": "",
    "Interpolate": "",
    "Fill with constant": "",
    "Remove columns": "",
    "Remove rows": "",
    "Remove rows by condition": "",
}
row_operation_dict = {
    "Filter by condition": "",
    "Remove by index": "",
    "Keep first N rows": "",
    "Random sample": "",
    "Manual selection": "",
}
duplicate_handling_dict = {
    "Remove duplicates": "",
    "Mark duplicates": "",
    "Keep first occurrence": "",
    "Keep last occurrence": "",
}
change_dtype_dict = {
    "To numeric": "",
    "To categorical": "",
    "To datetime": "",
    "To string": "",
    "Manual mapping": "",
}
outlier_handling_dict = {
    "Remove": "",
    "Cap (winsorize)": "",
    "Replace with min/max": "",
    "IQR filter": "",
    "Z-score clamp": "",
}
scaling_dict = {
    "None": "",
    "Standard (z-score)": "",
    "Min-Max (0-1)": "",
    "Robust (median/IQR)": "",
    "MaxAbs": "",
}
feature_eng_dict = {
    "One-hot encoding": "",
    "Create interaction terms": "",
    "Polynomial features": "",
    "Bin numeric features": "",
    "Extract date parts": "",
    "Drop correlated features": "",
}


def _coerce_value(value: str):
    text = value.strip()
    if text == "":
        return ""
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_manual_mapping(text: str) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            old_value, new_value = line.split(":", 1)
        elif "->" in line:
            old_value, new_value = line.split("->", 1)
        else:
            continue
        mapping[old_value.strip()] = _coerce_value(new_value)
    return mapping


def _selected_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _selected_rows(df: pd.DataFrame, start_index: int | None, end_index: int | None) -> list:
    if start_index is None or end_index is None or df.empty:
        return []
    start = max(0, start_index)
    end = min(end_index, len(df) - 1)
    if start > end:
        return []
    return df.iloc[start: end + 1].index.tolist()


def _apply_nan_handling(
    df: pd.DataFrame,
    operation_name: str,
    *,
    columns: list[str] | None = None,
    selected_rows: list | None = None,
    fill_constant_value: str = "",
    row_condition: str = "",
) -> pd.DataFrame:
    df = df.copy()
    columns = _selected_columns(df, columns or list(df.columns))
    selected_rows = selected_rows or []

    if operation_name == "Remove columns":
        return df.drop(columns=columns, errors="ignore")

    if operation_name in {"Replace with mean", "Replace with median", "Interpolate"}:
        numeric_columns = [
            column for column in columns if pd.api.types.is_numeric_dtype(df[column])]
        if operation_name == "Replace with mean":
            for column in numeric_columns:
                df[column] = df[column].fillna(df[column].mean())
        elif operation_name == "Replace with median":
            for column in numeric_columns:
                df[column] = df[column].fillna(df[column].median())
        elif operation_name == "Interpolate" and numeric_columns:
            df[numeric_columns] = df[numeric_columns].interpolate(
                limit_direction="both")
        return df

    if operation_name == "Fill with constant":
        fill_value = _coerce_value(fill_constant_value)
        if selected_rows:
            df.loc[selected_rows, columns] = df.loc[selected_rows,
                                                    columns].fillna(fill_value)
        else:
            df[columns] = df[columns].fillna(fill_value)
        return df

    if operation_name == "Remove rows":
        return df.drop(index=selected_rows) if selected_rows else df.dropna()

    if operation_name == "Remove rows by condition" and row_condition.strip():
        try:
            return df.drop(index=df.query(row_condition).index)
        except Exception:
            st.error("Invalid row condition. Use a valid pandas query.")
            return df

    return df


def _apply_dtype_handling(
    df: pd.DataFrame,
    operation_name: str,
    *,
    columns: list[str],
    manual_mapping_text: str = "",
) -> pd.DataFrame:
    df = df.copy()
    columns = _selected_columns(df, columns)

    if operation_name == "To numeric":
        for column in columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    elif operation_name == "To categorical":
        for column in columns:
            df[column] = df[column].astype("category")
    elif operation_name == "To datetime":
        for column in columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    elif operation_name == "To string":
        for column in columns:
            df[column] = df[column].astype("string")
    elif operation_name == "Manual mapping":
        mapping = _parse_manual_mapping(manual_mapping_text)
        if mapping:
            for column in columns:
                df[column] = df[column].replace(mapping)

    return df


def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _apply_outlier_handling(
    df: pd.DataFrame,
    operation_name: str,
    *,
    columns: list[str],
) -> pd.DataFrame:
    df = df.copy()
    columns = _selected_columns(df, columns)
    numeric_columns = [
        column for column in columns if pd.api.types.is_numeric_dtype(df[column])]

    if operation_name == "Z-score clamp":
        for column in numeric_columns:
            std_value = df[column].std()
            if std_value and not pd.isna(std_value):
                mean_value = df[column].mean()
                df[column] = df[column].clip(
                    lower=mean_value - 3 * std_value, upper=mean_value + 3 * std_value)
        return df

    if operation_name in {"Remove", "IQR filter"}:
        for column in numeric_columns:
            lower, upper = _iqr_bounds(df[column])
            if operation_name == "Remove":
                df = df[(df[column].isna()) | (
                    (df[column] >= lower) & (df[column] <= upper))]
            else:
                df[column] = df[column].clip(lower=lower, upper=upper)
        return df

    if operation_name == "Cap (winsorize)":
        for column in numeric_columns:
            lower, upper = _iqr_bounds(df[column])
            df[column] = df[column].clip(lower=lower, upper=upper)
        return df

    if operation_name == "Replace with min/max":
        for column in numeric_columns:
            lower, upper = _iqr_bounds(df[column])
            min_value = df[column].min()
            max_value = df[column].max()
            df[column] = df[column].mask(df[column] < lower, min_value)
            df[column] = df[column].mask(df[column] > upper, max_value)
        return df

    return df


def _apply_scaling(
    df: pd.DataFrame,
    operation_name: str,
    *,
    columns: list[str],
) -> pd.DataFrame:
    df = df.copy()
    columns = _selected_columns(df, columns)
    numeric_columns = [
        column for column in columns if pd.api.types.is_numeric_dtype(df[column])]

    if operation_name == "Standard (z-score)":
        for column in numeric_columns:
            std_value = df[column].std()
            if std_value and not pd.isna(std_value):
                df[column] = (df[column] - df[column].mean()) / std_value
    elif operation_name == "Min-Max (0-1)":
        for column in numeric_columns:
            min_value = df[column].min()
            max_value = df[column].max()
            if max_value != min_value and not pd.isna(max_value) and not pd.isna(min_value):
                df[column] = (df[column] - min_value) / (max_value - min_value)
    elif operation_name == "Robust (median/IQR)":
        for column in numeric_columns:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            if iqr and not pd.isna(iqr):
                df[column] = (df[column] - df[column].median()) / iqr
    elif operation_name == "MaxAbs":
        for column in numeric_columns:
            max_abs = df[column].abs().max()
            if max_abs and not pd.isna(max_abs):
                df[column] = df[column] / max_abs

    return df


def _apply_features(
    df: pd.DataFrame,
    operation_name: str,
    *,
    columns: list[str],
) -> pd.DataFrame:
    df = df.copy()
    columns = _selected_columns(df, columns)
    numeric_columns = [
        column for column in columns if pd.api.types.is_numeric_dtype(df[column])]

    if operation_name == "One-hot encoding":
        categorical_columns = [
            column for column in columns if not pd.api.types.is_numeric_dtype(df[column])]
        if categorical_columns:
            df = pd.get_dummies(
                df, columns=categorical_columns, dummy_na=False)
    elif operation_name == "Create interaction terms" and len(numeric_columns) >= 2:
        for left_column, right_column in combinations(numeric_columns, 2):
            df[f"{left_column}_x_{right_column}"] = df[left_column] * \
                df[right_column]
    elif operation_name == "Polynomial features":
        for column in numeric_columns:
            df[f"{column}_squared"] = df[column] ** 2
            df[f"{column}_cubed"] = df[column] ** 3
    elif operation_name == "Bin numeric features":
        for column in numeric_columns:
            try:
                df[f"{column}_binned"] = pd.cut(
                    df[column], bins=5, labels=False, duplicates="drop")
            except Exception:
                continue
    elif operation_name == "Extract date parts":
        for column in columns:
            converted = pd.to_datetime(df[column], errors="coerce")
            if converted.notna().any():
                df[f"{column}_year"] = converted.dt.year
                df[f"{column}_month"] = converted.dt.month
                df[f"{column}_day"] = converted.dt.day
    elif operation_name == "Drop correlated features" and len(numeric_columns) >= 2:
        correlation_matrix = df[numeric_columns].corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        correlated_columns = [column for column in upper_triangle.columns if any(
            upper_triangle[column] > 0.9)]
        df = df.drop(columns=correlated_columns, errors="ignore")

    return df


def _apply_row_operation(
    df: pd.DataFrame,
    operation_name: str,
    *,
    selected_rows: list | None,
    row_condition: str = "",
    n_rows: int = 100,
    sample_fraction: float = 0.1,
) -> pd.DataFrame:
    df = df.copy()
    selected_rows = selected_rows or []

    if operation_name == "Filter by condition" and row_condition.strip():
        try:
            return df.query(row_condition)
        except Exception:
            st.error("Invalid row condition. Use a valid pandas query.")
            return df
    if operation_name == "Remove by index" and selected_rows:
        return df.drop(index=selected_rows)
    if operation_name == "Keep first N rows":
        return df.head(n_rows)
    if operation_name == "Random sample" and 0 < sample_fraction <= 1:
        return df.sample(frac=sample_fraction, random_state=42)
    if operation_name == "Manual selection" and selected_rows:
        return df.loc[selected_rows]

    return df


def _apply_duplicates(df: pd.DataFrame, operation_name: str) -> pd.DataFrame:
    df = df.copy()
    if operation_name == "Remove duplicates":
        return df.drop_duplicates()
    if operation_name == "Mark duplicates":
        df["_is_duplicate"] = df.duplicated(keep=False)
        return df
    if operation_name == "Keep first occurrence":
        return df.drop_duplicates(keep="first")
    if operation_name == "Keep last occurrence":
        return df.drop_duplicates(keep="last")
    return df


def data_cleaning_page() -> None:

    df = get_active_dataframe()

    st.title("Data Cleaning")
    st.write("Data Cleaning")

    if df is None:
        st.info("No data loaded.")
        return

    menu_cols = st.columns([2, 3])

    with menu_cols[0].container(border=True):
        st.write("Data Handling")
        rows_or_columns = st.selectbox(
            "Do you want to clean rows or columns?",
            ["Rows", "Columns"],
        )

        if rows_or_columns == "Columns":
            columns_to_change = st.multiselect(
                "Which columns to change?",
                [col for col in df.columns],
            )

            nan_handling = st.selectbox(
                "NaN handling",
                list(nan_handling_dict.keys())[:5],
            )

            dtype_change = st.selectbox(
                "Change the datatype", list(change_dtype_dict.keys()))

            out_handling = st.selectbox(
                "Outlier handling", list(outlier_handling_dict.keys()))

            scaling_handling = st.selectbox(
                "Scaling", list(scaling_dict.keys()))

            feature_handling = st.selectbox(
                "Feature engineering", list(feature_eng_dict.keys()))

            fill_constant_value = "0"
            if nan_handling == "Fill with constant":
                fill_constant_value = st.text_input(
                    "Constant value",
                    value="0",
                    key="column_constant_value",
                )

            manual_mapping_text = ""
            if dtype_change == "Manual mapping":
                manual_mapping_text = st.text_area(
                    "Manual mapping (old:new per line)",
                    key="column_manual_mapping",
                )

        else:  # Rows

            rows_to_change = st.text_input(
                "Which rows to change? (start:end)")

            change_start_index = None
            change_end_index = None
            max_index_length = len(df)

            user_input = (rows_to_change or "").strip()
            if user_input:
                try:
                    nums = list(map(lambda x: int(x), user_input.split(":")))
                    if len(nums) == 1:
                        change_start_index = change_end_index = nums[0]
                    elif len(nums) == 2:
                        change_start_index, change_end_index = nums
                    else:
                        st.error("Invalid format")

                    if change_start_index > change_end_index:
                        st.error("Start index must be <= end index.")
                        change_start_index = None
                    elif change_end_index > max_index_length:
                        change_end_index = max_index_length
                        if change_start_index > max_index_length:
                            change_start_index = max_index_length
                        st.warning(
                            f"Selection clamped to valid range: {change_start_index}..{change_end_index}, inputted end index was bigger than data length")
                    elif 0 > change_end_index or 0 > change_start_index:
                        st.error("Indexes must be positive")
                        change_start_index = None
                except Exception:
                    st.error(
                        "Invalid row selection. Use a number or 'start:end'.")

            selected_rows = _selected_rows(
                df, change_start_index, change_end_index)

            if change_start_index is not None and change_end_index is not None:
                st.write(
                    f"Selected rows: {change_start_index} to {change_end_index}")

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
                "Row operation", list(row_operation_dict.keys()))
            row_condition = st.text_input(
                "Row condition (pandas query)",
                key="row_condition",
            )
            fill_constant_value = "0"
            if nan_handling == "Fill with constant":
                fill_constant_value = st.text_input(
                    "Constant value",
                    value="0",
                    key="row_constant_value",
                )

            if row_operation == "Keep first N rows":
                n_rows = st.number_input("N", min_value=1, value=100, step=1)
            else:
                n_rows = 100

            if row_operation == "Random sample":
                sample_fraction = st.slider(
                    "Sample fraction", min_value=0.01,
                    max_value=1.0, value=0.1, step=0.01)
            else:
                sample_fraction = 0.1

            duplicate_handling = st.selectbox(
                "Duplicates handling", list(duplicate_handling_dict.keys()))

        transform = st.button("Transform data")

        transform_steps = []
        if rows_or_columns == "Columns":
            transform_steps = [
                {
                    "name": f"NaN handling: {nan_handling}",
                    "handler": _apply_nan_handling,
                    "kwargs": {
                        "columns": columns_to_change,
                        "fill_constant_value": fill_constant_value,
                    },
                },
                {
                    "name": f"Data type: {dtype_change}",
                    "handler": _apply_dtype_handling,
                    "kwargs": {
                        "columns": columns_to_change,
                        "manual_mapping_text": manual_mapping_text,
                    },
                },
                {
                    "name": f"Outlier handling: {out_handling}",
                    "handler": _apply_outlier_handling,
                    "kwargs": {"columns": columns_to_change},
                },
                {
                    "name": f"Scaling: {scaling_handling}",
                    "handler": _apply_scaling,
                    "kwargs": {"columns": columns_to_change},
                },
                {
                    "name": f"Feature engineering: {feature_handling}",
                    "handler": _apply_features,
                    "kwargs": {"columns": columns_to_change},
                },
            ]
        else:
            transform_steps = [
                {
                    "name": f"NaN handling: {nan_handling}",
                    "handler": _apply_nan_handling,
                    "kwargs": {
                        "selected_rows": selected_rows,
                        "row_condition": row_condition,
                        "fill_constant_value": fill_constant_value,
                    },
                },
                {
                    "name": f"Row operation: {row_operation}",
                    "handler": _apply_row_operation,
                    "kwargs": {
                        "selected_rows": selected_rows,
                        "row_condition": row_condition,
                        "n_rows": int(n_rows),
                        "sample_fraction": float(sample_fraction),
                    },
                },
                {
                    "name": f"Duplicates: {duplicate_handling}",
                    "handler": _apply_duplicates,
                    "kwargs": {},
                },
            ]

        if transform:
            controller = get_controller()
            updated_df = df.copy()
            for step in transform_steps:
                updated_df = step["handler"](
                    updated_df, step["name"].split(": ", 1)[-1], **step["kwargs"])

            controller.current_df = updated_df.copy()
            st.success(f"Data transformed. New shape: {updated_df.shape}")
            st.rerun()

        st.button("Export your data")

    with menu_cols[1].container(border=True):
        st.write("col1")
