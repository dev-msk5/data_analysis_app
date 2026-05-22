"""Data cleaning helpers.

This module provides small, pure helpers that operate on pandas
DataFrame objects and return modified copies. Functions are intentionally
lightweight and do not mutate the input DataFrame.
"""

import pandas as pd


def fill_missing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy of ``df`` with missing values in ``columns`` filled.

    For each column name in ``columns`` the function fills NaNs with the
    column mean. The input DataFrame is not modified; a copy is
    returned.

    Parameters
    - df (pandas.DataFrame): Input dataframe.
    - columns (list[str]): List of column names whose missing values will
      be replaced with the respective column mean.

    Returns
    - pandas.DataFrame: A new DataFrame with filled values.
    """
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())
    return df


def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Min-max normalize ``column`` and return a new DataFrame.

    The values in ``column`` are scaled to the range [0, 1] using
    ``(x - min) / (max - min)``. If the column has constant values
    (``max == min``) the function currently returns a short string
    describing the condition rather than raising an exception.

    Parameters
    - df (pandas.DataFrame): Input dataframe.
    - column (str): Column name to normalize.

    Returns
    - pandas.DataFrame | str: A new DataFrame with the normalized column,
      or an explanatory string if the column cannot be normalized.
    """
    df = df.copy()
    col = df[column]
    try:
        df[column] = (col - col.min()) / (col.max() - col.min())
    except ZeroDivisionError:
        return "Col.max-col.min is zero"
    return df


def standardize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Standardize ``column`` to zero mean and unit variance.

    The transformation applied is ``(x - mean) / std``. If the column
    standard deviation is zero the function returns an explanatory string.

    Parameters
    - df (pandas.DataFrame): Input dataframe.
    - column (str): Column name to standardize.

    Returns
    - pandas.DataFrame | str: A new DataFrame with the standardized column,
      or an explanatory string when the operation cannot be performed.
    """
    df = df.copy()
    col = df[column]
    try:
        df[column] = (col - col.mean()) / (col.std())
    except ZeroDivisionError:
        return "Column std is zero"
    return df


def filter_rows(df: pd.DataFrame, condition) -> pd.DataFrame:
    """Return rows of ``df`` that match ``condition``.

    ``condition`` is typically a boolean mask or expression suitable for
    indexing a pandas DataFrame (for example ``df['col'] > 0`` or a
    boolean Series). The function returns the filtered view as a new
    DataFrame.

    Parameters
    - df (pandas.DataFrame): Input dataframe.
    - condition: Boolean mask or callable expression used to filter rows.

    Returns
    - pandas.DataFrame: Filtered DataFrame.
    """
    return df[condition]


def select_rows(df: pd.DataFrame, colums: list[str]) -> pd.DataFrame:
    """Select and return the specified columns from ``df``.

    Note: the parameter name matches the existing implementation
    (`colums`). Do not rename unless you intend to update all call sites.

    Parameters
    - df (pandas.DataFrame): Input dataframe.
    - colums (list[str]): List of column names to select.

    Returns
    - pandas.DataFrame: DataFrame containing only the requested columns.
    """
    return df[colums]


def add_column(df: pd.DataFrame, column_name: str, data) -> pd.DataFrame:
    """Return a new DataFrame with ``data`` added as ``column_name``.

    Parameters
    - df (pandas.DataFrame): Input dataframe.
    - column_name (str): Name of the new column to add.
    - data: Iterable or scalar to assign to the new column.

    Returns
    - pandas.DataFrame: Copy of ``df`` with the new column added.
    """
    df = df.copy()
    df[column_name] = data
    return df


def apply_function(df: pd.DataFrame, column: str, func) -> pd.DataFrame:
    """Apply ``func`` to values in ``column`` and return a new DataFrame.

    The function is applied element-wise using ``Series.apply``.

    Parameters
    - df (pandas.DataFrame): Input dataframe.
    - column (str): Column name on which to apply ``func``.
    - func (callable): Function to apply to each element of the column.

    Returns
    - pandas.DataFrame: Copy of ``df`` with the transformed column.
    """
    df = df.copy()
    df[column] = df[column].apply(func)
    return df
