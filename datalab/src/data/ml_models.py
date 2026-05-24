from __future__ import annotations

import importlib
import math

import streamlit as st

from gui.session_helpers import get_active_dataframe
from sklearn.base import clone

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from data.model_registry import (
    CLASSIFICATION_MODEL_NAMES,
    MODEL_OPTIONS,
    REGRESSION_MODEL_NAMES,
)

try:
    xgboost_module = importlib.import_module("xgboost")
    XGBClassifier = xgboost_module.XGBClassifier
    XGBRegressor = xgboost_module.XGBRegressor
except ImportError:  # pragma: no cover - optional dependency fallback
    XGBClassifier = GradientBoostingClassifier
    XGBRegressor = GradientBoostingRegressor


_CLASSIFICATION_MODELS = CLASSIFICATION_MODEL_NAMES
_REGRESSION_MODELS = REGRESSION_MODEL_NAMES


def _get_current_model_params() -> dict[str, object]:
    algorithm = st.session_state.get("algorithm")
    if not algorithm:
        return {}

    prefix = f"{algorithm}_"
    return {
        key.removeprefix(prefix): value
        for key, value in st.session_state.items()
        if key.startswith(prefix)
    }


def _build_model(algorithm: str, params: dict[str, object]):
    unsupported_algorithms = {"ARIMA", "SARIMA", "Prophet"}
    if algorithm in unsupported_algorithms:
        raise NotImplementedError(
            f"{algorithm} is listed in the UI, but this project does not yet have a runtime implementation for it."
        )

    if algorithm not in MODEL_OPTIONS:
        raise KeyError(f"No model factory registered for {algorithm!r}.")

    model = clone(MODEL_OPTIONS[algorithm])
    cleaned_params = {}
    for key, value in params.items():
        if value == "True":
            cleaned_params[key] = True
        elif value == "False":
            cleaned_params[key] = False
        elif value in ("None", "none"):
            cleaned_params[key] = None
        elif value == "-1":
            cleaned_params[key] = -1
        else:
            cleaned_params[key] = value

    if cleaned_params:
        model.set_params(**cleaned_params)
    return model


def _pick_target_column(dataframe, algorithm: str) -> str:
    target_column = st.session_state.get("target_column")
    if target_column in dataframe.columns:
        return target_column

    numeric_columns = list(dataframe.select_dtypes(include="number").columns)
    if algorithm in _REGRESSION_MODELS and numeric_columns:
        return numeric_columns[-1]

    return dataframe.columns[-1]


def _build_preprocessor(features):
    numeric_features = list(features.select_dtypes(include="number").columns)
    categorical_features = [
        column for column in features.columns if column not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers)


def _is_classification_algorithm(algorithm: str) -> bool:
    return algorithm in _CLASSIFICATION_MODELS


def run_model():
    dataframe = get_active_dataframe()
    if dataframe is None:
        st.error("Load a dataset first.")
        return None

    algorithm = st.session_state.get("algorithm")
    if not algorithm:
        st.error("Choose a model first.")
        return None

    try:
        target_column = _pick_target_column(dataframe, algorithm)
        features = dataframe.drop(columns=[target_column]).copy()
        target = dataframe[target_column].copy()

        if features.empty:
            st.error(
                "The selected dataframe does not have any feature columns after removing the target.")
            return None

        model_params = _get_current_model_params()
        model = _build_model(algorithm, model_params)

        is_classification = _is_classification_algorithm(algorithm)
        if is_classification:
            label_encoder = LabelEncoder()
            target_values = label_encoder.fit_transform(target.astype(str))
        else:
            try:
                target_values = target.astype(float)
            except ValueError:
                st.error(
                    f"{algorithm} expects a numeric target column, but {target_column!r} is not numeric."
                )
                return None

        stratify = target_values if is_classification and len(
            set(target_values)) > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target_values,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        preprocessor = _build_preprocessor(x_train)
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        if is_classification:
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions,
                          average="weighted", zero_division=0)
            st.success(f"Trained {algorithm} on {target_column!r}.")
            st.write({"accuracy": accuracy, "f1_weighted": f1})
            st.session_state["model_metrics"] = {
                "accuracy": accuracy, "f1_weighted": f1}
        else:
            rmse = math.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            st.success(f"Trained {algorithm} on {target_column!r}.")
            st.write({"rmse": rmse, "r2": r2})
            st.session_state["model_metrics"] = {"rmse": rmse, "r2": r2}

        st.session_state["trained_model"] = pipeline
        st.session_state["target_column"] = target_column
        st.session_state["model_predictions"] = predictions.tolist()
        return pipeline

    except NotImplementedError as exc:
        st.error(str(exc))
        return None
    except Exception as exc:
        st.error(f"Could not run the selected model: {exc}")
        return None
