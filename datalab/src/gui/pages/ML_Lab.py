import streamlit as st

import data.ml_models as models

from data.model_registry import MODEL_OPTIONS, MODEL_PARAMETERS
from gui.session_helpers import get_active_dataframe

DEFAULT_FORM_SPECIFICS = {}

CHART_OPTIONS = [
    "Bar",
    "Line",
    "Area",
    "Scatter",
    "Histogram",
    "Box",
    "Violin",
    "Pie",
    "Heatmap",
]


def _get_dataset_shape_from_session() -> tuple[int, int] | None:
    """Read active dataset dimensions from shared session state.

    Returns
    - tuple[int, int] | None: (rows, columns) if a dataset exists, otherwise None.
    """
    dataframe = get_active_dataframe()
    if dataframe is None:
        return None

    return int(dataframe.shape[0]), int(dataframe.shape[1])


def _bucket_sample_size(n_rows: int) -> str:
    if n_rows < 10_000:
        return "Small (<10,000 rows)"
    if n_rows <= 1_000_000:
        return "Medium (10,000 - 1,000,000 rows)"
    return "Large (>1,000,000 rows)"


def _bucket_feature_count(n_cols: int) -> str:
    if n_cols < 30:
        return "Low (<30 features)"
    if n_cols <= 150:
        return "Medium (30-150 features)"
    return "High (>150 features)"


def _recommend_models(
    task_type: str,
    optimization_goal: str,
    interpretability_need: str,
    sample_size: str,
    feature_count: str,
    data_noise: str,
    class_balance: str | None,
    time_signal: str | None,
) -> tuple[str, list[str]]:
    """Return a model group label and top-3 suggested models."""
    if task_type == "Time series forecasting":
        if time_signal == "Clear seasonality":
            return "Time series (seasonal)", ["Prophet", "SARIMA", "XGBoost with lag features"]
        return "Time series (general)", ["ARIMA", "Prophet", "LightGBM with lag features"]

    if interpretability_need == "High":
        if task_type == "Categorical (Classification)":
            return "Explainable linear", [
                "Logistic Regression",
                "Linear Regression",
                "Linear SVM",
            ]
        return "Explainable linear", ["Linear Regression", "ElasticNet", "Ridge Regression"]

    if optimization_goal == "Fast inference" or sample_size == "Large (>1,000,000 rows)":
        if task_type == "Categorical (Classification)":
            return "Scalable tabular", ["LightGBM", "Logistic Regression", "Linear Regression"]
        return "Scalable tabular", ["LightGBM", "Linear Regression", "SGD Regressor"]

    if feature_count == "High (>150 features)":
        if task_type == "Categorical (Classification)":
            return "High-dimensional", ["Linear SVM", "Logistic Regression", "LightGBM"]
        return "High-dimensional", ["Linear Regression", "ElasticNet", "Ridge Regression"]

    if task_type == "Categorical (Classification)" and class_balance == "Strong imbalance":
        return "Imbalanced classification", [
            "Balanced Random Forest",
            "XGBoost (class-weighted)",
            "Logistic Regression (class-weighted)",
        ]

    if data_noise in ("Medium", "High"):
        if task_type == "Categorical (Classification)":
            return "Robust tree ensemble", ["Random Forest", "Decision Tree", "XGBoost"]
        return "Robust tree ensemble", ["Random Forest Regressor", "Decision Tree Regressor", "XGBoost Regressor"]

    if task_type == "Categorical (Classification)":
        return "General tabular classification", ["XGBoost", "LightGBM", "Random Forest"]
    return "General tabular regression", ["Linear Regression", "XGBoost Regressor", "LightGBM Regressor"]


def _required_keys_for_task(task_type: str, has_uploaded_data: bool) -> list[str]:
    required_keys = [
        "task_type",
        "optimization_goal",
        "interpretability_need",
        "data_noise",
    ]

    if not has_uploaded_data:
        required_keys.extend(["sample_size", "feature_count"])

    if task_type == "Categorical (Classification)":
        required_keys.append("class_balance")

    if task_type == "Time series forecasting":
        required_keys.append("time_signal")

    return required_keys


def _is_form_specifics_complete(values: dict, required_keys: list[str]) -> bool:
    return all(values.get(key) not in (None, "") for key in required_keys)


def _render_model_parameters(model_name: str) -> list[str]:
    """Render one model chooser and return selections."""
    model_params = MODEL_PARAMETERS.get(model_name, [])
    for chooser in model_params:
        widget_key = f"{model_name}_{chooser['name']}"
        if chooser["control"] == "slider":
            st.slider(
                chooser["name"],
                min_value=chooser["min"],
                max_value=chooser["max"],
                value=chooser["default"],
                step=chooser["step"],
                key=widget_key,
            )
        elif chooser["control"] == "selectbox":
            st.selectbox(
                chooser["name"],
                chooser["options"],
                key=widget_key,
            )


def ml_page() -> None:
    st.title("ML Lab")
    st.write("Describe your ML setup and get a focused shortlist of models")

    ml_advisor, ml_explorer = st.tabs(["Model Advisor", "Model Explorer"])

    with ml_advisor:
        dataset_shape = _get_dataset_shape_from_session()
        has_uploaded_data = dataset_shape is not None

        if has_uploaded_data:
            n_rows, n_cols = dataset_shape
            st.success(
                f"Dataset detected from Home page: {n_rows:,} rows, {n_cols:,} columns. "
                "Some sizing questions are auto-filled."
            )
        else:
            st.info(
                "No uploaded dataset found in session. Please answer the data-size questions manually.")

        with st.form(key="models_form"):
            st.subheader("Enter the specifics of your data and task")
            form_specifics = st.session_state.setdefault(
                "ml_form_specifics", DEFAULT_FORM_SPECIFICS.copy())

            form_specifics["task_type"] = st.selectbox(
                "What is your primary ML task?",
                ["Numerical (Regression)", "Categorical (Classification)",
                 "Time series forecasting"],
            )

            form_specifics["optimization_goal"] = st.selectbox(
                "What matters most?",
                ["Best accuracy", "Fast training",
                    "Fast inference", "Interpretability"],
            )

            form_specifics["interpretability_need"] = st.selectbox(
                "How much interpretability do you need?",
                ["High", "Medium", "Low"],
            )

            form_specifics["data_noise"] = st.selectbox(
                "How noisy/outlier-heavy is the data?",
                ["Low", "Medium", "High", "Not sure"],
            )

            if has_uploaded_data:
                form_specifics["sample_size"] = _bucket_sample_size(n_rows)
                form_specifics["feature_count"] = _bucket_feature_count(n_cols)
                st.caption(
                    f"Sample size (auto): {form_specifics['sample_size']}")
                st.caption(
                    f"Feature count (auto): {form_specifics['feature_count']}")
            else:
                form_specifics["sample_size"] = st.selectbox(
                    "What is the estimated size of your training data?",
                    [
                        "Small (<10,000 rows)",
                        "Medium (10,000 - 1,000,000 rows)",
                        "Large (>1,000,000 rows)",
                    ],
                )
                form_specifics["feature_count"] = st.selectbox(
                    "How many input features do you expect?",
                    ["Low (<30 features)", "Medium (30-150 features)",
                     "High (>150 features)"],
                )

            form_specifics["class_balance"] = None
            if form_specifics["task_type"] == "Categorical (Classification)":
                form_specifics["class_balance"] = st.selectbox(
                    "How balanced are your classes?",
                    ["Balanced", "Mild imbalance", "Strong imbalance", "Not sure"],
                )

            form_specifics["time_signal"] = None
            if form_specifics["task_type"] == "Time series forecasting":
                form_specifics["time_signal"] = st.selectbox(
                    "What kind of temporal signal do you expect?",
                    ["Clear seasonality", "Trend only", "Unclear / noisy"],
                )

            submit_button = st.form_submit_button(label="Calculate best model")

        if submit_button:
            required_keys = _required_keys_for_task(
                form_specifics["task_type"],
                has_uploaded_data,
            )

            if not _is_form_specifics_complete(form_specifics, required_keys):
                st.error(
                    "Please complete all required fields before calculating recommendations.")
            else:
                model_group, suggested_models = _recommend_models(
                    task_type=form_specifics["task_type"],
                    optimization_goal=form_specifics["optimization_goal"],
                    interpretability_need=form_specifics["interpretability_need"],
                    sample_size=form_specifics["sample_size"],
                    feature_count=form_specifics["feature_count"],
                    data_noise=form_specifics["data_noise"],
                    class_balance=form_specifics["class_balance"],
                    time_signal=form_specifics["time_signal"],
                )

                st.subheader("Recommended model shortlist")
                st.write(f"Model group: **{model_group}**")
                st.write("Top 3 suggestions:")
                for idx, model_name in enumerate(suggested_models, start=1):
                    st.write(f"{idx}. {model_name}")

    with ml_explorer:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            m1.metric("F1 score", 1, border=True)
        with m2:
            m2.metric("Precision", 1, border=True)
        with m3:
            m3.metric("Accuracy", 1, border=True)
        with m4:
            m4.metric("AUC", 1, border=True)

        st.divider()

        menu_cols = st.columns([1, 3])

        with menu_cols[0].container(border=True, height="stretch", vertical_alignment="center"):
            st.markdown("### Algorithm")
            algorithm = st.selectbox(
                "Algorithm",
                list(MODEL_OPTIONS.keys()),
                label_visibility="collapsed",
            )

            st.session_state["algorithm"] = algorithm
            selected_model = MODEL_OPTIONS[algorithm]

            st.markdown("### Hyperparameters")
            _render_model_parameters(algorithm)
            st.caption(f"Model type: {selected_model.__class__.__name__}")

            execute = st.button("Run Model")

        with menu_cols[1].container(border=True, height="stretch", vertical_alignment="center"):
            if execute:
                models.run_model()
            else:
                st.write("Charts and results will go here.")
