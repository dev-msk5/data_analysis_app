import streamlit as st


form_specifics = {

}


def _get_dataset_shape_from_session() -> tuple[int, int] | None:
    """Read active dataset dimensions from shared session state.

    Returns
    - tuple[int, int] | None: (rows, columns) if a dataset exists, otherwise None.
    """
    controller = st.session_state.get("dataset_controller")
    if controller is None or getattr(controller, "original_df", None) is None:
        return None

    try:
        dataframe = controller.get_data()
    except Exception:
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
                "Linear SVM",
                "Explainable Boosting Machine",
            ]
        return "Explainable linear", ["ElasticNet", "Ridge Regression", "Lasso Regression"]

    if optimization_goal == "Fast inference" or sample_size == "Large (>1,000,000 rows)":
        if task_type == "Categorical (Classification)":
            return "Scalable tabular", ["LightGBM", "Logistic Regression", "Linear SVM"]
        return "Scalable tabular", ["LightGBM", "SGD Regressor", "Ridge Regression"]

    if feature_count == "High (>150 features)":
        if task_type == "Categorical (Classification)":
            return "High-dimensional", ["Linear SVM", "Logistic Regression", "LightGBM"]
        return "High-dimensional", ["ElasticNet", "Ridge Regression", "LightGBM"]

    if task_type == "Categorical (Classification)" and class_balance == "Strong imbalance":
        return "Imbalanced classification", [
            "Balanced Random Forest",
            "XGBoost (class-weighted)",
            "Logistic Regression (class-weighted)",
        ]

    if data_noise in ("Medium", "High"):
        if task_type == "Categorical (Classification)":
            return "Robust tree ensemble", ["Random Forest", "XGBoost", "CatBoost"]
        return "Robust tree ensemble", ["Random Forest Regressor", "XGBoost Regressor", "CatBoost Regressor"]

    if task_type == "Categorical (Classification)":
        return "General tabular classification", ["XGBoost", "LightGBM", "Random Forest"]
    return "General tabular regression", ["XGBoost Regressor", "LightGBM Regressor", "Random Forest Regressor"]


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


def ml_page() -> None:
    st.title("ML Lab")
    st.write("Describe your ML setup and get a focused shortlist of models.")

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

        form_specifics["task_type"] = st.selectbox(
            "What is your primary ML task?",
            ["Numerical (Regression)", "Categorical (Classification)",
             "Time series forecasting"],
        )

        form_specifics["optimization_goal"] = st.selectbox(
            "What matters most?",
            ["Best accuracy", "Fast training", "Fast inference", "Interpretability"],
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
            st.caption(f"Sample size (auto): {form_specifics['sample_size']}")
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

    if not submit_button:
        return

    required_keys = _required_keys_for_task(
        form_specifics["task_type"],
        has_uploaded_data,
    )

    if not _is_form_specifics_complete(form_specifics, required_keys):
        st.error(
            "Please complete all required fields before calculating recommendations.")
        return

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
