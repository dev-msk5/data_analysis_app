import streamlit as st
import pandas as pd

from gui.session_helpers import get_active_dataframe

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


def _render_chart_choosers(n_charts: int) -> list[str]:
    """Render one chart chooser per requested chart and return selections."""
    selected_charts: list[str] = []

    for index in range(n_charts):
        st.markdown(f"### Chart type {index + 1}")
        selected_chart = st.selectbox(
            "",
            CHART_OPTIONS,
            label_visibility="collapsed",
            key=f"chart_type_{index}",
        )
        selected_charts.append(selected_chart)

    return selected_charts


def _render_chart(chart_type: str, df: pd.DataFrame, x_column: str, y_column: str) -> None:
    """Render a single chart block for the selected chart type."""
    # Validate requested columns exist in the DataFrame
    if x_column not in df.columns or y_column not in df.columns:
        st.error(
            f"Selected axes not in dataframe columns: x={x_column!r}, y={y_column!r}."
        )
        st.write("Available columns:", list(df.columns))
        return

    chart_data = df[[x_column, y_column]].dropna().copy()
    chart_data = chart_data.rename(columns={x_column: "x", y_column: "y"})

    if chart_type == "Bar":
        st.bar_chart(chart_data.set_index("x")["y"])
        return

    if chart_type == "Line":
        st.line_chart(chart_data.set_index("x")["y"])
        return

    if chart_type == "Area":
        st.area_chart(chart_data.set_index("x")["y"])
        return

    if chart_type == "Scatter":
        st.scatter_chart(chart_data, x="x", y="y")
        return

    if chart_type == "Histogram":
        st.vega_lite_chart(
            {
                "data": {"values": chart_data.to_dict("records")},
                "mark": "bar",
                "encoding": {
                    "x": {"field": "y", "bin": True, "type": "quantitative"},
                    "y": {"aggregate": "count", "type": "quantitative"},
                },
            },
            use_container_width=True,
        )
        return

    if chart_type == "Box":
        st.vega_lite_chart(
            {
                "data": {"values": chart_data.to_dict("records")},
                "mark": {"type": "boxplot", "extent": "min-max"},
                "encoding": {"y": {"field": "y", "type": "quantitative"}},
            },
            use_container_width=True,
        )
        return

    if chart_type == "Violin":
        st.vega_lite_chart(
            {
                "data": {"values": chart_data.to_dict("records")},
                "transform": [{"density": "y", "as": ["y", "density"]}],
                "mark": {"type": "area", "orient": "horizontal"},
                "encoding": {
                    "y": {"field": "y", "type": "quantitative"},
                    "x": {"field": "density", "type": "quantitative"},
                },
            },
            use_container_width=True,
        )
        return

    if chart_type == "Pie":
        pie_source = chart_data["x"]
        if pd.api.types.is_numeric_dtype(pie_source):
            pie_source = pd.cut(pie_source, bins=min(
                5, max(2, pie_source.nunique())))
        pie_data = pie_source.value_counts().reset_index()
        pie_data.columns = ["bin", "count"]
        st.vega_lite_chart(
            {
                "data": {"values": pie_data.to_dict("records")},
                "mark": {"type": "arc", "innerRadius": 0},
                "encoding": {
                    "theta": {"field": "count", "type": "quantitative"},
                    "color": {"field": "bin", "type": "nominal"},
                },
            },
            use_container_width=True,
        )
        return

    if chart_type == "Heatmap":
        corr = df.select_dtypes(include="number").corr(
            numeric_only=True).reset_index().melt(id_vars="index")
        corr.columns = ["x", "y", "value"]
        st.vega_lite_chart(
            {
                "data": {"values": corr.to_dict("records")},
                "mark": "rect",
                "encoding": {
                    "x": {"field": "x", "type": "nominal"},
                    "y": {"field": "y", "type": "nominal"},
                    "color": {"field": "value", "type": "quantitative"},
                },
            },
            use_container_width=True,
        )
        return

    st.info(f"No renderer defined for {chart_type}.")


def data_visualization_page() -> None:
    st.title("Data Visualization")
    st.write("Visualize your data with different charts")

    df = get_active_dataframe()
    if df is None or df.empty:
        st.info("No data loaded.")
        return

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        st.info("No numeric columns available for descriptive statistics.")
        return

    st.session_state.setdefault("n_charts", 1)
    st.session_state.setdefault("show_charts", False)
    st.session_state.setdefault("x_axis_select", df.columns[0])
    st.session_state.setdefault("y_axis_select", numeric_df.columns[0])

    selected_column = st.selectbox(
        "Numeric column", options=numeric_df.columns)
    series = numeric_df[selected_column].dropna()

    mean, median, min, max, std, var, count, q3 = st.columns(8)

    mean.metric("Mean", f"{series.mean():.2f}")
    median.metric("Median", f"{series.median():.2f}")
    min.metric("Min", f"{series.min():.2f}")
    max.metric("Max", f"{series.max():.2f}")
    std.metric("Std", f"{series.std():.2f}")
    var.metric("Variance", f"{series.var():.2f}")
    count.metric("Count", f"{int(series.count())}")
    q3.metric("Q3 (75%)", f"{series.quantile(0.75):.2f}")

    menu_cols = st.columns([1, 3])

    with menu_cols[0].container(border=True, height="stretch", vertical_alignment="center"):
        st.markdown("### Number of charts")
        n_charts = st.slider(
            "n",
            min_value=1,
            max_value=5,
            key="n_charts",
            step=1,
        )

        selected_charts = _render_chart_choosers(n_charts)

        x_axis_select = st.selectbox(
            "X axis data", options=df.columns, key="x_axis_select")
        y_axis_select = st.selectbox(
            "Y axis data", options=numeric_df.columns, key="y_axis_select")

        if st.button("Show Charts"):
            st.session_state["show_charts"] = True

    with menu_cols[1].container(border=True, height="stretch", vertical_alignment="center"):
        st.session_state["selected_charts"] = selected_charts
        if st.session_state.get("show_charts"):
            st.subheader("Charts and results")
            for index, chart_type in enumerate(selected_charts, start=1):
                st.markdown(f"#### Chart {index}: {chart_type}")
                # Read latest axis selections from session state to be robust
                x_sel = st.session_state.get("x_axis_select", x_axis_select)
                y_sel = st.session_state.get("y_axis_select", y_axis_select)
                _render_chart(chart_type, df, x_sel, y_sel)
        else:
            st.write("Charts and results will go here.")
