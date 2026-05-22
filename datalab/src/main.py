import streamlit as st

from gui.pages.main_window import run_app as home_page
from gui.pages.ML_Lab import ml_page
from gui.pages.data_visualization import data_visualization_page


def main():
    st.set_page_config(page_title="DataLab", page_icon="📊")

    navigation = st.navigation(
        [
            st.Page(home_page, title="Home", icon="🏠", default=True),
            st.Page(data_visualization_page,
                    title="Data Visualization", icon="📈"),
            st.Page(ml_page, title="ML Lab", icon="🤖"),
        ]
    )
    navigation.run()


if __name__ == "__main__":
    main()
