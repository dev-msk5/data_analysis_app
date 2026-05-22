"""Streamlit entrypoint for DataLab."""

import streamlit as st
from data.data_loader import DataLoader


def run_app() -> None:
    st.title("DataLab")

    # upload_files = st.file_uploader("Choose the file to upload")
    # bytes_data = upload_files.read()
    # st.write("filename:", upload_files.name)
    # st.write(bytes_data)

    load_button = st.button("Load file")
    if load_button == True:
        data = DataLoader().load_data("testy.csv")
        st.write(data)


if __name__ == "__main__":
    run_app()
