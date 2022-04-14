import streamlit as st
from interface.overlaps import render_overlaps


def render():
    pages = {
        "Summary Overlaps": render_overlaps
    }

    st.sidebar.title("Summarization Highlights")
    selected_page = st.sidebar.radio("Select a page", options=list(pages.keys()))

    pages[selected_page]()


if __name__ == "__main__":
    render()