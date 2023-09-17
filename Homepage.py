import importlib.util

import streamlit as st

# Page Configurations
st.set_page_config(
    page_title="AI-Driven Customer Review Analysis",
    layout="wide",
)


# Initialize session state
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None


def homepage():
    # Title
    st.title("AI-Driven Customer Review Analysis")

    # Objective
    st.subheader("Objective 🎯")
    st.write(
        """
        Our primary goal is to help businesses analyse customer reviews
        more efficiently and effectively. Powered by state-of-the-art
        open-source language models, this tool classifies reviews into various
        sentiments, highlights areas for potential improvement, identifies
        common criticisms, and even automatically generates responses for you.
        """
    )

    # Features
    st.subheader("Features 🎁")

    # Using columns to arrange features
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Sentiment Classification 😃", unsafe_allow_html=True)
        st.write("Gauge the general sentiment about your product.")
        st.write("<br>", unsafe_allow_html=True)

        st.markdown("#### Potential Improvements 💡", unsafe_allow_html=True)
        st.write("Focus your improvement efforts effectively.")
        st.write("<br>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Potential Criticisms ⚠️", unsafe_allow_html=True)
        st.write("Proactively resolve recurring issues.")
        st.write("<br>", unsafe_allow_html=True)

        st.markdown("#### Auto-Responses 🗨️", unsafe_allow_html=True)
        st.write("Efficiently manage reviews and maintain brand voice.")
        st.write("<br>", unsafe_allow_html=True)

    # Instructions
    st.subheader("How to Use this Application 📘")
    # st.markdown("#### How to Use this Application 📘")
    st.write(
        """
        1. **Model Selection**: Choose the machine learning model you want to
        use.
        2. **Input Reviews**: Submit the customer reviews you want to analyze.
        3. **Results**: View the analysis results for each review. Download the
        results for further review or sharing.

        Navigate through these steps using the sidebar.👈
        """
    )


def import_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    model_selection_page = import_from_path(
        "pages/2_1._Model_Selection.py", "model_selection_page"
    )
    input_page = import_from_path("pages/3_2._Input_Reviews.py", "input_page")
    results_page = import_from_path("pages/4_3._Results.py", "results_page")

    homepage()
