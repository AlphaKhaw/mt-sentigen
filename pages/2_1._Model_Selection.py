from datetime import datetime

import streamlit as st


def model_selection_page():
    st.title("Model Selection")
    model = st.selectbox("Choose a model", ["LlaMA-7B-Chat-GGUF"])

    # Get current date
    current_date = datetime.now().strftime("%dth %B %Y")

    if model != "Select a model...":
        # st.write(f"You selected {model}.")
        st.info(
            f"""
            As of {current_date}, only the LlaMA-7B-Chat-GGUF model is
            deployed.
            """
        )

    if model == "LlaMA-7B-Chat-GGUF":
        st.subheader("About LlaMA-7B-Chat-GGUF")
        st.write(
            """
        The Llama2-7B-Chat is a state-of-the-art open-source language model
        designed and optimized for a range of dialogue and conversational
        use-cases. With 7 billion parameters, this model offers high-quality
        and nuanced language generation and understanding capabilities. Its
        expansive parameter count makes it one of the most versatile language
        models available for chat-based applications.
        """
        )


if __name__ == "__main__":
    # Page Configurations
    st.set_page_config(
        page_title="AI-Driven Customer Review Analysis",
        layout="wide",
    )
    model_selection_page()
