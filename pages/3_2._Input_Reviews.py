import logging
from io import StringIO
from typing import Dict

import pandas as pd
import requests
import streamlit as st
import yaml

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def input_page(endpoint: str):
    st.title("Input Reviews")

    # Choose input type
    input_type = st.selectbox(
        "Choose input type", ["Single Review", "Multiple Reviews", "CSV File"]
    )

    if input_type == "Single Review":
        single_text = st.text_input("Enter your text here")
        if st.button("Analyse Single Text"):
            with st.spinner("Analysing the single review..."):
                # Make API call
                response = requests.post(
                    f"{endpoint}/predict_one/", json={"text": single_text}
                )
                st.session_state.dataframe = convert_response_to_dataframe(
                    response.json()
                )
                st.write(st.session_state.dataframe)

    elif input_type == "Multiple Reviews":
        multiple_texts = st.text_area(
            "Enter multiple texts here separated by a new line"
        )
        if st.button("Analyse Multiple Texts"):
            with st.spinner("Analysing the multiple reviews..."):
                texts = multiple_texts.split("\n")

                # Make API call
                response = requests.post(
                    f"{endpoint}/predict_multiple/", json={"texts": texts}
                )
                st.session_state.dataframe = convert_response_to_dataframe(
                    response.json()
                )
                st.write(st.session_state.dataframe)

    elif input_type == "CSV File":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        # Describe the required structure of the CSV
        st.info(
            """
            The CSV file should either have a column named 'text' or have only
            one column.
            """
        )
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)

            # Check if it has a column named 'text', or take the only column
            if "text" in dataframe.columns:
                texts = dataframe["text"].tolist()
            elif len(dataframe.columns) == 1:
                texts = dataframe.iloc[:, 0].tolist()
            else:
                st.error(
                    """
                    The uploaded CSV file doesn't meet the requirements.
                    """
                )
                return

            if st.button("Analyse CSV File"):
                with st.spinner("Analysing the CSV file..."):
                    # Convert the DataFrame to CSV and then to string
                    csv_buffer = StringIO()
                    dataframe.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)

                    # Prepare the file payload
                    files = {
                        "file": ("uploaded_file.csv", csv_buffer, "text/csv")
                    }

                    # Make API call
                    response = requests.post(
                        f"{endpoint}/predict_csv/", files=files
                    )
                    st.session_state.dataframe = convert_response_to_dataframe(
                        response.json()
                    )
                    st.write(st.session_state.dataframe)

    # Test API endpoint
    if not test_api_endpoint(endpoint):
        st.info("The API endpoint is currently down. Please try again later.")
        return


def convert_response_to_dataframe(response: Dict[str, dict]) -> pd.DataFrame:
    dataframe = pd.DataFrame.from_dict(response, orient="index").reset_index()
    dataframe.columns = [
        "Review",
        "Sentiment",
        "Generated Response",
        "Potential Improvements",
        "Potential Criticisms",
    ]

    return dataframe


def test_api_endpoint(endpoint: str):
    try:
        response = requests.get(f"{endpoint}/")
        if response.status_code == 200:
            return True
    except Exception as e:
        st.warning(f"Exception encountered: {e}")

    return False


if __name__ == "__main__":
    with open("conf/base/pipelines.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
            endpoint = cfg["streamlit"]["fastapi_endpoint"]
        except Exception as e:
            logging.info(
                f"An error occurred while loading the configuration: {e}"
            )
            raise e
    input_page(endpoint)
