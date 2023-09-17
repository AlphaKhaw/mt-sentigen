from io import BytesIO

import streamlit as st


def results_page():
    st.title("Results")

    if hasattr(st.session_state, "dataframe"):
        if st.session_state.dataframe is not None:
            st.dataframe(st.session_state.dataframe)

            # Export DataFrame to CSV and create download link
            buffer = BytesIO()
            st.session_state.dataframe.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="Export as CSV",
                data=buffer,
                file_name="results.csv",
                mime="text/csv",
            )
        else:
            st.write("No data to display. Please input reviews first.")
    else:
        st.write("No data to display. Please input reviews first.")


if __name__ == "__main__":
    results_page()
