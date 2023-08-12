import logging
from pathlib import Path

import pandas as pd

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def read_from_csv(filepath: str) -> pd.DataFrame:
    """
    Read in CSV file without index and returns it as a Pandas DataFrame.
    Args:

        filepath (str): The filepath of the CSV file to be read.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    file_path = Path(filepath)
    if not Path(filepath).exists():
        raise FileNotFoundError(f"{filepath} not found.")

    if file_path.suffix.lower() != ".csv":
        raise TypeError("Invalid file type. Only .csv files are accepted.")

    dataframe = pd.read_csv(filepath, dtype=object)
    logging.info(f"Read CSV - {filepath}")

    return dataframe


def export_to_csv(dataframe: pd.DataFrame, filepath: str) -> None:
    """
    Export the DataFrame to a CSV file without index in the
    specified folder path.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be exported as a CSV file.
        filename (str): The file name of exported CSV file.
    """
    file_path = Path(filepath)

    try:
        file_path.parents[0].mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(file_path, index=False)
        logging.info(f"Export CSV to {file_path}")
    except Exception as error:
        raise error
