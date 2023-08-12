import gzip
import json
import logging
import os
from typing import Any, Dict, Iterator

import matplotlib.pyplot as plt
import pandas as pd
import requests

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def download_file(url: str, datapath: str) -> None:
    """
    Download a file from specified URL and save it to the specified datapath.

    Args:
        url (str): URL of file.
        datapath (str): Directory path where the downloaded file will be saved.

    Returns:
        None
    """
    if not os.path.exists(datapath):
        os.makedirs(datapath)
        logging.info(f"Created folder directory: {datapath}")
    filename = url.split("/")[-1]
    res = requests.get(url)
    if res.status_code == 200:
        filepath = os.path.join(datapath, filename)
        with open(filepath, "wb") as file:
            file.write(res.content)
        logging.info(f"Downloaded and saved to: {filepath}")
    else:
        logging.info(f"Failed to download the file from: {url}")


def parse(path: str) -> Iterator[Dict[str, Any]]:
    """
    Parse a Gzip-compressed JSON file and yield each JSON record as a
    dictionary.

    Args:
        path (str): The file path to the Gzip-compressed JSON file.

    Yields:
        Iterator[Dict[str, Any]]: An iterator that yields a dictionary
            representing each JSON record in the file.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        IsADirectoryError: If the provided path points to a directory instead
            of a file.
        gzip.BadGzipFile: If the file is not in Gzip format or is corrupted.
        json.JSONDecodeError: If there's an error decoding the JSON data.
        UnicodeDecodeError: If the file contains non-UTF-8 encoded data.
    """
    try:
        gzip_file = gzip.open(path, "r")
        for line in gzip_file:
            yield json.loads(line)
    except Exception as e:
        logging.info(f"An error occurred while parsing file {path}: {e}")
        return


def parse_into_dataframe(path: str) -> pd.DataFrame:
    """
    Parse a Gzip-compressed JSON file or read a Gzip-compressed CSV file and
    convert it into a pandas DataFrame.

    Args:
        path (str): The file path to the Gzip-compressed JSON or CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data parsed from the
            Gzip-compressed JSON file, or the data read from the
            Gzip-compressed CSV file.
    """
    if path.endswith(".json.gz"):
        return pd.DataFrame(parse(path))
    elif path.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    else:
        raise ValueError(
            "Unsupported file format. Only Gzip-compressed JSON (.json.gz) or \
                CSV (.csv.gz) files are supported."
        )


def parse_files_into_dataframe(path: str) -> pd.DataFrame:
    """
    Read in path of gzip files, parse files into DataFrame and concatenate
    into a single DataFrame.

    Args:
        path (str): Directory of raw gzip files.

    Returns:
        pd.DataFrame: Concatenated Pandas DataFrame.
    """
    concatenated_dataframes = []
    for dir in os.listdir(path):
        gzip_filepath = os.path.join(path, dir)
        dataframe = parse_into_dataframe(gzip_filepath)
        dataframe["state"] = dir.split(".")[0].split("-")[-1]
        concatenated_dataframes.append(dataframe)

        logging.info(f"Read {dir} into DataFrame")

    logging.info("Returning concatenated DataFrame")

    return pd.concat(concatenated_dataframes, ignore_index=True)


def remove_files_from_folder(folder_path: str) -> None:
    """
    Remove files from the specified folder.

    Args:
        folder_path (str): The path to the folder from which files will be
            removed.

    Raises:
        FileNotFoundError: If the provided folder_path does not exist.
        PermissionError: If the user does not have permission to remove files
            from the folder.

    Returns:
        None
    """
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                logging.info(f"File '{file}' has been removed.")
            else:
                logging.info(f"'{file}' is not a file. Skipping...")
        except Exception as e:
            logging.info(f"An error occurred while removing '{file}': {e}")


def map_rating_to_class(rating: float) -> str:
    """
    Map numeric ratings to corresponding sentiment classes.

    Args:
        rating (float): Numeric rating value.

    Returns:
        str: Sentiment class ('negative', 'neutral', or 'positive').
    """
    if rating <= 2.0:
        return "negative"
    elif rating == 3.0:
        return "neutral"
    else:
        return "positive"


def create_pie_chart(dataframe: pd.DataFrame, type_flag: str):
    """
    Create a pie chart visualization with percentage count based on input type
    flag:
    - `duplicate` - duplicated and and non-duplicated values
    - `nan` - NaN and non-NaN values

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        type_flag (str): Two values - Duplicate or NaN values.

    Returns:
        None
    """
    # Count duplicated and non-duplicated values
    if type_flag == "duplicate":
        count = dataframe.duplicated().sum()
    else:
        count = dataframe.isna().sum().sum()

    non_count = len(dataframe) - count

    # Create labels and sizes for the pie chart
    if type_flag == "duplicate":
        labels = ["Duplicated", "Non-Duplicated"]
    else:
        labels = ["NaN", "Non-NaN"]
    sizes = [count, non_count]

    # Create the pie chart
    plt.figure(figsize=(15, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.axis("equal")

    # Set chart title
    if type_flag == "duplicate":
        plt.title("Duplicated vs Non-Duplicated")
    else:
        plt.title("NaN vs Non-NaN")

    # Display the pie chart
    plt.show()
