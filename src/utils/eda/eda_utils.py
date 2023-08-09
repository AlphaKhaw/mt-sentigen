import gzip
import json
import logging
import os
from typing import Any, Dict, Iterator

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
