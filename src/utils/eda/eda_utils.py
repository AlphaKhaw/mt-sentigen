import gzip
import json
import logging
import os
from typing import Any, Dict, Iterator

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
    except json.JSONDecodeError:
        logging.info(f"JSONDecodeError occurred for file: {path}")
        return
    except Exception as e:
        logging.info(f"An error occurred while parsing file {path}: {e}")
        return
