import gzip
import json
import logging
import os
import random
import re
from typing import Any, Dict, Iterator

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

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


def read_json_file(filepath: str, sample_size: float) -> pd.DataFrame:
    """
    Read a JSON file and return its contents as a DataFrame.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        pd.DataFrame or None: DataFrame containing the JSON data, or None if an
            error occurs.
    """
    try:
        sampled_data = []
        with open(filepath, "r") as json_file:
            for line in json_file:
                if random.random() < sample_size:
                    sampled_data.append(json.loads(line))
        dataframe = pd.DataFrame(sampled_data)
        logging.info(f"Read JSON file in dataframe - {filepath}")
        return dataframe

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def read_json_files(path: str, seed: int, sample_size: float) -> pd.DataFrame:
    """
    Read and concatenate multiple JSON files from a directory.

    Args:
        path (str): Path to the directory containing JSON files.
        seed (int): Seed for random number generator.
        sample_size (float): Fraction of data to sample from each JSON file.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing the data from all JSON
            files.
    """
    random.seed(seed)
    dataframes = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            filepath = os.path.join(path, filename)
            dataframe = read_json_file(
                filepath=filepath, sample_size=sample_size
            )
            dataframes.append(dataframe)
            logging.info(f"Read JSON file into dataframe - {filepath}")

    return pd.concat(dataframes, ignore_index=True)


def create_pie_chart(dataframe: pd.DataFrame, type_flag: str) -> None:
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


def create_bar_plot(dataframe: pd.DataFrame, column: str) -> None:
    """
    Create a bar chart visualisation with count of each bar.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        column (str): Column of interest to visualise.

    Returns:
        None
    """
    plt.figure(figsize=(15, 6))

    # Create the count plot
    ax = sns.countplot(data=dataframe, x=column)

    # Add count labels to each bar
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

    # Set titles
    plt.title("Count of Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")

    plt.show()


def analyse_text(
    text_column: pd.Series, regex_dict: dict, target: str
) -> pd.DataFrame:
    """Analyse input text using provided regular expression patterns.
    Returns Pandas DataFrame containing count of matches for each
    regex pattern along with a list of the matches found for each pattern.

    Args:
        text_column (pandas.core.series.Series): Pandas series containing the
        text to be analyzed.
        regex_dict (dict): A dictionary containing the regex patterns to be
        matched along with a label for each pattern.
        target (str): Category of regular expressions.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the count of matches and a
        list of the matches found for each pattern.
    """
    count = {regex: 0 for regex in regex_dict}
    matches = {regex: [] for regex in regex_dict}

    for text in text_column:
        for regex in regex_dict:
            match = re.findall(regex_dict[regex], str(text))
            count[regex] += len(match)
            matches[regex].extend(match)

    results = (
        pd.DataFrame({"Count": count, "Matches": matches})
        .reset_index(drop=False)
        .rename(columns={"index": target})
    )

    return results


def visualise_regex_results(results: pd.DataFrame) -> None:
    """
    Visualize the distribution of regex results using a bar plot.

    Args:
        results (pd.DataFrame): DataFrame containing regex results with columns
            'Regex' and 'Count'.

    Returns:
        None
    """
    _, axis = plt.subplots(figsize=(15, 6))

    results.sort_values(by=["Count"], ascending=True, inplace=True)
    sns.barplot(
        data=results,
        x="Regex",
        y="Count",
    )

    # Set overall title
    plt.suptitle("Distribution of Results", fontsize=18, fontweight="bold")

    # Display respective bar count
    for p in axis.patches:
        axis.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

    # Display visualisation
    plt.show()
