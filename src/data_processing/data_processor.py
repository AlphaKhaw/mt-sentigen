import logging
import os
import re
import sys

import hydra
import pandas as pd
from omegaconf import DictConfig
from unidecode import unidecode

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from utils.dataframe.dataframe_utils import export_dataframe

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class DataProcessor:
    """
    DataProcessor class to preprocess extracted JSON files.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize a DataExtractor object and initialise database.

        Args:
            cfg (DictConfig): hydra configuration

        Returns:
            None
        """
        self.cfg = cfg

    def process_data(self) -> None:
        """
        Main function of DataProcessor class to:

        Args:
            None

        Returns:
            None
        """
        # Read in configurations from Hydra configuration file
        rating_column = self.cfg.general.rating_column
        response_column = self.cfg.general.response_column
        review_column = self.cfg.general.review_column
        input_folderpath = self.cfg.data_processor.input_folder
        output_folderpath = self.cfg.data_processor.output_folder
        output_suffix = self.cfg.data_processor.suffix
        remove_json_flag = self.cfg.data_processor.remove_json_files

        # Read in json files into dataframe
        dataframe = self._read_json_files(input_folderpath)
        logging.info("Read all JSON files into dataframe")

        # Extract review response
        dataframe[response_column] = dataframe[response_column].apply(
            lambda x: x["text"] if x is not None else None
        )
        logging.info("Extracted review response text")

        # Drop duplicates
        dataframe.drop_duplicates(inplace=True)
        logging.info("Dropped duplicated rows")

        # Drop NaN values
        dataframe.dropna(inplace=True)
        logging.info("Dropped rows with NaN values")

        # Categorize ratings into classes
        dataframe[rating_column] = dataframe[rating_column].apply(
            lambda x: self._map_rating_to_class(x)
        )
        logging.info("Categorized ratings into classes")

        # Process text columns - review
        dataframe = self._process_text_column(
            dataframe=dataframe, text_columns=[response_column, review_column]
        )

        # Output dataframe
        output_filepath = os.path.join(output_folderpath, output_suffix)
        export_dataframe(dataframe=dataframe, output_filepath=output_filepath)

        # Remove JSON files if True
        if remove_json_flag:
            json_file_list = [
                file
                for file in os.listdir(input_folderpath)
                if os.path.isfile(os.path.join(input_folderpath, file))
            ]
            for json_filename in json_file_list:
                os.remove(os.path.join(input_folderpath, json_filename))
                logging.info(f"Removed JSON file - {json_filename}")

    def _map_rating_to_class(self, rating: float) -> int:
        """
        Map numeric ratings to corresponding sentiment classes.

        Args:
            rating (float): Numeric rating value.

        Returns:
            str: Sentiment class
                ('negative': 0, 'neutral': 1, or 'positive': 2)
        """
        if rating <= 2.0:
            return 0
        elif rating == 3.0:
            return 1
        else:
            return 2

    def _read_json_files(self, path: str) -> pd.DataFrame:
        """
        Read and concatenate multiple JSON files from a directory using
        concurrency.

        Args:
            path (str): Path to the directory containing JSON files.

        Returns:
            pd.DataFrame: Concatenated DataFrame containing the data from all
                JSON files.
        """
        dataframes = []

        for filename in os.listdir(path):
            if filename.endswith(".json"):
                filepath = os.path.join(path, filename)
                dataframe = pd.read_json(filepath, lines=True)
                dataframes.append(dataframe)
                logging.info(f"Read JSON file into dataframe - {filepath}")

        return pd.concat(dataframes, ignore_index=True)

    def _process_text_column(
        self, dataframe: pd.DataFrame, text_columns: list
    ) -> pd.DataFrame:
        """
        - Remove unicode characters
        - Remove special characters except punctuations

        Args:
            text_column (list): _description_

        Returns:
            pd.DataFrame: _description_
        """
        regex_dict = self.cfg.data_processor.regex_pattern
        for text_column in text_columns:
            dataframe[text_column] = dataframe[text_column].apply(unidecode)
            dataframe[text_column] = dataframe[text_column].apply(
                lambda text: self._apply_regex_pattern(regex_dict, text)
            )
            dataframe[text_column] = dataframe[text_column].astype(str)
            logging.info(f"Processed text column - {text_column}")

        return dataframe

    def _apply_regex_pattern(self, regex_dict: DictConfig, text: str) -> str:
        """
        Apply a series of regular expression patterns to a given text.

        Args:
            regex_dict (OmegaConf): A dictionary containing regex patterns and
                their replacements.
            text (str): The input text to be processed.

        Returns:
            str: The processed text after applying the regular expression
                patterns.
        """
        for pattern, replacement in regex_dict.items():
            text = re.sub(rf"{pattern}", rf"{replacement}", text)

        return text


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run_standalone(cfg: DictConfig) -> None:
    """
    Initialize Hydra configuration and run standalone DataProcessor class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        None
    """
    logging.info(run(cfg))


def run(cfg: DictConfig) -> str:
    """
    Pass in Hydra configuration and run DataProcessor class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        str: Status of DataProcessor class.
    """
    processor = DataProcessor(cfg)
    processor.process_data()

    return "Complete data processing"


if __name__ == "__main__":
    run_standalone()
