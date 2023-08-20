import glob
import logging
import os
import re
import shutil
import sys
from typing import Union

import dask.dataframe as dd
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
        - Read in configurations from Hydra configuration file
        - Read in JSON files into Pandas DataFrame
        - Extract review response
        - Drop duplicated rows
        - Drop rows with NaN values
        - Categorise ratings into classes
        - Apply text cleaning
        - Output dataframe

        Optional flag: Remove JSON files if True.

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
        batch = self.cfg.data_processor.batch_to_process
        remove_json_flag = self.cfg.data_processor.remove_json_files

        # Read in total file size
        files = glob.glob(os.path.join(input_folderpath, "*.json"))
        intermediate_filepaths = []

        # Batch process
        for i, batch in enumerate(files):
            logging.info(f"Processing batch {i} - {batch}")

            # Read in json files into dataframe
            dataframe = self._read_json_files(input_folderpath, [batch])
            logging.info("Read batch JSON files into dataframe")

            # Extract text from review response
            dataframe[response_column] = dataframe[response_column].map(
                self._extract_text, meta=(response_column, "string")
            )
            dataframe = dataframe.persist()
            logging.info("Extracted review response text")

            # Remove missing values
            dataframe = dataframe.dropna(
                subset=[rating_column, response_column, review_column]
            )
            logging.info("Dropped rows with NaN values")

            # Drop duplicates
            dataframe.drop_duplicates(inplace=True)
            logging.info("Dropped duplicated rows")

            # Categorize ratings into classes
            dataframe[rating_column] = dataframe[rating_column].map_partitions(
                lambda partition: partition.apply(
                    lambda x: self._map_rating_to_class(x)
                )
            )
            logging.info("Categorized ratings into classes")

            # Process text columns - review
            dataframe = self._process_text_column(
                dataframe=dataframe,
                text_columns=[response_column, review_column],
            )

            # Output dataframe
            output_filepath = os.path.join(
                output_folderpath, f"{i}_{output_suffix}"
            )
            intermediate_filepaths.append(output_filepath)
            export_dataframe(
                dataframe=dataframe, output_filepath=output_filepath
            )

        # Read in intermediate files and output processed file
        processed_dataframe = self._read_intermediate_files(output_folderpath)
        logging.info("Read in intermediate processed files into dataframe")

        # Reset index
        processed_dataframe = processed_dataframe.reset_index(drop=True)

        # Output processed dataframe
        output_filepath = os.path.join(output_folderpath, output_suffix)
        export_dataframe(
            dataframe=processed_dataframe, output_filepath=output_filepath
        )

        # Remove intermediate files
        for file in intermediate_filepaths:
            if file.endswith(".csv"):
                os.remove(file)
            elif file.endswith(".parquet"):
                shutil.rmtree(file)
        logging.info("Removed all intermediate files")

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
            if isinstance(text, str):
                text = re.sub(rf"{pattern}", rf"{replacement}", text)

        return text

    def _extract_text(
        self, resp: Union[str, pd._libs.missing.NAType]
    ) -> Union[str, pd._libs.missing.NAType]:
        if isinstance(resp, str):
            return eval(resp).get("text")
        else:
            return resp

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

    def _process_text_column(
        self, dataframe: pd.DataFrame, text_columns: list
    ) -> dd.DataFrame:
        """
        Process the text columns in the DataFrame by removing unicode
        characters and special characters except punctuations.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing text
                columns.
            text_columns (list): A list of column names containing text to be
                processed.

        Returns:
            pd.DataFrame: A DataFrame with processed text columns.
        """
        regex_dict = self.cfg.data_processor.regex_pattern
        for text_column in text_columns:
            dataframe[text_column] = dataframe[text_column].apply(
                unidecode, meta=(text_column, "string")
            )
            dataframe[text_column] = dataframe[text_column].apply(
                lambda text: self._apply_regex_pattern(regex_dict, text)
            )
            logging.info(f"Processed text column - {text_column}")

        return dataframe

    def _read_intermediate_files(self, output_folderpath: str) -> dd.DataFrame:
        """
        Read and concatenate processed dataframes from CSV or Parquet files
        within the specified folder.

        Args:
            output_folderpath (str): The folder path containing the processed
                CSV and Parquet files.

        Returns:
            dd.DataFrame: A Dask dataframe containing the concatenated data
                from the processed files.
        """
        processed_dataframes = []
        for filename in os.listdir(output_folderpath):
            filepath = os.path.join(output_folderpath, filename)
            if filename.endswith(".csv"):
                processed_dataframes.append(dd.read_csv(filepath))
            elif filename.endswith(".parquet"):
                processed_dataframes.append(dd.read_parquet(filepath))

        processed_dataframe = dd.concat(processed_dataframes, ignore_index=True)
        return processed_dataframe

    def _read_json_files(self, path: str, files: list) -> dd.DataFrame:
        """
        Read and concatenate multiple JSON files from input batch.

        Args:
            path (str): Path to the directory containing JSON files.
            files (list): List containing names of JSON files.

        Returns:
            dd.DataFrame: Concatenated DataFrame containing the data from all
                JSON files.
        """
        dataframes = []

        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(path, filename)
                dataframe = dd.read_json(filepath, lines=True)
                dataframes.append(dataframe)
                logging.info(f"Read JSON file into dataframe - {filepath}")

        return dd.concat(dataframes, ignore_index=True)


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
