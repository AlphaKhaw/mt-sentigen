import gzip
import json
import logging
import os
import sys
from typing import Any, Dict, Iterator

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from utils.dataframe.dataframe_utils import export_to_csv

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class DataExtractor:
    """
    DataExtractor class to extract JSON/CSV files from downloaded gzip files.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize a DataExtractor object.

        Args:
            cfg (DictConfig): hydra configuration

        Returns:
            None
        """
        self.cfg = cfg

    def extract_data(self) -> None:
        """
        Main function of DataExtractor class to:
        - Read in configurations
        - Parse gzip files into a single concatenated dataframe
        - Export concatenated dataframe as CSV file

        Args:
            None

        Returns:
            None
        """
        # Read in configurations from Hydra configuration file
        input_folderpath = self.cfg.data_extractor.input_folder
        output_folderpath = self.cfg.data_extractor.output_folder
        suffix = self.cfg.data_extractor.suffix
        remove_gzip_flag = self.cfg.data_extractor.remove_gzip_files

        # Parse gzip files into concatenated dataframe
        for dir_ in os.listdir(input_folderpath):
            if os.path.isdir(os.path.join(input_folderpath, dir_)):
                logging.info(f"Reading {dir_} Gzip files")
                self._parse_files_into_dataframe(
                    input_folderpath=os.path.join(input_folderpath, dir_),
                    output_folderpath=os.path.join(output_folderpath, dir_),
                    output_suffix=suffix,
                    remove_gzip_flag=remove_gzip_flag,
                )

    def _parse(self, path: str) -> Iterator[Dict[str, Any]]:
        """
        Parse a Gzip-compressed JSON file and yield each JSON record as a
        dictionary.

        Args:
            path (str): The file path to the Gzip-compressed JSON file.

        Yields:
            Iterator[Dict[str, Any]]: An iterator that yields a dictionary
                representing each JSON record in the file.
        """
        try:
            gzip_file = gzip.open(path, "r")
            for line in gzip_file:
                yield json.loads(line)
        except Exception as error:
            logging.info(
                f"An error occurred while parsing file {path}: {error}"
            )
            return

    def _parse_files_into_dataframe(
        self,
        input_folderpath: str,
        output_folderpath: str,
        output_suffix: str,
        remove_gzip_flag: bool,
    ) -> pd.DataFrame:
        """
        Parse multiple gzip files in the specified directory into a
        concatenated DataFrame.

        Args:
            path (str): Filepath of input folder of downloaded gzip files.

        Returns:
            pd.DataFrame: Concatenated dataframe.
        """
        for dir_ in tqdm(
            os.listdir(input_folderpath), desc="Processing files", unit="file"
        ):
            gzip_filepath = os.path.join(input_folderpath, dir_)
            dataframe = self._parse_into_dataframe(gzip_filepath)
            dataframe["state"] = dir_.split(".")[0].split("-")[-1]

            logging.info(f"Read {dir_} into DataFrame")

            # Export to CSV
            output_filename = "_".join([dir_.split(".")[0], output_suffix])
            output_filepath = os.path.join(output_folderpath, output_filename)
            export_to_csv(dataframe=dataframe, filepath=output_filepath)

            if remove_gzip_flag:
                os.remove(gzip_filepath)
                logging.info(f"Removed file - {gzip_filepath}")

    def _parse_into_dataframe(self, path: str) -> pd.DataFrame:
        """
        Parse a Gzip-compressed JSON file or read a Gzip-compressed CSV file
        and convert it into a pandas DataFrame.

        Args:
            path (str): The file path to the Gzip-compressed JSON or CSV file.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the data parsed from
                the Gzip-compressed JSON file, or the data read from the
                Gzip-compressed CSV file.
        """
        if path.endswith(".json.gz"):
            return pd.DataFrame(self._parse(path))

        elif path.endswith(".csv.gz"):
            return pd.read_csv(path, compression="gzip")

        else:
            raise ValueError(
                "Unsupported file format. Only supports Gzip-compressed files."
            )


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run_standalone(cfg: DictConfig) -> None:
    """
    Initialize Hydra configuration and run standalone DataExtractor class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        None
    """
    logging.info(run(cfg))


def run(cfg: DictConfig) -> str:
    """
    Pass in Hydra configuration and run DataExtractor class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        str: Status of DataDownloader class.
    """
    extractor = DataExtractor(cfg)
    extractor.extract_data()

    return "Complete data extraction"


if __name__ == "__main__":
    run_standalone()
