import gzip
import json
import logging
from typing import Any, Dict, Iterator

import hydra
import pandas as pd
from omegaconf import DictConfig

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class DataExtractor:
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
        - Read in gzip files

        Args:
            None

        Returns:
            None
        """
        # Read in configurations from Hydra configuration file
        input_folder = self.cfg.data_extractor.input_folder
        output_folder = self.cfg.data_extractor.output_folder
        remove_gzip_flag = self.cfg.data_extractor.remove_gzip_files

        return

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
        except Exception as e:
            logging.info(f"An error occurred while parsing file {path}: {e}")
            return

    def _parse_into_dataframe(path: str) -> pd.DataFrame:
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
