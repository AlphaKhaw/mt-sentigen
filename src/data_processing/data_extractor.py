import gzip
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import hydra
import ujson
from omegaconf import DictConfig

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class DataExtractor:
    """
    DataExtractor class to extract JSON/CSV files from downloaded gzip files.
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

    def extract_data(self) -> None:
        """
        Main function of DataExtractor class to:
        - Read in configurations
        - Parse and write gzip files into json format

        Args:
            None

        Returns:
            None
        """
        # Read in configurations from Hydra configuration file
        input_folderpath = self.cfg.data_extractor.input_folder
        output_folder = self.cfg.data_extractor.output_folder
        output_suffix = self.cfg.data_extractor.suffix
        key_columns = self.cfg.data_extractor.key_columns
        batch_size = self.cfg.data_extractor.batch_size
        remove_gzip_flag = self.cfg.data_extractor.remove_gzip_files

        # Parse gzip files into database
        for dir_ in os.listdir(input_folderpath):
            if os.path.isdir(os.path.join(input_folderpath, dir_)):
                logging.info(f"Reading {dir_} Gzip files")
                self._parse_files_into_json(
                    input_folderpath=os.path.join(input_folderpath, dir_),
                    output_folderpath=os.path.join(output_folder, dir_),
                    output_suffix=output_suffix,
                    key_columns=key_columns,
                    batch_size=batch_size,
                    remove_gzip_flag=remove_gzip_flag,
                )

    def _parse_and_extract(
        self,
        path: str,
        output_folderpath: str,
        output_suffix: str,
        file_identifier: str,
        key_columns: list,
    ) -> None:
        """
        Parses a Gzip-compressed JSON file, extracts specified key columns,
        and writes to a JSON file.

        Args:
            path (str): Path to the Gzip-compressed JSON file.
            output_filepath (str): Path to the output JSON file.
            output_suffix (str): Desired suffix of the extracted JSON file.
            file_identifier (str): Name of file to parse and extract.
            key_columns (list): List of column names to extract.

        Returns:
            None
        """
        try:
            with gzip.open(path, "rt") as gzip_file:
                json_data = [ujson.loads(line) for line in gzip_file]

            file = path.split("/")[-1]
            logging.info(f"Parse {file} into JSON format")

            # Extract key columns
            extracted_data = [
                {col: json_obj.get(col) for col in key_columns}
                for json_obj in json_data
            ]
            logging.info("Extracted key-value pairs")

            # Construct output filepath
            output_filename = f"{file_identifier}_{output_suffix}"
            output_filepath = os.path.join(output_folderpath, output_filename)

            # Write JSON incrementally using a buffered writer
            with io.BufferedWriter(
                io.FileIO(output_filepath, "w")
            ) as output_file:
                for entry in extracted_data:
                    json_str = ujson.dumps(entry) + "\n"
                    output_file.write(json_str.encode("utf-8"))

            logging.info(f"JSON data written to {output_filepath}")

            return extracted_data

        except Exception as error:
            logging.info(
                f"An error occurred while parsing file {path}: {error}"
            )

    def _parse_files_into_json(
        self,
        input_folderpath: str,
        output_folderpath: str,
        output_suffix: str,
        key_columns: list,
        batch_size: int,
        remove_gzip_flag: bool,
    ) -> None:
        """
        Parses multiple gzip files in the specified directory,
        extracts key data, and writes JSON files.

        Args:
            input_folderpath (str): Path to input folder of downloaded gzip
                files.
            output_folderpath (str): Path to output folder for JSON files.
            output_suffix (str): Suffix for output JSON files.
            key_columns (list): List of column names to extract.
            batch_size (int): Number of files to process concurrently in
                each batch.
            remove_gzip_flag (bool): Flag to remove gzip files after
                processing.

        Returns:
            None
        """
        logging.info("Checking for existence of file")

        files_to_process = [
            (
                os.path.join(input_folderpath, dir_),
                dir_.split(".")[0].split("-")[-1],
            )
            for dir_ in os.listdir(input_folderpath)
        ]
        if not os.path.exists(output_folderpath):
            os.mkdir(output_folderpath)
        extracted_files = os.listdir(output_folderpath)
        document_existence = [
            f"{file_identifier}_{output_suffix}" in extracted_files
            for _, file_identifier in files_to_process
        ]
        pairs_to_process = [
            (gzip_filepath, file_identifier)
            for (gzip_filepath, file_identifier), document_exists in zip(
                files_to_process, document_existence
            )
            if not document_exists
        ]

        logging.info("Check completed")

        with ThreadPoolExecutor(max_workers=(os.cpu_count() // 2)) as executor:
            for batch_start in range(0, len(pairs_to_process), batch_size):
                batch = pairs_to_process[batch_start : batch_start + batch_size]
                futures = [
                    executor.submit(
                        self._parse_and_extract,
                        gzip_filepath,
                        output_folderpath,
                        output_suffix,
                        file_identifier,
                        key_columns,
                    )
                    for gzip_filepath, file_identifier in batch
                ]

                for future in futures:
                    future.result()

                if remove_gzip_flag:
                    for gzip_filepath, _ in batch:
                        os.remove(gzip_filepath)
                        logging.info(f"Removed file - {gzip_filepath}")


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
