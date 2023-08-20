import logging
import os
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from utils.dataframe.dataframe_utils import export_dataframe

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class DataSplitter:
    """
    DataSplitter class to split processed data into
    train, validation and test set.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize a DataSplitter object and initialise database.

        Args:
            cfg (DictConfig): Hydra configuration.

        Returns:
            None
        """
        self.cfg = cfg

    def split_data(self) -> None:
        """
        Main function of DataSplitter.

        Args:
            None

        Returns:
            None
        """
        # Read in configurations from Hydra configuration file
        input_filepath = self.cfg.data_splitter.input_filepath
        output_folderpath = self.cfg.data_splitter.output_folder
        suffix = self.cfg.data_splitter.suffix
        train_ratio = self.cfg.data_splitter.train_ratio
        val_ratio = self.cfg.data_splitter.val_ratio
        test_ratio = self.cfg.data_splitter.test_ratio
        seed = self.cfg.data_splitter.seed
        tolerance = self.cfg.data_splitter.split_tolerance
        rating_column = self.cfg.general.rating_column

        # Check input ratios
        self._check_input_ratio(train_ratio, val_ratio, test_ratio)

        # Read in processed file
        if input_filepath.endswith(".csv"):
            dataframe = pd.read_csv(input_filepath)
            logging.info("Read CSV file into dataframe")

        elif input_filepath.endswith(".parquet"):
            dataframe = pd.read_parquet(input_filepath)
            logging.info("Read Parquet file into dataframe")

        # Split dataframe into train, validation and test set
        split_data = self._train_val_test_split(
            dataframe,
            rating_column,
            [train_ratio, val_ratio, test_ratio],
            seed,
        )
        logging.info("Split data into train, validation and test set")

        # Check split data distribution
        self._check_split_distribution(
            split_data, [train_ratio, val_ratio, test_ratio], tolerance
        )

        # Export train, validation and test set
        for data, split in zip(split_data, suffix):
            output_filepath = os.path.join(output_folderpath, split)
            export_dataframe(dataframe=data, output_filepath=output_filepath)

    def _check_split_distribution(
        self,
        split_datasets: list,
        ratios: list[float],
        tolerance: float,
    ) -> None:
        """
        Check distribution of split datasets against input ratios.
        1. Compute total number of samples in train, validation and test set.
        2. Compute expected count based on user input ratios.
        3. Compute and log actual percentage of train, validation and test set.
        4. Compute ratio difference and compare each ratio to set tolerance.
            - Log successful message if all ratios fall under set tolerance
            - Raise ValueError if any of the ratios are above set tolerance

        Args:
            split_datasets (list): A list of split datasets.
            ratios (list[float]): List of train, valiation and test ratios.
            tolerance (float, optional): Tolerance value to allow deviations
                from the ratios.

        Raises:
            ValueError: If the distribution of data splits does not match the
                input ratios.

        Returns:
            None
        """
        total_samples = sum(len(dataset) for dataset in split_datasets)
        expected_counts = [int(total_samples * ratio) for ratio in ratios]

        percentage_counts = {
            split: round((len(data) / total_samples) * 100, 2)
            for data, split in zip(
                split_datasets, ["train", "validation", "test"]
            )
        }
        logging.info(f"Percentage count: {percentage_counts}")

        ratio_diffs = [
            abs((expected_count - len(data)) / total_samples)
            for data, expected_count in zip(split_datasets, expected_counts)
        ]

        if all(ratio <= tolerance for ratio in ratio_diffs):
            logging.info("Distribution of data splits matches input ratios")
        else:
            raise ValueError(
                "Distribution of data splits does not match input ratios"
            )

    def _check_input_ratio(
        self, train_ratio: float, val_ratio: float, test_ratio: float
    ) -> None:
        """
        Check validity of input train, validation and test ratios.

        Args:
            train_ratio (float): Ratio of training data.
            val_ratio (float): Ratio of validation data.
            test_ratio (float): Ratio of test data.

        Raises:
            ValueError: If val_ratio equals 0.0 or 1.0.
            ValueError: If sum of train_ratio and test_ratio does not add up
                to 1.

        Returns:
            None
        """
        if not 0.0 < val_ratio < 1.0:
            raise ValueError(
                "val_ratio should be a floating point - 0 < val_ratio < 1"
            )
        total_ratio = sum([train_ratio, val_ratio, test_ratio])

        if not np.isclose(total_ratio, 1.0, atol=1e-9):
            raise ValueError(
                "Sum of train_ratio, val_ratio and test_ratio must equal to 1"
            )

    def _train_val_test_split(
        self,
        dataframe: pd.DataFrame,
        stratify_column: str,
        ratios: list[str],
        seed: int,
    ) -> tuple[pd.DataFrame]:
        """
        Stratify dataset into Train, Validation and Test set based on input
        stratify column.

        Args:
            dataframe (pd.DataFrame): Input DataFrame with bucket column.
            stratify_column (str): The column to stratify the dataset on.
            ratios (list[float]): List of train, valiation and test ratios.
            seed (int): Seed value for random number generation.

        Returns:
            tuple[pd.DataFrame]: A tuple containing train, validation and
                test DataFrames.
        """
        train_ratio, val_ratio, test_ratio = ratios
        train_val_data, test_data = train_test_split(
            dataframe,
            test_size=test_ratio,
            random_state=seed,
            stratify=dataframe[stratify_column],
        )
        adjusted_val_ratio = round(val_ratio / sum([train_ratio, val_ratio]), 2)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=adjusted_val_ratio,
            random_state=seed,
            stratify=train_val_data[stratify_column],
        )

        return (train_data, val_data, test_data)


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run_standalone(cfg: DictConfig) -> None:
    """
    Initialize Hydra configuration and run standalone DataSplitter class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        None
    """
    logging.info(run(cfg))


def run(cfg: DictConfig) -> str:
    """
    Pass in Hydra configuration and run DataSplitter class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        str: Status of DataSplitter class.
    """
    processor = DataSplitter(cfg)
    processor.split_data()

    return "Complete data splitting"


if __name__ == "__main__":
    run_standalone()
