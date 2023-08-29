import logging
import os
import sys

import hydra
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from utils.dataframe.dataframe_utils import read_from_csv
from utils.enums.enums import DatasetType

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class CustomDataSet(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        review_column: str,
        rating_column: str,
        response_column: str,
        encoder_model: str,
        decoder_model: str,
    ) -> None:
        self.dataframe = dataframe
        self.review_column = review_column
        self.rating_column = rating_column
        self.response_column = response_column
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        review = self.dataframe.loc[idx, self.review_column]
        rating = self.dataframe.loc[idx, self.rating_column]
        response = self.dataframe.loc[idx, self.response_column]

        input_ids = self.encoder_tokenizer(review, return_tensors="pt")[
            "input_ids"
        ]
        attention_mask = self.encoder_tokenizer(review, return_tensors="pt")[
            "attention_mask"
        ]

        decoder_input_ids = self.decoder_tokenizer(
            response, return_tensors="pt"
        )["input_ids"]
        decoder_attention_mask = self.decoder_tokenizer(
            response, return_tensors="pt"
        )["attention_mask"]

        return {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "decoder_input_ids": decoder_input_ids.squeeze(),
            "decoder_attention_mask": decoder_attention_mask.squeeze(),
            "rating": rating,
        }


class DataPreparation:
    """
    DataPreparation class to load data into DataLoaders for modelling.
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

    def get_dataloaders(self, dataset: DatasetType) -> tuple[DataLoader]:
        """
        Main function to retrieve data loaders for
        training, validation, and testing.

        Returns:
            tuple: A tuple containing train, validation, and test dataloaders.
        """
        # Read in configurations from Hydra configuration file
        review_column = self.cfg.general.review_column
        rating_column = self.cfg.general.rating_column
        response_column = self.cfg.general.response_column
        input_folderpath = self.cfg.dataloader.input_folder
        batch_size = self.cfg.dataloader.batch_size
        encoder_model = self.cfg.dataloader.encoder_model
        decoder_model = self.cfg.dataloader.decoder_model

        # Construct DataPath
        dataframes = self._read_in_csv(input_folderpath, dataset)

        # Convert to model format
        datasets = self._convert_to_format(
            dataframes,
            [review_column, rating_column, response_column],
            encoder_model,
            decoder_model,
        )
        logging.info("Convert DataFrame to model format")

        return self._create_dataloaders(datasets, batch_size)

    def _convert_to_format(
        self,
        dataframes: tuple,
        columns: list,
        encoder_model: str,
        decoder_model: str,
    ) -> tuple[CustomDataSet]:
        """
        Convert data frames into the desired format for model input.

        Args:
            dataframes (tuple[pd.DataFrame]): A tuple of data frames
                containing train, validation, and test data.
            columns (list): A list of column names.
            encoder_model (str): Name of encoder model.
            decoder_model (str): Name of decoder model.

        Returns:
            tuple[CustomDataSet]: A tuple of CustomDataSet class instances.
        """
        train_data, val_data, test_data = dataframes
        review_column, rating_column, response_column = columns

        datasets = [
            CustomDataSet(
                train_data,
                review_column,
                rating_column,
                response_column,
                encoder_model,
                decoder_model,
            )
            for data in dataframes
        ]

        return datasets

    def _create_dataloaders(
        self, datasets: tuple, batch_size: int
    ) -> tuple[DataLoader]:
        """
        Create data loaders from the provided datasets.

        Args:
            datasets (tuple): A tuple of datasets.
            batch_size (int): Batch size for the data loaders.

        Returns:
            tuple: A tuple of data loaders containing
                train, validation, and test loaders.
        """
        train_data, val_data, test_data = datasets
        train_dataloader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False
        )
        test_dataloader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False
        )

        return train_dataloader, val_dataloader, test_dataloader

    def _read_in_csv(
        self, input_folderpath: str, dataset: DatasetType
    ) -> tuple[pd.DataFrame]:
        """
        Read data from CSV files and return data frames.

        Args:
            input_folderpath (str): Path to the input folder containing
                data files.
            dataset (Dataset): An object containing dataset file names.

        Returns:
            tuple: A tuple of data frames for train, validation, and test data.
        """
        train_datapath = os.path.join(input_folderpath, dataset.TRAIN.value)
        val_datapath = os.path.join(input_folderpath, dataset.VALIDATION.value)
        test_datapath = os.path.join(input_folderpath, dataset.TEST.value)

        train_data, val_data, test_data = (
            read_from_csv(train_datapath),
            read_from_csv(val_datapath),
            read_from_csv(test_datapath),
        )

        return train_data, val_data, test_data


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run_standalone(cfg: DictConfig) -> None:
    """
    Initialize Hydra configuration and run standalone DataPreparation class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        None
    """
    logging.info(run(cfg))


def run(cfg: DictConfig) -> tuple[DataLoader]:
    """
    Pass in Hydra configuration and run DataPreparation class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        tuple: A tuple containing train, validation, and test dataloaders.
    """
    custom = DataPreparation(cfg)
    train_dataloader, val_dataloader, test_dataloader = custom.get_dataloaders(
        DatasetType
    )
    logging.info("Returning DataLoaders")

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    run_standalone()
