import logging
import os
import sys
from typing import List

import hydra
import pandas as pd
import torch
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
    ) -> None:
        self.dataframe = dataframe
        self.review_column = review_column
        self.rating_column = rating_column
        self.response_column = response_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple:
        review = self.dataframe.loc[idx, self.review_column]
        review = f"generate response: {review}"
        rating = self.dataframe.loc[idx, self.rating_column]
        response = self.dataframe.loc[idx, self.response_column]

        return review, float(rating), response


class DataPreparation:
    """
    DataPreparation class to load data into DataLoaders for modelling.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize a DataPreparation object and initialise database.

        Args:
            cfg (DictConfig): Hydra configuration.

        Returns:
            None
        """
        self.cfg = cfg

    def collate_fn(self, batch: List[tuple]) -> tuple[torch.Tensor]:
        """
        Custom collate function to process a batch of data in the DataLoader.

        Args:
            batch : List[tuple]
                A list of tuples each containing:
                - input_text (str): The input text for the encoder.
                - label (int): The sentiment rating.
                - response_text (str): The target text for the decoder.

        Returns:
            Tuple[torch.Tensor]:
                Returns a tuple containing:
                - input_ids (torch.Tensor): Padded and encoded input text ids.
                - attention_masks (torch.Tensor): Attention masks for input
                    text.
                - sentiment_labels (torch.Tensor): Tensor containing sentiment
                    ratings.
                - response_labels (torch.Tensor): Padded and encoded response
                    text ids.
                - response_attention_mask (torch.Tensor): Attention masks for
                    response text.
        """
        input_texts, labels, response_texts = zip(*batch)
        max_length = self.cfg.dataloader.max_length
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.dataloader.encoder_decoder_model
        )

        # Input Text
        input_encodings = tokenizer(
            input_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = input_encodings["input_ids"]
        attention_masks = input_encodings["attention_mask"]

        # Sentiment Rating
        sentiment_labels = torch.tensor(labels, dtype=torch.long)

        # Response Text
        response_encodings = tokenizer(
            response_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        response_labels = response_encodings["input_ids"]
        response_attention_mask = response_encodings["attention_mask"]

        return (
            input_ids,
            attention_masks,
            sentiment_labels,
            response_labels,
            response_attention_mask,
        )

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

        # Construct DataPath
        dataframes = self._read_in_csv(input_folderpath, dataset)

        # Preprocess
        dataframes = self._preprocess_text_columns(
            dataframes, review_column, response_column
        )

        # Convert to model format
        datasets = self._convert_to_format(
            dataframes,
            [review_column, rating_column, response_column],
        )
        logging.info("Convert DataFrame to model format")

        return self._create_dataloaders(datasets, batch_size)

    def _convert_to_format(
        self,
        dataframes: tuple,
        columns: list,
    ) -> tuple[CustomDataSet]:
        """
        Convert data frames into the desired format for model input.

        Args:
            dataframes (tuple[pd.DataFrame]): A tuple of data frames
                containing train, validation, and test data.
            columns (list): A list of column names.

        Returns:
            tuple[CustomDataSet]: A tuple of CustomDataSet class instances.
        """
        review_column, rating_column, response_column = columns

        datasets = [
            CustomDataSet(
                data[:10000],
                review_column,
                rating_column,
                response_column,
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
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        val_dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
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

    def _preprocess_text_columns(
        self,
        dataframes: tuple[pd.DataFrame],
        review_column: str,
        response_column: str,
    ) -> tuple[pd.DataFrame]:
        for dataframe in dataframes:
            dataframe[review_column] = dataframe[review_column].astype(str)
            dataframe[response_column] = dataframe[response_column].astype(str)
            dataframe = dataframe[
                (dataframe[review_column] != "nan")
                & (dataframe[response_column] != "nan")
            ]

        return dataframes


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
    dataloaders = custom.get_dataloaders(DatasetType)
    logging.info("Returning DataLoaders")

    return dataloaders


if __name__ == "__main__":
    run_standalone()
