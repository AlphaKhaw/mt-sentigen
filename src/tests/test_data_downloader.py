import os
import sys

import pytest
from bs4 import BeautifulSoup
from hydra import compose, initialize
from omegaconf import DictConfig

project_root_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, project_root_dir)

from data_processing.data_downloader import DataDownloader


@pytest.fixture
def config() -> DictConfig:
    """
    Fixture to provide a sample configuration for testing.

    Returns:
        DictConfig: A sample configuration.
    """
    with initialize(
        config_path="../../conf/base",
        job_name="test_data_downloader",
        version_base="1.1",
    ):
        cfg = compose(config_name="test_pipelines.yaml")
    return cfg


@pytest.fixture
def data_downloader(config: DictConfig) -> DataDownloader:
    """
    Fixture to provide a DataDownloader instance for testing.

    Args:
        config (DictConfig): The sample configuration.

    Returns:
        DataDownloader: An instance of the DataDownloader class.
    """
    return DataDownloader(config)


def test_create_folders(data_downloader: DataDownloader) -> None:
    """
    Test the _create_folders method of the DataDownloader class.

    Args:
        data_downloader (DataDownloader): An instance of the DataDownloader
            class.

    Returns:
        None
    """
    folderpath = "test_folder"
    data_downloader._create_folders(folderpath)
    assert os.path.exists(folderpath)
    os.rmdir(folderpath)


@pytest.mark.parametrize("use_metadata", [True, False])
def test_extract_download_links(
    data_downloader: DataDownloader, use_metadata: bool
) -> None:
    """
    Test the _extract_download_links method of the DataDownloader class.

    Args:
        data_downloader (DataDownloader): An instance of the DataDownloader
            class.
        use_metadata (bool): Flag indicating whether to use metadata.

    Returns:
        None
    """
    data_table = BeautifulSoup(
        "<table><td><a href='review_link'></a></td></table>", "html.parser"
    )
    links = data_downloader._extract_download_links(use_metadata, data_table)
    if use_metadata:
        assert "review_link" in links
    else:
        assert "review_link" in links and "meta_link" not in links
