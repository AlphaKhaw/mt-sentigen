import os
import sys

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

project_root_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, project_root_dir)

from data_processing.data_extractor import DataExtractor


@pytest.fixture
def config() -> DictConfig:
    """
    Fixture to provide a sample configuration for testing.

    Returns:
        DictConfig: A sample configuration.
    """
    with initialize(
        config_path="../../conf/base",
        job_name="test_data_extractor",
        version_base="1.1",
    ):
        cfg = compose(config_name="test_pipelines.yaml")
    return cfg


@pytest.fixture
def data_extractor(config: DictConfig) -> DataExtractor:
    """
    Fixture to provide a DataExtractor instance for testing.

    Args:
        config (DictConfig): The sample configuration.

    Returns:
        DataExtractor: An instance of the DataExtractor class.
    """
    return DataExtractor(config)


def test_parse_and_extract(
    data_extractor: DataExtractor, config: DictConfig
) -> None:
    # Read in test configurations
    input_filename = config.data_extractor.input_filename
    expected_filename = config.data_extractor.expected_filename
    key_columns = config.data_extractor.key_columns
    output_suffix = config.data_extractor.suffix
    file_identifier = "mock_data"
    
    # Read in OS Environment Variables
    input_folderpath = os.getenv("MOCK_DATAPATH")
    print(input_folderpath)
    
    # Construct input filepath
    input_filepath = os.path.join(input_folderpath, input_filename)
    output_folderpath = input_folderpath
    
    # Construct expected filepath
    expected_filepath = os.path.join(output_folderpath, expected_filename)

    # Call test function
    data_extractor._parse_and_extract(
        input_filepath,
        output_folderpath,
        output_suffix,
        file_identifier,
        key_columns,
    )

    # Construct output filepath
    output_filename = f"{file_identifier}_{output_suffix}"
    output_filepath = os.path.join(output_folderpath, output_filename)

    # Read the content of the output file
    with open(output_filepath, "r") as f:
        output_contents = f.read()

    # Read the content of expected file
    with open(expected_filepath, "r") as f:
        expected_contents = f.read()

    # Assert a specific value in the output file
    assert expected_contents in output_contents
    os.remove(output_filepath)
