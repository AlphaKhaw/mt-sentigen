# import gzip
# import json
# import logging
# from typing import Any, Dict, Iterator

# import hydra
# import pandas as pd
# from omegaconf import DictConfig

# logging.warnings.filterwarnings("ignore")
# logging.basicConfig(level=logging.INFO)


# @hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
# def run_standalone(cfg: DictConfig) -> None:
#     """
#     Initialize Hydra configuration and run standalone DataExtractor class.

#     Args:
#         cfg (DictConfig): Hydra configuration.

#     Returns:
#         None
#     """
#     logging.info(run(cfg))


# def run(cfg: DictConfig) -> str:
#     """
#     Pass in Hydra configuration and run DataExtractor class.

#     Args:
#         cfg (DictConfig): Hydra configuration.

#     Returns:
#         str: Status of DataDownloader class.
#     """
#     extractor = DataExtractor(cfg)
#     extractor.extract_data()

#     return "Complete data extraction"


# if __name__ == "__main__":
#     run_standalone()
