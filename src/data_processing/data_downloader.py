import logging
import os

import bs4
import hydra
import requests
from bs4 import BeautifulSoup
from omegaconf import DictConfig
from tqdm import tqdm

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class DataDownloader:
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize a DataDownloader object.

        Args:
            cfg (DictConfig): hydra configuration

        Returns:
            None
        """
        self.cfg = cfg

    def download_data(self) -> None:
        """
        Main function of DataDownloader class to:
        - Read in configurations
        - Extracts data from a specified URL
        - Downloads and store relevant files in intermediate gzip folder
        - Parse gzip files and store in output folder

        Args:
            None

        Returns:
            None
        """
        # Read in configurations from Hydra configuration file
        url = self.cfg.data_downloader.base_url
        use_subset_flag = self.cfg.data_downloader.use_subset
        use_metadata_flag = self.cfg.data_downloader.use_metadata
        output_folderpath = self.cfg.data_downloader.output_folder

        # Send a GET request to the specified URL to retrieve webpage content
        response = requests.get(url)

        # Create a BeautifulSoup object to parse the HTML content of webpage
        soup = BeautifulSoup(response.content, "html.parser")

        # Read in complete or subset data table
        data_table = self._read_in_data_table(
            use_subset=use_subset_flag, soup=soup
        )
        logging.info("Read in data table from webpage")

        # Extract download links
        review_links = self._extract_download_links(
            use_metadata=use_metadata_flag, data_table=data_table
        )
        logging.info("Extracted download links")

        # Download files
        self._download_files(
            links=review_links,
            datapath=output_folderpath,
            use_metadata=use_metadata_flag,
        )

    def _create_folders(self, folderpath: str) -> None:
        """
        Checks if specified folderpath exists and creates a folder directory
        if it does not exist.

        Args:
            folderpath (str): The path to the folder directory to be created.

        Returns:
            None
        """
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
            logging.info(f"Created folder directory: {folderpath}")

    def _download_file(self, url: str, folderpath: str) -> None:
        """
        Download a file from specified URL and save it to the specified
        datapath.

        Args:
            url (str): URL of file.
            folderpath (str): Directory path where the downloaded file will be
                saved.

        Returns:
            None
        """
        filepath = os.path.join(folderpath, url.split("/")[-1])
        if os.path.exists(filepath):
            logging.info(f"File already exits: {filepath}")
            return

        res = requests.get(url)
        if res.status_code == 200:
            with open(filepath, "wb") as file:
                file.write(res.content)
            logging.info(f"Downloaded and saved to: {filepath}")
        else:
            logging.info(f"Failed to download the file from: {url}")

    def _download_files(
        self, links: list, datapath: str, use_metadata: bool
    ) -> None:
        """
        Downloads files from the provided links and stores them in the
        specified data path.

        Args:
            links (list): A list of download links.
            datapath (str): The path to the intermediate folder where gzip
                files will be stored.
            use_metadata (bool): If True, downloads metadata files in addition
                to review files.

        Returns:
            None
        """
        # Create gzip folderpath
        self._create_folders(folderpath=datapath)

        # Download reviews
        review_folderpath = os.path.join(datapath, "review")
        self._create_folders(folderpath=review_folderpath)
        review_links = [link for link in links if "review" in link]
        for link in tqdm(review_links, desc="Downloading reviews"):
            self._download_file(url=link, folderpath=review_folderpath)

        # Download metadata
        if use_metadata:
            metadata_links = [link for link in links if "meta" in link]
            metadata_folderpath = os.path.join(datapath, "metadata")
            self._create_folders(folderpath=metadata_folderpath)
            for link in tqdm(metadata_links, desc="Downloading metadata"):
                self._download_file(url=link, folderpath=metadata_folderpath)

    def _extract_download_links(
        self, use_metadata: bool, data_table: bs4.element.Tag
    ) -> list:
        """
        Extracts download links from the data table based on the specified
        metadata flag.

        Args:
            use_metadata (bool): If True, extracts both review and metadata
                links, otherwise only review links.
            data_table (bs4.element.Tag): The data table as a BeautifulSoup Tag
                object.

        Returns:
            list: A list of download links.
        """
        rows = data_table.findAll(lambda tag: tag.name == "td")
        href_links = [row.find("a")["href"] for row in rows if row.find("a")]
        review_links = [link for link in href_links if "review" in link]

        if use_metadata:
            meta_links = [link for link in href_links if "meta" in link]
            review_links.extend(meta_links)

        return review_links

    def _read_in_data_table(
        self, use_subset: bool, soup: bs4.BeautifulSoup
    ) -> bs4.element.Tag:
        """
        Reads in the data table from a BeautifulSoup object based on the
        specified subset flag.

        Args:
            use_subset (bool): If True, reads in a subset data table,
                otherwise reads in the complete data table.
            soup (bs4.BeautifulSoup): The BeautifulSoup object containing the
                HTML content.

        Returns:
            bs4.element.Tag: The data table as a BeautifulSoup Tag object.
        """
        if use_subset:
            data_table = soup.find(
                lambda tag: tag.name == "table"
                and "10-core" in tag.findChildren("td")[1].get_text()
            )
        else:
            data_table = soup.find(
                lambda tag: tag.name == "table"
                and "reviews" in tag.findChildren("td")[1].get_text()
            )

        return data_table


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run_standalone(cfg: DictConfig) -> None:
    """
    Initialize Hydra configuration and run standalone DataDownloader class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        None
    """
    logging.info(run(cfg))


def run(cfg: DictConfig) -> str:
    """
    Pass in Hydra configuration and run DataDownloader class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        str: Status of DataDownloader class.
    """
    extractor = DataDownloader(cfg)
    extractor.download_data()

    return "Complete download of data"


if __name__ == "__main__":
    run_standalone()
