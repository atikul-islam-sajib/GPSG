import argparse
import logging
import joblib
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/dataset.log",
)


class Loader:
    """
    This script facilitates the downloading and loading of the MNIST dataset for machine learning tasks.

    The script uses argparse to accept command-line arguments for batch size and a flag to trigger the download of the MNIST dataset. It defines a `Loader` class responsible for downloading the MNIST dataset and creating a DataLoader object from it. The DataLoader object is then serialized and saved using joblib.

    The MNIST dataset is a collection of 28x28 pixel grayscale images of handwritten digits (0-9), commonly used for training and testing in the field of machine learning.

    Command Line Arguments:
    - --batch_size (int): The batch size for the DataLoader. Default is 64.
    - --download_mnist (flag): Flag to trigger the download of the MNIST dataset.

    Features:
    - Downloading the MNIST dataset from the torchvision package.
    - Applying necessary transformations to the dataset.
    - Creating a DataLoader to facilitate batch processing during training.
    - Serializing and saving the DataLoader object for future use.

    Examples:
    python script.py --batch_size 64 --download_mnist
    python script.py --download_mnist

    Note:
    - The script uses the 'logging' module for logging information and errors.
    - The MNIST dataset is stored in './data/raw/' and the DataLoader object in './data/processed/'.
    - The script is intended to be executed in an environment where Python packages like 'argparse', 'logging', 'joblib', 'torch', and 'torchvision' are installed.
    """

    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def download_dataset(self):
        logging.info("Downloading dataset...".capitalize())

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        mnist_data = datasets.MNIST(
            root="./data/raw/",
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )
        return mnist_data

    def create_dataloader(self, mnist_data):
        logging.info("Creating dataloader...".capitalize())

        dataloader = DataLoader(mnist_data, batch_size=self.batch_size, shuffle=True)

        joblib.dump(
            value=dataloader,
            filename="./data/processed/dataloader.pkl",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset download")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for dataloader",
        required=True,
    )
    parser.add_argument(
        "--download_mnist", action="store_true", help="Download MNIST dataset"
    )

    args = parser.parse_args()
    if args.download_mnist:
        if args.batch_size:
            logging.info("Batch size: %d".capitalize(), args.batch_size)
            loader = Loader(batch_size=args.batch_size)

            logging.info("Downloading dataset...".capitalize())
            loader.download_dataset()
            loader.create_dataloader(loader.download_dataset())
        else:
            logging.error("Batch size is required".capitalize())
    else:
        logging.error("Download MNIST dataset flag is required".capitalize())
