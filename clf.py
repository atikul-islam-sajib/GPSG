import sys
import logging
import argparse

sys.path.append("./src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/clf.log",
)
from trainer import Trainer
from data_loader import Loader


def clf_code():
    """
    Facilitates the configuration and initiation of the training process for a Generative Adversarial Network (GAN) using command-line arguments.

    This function uses the argparse library to parse command-line arguments that specify training parameters and controls, such as batch size, learning rate, number of epochs, latent space dimension, and a flag to download the MNIST dataset.

    The MNIST dataset, a collection of handwritten digits, is a standard benchmark in machine learning for image processing tasks. The function sets up a DataLoader for this dataset and initializes a Trainer class to train the GAN.

    Command-Line Arguments:
    - --batch_size (int): Batch size for the DataLoader. Default is 64.
    - --download_mnist (flag): Triggers the download of the MNIST dataset.
    - --lr (float): Learning rate for the GAN's training process. Default is 0.0002.
    - --epochs (int): Number of epochs for training the GAN. Default is 100.
    - --latent_space (int): Dimensionality of the latent space for the GAN's generator. Default is 100.

    Usage:
    Execute the script with the required arguments to initiate the GAN training process. For example:
        python script.py --batch_size 64 --download_mnist --lr 0.0002 --epochs 100 --latent_space 100

    Functionality:
    - Parses and validates command-line arguments.
    - Downloads and prepares the MNIST dataset for training.
    - Initializes and starts the GAN training using the specified parameters.
    - Logs the process and any errors or exceptions encountered.

    Note:
    - The script assumes the presence of 'Loader' and 'Trainer' classes in the '../GPSG/src' directory.
    - Logging is configured to capture the flow and any issues, outputting to a file './logs/clf.log'.
    - This function is intended to be the main entry point of a Python script designed for GAN training.
    """
    parser = argparse.ArgumentParser(description="Command line coding".title())
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
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate".capitalize()
    )
    parser.add_argument("--epochs", type=int, default=100, help="Epochs".capitalize())
    parser.add_argument(
        "--latent_space", type=int, default=100, help="Latent space".capitalize()
    )

    args = parser.parse_args()

    if args.download_mnist:
        if args.batch_size and args.lr and args.epochs and args.latent_space:
            logging.info("Downloading MNIST dataset")

            loader = Loader(batch_size=args.batch_size)
            loader.download_dataset()
            loader.create_dataloader(loader.download_dataset())

            logging.info("Start training".capitalize())

            trainer = Trainer(
                lr=args.lr, epochs=args.epochs, latent_space=args.latent_space
            )
            trainer.train_simple_gan()

            logging.info("Training complete".capitalize())

        else:
            logging.exception("Please provide all the required arguments")

    else:
        logging.exception("Please provide --download_mnist flag")


if __name__ == "__main__":
    clf_code()
