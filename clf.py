import sys
import logging
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

sys.path.append("./src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/clf.log",
)
from trainer import Trainer
from data_loader import Loader
from generator import Generator


def get_the_best_model():
    """
    Loads and returns the latest trained generator model from a specified directory of checkpoints.

    This function identifies the latest model checkpoint based on the naming convention and the number of
    files in the './models/checkpoints/' directory. It assumes that the checkpoints are saved in the format
    'generator_{model_number}.pth', where '{model_number}' is an incrementing integer. The function loads
    the state dictionary of the latest generator model into a new Generator instance and sets it to
    evaluation mode.

    Returns:
    - generator (Generator): An instance of the Generator class with loaded weights from the latest checkpoint.

    Note:
    - The function requires the 'Generator' class to be defined and accessible within its scope.
    - The directory path './models/checkpoints/' is hardcoded and should contain the saved model checkpoints.
    - PyTorch's 'torch.load' method is used to load the state dictionary from the checkpoint file.
    - The function assumes that the latest model is the best model. If a different model selection strategy
    is required (like based on validation performance), the function needs to be modified accordingly.
    """
    best_model_num = len(os.listdir("./models/checkpoints/")) - 1
    generator = Generator()

    state_dict = torch.load(
        "./models/checkpoints//generator_{}.pth".format(best_model_num)
    )
    generator.load_state_dict(state_dict)
    generator.eval()

    return generator


def plot_images(images, num_rows, num_cols):
    """
    Plots a grid of images using Matplotlib.

    This function takes a list of image tensors and displays them in a specified grid format. Each image is
    plotted in a subplot without axes. The function is designed to handle image tensors directly from a
    PyTorch model's output.

    Parameters:
    - images (list of torch.Tensor): A list of image tensors to be plotted.
    - num_rows (int): The number of rows in the grid.
    - num_cols (int): The number of columns in the grid.

    The images are assumed to be in a format compatible with Matplotlib's imshow function (e.g., numpy arrays).
    If the images are PyTorch tensors, they are converted to numpy arrays and detached from the current
    computation graph. This function is useful for visualizing the outputs of a generative model.

    Note:
    - The function requires Matplotlib for plotting.
    - The images should be in a format that can be directly used by 'plt.imshow' (like numpy arrays). If the images
      are tensors, they are converted within the function.
    - The figsize is set to (64, 6), which might need to be adjusted depending on the resolution and number of images.
    """
    plt.figure(figsize=(64, 6))

    for index, image in enumerate(images):
        plt.subplot(num_rows, num_cols, index + 1)
        plt.imshow(image.detach().cpu())
        plt.axis("off")

    plt.show()


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

    parser.add_argument(
        "--generate", help="Create the Generated data".capitalize(), action="store_true"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to generate"
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

    if args.generate:
        if args.num_samples > 1:
            logging.info("Generating samples".capitalize())

            noise = torch.randn(args.num_samples, 100)
            generator = get_the_best_model()
            generated_images = generator(noise)
            images = generated_images.reshape(-1, 28, 28)
            num_cols = int(images.shape[0] // 2)
            num_rows = int(np.round(images.shape[0] / num_cols))

            logging.info("Saving generated images")
            plot_images(images=images, num_rows=num_rows, num_cols=num_cols)


if __name__ == "__main__":
    clf_code()
