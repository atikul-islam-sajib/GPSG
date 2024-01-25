import logging
import argparse
import sys
import torch
import torch.nn as nn
from collections import OrderedDict

sys.path.append("/src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="./logs/logs.log/",
)
logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """
    A generative neural network model for generating images using the DCGAN architecture.

    Args:
        latent_space (int): The dimensionality of the latent space noise vector. Default is 100.

    Attributes:
        latent_space (int): The dimensionality of the latent space noise vector.
        model (nn.Sequential): The generator model composed of several layers.

    Example:
        >>> generator = Generator(latent_space=100)
        >>> noise = torch.randn(64, 100)
        >>> generated_images = generator(noise)
    """

    def __init__(self, latent_space=100):
        """
        Initialize the Generator.

        Args:
            latent_space (int, optional): The dimensionality of the latent space noise vector. Default is 100.
        """
        self.latent_space = latent_space
        super(Generator, self).__init__()
        layers_config = [
            (self.latent_space, 256, 0.2),
            (256, 512, 0.2),
            (512, 1024, 0.2),
            (1024, 28 * 28),
        ]
        self.model = self.generate_layer(layers_config=layers_config)

    def generate_layer(self, layers_config):
        """
        Create the layers of the generator model based on the provided configuration.

        Args:
            layers_config (list): A list of tuples specifying the layer configurations.

        Returns:
            nn.Sequential: A sequential model containing the specified layers.

        Example:
            >>> layers_config = [(100, 256, 0.02), (256, 512, 0.02), (512, 1024, 0.02), (1024, 28*28)]
            >>> generator = Generator()
            >>> generator_model = generator.generate_layer(layers_config)
        """
        layers = OrderedDict()
        for index, (input_feature, out_feature, negative_slope) in enumerate(
            layers_config[:-1]
        ):
            layers[f"layer_{index}"] = nn.Linear(
                in_features=input_feature, out_features=out_feature
            )
            layers[f"layer_{index}_activation"] = nn.LeakyReLU(
                negative_slope=negative_slope
            )

        layers[f"output_layer"] = nn.Linear(
            in_features=layers_config[-1][0], out_features=layers_config[-1][1]
        )
        layers[f"output_layer_activation"] = nn.Tanh()

        return nn.Sequential(layers)

    def forward(self, x):
        """
        Forward pass of the generator model.

        Args:
            x (torch.Tensor): Input noise tensor sampled from the latent space.

        Returns:
            torch.Tensor: Generated images.

        Example:
            >>> noise = torch.randn(64, 100)
            >>> generated_images = generator(noise)
        """
        if x is not None:
            x = self.model(x)
        else:
            x = "ERROR"

        return x.reshape(-1, 1, 28, 28)


if __name__ == "__main__":
    """
    This script facilitates the generation of images using a Generative Adversarial Network (GAN) model.
    It utilizes command-line arguments to configure the GAN model's parameters, specifically focusing
    on the generator component.

    The script allows users to specify the dimensionality of the latent space and the batch size for
    generating synthetic images. Additionally, it includes an option to define the generator model.
    If the generator model is defined, the script proceeds to create a generator instance, generate
    a batch of fake samples, and log their shape. In case of missing required arguments or if the
    generator model is not specified, the script logs an appropriate exception.

    Command Line Arguments:
    - --latent_space (int): Mandatory. Dimensionality of the latent space.
    - --batch_size (int): Batch size for generating samples. Default is 64.
    - --generator (flag): Specify this flag to define and utilize the generator model.

    The script expects a predefined `Generator` class and uses the `torch.randn` function to create
    noise samples. Logging is performed using a pre-configured logger.

    Examples:
    python script.py --latent_space 100 --batch_size 64 --generator
    python script.py --latent_space 100 --generator

    Note:
    - The script assumes the availability of a logging module (`logger`) configured for logging information
    and exceptions.
    - It is essential that the `Generator` class is defined and accessible within the script's scope.
    - The script is designed to be executed in an environment where `argparse` and `torch` libraries are available.
    """
    parser = argparse.ArgumentParser(
        description="Generate images using a GAN model.".capitalize()
    )
    logger.info("Starting GAN model training.")

    parser.add_argument(
        "--latent_space",
        type=int,
        default=100,
        help="Dimensionality of the latent space.",
        required=True,
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size.".capitalize()
    )
    parser.add_argument(
        "--generator", action="store_true", help="Define the model".capitalize()
    )

    args = parser.parse_args()

    if args.generator:
        if args.latent_space and args.batch_size:
            logger.info("Generator model defined".title())

            generator = Generator(args.latent_space)
            noise_samples = torch.randn(args.batch_size, args.latent_space)
            fake_samples = generator(noise_samples)

            print("The shape of the Generated dataset # {} ".format(fake_samples.shape))

            logger.info("Generator data shape # {} ".title().format(fake_samples.shape))

        else:
            logger.exception("Latent space is not defined".title())
    else:
        logger.exception("Generator model is not defined".title())
