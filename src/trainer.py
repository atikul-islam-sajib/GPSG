import sys
import logging
import argparse
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("/src")

from generator import Generator
from discriminator import Discriminator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/train.log",
)


class Trainer:
    """
    A Trainer class for setting up and training a Generative Adversarial Network (GAN).

    This class is responsible for initializing the GAN components, including the generator and
    discriminator models, loss functions, optimizers, and the dataloader for input data. It
    also configures the training environment (device selection) and training parameters.

    Attributes:
    - lr (float): Learning rate for the Adam optimizers of both generator and discriminator.
    - epochs (int): Number of epochs for training the GAN.
    - betas (tuple): Beta coefficients used for computing running averages of gradient and its square
                     in the Adam optimizer.
    - latent_space (int): Dimensionality of the latent space for the generator.
    - dataloader (DataLoader): DataLoader object for loading the training data.
    - device (torch.device): The computation device (CPU or GPU/MPS) for training.
    - loss_function (nn.Module): The loss function used during training (Binary Cross Entropy Loss).
    - generator (nn.Module): The generator model of the GAN.
    - discriminator (nn.Module): The discriminator model of the GAN.
    - optimizer_discriminator (optim.Optimizer): The optimizer for the discriminator.
    - optimizer_generator (optim.Optimizer): The optimizer for the generator.

    The class assumes the presence of a 'Generator' and 'Discriminator' class for initializing
    the respective models. The training data is expected to be pre-processed and serialized using
    joblib, and is loaded into a DataLoader.

    Usage:
    To use this class, create an instance and then call its training method (not implemented in
    this snippet). For example:
        trainer = Trainer(lr=0.0002, epochs=100, betas=(0.5, 0.999), latent_space=100)
        trainer.train()  # Assuming a 'train' method is implemented

    Note:
    - This class requires the 'torch', 'torchvision', 'torch.nn', 'torch.optim', and 'joblib' libraries.
    - The trainer is designed to work with image data, specifically configured here for the MNIST dataset.
    - The device selection is automatically determined based on the availability of Apple's Metal
      Performance Shaders (MPS) for acceleration on compatible hardware.
    """

    def __init__(self, lr=0.0002, epochs=100, betas=(0.5, 0.999), latent_space=100):
        self.lr = lr
        self.epochs = epochs
        self.betas = betas
        self.latent_space = latent_space
        self.dataloader = joblib.load(filename="./data/processed/dataloader.pkl")

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.loss_function = nn.BCELoss()
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.optimizer_discriminator = optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=self.betas
        )

        self.optimizer_generator = optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=self.betas
        )

    logging.info("Define the discriminator training function".capitalize())

    def saved_checkpoint(self, epoch):
        """
        Saves the current state (weights and biases) of the generator model as a checkpoint file.

        This method is intended to be used during the training process of a Generative Adversarial Network (GAN)
        to periodically save the state of the generator. The saved state can be used for resuming training,
        evaluation, or generating data at a later time. The checkpoint is saved in the current directory with
        a filename that includes the epoch number.

        Parameters:
        - epoch (int): The current epoch number in the training process. This is used to tag the saved file
        for easy identification.

        The method assumes the presence of a 'generator' attribute in the class instance, which is the
        neural network model to be saved. It also utilizes Python's logging module to log the success or
        failure of the save operation.

        Upon successful saving, a message is logged with the checkpoint file's name. If an exception occurs
        during saving, an error message is logged with the exception details.

        Example:
        - To save a checkpoint of the generator at epoch 5:
        instance.saved_checkpoint(5)

        Note:
        - The method uses 'torch.save' for saving the model state and requires the PyTorch library.
        - The saved files follow the naming convention 'generator_{epoch}.pth', where {epoch} is replaced
        with the actual epoch number.
        - Exception handling is used to manage any errors that might occur during the save operation, with
        details logged as error messages.
        """
        logging.info("Save the checkpoint of the discriminator".capitalize())

        try:
            torch.save(
                self.generator.state_dict(),
                "./models/checkpoints/generator_{}.pth".format(epoch),
            )
        except Exception as e:
            logging.error("Error saving the checkpoint of the generator: {}".format(e))

    def train_discriminator(self, real_samples, fake_samples, real_labels, fake_labels):
        """
        Train the discriminator of a Generative Adversarial Network (GAN).

        This function updates the discriminator by training it on both real and fake samples.
        It calculates the loss for both real and fake predictions and backpropagates the total loss
        to update the discriminator's weights.

        Parameters:
        - real_samples (Tensor): A batch of real samples from the dataset.
        - fake_samples (Tensor): A batch of fake samples generated by the GAN's generator.
        - real_labels (Tensor): A batch of labels, typically ones, representing real samples.
        - fake_labels (Tensor): A batch of labels, typically zeros, representing fake samples.

        The function assumes that the discriminator and loss_function are globally accessible,
        and it also utilizes the optimizer_discriminator for the backpropagation process.

        Returns:
        - float: The total loss incurred by the discriminator for the current batch of real and fake samples.
        """
        self.optimizer_discriminator.zero_grad()

        real_predicted = self.discriminator(real_samples)
        fake_predicted = self.discriminator(fake_samples.detach())

        real_predicted_loss = self.loss_function(real_predicted, real_labels)
        fake_predicted_loss = self.loss_function(fake_predicted, fake_labels)

        total_discriminator_loss = real_predicted_loss + fake_predicted_loss

        total_discriminator_loss.backward()
        self.optimizer_discriminator.step()

        return total_discriminator_loss.item()

    logging.info("Define the generator training function".capitalize())

    def train_generator(self, fake_samples, real_labels):
        """
        Train the generator of a Generative Adversarial Network (GAN).

        This function trains the generator by attempting to fool the discriminator. It updates
        the generator based on how well it can trick the discriminator into classifying the
        generated (fake) samples as real. The function calculates the loss by comparing the
        discriminator's predictions on the fake samples against the 'real' labels and then
        performs backpropagation to update the generator's weights.

        Parameters:
        - fake_samples (Tensor): A batch of fake samples generated by the GAN's generator.
        - real_labels (Tensor): A batch of labels, typically ones, used as targets for training the generator.

        This function assumes the availability of a globally accessible discriminator model,
        a loss function, and the optimizer for the generator (optimizer_generator).

        Returns:
        - float: The loss incurred by the generator for the current batch of fake samples, indicating
                how well the generator is able to fool the discriminator.
        """
        self.optimizer_generator.zero_grad()

        fake_predict = self.discriminator(fake_samples)

        generated_loss = self.loss_function(fake_predict, real_labels)

        generated_loss.backward()
        self.optimizer_generator.step()

        return generated_loss.item()

    def train_simple_gan(self):
        """
        Train a Generative Adversarial Network (GAN) consisting of a generator and discriminator.

        This function orchestrates the training process of a GAN over a specified number of epochs.
        Training involves alternating between training the discriminator and the generator.
        The discriminator is trained to distinguish real data from fake data generated by the generator,
        while the generator is trained to produce data that appears real to the discriminator.

        Parameters:
        - epochs (int): The number of training epochs.
        - latent_space (int): The size of the latent space used to generate noise samples for the generator.
        - print_interval (int): Interval of steps for printing training progress within each epoch.
        - dataloader (DataLoader): DataLoader object providing access to the dataset.
        - device (torch.device): The device (CPU/GPU) on which the training is performed.

        The function assumes that 'train_discriminator' and 'train_generator' are pre-defined functions
        that handle the training of the discriminator and generator, respectively. Additionally,
        'generator' and 'discriminator' should be predefined models. The function also requires
        'np.mean' for calculating average losses, and it prints the training progress at regular intervals.

        Returns:
        None: This function does not return a value but prints the training progress and average losses per epoch.
        """
        print_interval = 100

        for epoch in range(self.epochs):
            discriminator_loss = []
            generator_loss = []

            for i, (real_samples, _) in enumerate(self.dataloader):
                real_samples = real_samples.to(self.device)
                batch_size = real_samples.shape[0]

                logging.info("Generate noise samples and fake samples".capitalize())

                noise_samples = torch.randn(batch_size, self.latent_space).to(
                    self.device
                )
                fake_samples = self.generator(noise_samples)

                logging.info("Define labels for real and fake samples".capitalize())

                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                logging.info("Train the discriminator".capitalize())

                d_loss = self.train_discriminator(
                    real_samples=real_samples,
                    fake_samples=fake_samples,
                    real_labels=real_labels,
                    fake_labels=fake_labels,
                )

                logging.info("Train the generator".capitalize())

                g_loss = self.train_generator(
                    fake_samples=fake_samples, real_labels=real_labels
                )

                discriminator_loss.append(d_loss)
                generator_loss.append(g_loss)

                logging.info(
                    "Print training progress every 'print_interval' iterations".capitalize()
                )
                if i % print_interval == 0:
                    print(
                        f"Epoch [{epoch+1}/{self.epochs}], Step [{i}/{len(self.dataloader)}], d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}"
                    )

            logging.info("Output average loss at the end of each epoch".capitalize())
            print(f"Epoch [{epoch + 1}/{self.epochs}] Completed")
            print(
                f"[==============] Average d_loss: {np.mean(discriminator_loss):.4f} - Average g_loss: {np.mean(generator_loss):.4f}"
            )

            logging.info("Save checkpoint at the end of each epoch".capitalize())
            self.saved_checkpoint(epoch=epoch + 1)


if __name__ == "__main__":
    """
    This script is designed to facilitate the training of a Simple Generative Adversarial Network (GAN)
    by parsing command-line arguments to configure the training parameters.

    The script uses the argparse library to accept user-specified values for the learning rate, number of
    training epochs, and dimensionality of the latent space. These parameters are crucial in defining how
    the GAN will be trained in terms of its learning dynamics and the complexity of the generated data.

    Upon receiving valid parameters, the script initializes a Trainer instance with the specified settings
    and starts the training process. If any of the required parameters are not provided, the script logs
    an appropriate error message.

    Command Line Arguments:
    - --lr (float): Learning rate for the optimizers. Defaults to 0.0002.
    - --epochs (int): Number of epochs for training the GAN. Defaults to 100.
    - --latent_space (int): Dimensionality of the latent space for the generator. Defaults to 100.

    Usage:
    The script is executed from the command line with optional arguments for learning rate, epochs,
    and latent space. For example:
    python script.py --lr 0.0002 --epochs 100 --latent_space 100

    Functionality:
    - Parses command-line arguments for GAN training configuration.
    - Initializes and invokes a Trainer class to start the GAN training process.
    - Logs the start of the training process and any configuration errors.

    Note:
    - The script assumes the presence of a 'Trainer' class with a method 'train_simple_gan' to handle
    the training process.
    - Logging is configured to capture informational messages and exceptions.
    - The script is designed to be executed in an environment where Python libraries like 'argparse',
    'logging', and 'torch' are installed.
    """
    parser = argparse.ArgumentParser(description="Simple GAN".title())
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate".capitalize()
    )
    parser.add_argument("--epochs", type=int, default=100, help="Epochs".capitalize())
    parser.add_argument(
        "--latent_space", type=int, default=100, help="Latent space".capitalize()
    )

    args = parser.parse_args()

    if args.lr and args.epochs and args.latent_space:
        logging.info("Start training".capitalize())

        trainer = Trainer(
            lr=args.lr, epochs=args.epochs, latent_space=args.latent_space
        )
        trainer.train_simple_gan()
    else:
        logging.exception(
            "lr, epochs, and latent space needs to be defined".capitalize()
        )
