
# GPSG - Generative Adversarial Network for MNIST Dataset

## Overview
GPSG (GAN-based Project for Synthesizing Grayscale images) is a machine learning project focused on generating synthetic images using Generative Adversarial Networks (GANs). Specifically, it is designed to work with the MNIST dataset, a large database of handwritten digits commonly used for training various image processing systems.

## Features
- Utilizes PyTorch for implementing GAN models.
- Provides scripts for easy training and generating synthetic images.
- Includes a custom data loader for the MNIST dataset.
- Customizable training parameters for experimenting with GAN.

## Installation
Clone the repository:
```
git clone https://github.com/atikul-islam-sajib/GPSG.git
cd GPSG
```

## Core Script Usage
The core script sets up the necessary components for training the GAN. Here's a quick overview of what each part does:

```python
from src.data_loader import Loader
from src.discriminator import Discriminator
from src.generator import Generator
from src.trainer import Trainer

# Initialize the data loader with batch size
loader = Loader(batch_size=64)
mnist_data = loader.download_dataset()
loader.create_dataloader(mnist_data = mnist_data)

# Set up the trainer with learning rate, epochs, and latent space size
trainer = Trainer(lr = 0.0002, epochs = 10, latent_space = 100)
trainer.train_simple_gan()
```

This script initializes the data loader, downloads the MNIST dataset, and prepares the data loader. It then sets up and starts the training process for the GAN model.

## Training and Generating Images
### Training the GAN Model
To train the GAN model with default parameters:
```
python /content/GPSG/clf.py --batch_size 64 --lr 0.0002 --epochs 1 --latent_space 100 --download_mnist
```

### Generating Images
To generate images using the trained model:
```
python /content/GPSG/clf.py --num_samples 64 --generate
```

### Viewing Generated Images
Check the specified output directory for the generated images.

## Documentation
For detailed documentation on the implementation and usage, visit the [GPSG Documentation](https://atikul-islam-sajib.github.io/PageDep/).

## Contributing
Contributions to improve the project are welcome. Please follow the standard procedures for contributing to open-source projects.

## License
This project is licensed under [MIT LICENSE]. Please see the LICENSE file for more details.

## Acknowledgements
Thanks to all contributors and users of the GPSG project. Special thanks to those who have provided feedback and suggestions for improvements.

## Contact
For any inquiries or suggestions, feel free to reach out to [atikulislamsajib137@gmail.com].

## Additional Information
- This project is a work in progress and subject to changes.
- Feedback and suggestions are highly appreciated.
