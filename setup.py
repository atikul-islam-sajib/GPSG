from setuptools import setup, find_packages

setup(
    name="simple_gan",
    version="0.1.0",  # Update with the version number of your package
    description="A deep learning project that is build for GAN for the Mnist dataset",
    author="Atikul Islam Sajib",
    author_email="atikul.sajib@ptb.de",
    url="https://github.com/atikul-islam-sajib/GPSG.git",  # Update with your project's GitHub repository URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="Alzheimer classification machine-learning",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/GPSG.git/issues",
        "Documentation": "https://github.com/atikul-islam-sajib/GPSG.git/blob/main/README.md",
        "Source Code": "https://github.com/atikul-islam-sajib/GPSG.git",
    },
)
