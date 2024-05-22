from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mljar-supervised",
    version="1.1.8",
    description="Automated Machine Learning for Humans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mljar/mljar-supervised",
    author="MLJAR, Sp. z o.o.",
    author_email="contact@mljar.com",
    license="MIT",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=open("requirements.txt").readlines(),
    include_package_data=True,
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=[
        "automated machine learning",
        "automl",
        "machine learning",
        "data science",
        "data mining",
        "mljar",
        "random forest",
        "decision tree",
        "xgboost",
        "lightgbm",
        "catboost",
        "neural network",
        "extra trees",
        "linear model",
        "features selection",
        "features engineering"
    ],
)
