from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mljar-supervised",
    version="1.4.0rc1",
    description="Automated Machine Learning for Humans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mljar/mljar-supervised",
    author="MLJAR, Sp. z o.o.",
    author_email="contact@mljar.com",
    license="MIT",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=open("requirements.txt").readlines(),
    extras_require={
        # SDM itself currently requires the separately documented pinned Git install.
        "kumo": ["torch>=2.7,<3", "huggingface_hub>=1,<3", "psutil>=5.0"],
        "tabiclv2": ["torch>=2.7,<3", "huggingface_hub>=1,<3", "psutil>=5.0"],
        "tfm": ["torch>=2.7,<3", "huggingface_hub>=1,<3", "psutil>=5.0"],
        "realmlp": ["pytabkit>=1.7.2,<1.8", "torch>=2.7,<3", "psutil>=5.0"],
        "tabm": ["pytabkit>=1.7.2,<1.8", "torch>=2.7,<3", "psutil>=5.0"],
        "all": ["pytabkit>=1.7.2,<1.8", "torch>=2.7,<3", "psutil>=5.0", "huggingface_hub>=1,<3"],
    },
    include_package_data=True,
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
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
