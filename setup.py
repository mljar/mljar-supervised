from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mljar-supervised",
    version="0.8.1",
    description="Automates Machine Learning Pipeline with Feature Engineering and Hyper-Parameters Tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mljar/mljar-supervised",
    author="MLJAR, Inc.",
    author_email="contact@mljar.com",
    license="MIT",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=open("requirements.txt").readlines(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
    keywords=[
        "automl",
        "machine learning",
        "random forest",
        "keras",
        "xgboost",
        "lightgbm",
        "catboost",
        "mljar",
        "data science",
        "feature engineering"
    ],
)
