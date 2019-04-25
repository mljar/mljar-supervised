from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mljar-supervised',
    version='0.1.7',
    description='Automated Machine Learning for Supervised tasks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mljar/mljar-supervised',
    author='MLJAR, Inc.',
    author_email='contact@mljar.com',
    license='MIT',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=["numpy>=1.15.4", "pandas>=0.23.4", "scipy>=1.1.0", "scikit-learn>=0.20.0",
                        "xgboost==0.80", "lightgbm==2.2.3", "catboost==0.13.1",
                        "h5py>=2.9.0", "tensorflow==1.13.1", "Keras==2.2.4", "tqdm==4.31.1"],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6'
    ],
    keywords=['automl', 'machine learning', "random forest", 'keras', 'xgboost'],
)
