# mljar-supervised

[![Build Status](https://travis-ci.org/mljar/mljar-supervised.svg?branch=master)](https://travis-ci.org/mljar/mljar-supervised)
[![PyPI version](https://badge.fury.io/py/mljar-supervised.svg)](https://badge.fury.io/py/mljar-supervised)
[![Coverage Status](https://coveralls.io/repos/github/mljar/mljar-supervised/badge.svg?branch=master)](https://coveralls.io/github/mljar/mljar-supervised?branch=master)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mljar-supervised.svg)](https://pypi.python.org/pypi/mljar-supervised/)

[![Machine Learning for Humans](images/the-mljar.svg =100x)](https://mljar.com)

**The new standard in Machine Learning!** Always have best model which is selected and tuned.

Collection of supervised methods (including processing), used in MLJAR AutoML solution.

With this package you can handle any supervised ML algorithm with the same interface.

## Installation

From PyPi repository:

```
pip install mljar-supervised
```

From source code:

```
git clone https://github.com/mljar/mljar-supervised.git
cd mljar-supervised
python setup.py install
```

Python 3.6 is required.

## Usage

This is Automated Machine Learning package, so all hard tasks is done for you. The interface is simple but if necessary allows you to control the training process.

#### Train and predict

```
automl = AutoML()
automl.fit(X, y)
predictions = automl.predict(X)
```

By the default, the training should finish in less than 1 hour and as ML algorithms will be checked:

- Random Forest
- Xgboost
- CatBoost
- LightGBM
- Neural Network
- Ensemble

The parameters that you can use to control the training process are:

- **total_time_limit** - it is a total time limit that AutoML can spend for searching to the best ML model. It is in seconds. _Default is set to 3600 seconds._

## Development

### Installation

```
git clone https://github.com/mljar/mljar-supervised.git
virtualenv venv --python=python3.6
source venv/bin/activate
pip install -r requirements.txt
```

### Testing

```
cd supervised
python -m tests.run_all
```

## Newsletter

Don't miss updates and news from us.
[Subscribe to newsletter!](https://tinyletter.com/mljar)

## Roadmap

The package is under active development! Please expect a lot of changes!
For this package the graphical interface will be provided soon (also open source!). Please be tuned.
