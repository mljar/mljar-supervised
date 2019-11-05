# mljar-supervised

[![Build Status](https://travis-ci.org/mljar/mljar-supervised.svg?branch=master)](https://travis-ci.org/mljar/mljar-supervised)
[![PyPI version](https://badge.fury.io/py/mljar-supervised.svg)](https://badge.fury.io/py/mljar-supervised)
[![Coverage Status](https://coveralls.io/repos/github/mljar/mljar-supervised/badge.svg?branch=master)](https://coveralls.io/github/mljar/mljar-supervised?branch=master)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mljar-supervised.svg)](https://pypi.python.org/pypi/mljar-supervised/)

<p align="center">
<img src="https://raw.githubusercontent.com/mljar/mljar-supervised/master/images/the-mljar.svg" width=300 />
</p>

## The new standard in Machine Learning!

Thanks to Automated Machine Learning you don't need to worry about different machine learning interfaces. You don't need to know all algorithms and their hyper-parameters. With AutoML model tuning and training is painless.

In the current version only binary classification is supported with optimization of LogLoss metric.

## Quick example

```python
import pandas as pd
from supervised.automl import AutoML

df = pd.read_csv("https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv", skipinitialspace=True)

X = df[df.columns[:-1]]
y = df["income"]

automl = AutoML()
automl.fit(X, y)

predictions = automl.predict(X)
```

## The tuning algorithm

The tuning algorithm was created and developed by Piotr Płoński. It is heuristic algorithm created from combination of:

- **not-so-random** approach
- and **hill-climbing**

The approach is **not-so-random** because each algorithm has a defined set of hyper-parameters that usually works. At first step from not so random parameters an initial set of models is drawn. Then the hill climbing approach is used to pick best performing algorithms and tune them.

For each algorithm used in the AutoML the early stopping is applied.

The ensemble algorithm was implemented based on [Caruana paper](http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf).

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

This is Automated Machine Learning package, so all hard tasks is done for you. The interface is simple but if necessary it gives you ability to control the training process.

#### Train and predict

```python
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
- **learner_time_limit** - the time limit for training single model, in case of `k`-fold cross validation, the time spend on training is `k*learner_time_limit`. This parameter is only considered when `total_time_limit` is set to None. _Default is set to 120 seconds_.
- **algorithms** - the list of algorithms that will be checked. _Default is set to ["CatBoost", "Xgboost", "RF", "LightGBM", "NN"]_.
- **start_random_models** - the number of models to check with _not so random_ algorithm. _Default is set to 10_.
- **hill_climbing_steps** - number of hill climbing steps used in models tuning. _Default is set to 3_.
- **top_models_to_improve** - number of models considered for improvement in each hill climbing step. _Default is set to 5_.
- **train_ensemble** - decides if ensemble model is trained at the end of AutoML fit procedure. _Default is set to True_.
- **verbose** - controls printouts, _Default is set to True_.

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

To be added:
- training single decision tree
- create text report from trained models (maybe with plots from learning)
- compute threshold for model prediction and predicting discrete output (label)
- add model/predictions explanations
- add support for multiclass classification
- add support for regressions
