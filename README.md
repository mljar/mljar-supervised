# mljar-supervised

[![Build Status](https://travis-ci.org/mljar/mljar-supervised.svg?branch=master)](https://travis-ci.org/mljar/mljar-supervised)
[![PyPI version](https://badge.fury.io/py/mljar-supervised.svg)](https://badge.fury.io/py/mljar-supervised)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mljar-supervised.svg)](https://pypi.python.org/pypi/mljar-supervised/)

## Automated Machine Learning 

`mljar-supervised` is Automated Machine Learning package. It can train ML models for:

- binary classification,
- multi-class classification,
- regression.

## What's good in it?

- `mljar-supervised` creates markdown reports from AutoML training. The report showing AutoML leaderboard:

![AutoML leaderboard](https://github.com/mljar/mljar-examples/blob/master/media/automl_summary.gif)

The report showing example for `Decision Tree` summary:
![Decision Tree summary](https://github.com/mljar/mljar-examples/blob/master/media/decision_tree_summary.gif)

The report showing example for `LightGBM` summary:
![Decision Tree summary](https://github.com/mljar/mljar-examples/blob/master/media/lightgbm_summary.gif)

- This package is computing `Baseline` for your data. So you will know if you need Machine Learning or not! You will know how good are your ML models comaring to the `Baseline`. The `Baseline` is computed based on prior class distribution for classification, and simple mean for regression.
- This package is training simple `Decision Trees` with `max_depth <= 5` so you can easily visualize them with amazing [dtreeviz](https://github.com/parrt/dtreeviz), so you can better understand your data.
- The `mljar-supervised` is using simple linear regression and include its coefficients in summary report, so you can check which features are used the most in linear model.
- It is using a vast set of algorithms: `Random Forest`, `Extra Trees`, `LightGBM`, `Xgboost`, `CatBoost`. (`Neural Networks` will be added soon)
- It can do features preprocessing, like missing values imputation and converting categoricals. What is more, it can also handle target values preprocessing (You won't belive how often it is needed!).
- It can tune hyper-parameters with not-so-random-search algorithm (over defined set of values) and hill climbing to fine-tune final models.
- It can compute Ensemble based on [Caruana paper](http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)
- It cares about explainability of models: for every algorithm the permutation based feature importance is computed. Additionally, for every algorithm the SHAP explanations are computed: feature importance, dependence plots, and decision plots (explanations can be swiched off with `explain_level` parameter).

## Quick example

There is simple interface available with `fit` and `predict` methods.

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

For details please check [AutoML API Docs](docs/api.md).

## Installation

From source code:

```
git clone https://github.com/mljar/mljar-supervised.git
cd mljar-supervised
python setup.py install
```

From PyPi repository (PyPi can be not updated, it is better to install from source):

```
pip install mljar-supervised
```

Installation for development
```
git clone https://github.com/mljar/mljar-supervised.git
virtualenv venv --python=python3.6
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
```


