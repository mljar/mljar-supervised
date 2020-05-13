# mljar-supervised

[![Build Status](https://travis-ci.org/mljar/mljar-supervised.svg?branch=master)](https://travis-ci.org/mljar/mljar-supervised)
[![PyPI version](https://badge.fury.io/py/mljar-supervised.svg)](https://badge.fury.io/py/mljar-supervised)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mljar-supervised.svg)](https://pypi.python.org/pypi/mljar-supervised/)

## Automated Machine Learning 

`mljar-supervised` is an Automated Machine Learning python package. It can train ML models for:

- binary classification,
- multi-class classification,
- regression.

## What's good in it?

- `mljar-supervised` creates markdown reports from AutoML training. The example of AutoML leaderboard summary:

![AutoML leaderboard](https://github.com/mljar/mljar-examples/blob/master/media/automl_summary.gif)

The example for `Decision Tree` summary:
![Decision Tree summary](https://github.com/mljar/mljar-examples/blob/master/media/decision_tree_summary.gif)

The example for `LightGBM` summary:
![Decision Tree summary](https://github.com/mljar/mljar-examples/blob/master/media/lightgbm_summary.gif)

- This package is computing `Baseline` for your data. So you will know if you need Machine Learning or not! You will know how good are your ML models comparing to the `Baseline`. The `Baseline` is computed based on prior class distribution for classification, and simple mean for regression.
- This package is training simple `Decision Trees` with `max_depth <= 5`, so you can easily visualize them with amazing [dtreeviz](https://github.com/parrt/dtreeviz) to better understand your data.
- The `mljar-supervised` is using simple linear regression and include its coefficients in the summary report, so you can check which features are used the most in the linear model.
- It is using a vast set of algorithms: `Random Forest`, `Extra Trees`, `LightGBM`, `Xgboost`, `CatBoost` (`Neural Networks` will be added soon).
- It can do features preprocessing, like: missing values imputation and converting categoricals. What is more, it can also handle target values preprocessing (You won't believe how often it is needed!). For example, converting categorical target into numeric.
- It can tune hyper-parameters with `not-so-random-search` algorithm (random-search over defined set of values) and hill climbing to fine-tune final models.
- It can compute Ensemble based on greedy algorithm from [Caruana paper](http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf).
- It cares about explainability of models: for every algorithm, the feature importance is computed based on permutation. Additionally, for every algorithm the SHAP explanations are computed: feature importance, dependence plots, and decision plots (explanations can be switched off with `explain_level` parameter).

## Quick example

There is a simple interface available with `fit` and `predict` methods.

```python
import pandas as pd
from supervised.automl import AutoML

df = pd.read_csv("https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv", skipinitialspace=True)

X = df[df.columns[:-1]]
y = df["income"]

automl = AutoML(results_path="directory_with_reports")
automl.fit(X, y)

predictions = automl.predict(X)
```

For details please check [AutoML API Docs](docs/api.md).

## Examples

- [**Income classification**](https://github.com/mljar/mljar-examples/tree/master/Income_classification) - it is a binary classification task on census data
- [**Iris classification**](https://github.com/mljar/mljar-examples/tree/master/Iris_classification) - it is a multiclass classification on Iris flowers data
- [**House price regression**](https://github.com/mljar/mljar-examples/tree/master/House_price_regression) - it is a regression task on Boston houses data

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

Installation for development
```
git clone https://github.com/mljar/mljar-supervised.git
virtualenv venv --python=python3.6
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

## MLJAR
<p align="center">
  <img src="https://github.com/mljar/mljar-examples/blob/master/media/large_logo.png" width="314" />
</p>

The `mljar-supervised` is an open-source project created by [MLJAR](https://mljar.com). We care about ease of use in the Machine Learning. 
The [mljar.com](https://mljar.com) provides a beautiful and simple user interface for building machine learning models.
