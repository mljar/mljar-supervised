# Get started 

[![Build Status](https://travis-ci.org/mljar/mljar-supervised.svg?branch=master)](https://travis-ci.org/mljar/mljar-supervised)
[![Coverage Status](https://coveralls.io/repos/github/mljar/mljar-supervised/badge.svg?branch=master)](https://coveralls.io/github/mljar/mljar-supervised?branch=master)
[![PyPI version](https://badge.fury.io/py/mljar-supervised.svg)](https://badge.fury.io/py/mljar-supervised)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mljar-supervised.svg)](https://pypi.python.org/pypi/mljar-supervised/)



The `mljar-supervised` is an Automated Machine Learning Python package that works with tabular data. It is designed to save time for a data scientist :sunglasses:. It abstracts the common way to preprocess the data, construct the machine learning models, and perform hyper-parameters tuning to find the best model :trophy:. It is no black-box as you can see exactly how the ML pipeline is constructed (with a detailed Markdown report for each ML model).

The `mljar-supervised` will help you with:

- explaining and understanding your data,
- trying many different machine learning models,
- creating Markdown reports from analysis with details about all models,
- saving, re-running and loading the analysis and ML models.

## What's good in it? 

- `mljar-supervised` creates markdown reports from AutoML training full of ML details and charts. 
- It can compute the `Baseline` for your data. So you will know if you need Machine Learning or not! You will know how good are your ML models comparing to the `Baseline`. The `Baseline` is computed based on prior class distribution for classification, and simple mean for regression.
- This package is training simple `Decision Trees` with `max_depth <= 5`, so you can easily visualize them with amazing [dtreeviz](https://github.com/parrt/dtreeviz) to better understand your data.
- The `mljar-supervised` is using simple linear regression and include its coefficients in the summary report, so you can check which features are used the most in the linear model.
- It is using a many algorithms: `Baseline`, `Linear`, `Random Forest`, `Extra Trees`, `LightGBM`, `Xgboost`, `CatBoost`, `Neural Networks`, and `Nearest Neighbors`.
- It can do features preprocessing, like: missing values imputation and converting categoricals. What is more, it can also handle target values preprocessing (You won't believe how often it is needed!). For example, converting categorical target into numeric.
- It can tune hyper-parameters with `not-so-random-search` algorithm (random-search over defined set of values) and hill climbing to fine-tune final models.
- It can compute Ensemble based on greedy algorithm from [Caruana paper](http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf).
- It can stack models to build level 2 ensemble (available in `Compete` mode or after setting `stack_models` parameter).
- It cares about explainability of models: for every algorithm, the feature importance is computed based on permutation. Additionally, for every algorithm the SHAP explanations are computed: feature importance, dependence plots, and decision plots (explanations can be switched off with `explain_level` parameter).


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

Running in docker with Jupyter notebook:
```
FROM python:3.7-slim-buster
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN pip3 install mljar-supervised jupyter
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
```


## Basic usage

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML

df = pd.read_csv(
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv",
    skipinitialspace=True,
)
X_train, X_test, y_train, y_test = train_test_split(
    df[df.columns[:-1]], df["income"], test_size=0.25
)

automl = AutoML()
automl.fit(X_train, y_train)

predictions = automl.predict(X_test)
```