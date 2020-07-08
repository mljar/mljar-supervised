# `mljar-supervised` AutoML Python Package

[![Build Status](https://travis-ci.org/mljar/mljar-supervised.svg?branch=master)](https://travis-ci.org/mljar/mljar-supervised)
[![PyPI version](https://badge.fury.io/py/mljar-supervised.svg)](https://badge.fury.io/py/mljar-supervised)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mljar-supervised.svg)](https://pypi.python.org/pypi/mljar-supervised/)

## Automated Machine Learning :computer: :bulb: :rocket:

The `mljar-supervised` is an Automated Machine Learning Python package that works with tabular data. It is designed to save time for a data scientist :sunglasses:. It abstracts the common way to preprocess the data, construct the machine learning models, and perform hyper-parameters tuning to find the best model :trophy:. It is no black-box as you can see exactly how the ML pipeline is constructed (with a detailed Markdown report for each ML model). 

The `mljar-supervised` will help you with:
 - explaining and understanding your data,
 - trying many different machine learning models,
 - creating Markdown reports from analysis with details about all models,
 - saving, re-running and loading the analysis and ML models.

It has three built-in modes of work:
 - `Explain` mode, which is ideal for explaining and understanding the data, with many data explanations, like decision trees visualization, linear models coefficients display, permutation importances and SHAP explanations of data,
 - `Perform` for building ML pipelines to use in production,
 - `Compete` mode that trains highly-tuned ML models with ensembling and stacking, with a purpose to use in ML competitions.

Of course, you can further customize the details of each mode to meet requirements.

It integrates many popular frameworks:

 - pandas
 - scikit-learn
 - xgboost
 - lightGBM
 - CatBoost
 - Tensorflow
 - Keras

## What's good in it? :eyes: :cake: :boom:

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

## Available Modes :school: :books:

### Explain 

It is aimed to be used when the user wants to explain and understand the data.
 - It is using 75%/25% train/test split. 
 - It is using: `Baseline`, `Linear`, `Decision Tree`, `Random Forest`, `Xgboost`, `Neural Network` algorithms and ensemble. 
 - It has full explanations: learning curves, importance plots, and SHAP plots.

### Perform

It should be used when the user wants to train a model that will be used in real-life use cases.
 - It is using 5-fold CV.
 - It is using: `Linear`, `Random Forest`, `LightGBM`, `Xgboost`, `CatBoost` and `Neural Network`. It uses ensembling. 
 - It has learning curves and importance plots in reports.

### Compete

It should be used for machine learning competitions.
 - It is using 10-fold CV. 
 - It is using: `Linear`, `Decision Tree`, `Random Forest`, `Extra Trees`, `LightGBM`, `Xgboost`, `CatBoost`, `Neural Network` and `Nearest Neighbors`. It uses ensemble and **stacking**. 
 - It has only learning curves in the reports.

# Examples

## :point_right: Classification example

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


## :point_right: Multi-Class Classification Example

The example code for classification of the optical recognition of handwritten digits dataset. Running this code in less than 30 minutes will result in test accuracy ~98%.

```python
import pandas as pd 
# scikit learn utilites
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# mljar-supervised package
from supervised.automl import AutoML

# load the data
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    pd.DataFrame(digits.data), digits.target, stratify=digits.target, test_size=0.25,
    random_state=123
)

# train models with AutoML
automl = AutoML(mode="Perform")
automl.fit(X_train, y_train)

# compute the accuracy on test data
predictions = automl.predict(X_test)
print(predictions.head())
print("Test accuracy:", accuracy_score(y_test, predictions["label"].astype(int)))
```

## :point_right: Regression Example

Regression example on Boston house prices data. On test data it scores ~ 10.85 mean squared error (MSE).

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from supervised.automl import AutoML # mljar-supervised

# Load the data
housing = load_boston()
X_train, X_test, y_train, y_test = train_test_split(
    pd.DataFrame(housing.data, columns=housing.feature_names),
    housing.target,
    test_size=0.25,
    random_state=123,
)

# train models with AutoML
automl = AutoML(mode="Explain")
automl.fit(X_train, y_train)

# compute the MSE on test data
predictions = automl.predict(X_test)
print(predictions.head())
print("Test MSE:", mean_squared_error(y_test, predictions["prediction"]))
```

## :point_right: More Examples

- [**Income classification**](https://github.com/mljar/mljar-examples/tree/master/Income_classification) - it is a binary classification task on census data
- [**Iris classification**](https://github.com/mljar/mljar-examples/tree/master/Iris_classification) - it is a multiclass classification on Iris flowers data
- [**House price regression**](https://github.com/mljar/mljar-examples/tree/master/House_price_regression) - it is a regression task on Boston houses data


# Markdown reports

## The AutoML Report

The report from running AutoML will contain the table with infomation about each model score and time needed to train the model. For each model there is a link, which you can click to see model's details. The performance of all ML models is presented as scatter and box plots so you can visually inspect which algorithms perform the best :throphy:.

![AutoML leaderboard](https://github.com/mljar/mljar-examples/blob/master/media/automl_summary.gif)

## The `Decision Tree` Report

The example for `Decision Tree` summary with trees visualization. For classification tasks additional metrics are provided:
- confusion matrix
- threshold (optimized in the case of binary classification task)
- F1 score
- Accuracy
- Precision, Recall, MCC

![Decision Tree summary](https://github.com/mljar/mljar-examples/blob/master/media/decision_tree_summary.gif)

## The `LightGBM` Report

The example for `LightGBM` summary:

![Decision Tree summary](https://github.com/mljar/mljar-examples/blob/master/media/lightgbm_summary.gif)


# Installation :package: 

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

# Contributing

To get started take a look at our [Contribution Guide](docs/contributing.md) for information about our process and where you can fit in!

# License :necktie:

The `mljar-supervised` is provided with [MIT license](https://github.com/mljar/mljar-supervised/blob/master/LICENSE).

# MLJAR :heart:
<p align="center">
  <img src="https://github.com/mljar/mljar-examples/blob/master/media/large_logo.png" width="314" />
</p>

The `mljar-supervised` is an open-source project created by [MLJAR](https://mljar.com). We care about ease of use in the Machine Learning. 
The [mljar.com](https://mljar.com) provides a beautiful and simple user interface for building machine learning models.
