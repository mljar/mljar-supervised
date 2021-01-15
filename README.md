# MLJAR Automated Machine Learning

[![Build Status](https://travis-ci.org/mljar/mljar-supervised.svg?branch=master)](https://travis-ci.org/mljar/mljar-supervised)
[![Coverage Status](https://coveralls.io/repos/github/mljar/mljar-supervised/badge.svg?branch=master)](https://coveralls.io/github/mljar/mljar-supervised?branch=master)
[![PyPI version](https://badge.fury.io/py/mljar-supervised.svg)](https://badge.fury.io/py/mljar-supervised)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mljar-supervised.svg)](https://pypi.python.org/pypi/mljar-supervised/)


<p align="center">
  <img src="https://raw.githubusercontent.com/mljar/mljar-examples/master/media/AutoML_overview_mljar_v3.svg" width="100%" />
</p>

---

**Documentation**: <a href="https://supervised.mljar.com/" target="_blank">https://supervised.mljar.com/</a>

**Source Code**: <a href="https://github.com/mljar/mljar-supervised" target="_blank">https://github.com/mljar/mljar-supervised</a>

---

## Table of Contents

 - [Automated Machine Learning](https://github.com/mljar/mljar-supervised#automated-machine-learning-rocket)
 - [What's good in it?](https://github.com/mljar/mljar-supervised#whats-good-in-it-boom)
 - [Atomatic Documentation](https://github.com/mljar/mljar-supervised#automatic-documentation)
 - [Available Modes](https://github.com/mljar/mljar-supervised#available-modes-books)
 - [Examples](https://github.com/mljar/mljar-supervised#examples)
 - [Documentation](https://github.com/mljar/mljar-supervised#documentation)
 - [Installation](https://github.com/mljar/mljar-supervised#installation-package)
 - [Contributing](https://github.com/mljar/mljar-supervised#contributing)
 - [License](https://github.com/mljar/mljar-supervised#license-necktie)
 - [MLJAR](https://github.com/mljar/mljar-supervised#mljar-heart)

## Automated Machine Learning :rocket: 

The `mljar-supervised` is an Automated Machine Learning Python package that works with tabular data. It is designed to save time for a data scientist. It abstracts the common way to preprocess the data, construct the machine learning models, and perform hyper-parameters tuning to find the best model :trophy:. It is no black-box as you can see exactly how the ML pipeline is constructed (with a detailed Markdown report for each ML model). 

The `mljar-supervised` will help you with:
 - explaining and understanding your data (Automatic Exploratory Data Analysis),
 - trying many different machine learning models (Algorithm Selection and Hyper-Parameters tuning),
 - creating Markdown reports from analysis with details about all models (Atomatic-Documentation),
 - saving, re-running and loading the analysis and ML models.

It has three built-in modes of work:
 - `Explain` mode, which is ideal for explaining and understanding the data, with many data explanations, like decision trees visualization, linear models coefficients display, permutation importances and SHAP explanations of data,
 - `Perform` for building ML pipelines to use in production,
 - `Compete` mode that trains highly-tuned ML models with ensembling and stacking, with a purpose to use in ML competitions.

Of course, you can further customize the details of each `mode` to meet the requirements.

## What's good in it? :boom:

- It is using many algorithms: `Baseline`, `Linear`, `Random Forest`, `Extra Trees`, `LightGBM`, `Xgboost`, `CatBoost`, `Neural Networks`, and `Nearest Neighbors`.
- It can compute Ensemble based on greedy algorithm from [Caruana paper](http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf).
- It can stack models to build level 2 ensemble (available in `Compete` mode or after setting `stack_models` parameter).
- It can do features preprocessing, like: missing values imputation and converting categoricals. What is more, it can also handle target values preprocessing.
- It can do advanced features engineering, like: [Golden Features](https://supervised.mljar.com/features/golden_features/), [Features Selection](https://supervised.mljar.com/features/features_selection/), Text and Time Transformations.
- It can tune hyper-parameters with `not-so-random-search` algorithm (random-search over defined set of values) and hill climbing to fine-tune final models.
- It can compute the `Baseline` for your data. That you will know if you need Machine Learning or not!
- It has extensive explanations. This package is training simple `Decision Trees` with `max_depth <= 5`, so you can easily visualize them with amazing [dtreeviz](https://github.com/parrt/dtreeviz) to better understand your data.
- The `mljar-supervised` is using simple linear regression and include its coefficients in the summary report, so you can check which features are used the most in the linear model.
- It cares about explainability of models: for every algorithm, the feature importance is computed based on permutation. Additionally, for every algorithm the SHAP explanations are computed: feature importance, dependence plots, and decision plots (explanations can be switched off with `explain_level` parameter).
- There is automatic documnetation for every ML experiment run with AutoML. The `mljar-supervised` creates markdown reports from AutoML training full of ML details, metrics and charts. 

# Automatic Documentation

## The AutoML Report

The report from running AutoML will contain the table with infomation about each model score and time needed to train the model. For each model there is a link, which you can click to see model's details. The performance of all ML models is presented as scatter and box plots so you can visually inspect which algorithms perform the best :trophy:.

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


## Available Modes :books:

In the [docs](https://supervised.mljar.com/features/modes/) you can find details about AutoML modes are presented in the table .

### Explain 

```py
automl = AutoML(mode="Explain")
```

It is aimed to be used when the user wants to explain and understand the data.
 - It is using 75%/25% train/test split. 
 - It is using: `Baseline`, `Linear`, `Decision Tree`, `Random Forest`, `Xgboost`, `Neural Network` algorithms and ensemble. 
 - It has full explanations: learning curves, importance plots, and SHAP plots.

### Perform

```py
automl = AutoML(mode="Perform")
```

It should be used when the user wants to train a model that will be used in real-life use cases.
 - It is using 5-fold CV.
 - It is using: `Linear`, `Random Forest`, `LightGBM`, `Xgboost`, `CatBoost` and `Neural Network`. It uses ensembling. 
 - It has learning curves and importance plots in reports.

### Compete

```py
automl = AutoML(mode="Compete")
```

It should be used for machine learning competitions.
 - It adapts the validation strategy depending on dataset size and `total_time_limit`. It can be: train/test split (80/20), 5-fold CV or 10-fold CV. 
 - It is using: `Linear`, `Decision Tree`, `Random Forest`, `Extra Trees`, `LightGBM`, `Xgboost`, `CatBoost`, `Neural Network` and `Nearest Neighbors`. It uses ensemble and **stacking**. 
 - It has only learning curves in the reports.

# Examples

## :point_right: Binary Classification Example

There is a simple interface available with `fit` and `predict` methods.

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

AutoML `fit` will print:
```py
Create directory AutoML_1
AutoML task to be solved: binary_classification
AutoML will use algorithms: ['Baseline', 'Linear', 'Decision Tree', 'Random Forest', 'Xgboost', 'Neural Network']
AutoML will optimize for metric: logloss
1_Baseline final logloss 0.5519845471086654 time 0.08 seconds
2_DecisionTree final logloss 0.3655910192804364 time 10.28 seconds
3_Linear final logloss 0.38139916864708445 time 3.19 seconds
4_Default_RandomForest final logloss 0.2975204390214936 time 79.19 seconds
5_Default_Xgboost final logloss 0.2731086827200411 time 5.17 seconds
6_Default_NeuralNetwork final logloss 0.319812276905242 time 21.19 seconds
Ensemble final logloss 0.2731086821194617 time 1.43 seconds
```

- the AutoML results in [Markdown report](https://github.com/mljar/mljar-examples/tree/master/Income_classification/AutoML_1#automl-leaderboard)
- the Xgboost [Markdown report](https://github.com/mljar/mljar-examples/blob/master/Income_classification/AutoML_1/5_Default_Xgboost/README.md), please take a look at amazing dependence plots produced by SHAP package :sparkling_heart:
- the Decision Tree [Markdown report](https://github.com/mljar/mljar-examples/blob/master/Income_classification/AutoML_1/2_DecisionTree/README.md), please take a look at beautiful tree visualization :sparkles:
- the Logistic Regression [Markdown report](https://github.com/mljar/mljar-examples/blob/master/Income_classification/AutoML_1/3_Linear/README.md), please take a look at coefficients table, and you can compare the SHAP plots between (Xgboost, Decision Tree and Logistic Regression) :coffee:


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
predictions = automl.predict_all(X_test)
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
print("Test MSE:", mean_squared_error(y_test, predictions))
```

## :point_right: More Examples

- [**Income classification**](https://github.com/mljar/mljar-examples/tree/master/Income_classification) - it is a binary classification task on census data
- [**Iris classification**](https://github.com/mljar/mljar-examples/tree/master/Iris_classification) - it is a multiclass classification on Iris flowers data
- [**House price regression**](https://github.com/mljar/mljar-examples/tree/master/House_price_regression) - it is a regression task on Boston houses data

# Documentation :books:

For details please check [mljar-supervised docs](https://supervised.mljar.com).

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

Running in the docker:
```
FROM python:3.7-slim-buster
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN pip3 install mljar-supervised jupyter
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
```

# Contributing

To get started take a look at our [Contribution Guide](https://supervised.mljar.com/contributing/) for information about our process and where you can fit in!

### Contributors
<a href="https://github.com/mljar/mljar-supervised/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=mljar/mljar-supervised" />
</a>


# License :necktie:

The `mljar-supervised` is provided with [MIT license](https://github.com/mljar/mljar-supervised/blob/master/LICENSE).

# MLJAR :heart:
<p align="center">
  <img src="https://github.com/mljar/mljar-examples/blob/master/media/large_logo.png" width="314" />
</p>

The `mljar-supervised` is an open-source project created by [MLJAR](https://mljar.com). We care about ease of use in the Machine Learning. 
The [mljar.com](https://mljar.com) provides a beautiful and simple user interface for building machine learning models.
