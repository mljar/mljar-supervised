---
description: Automated machine learning for Python with support for tabular classification and regression, model explainability, fairness-aware learning, and generated Markdown reports.
social:
  cards_layout: default
---

# MLJAR AutoML for Python

**MLJAR AutoML 1.4.0b1** is available from PyPI and requires Python 3.11+.
Feature discovery is included in the base installation:

```bash
python -m pip install "mljar-supervised==1.4.0b1"
```

An unpinned install selects the stable release. For the beta's optional neural
models, choose [CPU](install.md#realmlp-on-cpu) or
[NVIDIA GPU](install.md#realmlp-on-nvidia-gpu) PyTorch first, then install
`mljar-supervised[realmlp]==1.4.0b1` or `mljar-supervised[tabm]==1.4.0b1`.
Both [RealMLP](features/realmlp.md) and [TabM mini](features/tabm.md) are available
in Compete mode; no source checkout is required.

[Kumo and TabICLv2](features/sdm.md) are separate optional Compete algorithms.
`mljar-supervised[tfm]==1.4.0b1` installs their SDM 0.2.1 backend from PyPI;
model weights download on first use. `[all]` includes all optional model backends.
See [Installation](install.md) for versioned commands, hardware requirements,
and the macOS native-runtime compatibility checks.

[![Tests status](https://github.com/mljar/mljar-supervised/actions/workflows/run-tests.yml/badge.svg)](https://github.com/mljar/mljar-supervised/actions/workflows/run-tests.yml)
[![PyPI version](https://badge.fury.io/py/mljar-supervised.svg)](https://badge.fury.io/py/mljar-supervised)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/mljar-supervised/badges/version.svg)](https://anaconda.org/conda-forge/mljar-supervised)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mljar-supervised.svg)](https://pypi.python.org/pypi/mljar-supervised/)


[![Anaconda-Server Badge](https://anaconda.org/conda-forge/mljar-supervised/badges/platforms.svg)](https://anaconda.org/conda-forge/mljar-supervised)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/mljar-supervised/badges/license.svg)](https://anaconda.org/conda-forge/mljar-supervised)
[![Downloads](https://pepy.tech/badge/mljar-supervised)](https://pepy.tech/project/mljar-supervised)


`mljar-supervised` is an automated machine learning library for Python that works with tabular data. It helps train machine learning models, tune hyperparameters, explain predictions, evaluate fairness, and generate detailed Markdown reports for classification and regression tasks.

The `mljar-supervised` package is designed to save time for a data scientist :sunglasses:. It abstracts the common way to preprocess the data, construct machine learning models, and perform hyper-parameter tuning to find the best model :trophy:. It is no black-box as you can see exactly how the ML pipeline is constructed (with a detailed Markdown report for each ML model).

If you want to understand what happens to raw input data before training, see [Preprocessing](features/preprocessing.md).

The `mljar-supervised` will help you with:

- explaining and understanding your data,
- trying many different machine learning models,
- creating Markdown reports from analysis with details about all models,
- saving, re-running and loading the analysis and ML models.

It can also generate Mercury apps from trained models for local use, quick publishing, or deployment on your own server. See the [Apps](features/apps.md) guide.

## What's good in it? 

- `mljar-supervised` creates markdown reports from AutoML training full of ML details and charts. 
- It can compute the `Baseline` for your data. Compare trained models against this reference on the same evaluation data. An improvement alone does not establish that a model is useful for your application. The `Baseline` is computed based on prior class distribution for classification, and simple mean for regression.
- This package is training simple `Decision Trees` with `max_depth <= 5`, so you can easily visualize them with amazing [dtreeviz](https://github.com/parrt/dtreeviz) to better understand your data.
- The `mljar-supervised` is using simple linear regression and include its coefficients in the summary report, so you can check which features are used the most in the linear model.
- It is using a many algorithms: `Baseline`, `Linear`, `Random Forest`, `Extra Trees`, `LightGBM`, `Xgboost`, `CatBoost`, `Neural Networks`, and `Nearest Neighbors`.
- It can do features preprocessing, like: missing values imputation and converting categoricals. What is more, it can also handle target values preprocessing (You won't believe how often it is needed!). For example, converting categorical target into numeric.
- It can tune hyper-parameters with `not-so-random-search` algorithm (random-search over defined set of values) and hill climbing to fine-tune final models.
- It can compute Ensemble based on greedy algorithm from [Caruana paper](http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf).
- It can stack models to build level 2 ensemble (available in `Compete` mode or after setting `stack_models` parameter).
- It cares about explainability of models: for every algorithm, the feature importance is computed based on permutation. SHAP explanations are available for supported algorithms, depending on `explain_level`; they are not enabled for every backend. See [Explainability](features/explain.md).


## Installation 

Install the beta from PyPI:

```bash
python -m pip install "mljar-supervised==1.4.0b1"
```

From source code:

```
git clone https://github.com/mljar/mljar-supervised.git
cd mljar-supervised
pip install .
```

Installation for development
```
git clone https://github.com/mljar/mljar-supervised.git
cd mljar-supervised
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -e .
```

Running in docker with Jupyter notebook:
```
FROM python:3.12-slim
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN pip3 install "mljar-supervised==1.4.0b1" jupyter
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
