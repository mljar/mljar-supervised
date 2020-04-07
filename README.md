# mljar-supervised

[![Build Status](https://travis-ci.org/mljar/mljar-supervised.svg?branch=master)](https://travis-ci.org/mljar/mljar-supervised)
[![PyPI version](https://badge.fury.io/py/mljar-supervised.svg)](https://badge.fury.io/py/mljar-supervised)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mljar-supervised.svg)](https://pypi.python.org/pypi/mljar-supervised/)


## Automated Machine Learning 

`mljar-supervised` is Automated Machine Learning package. It can train ML models for:

- binary classification,
- multi-class classification,
- regression.

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


