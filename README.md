![Machine Learning for Humans](images/the-mljar.svg)

[![Build Status](https://travis-ci.org/mljar/mljar-supervised.svg?branch=master)](https://travis-ci.org/mljar/mljar-supervised)
[![PyPI version](https://badge.fury.io/py/mljar-supervised.svg)](https://badge.fury.io/py/mljar-supervised)
[![Coverage Status](https://coveralls.io/repos/github/mljar/mljar-supervised/badge.svg?branch=master)](https://coveralls.io/github/mljar/mljar-supervised?branch=master)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mljar-supervised.svg)](https://pypi.python.org/pypi/mljar-supervised/)

# mljar-supervised

Collection of supervised methods (including processing), used in MLJAR AutoML solution.

With this package you can handle any supervised ML algorithm with the same interface.

# The package is under active development! Please expect a lot of changes!

Please check our [platform](https://github.com/mljar/mljar)!

# Development

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
