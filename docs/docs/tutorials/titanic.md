---
description: End-to-end Titanic classification tutorial with MLJAR AutoML, including training, leaderboard interpretation, explainability, and prediction accuracy.
social:
  cards_layout: default/variant
---

# Classify Titanic passengers

!!! note "Historical example — not a 1.4.0b1 benchmark"
    The tables, screenshots, and console output below reproduce an earlier run
    from the [archived examples repository](https://github.com/mljar/mljar-examples/tree/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification).
    Report links are pinned to snapshot `3035b1e`. The original training
    date, exact package versions, and hardware are not established here. These
    results have not been rerun with `1.4.0b1`; use them to learn how to read a
    report, not to predict current accuracy or runtime. For a new run, use the
    [beta installation instructions](../install.md) and record your environment.


This historical example shows how to read an AutoML classification report, inspect model explanations, and make predictions.

All the code and results are available at the [GitHub](https://github.com/mljar/mljar-examples/tree/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification)


## The code

What does python code do:

- reads Titanic train dataset (the same data as in [Kaggle platform](https://www.kaggle.com/c/titanic)),
- trains `AutoML` object,
- computes predictions using a separately hosted file named `test_with_Survived.csv`. The label provenance of that file is not verified here; the reported accuracy is not a verified Kaggle score or a current benchmark.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from supervised import AutoML

train = pd.read_csv("https://raw.githubusercontent.com/pplonski/datasets-for-start/master/Titanic/train.csv")

X = train[train.columns[2:]]
y = train["Survived"]

automl = AutoML(results_path="AutoML_3")
automl.fit(X, y)

test = pd.read_csv("https://raw.githubusercontent.com/pplonski/datasets-for-start/master/Titanic/test_with_Survived.csv")
predictions = automl.predict(test)
print(f"Accuracy: {accuracy_score(test['Survived'], predictions)*100.0:.2f}%" )
```

As you see from above example the heavy job is done in exactly `2` lines of code:

```python
automl = AutoML(results_path="AutoML_3")
automl.fit(X, y)
```

I will show you step by step what above code produced based on the training data.

## The `Explain` mode

The default `mode` for [`mljar-supervised`](https://github.com/mljar/mljar-supervised) is `Explain`, which means that:

- there will be used `75% / 25%` for train / test split for model training and evaluation,
- there will be trained following algorithms: `Baseline`, `Decision Tree`, `Linear`, `Random Forest`, `Xgboost`, `Neural Network`, and `Ensemble`,
- the [full explanations](../features/explain.md) will be created.

All results created during `AutoML` training will be saved to the hard drive. There will be **Markdown report** in the `README.md` file for each model available (no black-boxes!).

## The AutoML leaderboard report

The main [`README.md`](https://github.com/mljar/mljar-examples/tree/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3) in the report will contain:

- table will all models performance,
- performance plotted as scatter plot and box plot.

The leaderbord:

| Best model   | name                    | model_type     | metric_type   |   metric_value |   train_time | Link                                              |
|:-------------|:------------------------|:---------------|:--------------|---------------:|-------------:|:--------------------------------------------------|
|              | 1_Baseline              | Baseline       | logloss       |       0.666775 |         0.26 | [Results link](https://github.com/mljar/mljar-examples/tree/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/1_Baseline/README.md)              |
|              | 2_DecisionTree          | Decision Tree  | logloss       |       0.648504 |        18    | [Results link](https://github.com/mljar/mljar-examples/tree/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/2_DecisionTree/README.md)          |
|              | 3_Linear                | Linear         | logloss       |       0.593649 |        12.2  | [Results link](https://github.com/mljar/mljar-examples/tree/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/3_Linear/README.md)                |
|              | 4_Default_RandomForest  | Random Forest  | logloss       |       0.448691 |        22.24 | [Results link](https://github.com/mljar/mljar-examples/tree/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/4_Default_RandomForest/README.md)  |
|              | 5_Default_Xgboost       | Xgboost        | logloss       |       0.458922 |        12.63 | [Results link](https://github.com/mljar/mljar-examples/tree/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/5_Default_Xgboost/README.md)       |
|              | 6_Default_NeuralNetwork | Neural Network | logloss       |       0.733411 |        23.84 | [Results link](https://github.com/mljar/mljar-examples/tree/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/6_Default_NeuralNetwork/README.md) |
| **the best** | Ensemble                | Ensemble       | logloss       |       0.436319 |         0.83 | [Results link](https://github.com/mljar/mljar-examples/tree/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/Ensemble/README.md)                |

From the above table you can check what was the performance of the models and how long was the training. There is a `Results link` in the table for each model (please scroll this table if you don't see it), which you can **click** and go into model details :+1: :tada:

The performance is presented in the plots:

![AutoML Performance](https://raw.githubusercontent.com/mljar/mljar-examples/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/ldb_performance.png)

![AutoML Performance Boxplot](https://raw.githubusercontent.com/mljar/mljar-examples/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/ldb_performance_boxplot.png)

## The `Baseline`

The `Baseline` provides a reference on the same evaluation rows. For classification logloss, it predicts training class probabilities. Comparing against it helps interpret a result, but does not by itself establish data quality or whether a model should be used.

Let's compute the percentage difference between the best model (`Ensemble`) and the `Baseline`:

```text
Relative reduction in validation logloss = (0.667 - 0.436) / 0.667 * 100 = 34.6%
```

The selected ensemble has about 34.6% lower validation logloss than the baseline
in the rounded historical values above. This is a reduction in this metric,
not a 34.6-percentage-point increase in accuracy or proof of practical usefulness.

!!! note "Assess improvement on independent data"
    There is no universal 5% threshold that separates useful data from random
    data. Model selection can make validation results optimistic. Use a held-out
    test set with verified labels and a split appropriate for the problem;
    assess uncertainty and the practical consequences of errors. Keep that test
    set out of further tuning decisions.

## `Decision Tree`

Let's look closer into `Decision Tree` [report](https://github.com/mljar/mljar-examples/tree/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/2_DecisionTree/README.md).

The part of report is below:

### Decision Tree hyperparameters
- **criterion**: gini
- **max_depth**: 3
- **explain_level**: 2

### Validation
 - **validation_type**: split
 - **train_ratio**: 0.75
 - **shuffle**: True
 - **stratify**: True

### Optimized metric
logloss

### Training time

17.1 seconds

### Metric details
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.648504 |  nan        |
| auc       | 0.814293 |  nan        |
| f1        | 0.728261 |    0.351143 |
| accuracy  | 0.775785 |    0.351143 |
| precision | 0.843137 |    0.597938 |
| recall    | 0.965116 |    0        |
| mcc       | 0.54213  |    0.351143 |


### Confusion matrix (at threshold=0.351143)
|                     |   Predicted as negative |   Predicted as positive |
|:--------------------|------------------------:|------------------------:|
| Labeled as negative |                     106 |                      31 |
| Labeled as positive |                      19 |                      67 |

There are many metrics and confusion matrix pre-computed.

Additionally, there is a `Decision Tree` visualization:

![Decision Tree visualization](https://raw.githubusercontent.com/mljar/mljar-examples/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/2_DecisionTree/learner_1_tree.svg)

There are created many explanations for each model. Let's check how they look like for `Xgboost` (the best single model).

## The `Xgboost` model

You can check details of `Xgboost` model in the Markdown [report](https://github.com/mljar/mljar-examples/tree/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/5_Default_Xgboost/README.md). Here I will show some parts of the report with short comment.

### Learning curves

The vertical line indicates the optimal number of trees in the `Xgboost` (found with early stopping). This number of trees will be used during computning predictions.

![Xgboost learning curve](https://raw.githubusercontent.com/mljar/mljar-examples/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/5_Default_Xgboost/learning_curves.png)

### Feature Importance

The permutation-based feature importance:

![Permutation based feature importance](https://raw.githubusercontent.com/mljar/mljar-examples/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/5_Default_Xgboost/permutation_importance.png)

From the plot you can see that the most used feature is `Name_mr`. There wasn't such feature in the training data. There was `Name` feature. The `AutoML` used `TF-IDF` transformation (scikit-learn [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)) to construct new features from `Name` text feature.

### SHAP dependence plots

![SHAP dependece plots](https://raw.githubusercontent.com/mljar/mljar-examples/3035b1e39208e71efa979b002b044e923a82bc40/Titanic_Classification/AutoML_3/5_Default_Xgboost/learner_1_shap_dependence.png)


## The test accuracy

The `AutoML` is used to predict the labels for test data samples. The archived console output reports accuracy using the separately hosted
`test_with_Survived.csv` file. Its label provenance is not verified here.
The following numbers are historical output, not beta acceptance criteria:

```python
AutoML directory: AutoML_3
The task is binary_classification with evaluation metric logloss
AutoML will use algorithms: ['Baseline', 'Linear', 'Decision Tree', 'Random Forest', 'Xgboost', 'Neural Network']
AutoML will ensemble availabe models
AutoML steps: ['simple_algorithms', 'default_algorithms', 'ensemble']
* Step simple_algorithms will try to check up to 3 models
1_Baseline logloss 0.666775 trained in 0.23 seconds
2_DecisionTree logloss 0.648504 trained in 17.06 seconds
3_Linear logloss 0.593649 trained in 11.04 seconds
* Step default_algorithms will try to check up to 3 models
4_Default_RandomForest logloss 0.448691 trained in 21.72 seconds
5_Default_Xgboost logloss 0.458922 trained in 17.47 seconds
6_Default_NeuralNetwork logloss 0.718124 trained in 22.08 seconds
* Step ensemble will try to check up to 1 model
Ensemble logloss 0.436478 trained in 0.71 seconds
AutoML fit time: 96.77 seconds
Accuracy: 77.99%
```

## Adapting this example

Choose input columns that will be available when making a prediction. Reserve
held-out evaluation rows with verified labels before training, choose a suitable
validation strategy, and compare against a baseline on the same rows. Inspect
errors and reports, then verify predictions after reloading the saved run.
The historical tables above illustrate the reports; measure results again for
your own data and installed version.
