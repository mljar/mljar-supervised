---
description: Python API reference for MLJAR AutoML, including the AutoML class, training options, prediction methods, reports, fairness settings, and app generation helpers.
social:
  cards_layout: default/variant
---

# API documentation

If you are looking for how trained models are stored and reloaded, see [Save and Load models](features/save-and-load-models.md).
If you need a user-defined evaluation function, see [Custom eval metric](features/custom-eval-metric.md).

## F1 metrics and classification scoring

Use `eval_metric="f1_macro"` to give each class equal importance during early
stopping, tuning, model selection, and ensemble selection:

```python
from supervised import AutoML

automl = AutoML(ml_task="multiclass_classification", eval_metric="f1_macro")
automl.fit(X_train, y_train)
```

| Metric | Averaging |
| --- | --- |
| `f1` | Positive-class F1 for binary classification; micro F1 for multiclass classification. Existing behavior is preserved. |
| `f1_micro` | Aggregate true positives, false positives, and false negatives across classes. |
| `f1_macro` | Mean of per-class F1 scores, with equal weight for each class. |
| `f1_weighted` | Mean of per-class F1 scores, weighted by class support. |

Micro F1 equals accuracy for single-label classification. Macro F1 can reveal poor
performance on minority classes even when accuracy is high. Explicit F1 variants
include all classes represented by the probability outputs, including classes
absent from a validation fold. Undefined per-class F1 scores contribute zero.
Sample weights are supported.

`automl.score(X_test, y_test)` returns accuracy for classification and R² for
regression, following scikit-learn conventions. It does not use `eval_metric`.
To compute macro F1 on held-out predictions:

```python
from sklearn.metrics import f1_score

predictions = automl.predict(X_test)
macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
```

The leaderboard and model summary identify the optimization metric. Classification
reports also show auxiliary metrics, including accuracy; those do not change the
selected optimization metric. See [issue #646](https://github.com/mljar/mljar-supervised/issues/646).

## `AutoML` class

::: supervised.automl.AutoML
