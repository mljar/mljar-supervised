---
description: How to use a custom evaluation metric in MLJAR AutoML by passing a Python function directly as eval_metric.
social:
  cards_layout: default/variant
---

# Custom eval metric

`mljar-supervised` supports custom evaluation metrics.

You can pass your own Python function directly as the `eval_metric` argument in `AutoML`.

## Basic usage

The function should have this interface:

```python
def my_custom_metric(y_true, y_predicted, sample_weight=None):
    # compute score
    return score
```

Then use it directly:

```python
from supervised import AutoML

automl = AutoML(
    results_path="AutoML_custom_metric",
    eval_metric=my_custom_metric,
)
automl.fit(X, y)
```

## Important rule: the metric must be minimized

Custom metrics in `mljar-supervised` are always treated as metrics to minimize.

This means:

- if lower is better, return the value directly
- if higher is better, return its negative value

For example:

- MSE can be returned directly
- precision, F1, or AUC should usually return `-value`

## Regression example

```python
import numpy as np
from supervised import AutoML

def custom_mse(y_true, y_predicted, sample_weight=None):
    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)
    return np.mean((y_true - y_predicted) ** 2)

automl = AutoML(
    results_path="AutoML_regression_custom_metric",
    eval_metric=custom_mse,
)
automl.fit(X, y)
```

## Classification example

For classification, `y_predicted` can contain probabilities, so you may need to apply thresholding or `argmax` inside your metric.

```python
import numpy as np
from sklearn.metrics import precision_score
from supervised import AutoML

def positive_class_precision(y_true, y_predicted, sample_weight=None):
    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)

    if y_predicted.ndim == 2 and y_predicted.shape[1] == 1:
        y_predicted = y_predicted.ravel()

    if y_predicted.ndim == 1:
        y_predicted = (y_predicted > 0.5).astype(int)
    else:
        y_predicted = np.argmax(y_predicted, axis=1)

    value = precision_score(y_true, y_predicted, sample_weight=sample_weight)

    # higher precision is better, so return negative value
    return -value

automl = AutoML(
    results_path="AutoML_classification_custom_metric",
    eval_metric=positive_class_precision,
)
automl.fit(X, y)
```

## Notes

- the metric function must return a single numeric value
- the metric should handle `sample_weight=None`
- the metric will be used for early stopping and model selection
- the metric should be deterministic and reasonably fast

## FAQ

### Can I pass a function directly?

Yes. This is the supported public interface:

```python
automl = AutoML(eval_metric=my_custom_metric)
```

### Should I pass `eval_metric="user_defined_metric"`?

No. That name is used internally. In user code, pass the function itself.

### Can I maximize my metric directly?

No. Convert it to a minimization target, usually by returning `-value`.

### Why do I need thresholding for some classification metrics?

Because many classification metrics such as precision or F1 expect class labels, while model predictions during evaluation can be probabilities.

## Related pages

- [AutoML API](../api.md)
- [Save and Load models](save-and-load-models.md)
- [Preprocessing](preprocessing.md)
