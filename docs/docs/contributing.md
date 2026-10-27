---
title: "Contributing and Community"
description: "Contribute to mljar-supervised by reporting bugs, improving documentation, adding algorithms, writing tests, or helping the AutoML community."
---
# Contributing 

Thank you! :tada:

Thank you for your interest in [`mljar-supervised`](https://github.com/mljar/mljar-supervised) and for willing to contribute! :+1: :smiley:

## How can I contribute? :muscle:

### Reporting Bugs

When creating a new bug report, please make sure to:

- search for existing issues first. If you find a previous report of your issue, please update the existing issue with additional information instead of creating a new one.

- please review the template and guidelines and include as many details as possible.

### Suggest Enhancements

Please try first to see if an issue exists which describes your enhancement. If there is none, please create the new one with a detailed description. Thank you!

### Pull Requests

- Before any contribution, please reach out and discuss what do you want to implement and how (ideally in the created issue).
- For code formatting we use [Black](https://github.com/psf/black).

To run all tests locally:

```
pytest tests 
```

To run only tests for the selected file (for example, NN algorithm):
```
pytest tests/tests_algorithms/test_nn.py -vsx
```

I add `vsx` flags to see what's going on.

### Adding an algorithm

Algorithm implementations inherit from `BaseAlgorithm` in
`supervised/algorithms/algorithm.py`. It uses Python's `ABC` and requires these
methods, either implemented directly or inherited from an adapter:

| Method | Responsibility |
| --- | --- |
| `fit` | Train using the training data and supported validation arguments. |
| `predict` | Return task-appropriate predictions (positive-class probabilities for binary classification, class probabilities for multiclass classification, or numeric predictions for regression). |
| `is_fitted` | Report whether the model is loaded and ready for prediction. |
| `file_extension` | Return the saved model's extension without a leading dot. |
| `copy` | Return an independent copy of the learner and its current model state. |
| `save` | Persist the model to the supplied path. |
| `load` | Restore the model from the supplied path. |

Incomplete subclasses cannot be instantiated. Shared adapters such as
`SklearnAlgorithm` can remain abstract until a concrete algorithm implements the
remaining methods.

The base class provides metadata serialization, lazy loading, interpretation,
and probability formatting. `update` is an optional no-op hook;
`get_metric_name` returns `None` unless the algorithm supplies a metric name.
Override these defaults only when the algorithm needs different behavior.

The abstract contract checks that methods are implemented; it does not validate
their signatures or behavior. Add tests for training and prediction, independent
copies, saving and loading, and prediction after lazy loading. Registered
algorithms must also pass `tests/tests_algorithms/test_algorithm_contract.py`.

## Documentation :scroll:

The project documentation source is in this repository under `docs/`. Deployed docs are available at [https://supervised.mljar.com](https://supervised.mljar.com).

## Help the community

You can contribute without writing code:

- Help other users with their [GitHub issues](https://github.com/mljar/mljar-supervised/issues).
- Write a tutorial about how you use AutoML. Publish it on your own blog or a community site, or contact us about sharing it on the [MLJAR blog](https://mljar.com/blog).
- Star the [repository](https://github.com/mljar/mljar-supervised) to help others discover it. Select **Watch → Custom → Releases** to follow new releases.
- Share your projects and connect with [MLJAR on Twitter](https://twitter.com/MLJARofficial), or with Piotr Płoński on [LinkedIn](https://www.linkedin.com/in/piotr-plonski-mljar/) and [GitHub](https://github.com/pplonski).
- Join the [Slack community](https://mljar-supervised.slack.com/join/shared_invite/zt-gkhfsvhw-H6LMKxxV5adeTmn9V7nbZw#/) to discuss machine learning and AutoML development.

## Contributor License Agreement :necktie:

Before merging any pull request with code updates, we'll need each person contributing to the code base to sign a Contributor License Agreement [here](https://forms.gle/au2qhFWjuuvEmKfM8).
