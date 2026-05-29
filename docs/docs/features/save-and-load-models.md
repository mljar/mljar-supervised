---
description: How MLJAR AutoML saves trained models, what is stored in results_path, how to load a trained run, and what files like learner_fold_0.catboost mean.
social:
  cards_layout: default/variant
---

# Save and Load models

`mljar-supervised` saves trained models automatically during `AutoML.fit()`.

In normal usage, you do **not** save or load individual learner files yourself. The main artifact is the whole AutoML directory stored in `results_path`.

## The normal workflow

Train a model:

```python
from supervised import AutoML

automl = AutoML(results_path="AutoML_1")
automl.fit(X, y)
```

Later, load it again:

```python
from supervised import AutoML

automl = AutoML(results_path="AutoML_1")
```

If `results_path` already contains a trained AutoML run, it is loaded automatically.

Then you can use it normally:

```python
predictions = automl.predict(X_test)
```

## What is saved

The full AutoML run is saved in the directory given by `results_path`.

That directory contains items such as:

- `params.json`
- `leaderboard.csv`
- top-level `README.md`
- model subdirectories
- preprocessing information
- validation artifacts
- algorithm-specific model files

Think of `results_path` as the saved AutoML project directory.

## Important file: `params.json`

The key file used to recognize a trained AutoML directory is:

```text
params.json
```

This is why `AutoML(results_path="AutoML_1")` can load a trained run automatically. The directory is the public loading entrypoint, not a single learner file.

## What are files like `learner_fold_0.catboost`?

Files such as:

```text
learner_fold_0.catboost
```

are backend model files created by a specific algorithm.

For example:

- CatBoost models are stored in CatBoost-native format
- scikit-learn based models use their own saved format
- other learners can have their own native artifacts

These files are **not** the main public interface of `mljar-supervised`.

## Can I load `learner_fold_0.catboost` as JSON?

No.

A file like `learner_fold_0.catboost` is not a JSON file. It is a CatBoost model file in CatBoost’s native format.

If you try to read it as JSON, it will not work.

## Which object should I load in normal usage?

In normal usage, load the full AutoML run by pointing `AutoML` at the saved directory:

```python
automl = AutoML(results_path="AutoML_1")
```

This is the recommended way because it restores:

- the best model selection
- model orchestration
- preprocessing pipeline
- prediction interface
- reports and metadata

## Why not load learner files directly?

You usually should not start from an individual learner file because `mljar-supervised` adds important context around the raw backend model:

- preprocessing
- target transformations
- class label handling
- model selection
- ensemble or stacked-model logic

Loading only the backend learner can bypass that context.

## When direct loading can make sense

Direct loading is advanced usage.

You might do it if you want to inspect a backend model with the original library, for example CatBoost. But in that case, you are working at the backend-model level, not the full AutoML level.

For standard prediction and reuse, load `results_path` with `AutoML`.

## Where to inspect model details

If your goal is to understand the trained model, start here:

- the main `README.md` in `results_path`
- the `README.md` inside each model subdirectory
- feature importance plots
- SHAP explanations, if enabled
- leaderboard and structured reports

This is usually more useful than reading backend model files directly.

## FAQ

### Do I need to call `save()` manually?

No. Models are saved automatically during training.

### How do I reload a trained AutoML run?

Use:

```python
automl = AutoML(results_path="AutoML_1")
```

### Can I import `.catboost` as JSON?

No. It is not JSON.

### What is the main saved artifact?

The whole `results_path` directory.

### What should I share or archive?

Share or archive the full AutoML directory, not just a single learner file.

## Related pages

- [Preprocessing](preprocessing.md)
- [Apps](apps.md)
- [AutoML API](../api.md)
