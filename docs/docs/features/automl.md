# Steps of AutoML

AutoML searches for models in stages. The stages available in a run depend on its
mode, algorithms, validation strategy, data, and remaining time. The terminal log
shows the scheduled stages; a scheduled stage can still be skipped when no
eligible candidate fits its constraints.

!!! note "Names of steps (`fit_level`)"
    The documentation uses exactly the same names as used in the code to describe each training level.
    You can see these names in the terminal during `AutoML` training.


## `adjust_validation`

This step is used only in selected `Compete` mode runs.

Its purpose is to quickly estimate how expensive full validation will be and adjust the validation strategy to the available time budget.

In practice, `mljar-supervised`:

- trains a fast `Decision Tree` first,
- measures how long it takes,
- uses that as a rough estimate of training cost,
- then decides whether the available budget can afford:
  - split validation,
  - `5`-fold cross-validation,
  - or `10`-fold cross-validation.

If there is enough time, AutoML switches from fast split validation to cross-validation.

If there is not enough time, it stays with split validation.

So this step is not an extra modeling trick. It is a budget-aware calibration step used to choose a realistic validation strategy.

### When does it appear?

You can see `adjust_validation` when:

- `mode="Compete"`
- `total_time_limit` is set
- `validation_strategy="auto"`
- validation was not already forced to the fastest split mode

### What can happen after this step?

Depending on the estimated budget:

- AutoML can switch to `5`-fold CV
- AutoML can switch to `10`-fold CV
- AutoML can stay with split validation

There are also two side effects that can surprise users:

- the first `Decision Tree` used only for estimation can be removed
- stacking can be disabled:
  - for small datasets
  - or when split validation is kept

### Why can the first `Decision Tree` disappear?

If AutoML decides to switch from split validation to cross-validation, it removes the initial `Decision Tree` that was trained only to estimate speed. After that, training continues with the adjusted validation strategy.

### Why can stacking be disabled?

Stacking needs validation outputs that are not always practical in the fastest split-based setup. It can also be disabled for small datasets after validation adjustment.

### How to avoid this step?

If you do not want automatic validation adjustment, set validation explicitly in `AutoML`:

```python
automl = AutoML(
    mode="Compete",
    validation_strategy={
        "validation_type": "kfold",
        "k_folds": 5,
        "shuffle": True,
        "stratify": True,
    },
)
```

If validation is explicitly provided, AutoML will use it instead of trying to adjust it automatically.

## `simple_algorithms`

After any validation adjustment, AutoML checks eligible simple algorithms:

- `Baseline`,
- `Decision Tree`,
- `Linear`.

The stage proposes one configuration for each selected simple algorithm. Cross-validation can train multiple folds for a configuration:

- `Baseline` will provide the baseline result without complex ML involved.
    - For classification, `Baseline` returns training class proportions as probabilities and the most frequent class as its predicted label.
    - `Baseline` returns the mean of the target from training data for regression tasks.
    - Compare models with this reference on the same evaluation data. Check whether any improvement persists on held-out data and matters for your application; there is no universal percentage threshold. See the [random-data example](../tutorials/random.md).
- `Decision Tree` fits a small tree. When explanations are enabled, its report can include a tree visualization to help inspect the learned rules.
- `Linear` will provide simple ML model. You can inspect coefficients of the model.

The models in this step should be quickly trained, so you will get fast intuition about your data and the solved problem.

## `default_algorithms`

This stage starts from each eligible algorithm's default configuration for the selected task. It provides initial models to compare and improve during later stages.

The following algorithms can be fitted in this step:

- `Random Forest`,
- `Extra Trees`,
- `Xgboost`,
- `LightGBM`,
- `CatBoost`,
- `Neural Network`,
- `Nearest Neighbors`.

Eligible optional Compete algorithms (`RealMLP`, `TabM`, `Kumo`, and `TabICLv2`)
can also run here when their dependencies and runtime requirements are satisfied.
The stage proposes one default configuration per eligible algorithm. A candidate
can be skipped or fail, and cross-validation can train several folds for that
configuration.

## `not_so_random`

This step performs Random Search over defined set of hyperparameters (hence the name).

The following algorithms can be fitted in this step:

- `Random Forest`,
- `Extra Trees`,
- `Xgboost`,
- `LightGBM`,
- `CatBoost`,
- `Neural Network`,
- `Nearest Neighbors`.

Eligible optional Compete algorithms can also participate. For each algorithm, up to `start_random_models-1` models are tuned (there is `-1` because one model is used in the `default_algorithms` step). The exact number cannot be given in advance, because in this step there can be models drawn with the same hyperparameters set, in such case the duplicate models are omitted.

For example, if you set:

```py
automl = AutoML(algorithms=["Xgboost", "CatBoost"], start_random_models=3)
```

If both algorithms remain eligible and the budget permits, the example proposes:

- `1` model for `Xgboost` and `1` model for `CatBoost` with default hyperparameters (`default_algorithms` step)
- `2` models for `Xgboost` and `2` models for `CatBoost` with randomly selected hyperparameters (`not_so_random` step)


!!! tip "Hyperparameters set for algorithm"
    You can check which hyperparameters will be optimized for each algorithm in `mljar-supervised` code.

    - To get list of all algorithms: check [`supervised/algorithms`](https://github.com/mljar/mljar-supervised/tree/master/supervised/algorithms) directory
    - To check hyperparameters for `Xgboost`: check [`supervised/algorithms/xgboost.py`](https://github.com/mljar/mljar-supervised/tree/master/supervised/algorithms/xgboost.py#L170). The hyperparameters are defined at the end of the file.



!!! info "To skip `not_so_random` step"
    Please set `start_random_models=1` in the `AutoML` constructor to skip `not_so_random` step.


## `mix_encoding`

When eligible categorical columns and algorithms are available, this stage tries
alternative categorical encodings and compares the resulting models on the same
validation strategy. It does not replace the encodings of already trained models.
Set `mix_encoding=False` to disable it.

## `golden_features`

Golden features combine numerical input columns with arithmetic operations and
rank the resulting candidates. AutoML tries these features with up to three
eligible high-performing models, reusing their hyperparameters. Generated
features are candidates for validation, not a guarantee of better predictions.
Definitions are saved in `golden_features.json` in the results directory, and
candidate names use the `_GoldenFeatures` suffix.

See [Golden Features](golden_features.md) for transformation details.
Set `golden_features=False` to disable this stage.

## `feature_discovery`

This stage uses the included `features-goldmine` package to propose up to three
feature sets. It follows `golden_features` and precedes `kmeans_features`.
`feature_discovery="auto"` enables it in eligible Compete runs; no optional
installation extra is needed. Set `feature_discovery=False` to disable it, or
explicitly enable it in another supported mode with `True`.

AutoML selects up to three eligible parent models, subject to
`top_models_to_improve` and the remaining budget. It reuses each parent's
hyperparameters and preprocessing, which may already include golden features.
The generator is fitted separately inside each training fold. Training uses
out-of-fold target encodings; validation and prediction use frozen mappings.
Shuffled split and K-fold validation are supported; custom, grouped, and
time-ordered validation are not supported by this stage.

Candidates use the `_FeatureDiscovery_<SetName>` suffix. Only sets available
across the required folds are tried. Model reports link to Markdown and JSON
transformation descriptions under `feature_discovery/` in the results directory.
Reloading a model reuses the saved transformations; it does not repeat discovery.

The scheduler reserves time for training the resulting candidates. When all
three feature-engineering stages are enabled, their combined scheduling
allocation is shared **30% golden features, 50% feature discovery, and 20% KMeans**.
These are proportions of the feature-engineering allocation, not of the entire
run or guaranteed wall-clock durations. See [Modes](modes.md) for scheduling
weights and [Feature discovery](feature_discovery.md) for eligibility, profiles,
resource limits, and saved artifacts.

## `kmeans_features`

This stage adds distances to cluster centers and a cluster assignment derived
from numerical columns. Imputation, scaling, and clustering are fitted on each
training fold and saved for validation and prediction. AutoML tries up to three
eligible parent configurations; model names use `_KMeansFeatures`.
Set `kmeans_features=False` to disable this stage.

## `insert_random_feature`

This step is a first part of Feature Selection procedure. Please refer to the [Features Selection](features_selection.md) section in the documentation for details.

During this step:

- The random feature is added to the original data. (New column `random_feature` will appear in the data)
- The best model is selected from previous steps and its hyperparameters are used to train model with newly inserted feature.
- After the training, the feature importance (permutation-based) is computed (forced by setting `explain_level=1`).
- Features with lower importance than `random_feature` are saved to `drop_features.json` file (available in `results_path`).

!!! info "To skip `insert_random_feature` step"
    Please set `features_selection=False` in the `AutoML` constructor to skip `insert_random_feature` step.
    If you disable this step, the `features_selection` step will be skipped as well.

## `features_selection`

This step is a second part of Feature Selection procedure. Please refer to the [Features Selection](features_selection.md) section in the documentation for details.

In this step:

- The best model for each algorithm is selected (from previous steps).
- Its hyperparameters are reused and it is trained using only selected features.
- If all features are important, this step will be skipped. (No models will be trained)

There are considered the following algorithms during this step:

- `Random Forest`,
- `Extra Trees`,
- `Xgboost`,
- `LightGBM`,
- `CatBoost`,
- `Neural Network`.


!!! info "To skip `features_selection` step"
    Please set `features_selection=False` in the `AutoML` constructor to skip `features_selection` step.
    If you disable this step, the `insert_random_feature` step will be skipped as well.


## `hill_climbing`

In the `hill_climbing` step the fine tuning of models is done :muscle:.

There can be several `hill_climbing` steps, they are described with counter at the end of step name. For example, `hill_climbing_1`, `hill_climbing_2`, ... The number of `hill_climbing` steps is controlled with `hill_climbing_steps` parameter in `AutoML` constructor.

In each `hill_climbing` step, the top performing models from each algorithm are tuned further. The number of selected top models is controlled with `top_models_to_improve` parameter from `AutoML.__init__()`.

If a model is selected for further tuning, then only one randomly selected hyperparameter from its setting is changed. The selected hyperparameter will be changed in two directions.

!!! example "How are hyperparameters fine tuned?"
    Example:

    - We have top model `Xgboost`, with `max_depth=5`.
    - The `max_depth` is selected for further tuning. The `max_depth` can take values from array `[1, 2, 3, 4, 5, 6, 7, 8, 9]`.
    - There will be created two new `Xgboost` models with `max_depth=4` and `max_depth=6`.

!!! info "To skip `hill_climbing` steps"
    Please set `hill_climbing_steps=0` in the `AutoML` constructor to skip `hill_climbing` steps.

## `ensemble`

This stage selects a weighted combination of eligible models using their saved
validation or out-of-fold predictions. It runs up to **30 greedy selection
rounds**, retaining the best validation combination. A model may be selected
more than once to increase its weight, so the final ensemble can contain fewer
than 30 distinct members. It does not necessarily include every trained model.
The round limit is fixed and is not a public parameter.

!!! info "To skip `ensemble` step"
    Please set `train_ensemble=False` in the `AutoML` constructor to skip `ensemble` step.

## `stack`

Stacking trains additional models on the original input columns plus predictions
from selected earlier models. For training rows, these added columns come from
saved **out-of-fold predictions**: the corresponding base fold did not train on
those rows. AutoML reuses these evaluation predictions rather than predicting
the training rows with a model fitted on all of them.

AutoML selects at most **five valid base models per algorithm**, ranked by
validation loss. This is an input limit per algorithm, not a five-model limit
for the whole run. Baseline models, random-feature probes, ensembles, and existing
stacks are excluded; fairness constraints apply when enabled. Eligible selected
configurations are retrained on the expanded input, with names ending in
`_Stacked`. Training-cost checks can skip unaffordable candidates.

At inference, AutoML computes the required base predictions, builds the saved
stacked-column schema, and then predicts with the stacked model. Predictions
shared by ensemble or stacking dependencies are reused within that prediction
request. The cache is not retained between separate prediction calls.

Split and repeated validation disable stacking. Automatic validation adjustment
can also disable it for small datasets. The number of available models, remaining
time, class count, and fairness constraints can prevent stacking from starting.
Setting `stack_models=True` requests stacking but does not override these checks.
Keep the complete saved results directory so the base models remain available.
See [Stacking and ensembles](stacking-ensemble.md) for dependency selection.

!!! info "To skip `stack` step"
    Please set `stack_models=False` in the `AutoML` constructor to skip `stack` step.


## `ensemble_stacked`

This stage selects a weighted combination from eligible ordinary and stacked
models using the same limit of 30 selection rounds. A selected stacked member
also needs its upstream base models, so this limit does not bound the total
prediction dependency count. The stage is scheduled when both `stack_models`
and `train_ensemble` are enabled and remains subject to stacking constraints.

!!! info "To skip `ensemble_stacked` step"
    Please set `train_ensemble=False` in the `AutoML` constructor to skip `ensemble` step.
