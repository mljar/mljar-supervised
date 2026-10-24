# AutoML Modes

## Algorithm early stopping in Compete

Starting with 1.4.0rc7, new ordinary Compete runs automatically stop further
original-feature tuning of algorithms that remain uncompetitive after at least
three successful, distinct configurations. There is no additional argument.
The existing search settings and time budget still determine how many initial
configurations can run. Changing only a random seed does not count as a new
configuration; failed and resource-skipped trials do not count.

The first successful, finite-scoring individual model after validation adjustment
provides a fixed reference. With every metric expressed in AutoML's lower-is-better
convention, further trials are skipped when:

```text
algorithm_best_loss - leader_loss > 0.20 * (reference_loss - leader_loss)
```

An algorithm must retain at least 80% of the leader's improvement over the
reference. This is **not** the same as requiring its raw error to be within 20%
of the leader. For example, reference log loss 1.0 and leader log loss 0.4 permit
an algorithm's best log loss up to 0.52. The same rule handles all built-in and
custom metrics, including negative scores, and is unchanged by a positive scale
factor or a score offset. Invalid comparisons and a leader that has not improved
over the reference do not trigger early stopping.

The check runs during random search and original-feature hill climbing, including
immediately before each fit. It compares completed models with the same validation
setup and sample weights. Changed-feature trials, their hill climbing, stacking,
and ensemble selection keep their existing opportunities. Previously trained
models and predictions remain available. Other modes, Optuna searches, and
fairness-driven searches retain their existing policies.

The stage ratios stay the same. Eligible algorithms share the tuning allowance,
subject to remaining wall-clock limits; unused time can carry into later stages.
This can improve how a fixed budget is spent without shortening the whole run.

AutoML prints one concise message per paused algorithm. The Markdown report links
to `algorithm_early_stopping.json`, containing reference/leader scores and skipped
candidates. Policy state is also saved in `progress.json`, so resumed runs retain
their decisions. Older interrupted runs without policy metadata keep their
original search behavior. Loading a completed model for prediction requires no
new calculations.

## Candidate time budgets in Compete

With a total time limit, ordinary Compete searches share each stage's allowance
among the participating algorithms. Default trials receive a whole-model budget,
including all validation folds and repeats, instead of a minimum of 60 seconds
per fold. Time spent on failed and timed-out attempts also consumes the algorithm's
stage allowance. This spending safeguard is separate from accuracy early stopping:
it does not label an algorithm uncompetitive before three successful configurations.

Before each fold, AutoML checks whether the remaining folds can fit the candidate
allowance. Backend fitting receives at most 80% of its current per-fold allowance,
leaving room for preparation and prediction. Deadlines are cooperative: individual
noninterruptible operations may overrun them. A timed-out configuration is retried
only when its allowance exceeds the previously observed timeout duration; changed
configurations and feature representations have separate histories.

Expected deadline stops are saved as skipped candidates in `budget_skips.json`,
linked from the model report, and summarized in the console. Genuine failures
remain in `errors.md`. Timing history is restored on resume. An explicit
`model_time_limit` retains control over per-model fitting, and searches without a
total time limit do not receive the new stage spending caps.

Stacking admission also reuses completed-fold timing from candidates that could
not finish. Compatible subsequent candidates are skipped before training when
the measured full-model estimate exceeds their allowance. For RealMLP, equal or
larger depth and width can reuse an estimate when input representation, validation,
device, batch settings and other cost settings match. Smaller networks or mixed
depth/width changes receive a new probe; no linear speedup is assumed. Other
algorithms initially require equivalent cost settings. A sufficiently larger
allowance permits another attempt. Successful measurements can revise estimates.

Observations persist in `progress.json`; budget-skip evidence identifies the source
attempt, estimated completion time and allowance. Admission skips do not create
observations or count toward accuracy early stopping. Existing model names and
stacking limits are unchanged.

## Feature-engineering stage ratios

Golden features, feature discovery, and KMeans remain separate, sequential stages.
Their combined scheduling budget is distributed as 30% golden features, 50%
feature discovery, and 20% KMeans. Each allocation covers feature preparation and
training the models that use those features; it is not a guaranteed runtime.

With all stages enabled, including mixed categorical encoding, the weights are:

| Stage | Weight | Approximate allocation in a one-hour Compete run |
| --- | ---: | ---: |
| Golden features | 0.0375 | 100 seconds |
| Feature discovery | 0.0625 | 167 seconds |
| KMeans features | 0.0250 | 67 seconds |

Their combined weight remains 0.125. Other stages keep their existing allocations.
If a feature stage is disabled or ineligible, its previous allocation is omitted
from the combined budget, and the 30:50:20 proportions are normalized among the
remaining feature stages. For example, golden features and feature discovery
without KMeans share their available budget in a 3:5 ratio. Unused time can carry
forward to later stages.

No API arguments or model-name postfixes change. These scheduling weights apply
when using `total_time_limit`; an explicit `model_time_limit` retains its existing
behavior. Previously built release wheels and their benchmark runs are unchanged.

## Built-in modes

There are 3 built-in modes available in `AutoML`:

- **Explain** - to be used when the user wants to explain and understand the data. 
- **Perform** - to be used when the user wants to train a model that will be used in real-life use cases.
- **Compete** - To be used for machine learning competitions (maximum performance!).

Kumo and TabICLv2 require [the optional TFM dependencies](../install.md#kumo-and-tabiclv2-on-cpu-or-nvidia-gpu).
Compete selects them automatically when the tested backend and CUDA are available.
For CPU runs, select `algorithms=["Kumo", "TabICLv2"]` explicitly in Compete mode.
They are not available in Explain, Perform, or Optuna mode.

| | |**AutoML Modes**| |
|--- |--- |--- |--- |
||**Explain**|**Perform**|**Compete**|
|| | ***Algorithms***| |
|Baseline|:fontawesome-solid-check:{: .green } |||
|Linear|:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } ||
|Decision Tree|:fontawesome-solid-check:{: .green } ||:fontawesome-solid-check:{: .green } |
|Random Forest|:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|Extra Trees|||:fontawesome-solid-check:{: .green } |
|XGBoost|:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|LightGBM||:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green }|
|CatBoost||:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|Neural Network|:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|Nearest Neighbors|||:fontawesome-solid-check:{: .green } |
|[Kumo](sdm.md) (optional)|||:fontawesome-solid-check:{: .green } |
|[TabICLv2](sdm.md) (optional)|||:fontawesome-solid-check:{: .green } |
|Ensemble|:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|Stacking|||:fontawesome-solid-check:{: .green } |
|||***Steps***||
|simple_algorithms|:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|default_algorithms|:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|not_so_random||:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|golden_features||:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|insert_random_feature||:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|feature_selection||:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|hill_climbing_1||:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|hill_climbing_2||:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|ensemble|:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|stack|||:fontawesome-solid-check:{: .green } |
|ensemble_stacked|||:fontawesome-solid-check:{: .green } |
|| | ***Validation***||
||75%/25% train/test split|5-fold CV, Shuffle, Stratify| 10-fold CV, Shuffle, Stratify|
|||***Explanations***||
||`explain_level=2`|`explain_level=1`|`explain_level=0`|
|Learning   curves|:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } |
|Importance   plots|:fontawesome-solid-check:{: .green } |:fontawesome-solid-check:{: .green } ||
|SHAP   plots|:fontawesome-solid-check:{: .green } |||
| | |***Tuning***||
|Parameters |`start_random_models=1`, `hill_climbing_steps=0`, `top_models_to_improve=0` |`start_random_models=5`, `hill_climbing_steps=2`, `top_models_to_improve=2`|`start_random_models=10`, `hill_climbing_steps=2`, `top_models_to_improve=3`|
|Models with default hyperparemeters | `1`|`1`|`1`|
|Models with **not_so_random** hyperparemeters | `0` |`4`|`9`|
|`hill_climbing` steps | `0`|`2`|`2`|
|Top models imporoved in each `hill_climbing` step | `0`|`2`|`3`|
|**Total models** tuned for each algorithm[^1] | `1`|about `13`[^2]|about `22`[^2]|

!!! note
    In `Compete` mode, AutoML can automatically adjust validation based on the observed training speed and the available `total_time_limit`. This can introduce the `adjust_validation` step in logs and may switch validation between split, `5`-fold CV, and `10`-fold CV. See [Steps of AutoML](automl.md#adjust_validation).


## Custom modes

User can define custom modes by setting the parameters in the `AutoML` constructor ([`AutoML` API](../api.md)).

Example setting:

```python
automl = AutoML(
    algorithms=["CatBoost", "Xgboost", "LightGBM"],
    model_time_limit=30*60,
    start_random_models=10,
    hill_climbing_steps=3,
    top_models_to_improve=3,
    golden_features=True,
    features_selection=False,
    stack_models=True,
    train_ensemble=True,
    explain_level=0,
    validation_strategy={
        "validation_type": "kfold",
        "k_folds": 4,
        "shuffle": False,
        "stratify": True,
    }
)
```

- It will train models with `CatBoost`, `Xgboost` and `LightGBM` algorithms.
- Each model will be trained for 30 minutes (`30*60` seconds). `total_time_limit` is not set.
- There will be trained about `10+3*3*2=28` unstacked models and `10` stacked models for each algorithm. (There is stacked up to `10` models for each algorithm)
- There will trained `Ensemble` based on unstacked models and `Ensemble_Stacked` from unstacked and stacked models.
- In total there will be about `3*28+2=86` models trained.
- `explain_level=0` means that there will be only learning curves saved. No other explanations will be computed.


[^1]: Not every algorithm is tuned. Models which are not tuned: `Baseline`, `Decision Tree`, `Linear`, `Nearest Neighbors`.
[^2]: 
    The exact number cannot be given, because sometimes the newly generated hyperparameters are rejected 
    during `not_so_random` or `hill_climbing` steps because of model duplicates or invalid hyperparameters set.
