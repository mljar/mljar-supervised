# AutoML Modes

## Built-in modes

There are 3 built-in modes available in `AutoML`:

- **Explain** - to be used when the user wants to explain and understand the data. 
- **Perform** - to be used when the user wants to train a model that will be used in real-life use cases.
- **Compete** - To be used for machine learning competitions (maximum performance!).

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


## Custom modes

User can define his own modes by setting the parameters in `AutoML` constructor ([`AutoML` API](/api)).

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
