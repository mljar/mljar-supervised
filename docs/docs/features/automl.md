# Steps of AutoML

The training of `mljar-supervised` AutoML is divided into steps. Each step represents the actions that are common in the process of searching for the best performing Machine Learning model in the ML pipeline. Below are described steps of AutoML. 

!!! note "Names of steps (`fit_level`)"
    The documentation uses exactly the same names as used in the code to describe each training level.
    You can see these names in the terminal during `AutoML` training.


## `simple_algorithms`

The first step in the `AutoML` training is to check the simplest algorithms to get quick insights:

- `Baseline`,
- `Decision Tree`,
- `Linear`.

One model is trained for each of the algorithms. By analyzing results of each model you can learn a lot about the analyzed dataset:

- `Baseline` will provide the baseline result without complex ML involved. 
    - `Baseline` returns the most frequent label from the training data for classification tasks.
    - `Baseline` returns the mean of the target from training data for regression tasks.
    - If further results are much better than `Baseline` results, **it justify the use of Machine Learning**. Take a look at [tutorial with classification of random data](/tutorials/random/).
- `Decision Tree` will provide results for simple tree (with `max_depth` up to 4). The tree will be visualized with [`dtreeviz`](https://github.com/parrt/dtreeviz) package, so you can easily check in the Markdown report what are the rules in the tree. 
- `Linear` will provide simple ML model. You can inspect coefficients of the model.

The models in this step should be quickly trained, so you will get fast intuition about your data and the solved problem.

## `default_algorithms`

In this step, the models are trained with default hyperparameters. Regardless of the data, the hyperparameter values used are always the same for each algorithm. In this step, you can compare the results of default models from other datasets and get intuition about your problem complexity.

The following algorithms can be fitted in this step:

- `Random Forest`,
- `Extra Trees`,
- `Xgboost`,
- `LightGBM`,
- `CatBoost`,
- `Neural Network`,
- `Nearest Neighbors`.

There is exactly one model fitted for each algorithm in this step. (Each algorithm has one set of default hyperparameter values for each ML task).

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

For each algorithm up to `start_random_models-1` models are tuned (there is `-1` because one model is used in the `default_algorithms` step). The exact number cannot be given in advance, because in this step there can be models drawn with the same hyperparameters set, in such case the duplicate models are omitted. 

For example, if you set:

```py
automl = AutoML(algorithms=["Xgboost", "CatBoost"], start_random_models=3)
```

There will be trained:

- `1` model for `Xgboost` and `1` model for `CatBoost` with default hyperparameters (`default_algorithms` step)
- `2` models for `Xgboost` and `2` models for `CatBoost` with randomly selected hyperparameters (`not_so_random` step)


!!! tip "Hyperparameters set for algorithm"
    You can check which hyperparameters will be optimized for each algorithm in `mljar-supervised` code.
    
    - To get list of all algorithms: check [`supervised/algorithms`](https://github.com/mljar/mljar-supervised/tree/master/supervised/algorithms) directory
    - To check hyperparameters for `Xgboost`: check [`supervised/algorithms/xgboost.py`](https://github.com/mljar/mljar-supervised/tree/master/supervised/algorithms/xgboost.py#L170). The hyperparameters are defined at the end of the file.



!!! info "To skip `not_so_random` step"
    Please set `start_random_models=1` in the `AutoML` constructor to skip `not_so_random` step.


## `golden_features`

Golden Features are new features constructed from original data which have great predictive power. Please see the [Golden Features](/features/golden_features/) section in the documentation for more details about how are they constructed. 

The Golden Features are constructed only once during `AutoML` fit. They are saved in `results_path` in `golden_features.json` file.

After creation of the Golden Features they are added to the original data and following algorithms are trained:

- `Xgboost`
- `CatBoost`
- `LightGBM`.

There is trained `1` model for each algorithm. The hyperparameters used in the model are selected from the best performing models from previous steps (`default_algorithms` and `not_so_random`).

!!! info "To skip `golden_features` step"
    Please set `golden_features=False` in the `AutoML` constructor to skip `golden_features` step.

## `insert_random_feature`

This step is a first part of Feature Selection procedure. Please refer to the [Features Selection](/features/features_selection/) section in the documentation for details.

During this step:

- The random feature is added to the original data. (New column `random_feature` will appear in the data)
- The best model is selected from previous steps and its hyperparameters are used to train model with newly inserted feature.
- After the training, the feature importance (permutation-based) is computed (forced by setting `explain_level=1`).
- Features with lower importance than `random_feature` are saved to `drop_features.json` file (available in `results_path`).

!!! info "To skip `insert_random_feature` step"
    Please set `feature_selection=False` in the `AutoML` constructor to skip `insert_random_feature` step.
    If you disable this step, the `features_selection` step will be skipped as well.

## `features_selection`

This step is a second part of Feature Selection procedure. Please refer to the [Features Selection](/features/features_selection/) section in the documentation for details.

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
    Please set `feature_selection=False` in the `AutoML` constructor to skip `features_selection` step.
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

During `ensemble` step all models from previous steps are ensembled.

!!! info "To skip `ensemble` step"
    Please set `train_ensemble=False` in the `AutoML` constructor to skip `ensemble` step.
    
## `stack`

In this step models are stacked (level `1`).

TODO: need add more decription.

!!! info "To skip `stack` step"
    Please set `stack_models=False` in the `AutoML` constructor to skip `stack` step.
    

## `ensemble_stacked`

During `ensemble_stacked` all models from previous steps are ensembled (stacked and unstacked models). To have this step enabled, the `stack_models` paramter in `AutoML` should be `True`.

!!! info "To skip `ensemble_stacked` step"
    Please set `train_ensemble=False` in the `AutoML` constructor to skip `ensemble` step.

