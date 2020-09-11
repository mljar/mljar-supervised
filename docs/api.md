<a name=".supervised.automl"></a>
## AutoML API Docs

<a name=".supervised.automl.AutoML"></a>
### AutoML

```python
class AutoML()
```

Automated Machine Learning for supervised tasks (binary classification, multiclass classification, regression).

<a name=".supervised.automl.AutoML.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(results_path=None, 
            total_time_limit=60 * 60, 
            model_time_limit=None, 
            algorithms=["Random Forest", "Xgboost"], 
            train_ensemble=True, 
            optimize_metric=None, 
            validation={"validation_type": "kfold", "k_folds": 5, "shuffle": True}, 
            verbose=True, 
            ml_task=None, 
            explain_level=2,
            seed=1)
```

Create the AutoML object. Initialize directory for results.

**Arguments**:

- `results_path`: The path where all results will be saved.
If left `None` then the name of directory will be generated, with schema: AutoML_{number},
where number can be from 1 to 100 - depends which direcory name will be available.

If the `results_path` will point to directory with AutoML results, then all models will be loaded.

- `total_time_limit`: The time limit in seconds for AutoML training. It is not used when `model_time_limit` is not `None`.

- `model_time_limit`: The time limit in seconds for training single model.
If `model_time_limit` is set, the `total_time_limit` is not respected.
Single model can contain several learners, for example in the case of 10-fold cross-validation, one model will have 10 learners.
Based on `model_time_limit` the time limit for single learner is computed.

- `algorithms`: The list of algorithms that will be used in the training.

- `train_ensemble`: If true then at the end of models training the ensemble will be created.

- `optimize_metric`: The metric to be optimized. (not implemented yet, please left `None`)

- `validation`: The JSON with validation type. Right now only Cross-Validation is supported.
The example JSON parameters for validation:
```
{"validation_type": "kfold", "k_folds": 5, "shuffle": True, "stratify": True, "random_seed": 123}
```
- `verbose`: Not implemented yet.
- `ml_task`: The machine learning task that will be solved. Can be: `"binary_classification", "multiclass_classification", "regression"`.
If left `None` AutoML will try to guess the task based on target values.
If there will be only 2 values in the target, then task will be set to `"binary_classification"`.
If number of values in the target will be between 2 and 20 (included), then task will be set to `"multiclass_classification"`.
In all other casses, the task is set to `"regression"`.
- `explain_level`: The level of explanations included to each model.

    - `explain_level = 0` means no explanations,
    - `explain_level = 1` means produce importance plot (with permutation method), for decision trees produce tree plots, for linear models save coefficients
    - `explain_level = 2` the same as for `1` plus SHAP explanations

- `seed`: The seed for random generator.

<a name=".supervised.automl.AutoML.set_advanced"></a>
#### set\_advanced

```python
 | set_advanced(start_random_models=1, hill_climbing_steps=0, top_models_to_improve=0)
```

Advanced set of tuning parameters.

**Arguments**:

- `start_random_models`: Number of not-so-random models to check for each algorithm.
- `hill_climbing_steps`: Number of hill climbing steps during tuning.
- `top_models_to_improve`: Number of top models (of each algorithm) which will be considered for improving in hill climbing steps.

<a name=".supervised.automl.AutoML.fit"></a>
#### fit

```python
 | fit(X_train, y_train, X_validation=None, y_validation=None)
```

Fit AutoML

**Arguments**:

- `X_train`: Pandas DataFrame with training data.
- `y_train`: Numpy Array with target training data.

- `X_validation`: Pandas DataFrame with validation data. (Not implemented yet)
- `y_validation`: Numpy Array with target of validation data. (Not implemented yet)

<a name=".supervised.automl.AutoML.predict"></a>
#### predict

```python
 | predict(X)
```

Computes predictions from AutoML best model.

**Arguments**:

- `X`: The Pandas DataFrame with input data. The input data should have the same columns as data used for training, otherwise the `AutoMLException` will be raised.

