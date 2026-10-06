# Algorithms

## Baseline

### Classification

The `Baseline` algorithm is using scikit-learn algorithm: [`DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html). It is using strategy `prior` which returns most frequent class as label and class prior for `predict_proba()`.

### Regression

The `Baseline` algorithm is using scikit-learn algorithm: [`DummyRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html). It is using strategy `mean` which returns mean of the target from training data.

!!! note "`Baseline` is not tuned"
    There will be only one model for algorithm `Baseline`. This algorithm has no hyperparameters.

## Decision Tree

### Classification 

The `Decision Tree` is using scikit-learn [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

!!! note "`Decision Tree` hyperparameters for classification"
    The allowed values of hyperparameters:
    ```python
    dt_params = {"criterion": ["gini", "entropy"], 
                 "max_depth": [2, 3, 4]}
    ```
    The default set of hyperparameters:
    ```python
    classification_default_params = {"criterion": "gini", "max_depth": 3}
    ```

### Regression

The `Decision Tree` is using scikit-learn [`DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).

!!! note "`Decision Tree` hyperparameters for regression"
    The allowed values of hyperparameters:
    ```python
    dt_params = {
        "criterion": ["mse", "friedman_mse"], 
        "max_depth": [2, 3, 4]
    }
    ```
    The default set of hyperparameters:
    ```python
    classification_default_params = {"criterion": "mse", "max_depth": 3}
    ```

For `Decision Tree` a visualization can be created with `dtreeviz` package (not to have `explain_level > 0`).

## Linear

### Classification

The `Linear` is using scikit-learn [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

!!! note "`Linear` hyperparameters for classification"
    Thera are no hyperparameters for `Linear` model. The parameters used in `LogisticRegression` initialization: `max_iter=500, tol=5e-4, n_jobs=-1`.

### Regression

The `Linear` is using scikit-learn [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

!!! note "`Linear` hyperparameters for regression"
    Thera are no hyperparameters for `Linear` model. The parameters used in `LinearRegression` initialization: `n_jobs=-1`.

The coefficients are saved in Markdown report if `explain_level > 0`.

## Random Forest

### Classification

The `Random Forest` is using scikit-learn [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

!!! note "`Random Forest` hyperparameters for classification"
    The allowed hyperparameters values:
    ```python
    rf_params = {
        "criterion": ["gini", "entropy"],
        "max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [10, 20, 30, 40, 50],
        "max_depth": [4, 6, 8, 10, 12],
    }
    ```
    The default hyperparameters:
    ```python
    classification_default_params = {
        "criterion": "gini",
        "max_features": 0.6,
        "min_samples_split": 30,
        "max_depth": 6,
    }
    ```

### Regression

The `Random Forest` is using scikit-learn [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

!!! note "`Random Forest` hyperparameters for regression"
    The allowed hyperparameters values:
    ```python
    regression_rf_params = {
        "criterion": ["mse"],
        "max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [10, 20, 30, 40, 50],
        "max_depth": [4, 6, 8, 10, 12],
    }
    ```
    The default hyperparameters:
    ```python
    regression_default_params = {
        "criterion": "mse",
        "max_features": 0.6,
        "min_samples_split": 30,
        "max_depth": 6,
    }
    ```

## Extra Trees

### Classification

The `Extra Trees` is using scikit-learn [`ExtraTreesClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).

!!! note "`Extra Trees` hyperparameters for classification"
    The allowed hyperparameters values:
    ```python
    et_params = {
        "criterion": ["gini", "entropy"],
        "max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [10, 20, 30, 40, 50],
        "max_depth": [4, 6, 8, 10, 12],
    }
    ```
    The default hyperparameters:
    ```python
    classification_default_params = {
        "criterion": "gini",
        "max_features": 0.6,
        "min_samples_split": 30,
        "max_depth": 6,
    }
    ```

### Regression

The `Extra Trees` is using scikit-learn [`ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html).

!!! note "`Extra Trees` hyperparameters for regression"
    The allowed hyperparameters values:
    ```python
    regression_et_params = {
        "criterion": ["mse"],
        "max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [10, 20, 30, 40, 50],
        "max_depth": [4, 6, 8, 10, 12],
    }
    ```
    The default hyperparameters:
    ```python
    regression_default_params = {
        "criterion": "mse",
        "max_features": 0.6,
        "min_samples_split": 30,
        "max_depth": 6,
    }
    ```

## Xgboost

The AutoML is using [`Xgboost`](https://xgboost.readthedocs.io/en/latest/index.html) package.

### Binary Classification

!!! note "`Xgboost` hyperparameters for binary classification"
    The allowed hyperparameters values:
    ```python
    xgb_bin_class_params = {
        "objective": ["binary:logistic"],
        "eval_metric": ["logloss"],
        "eta": [0.05, 0.075, 0.1, 0.15],
        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "subsample": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    ```
    The default hyperparameters:
    ```python
    classification_bin_default_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }
    ```

### Multi-class Classification

!!! note "`Xgboost` hyperparameters for multi-class classification"
    The allowed hyperparameters values:
    ```python
    xgb_multi_class_params = {
        "objective": ["multi:softprob"],
        "eval_metric": ["mlogloss"],
        "eta": [0.05, 0.075, 0.1, 0.15],
        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "subsample": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    
    ```
    The default hyperparameters:
    ```python       
    classification_multi_default_params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "eta": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }
    ```

### Regression

!!! note "`Xgboost` hyperparameters for regression"
    The allowed hyperparameters values:
    ```python
    xgb_regression_params = {
        "objective": ["reg:squarederror"],
        "eval_metric": ["rmse"],
        "eta": [0.05, 0.075, 0.1, 0.15],
        "max_depth": [1, 2, 3, 4],
        "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "subsample": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    
    ```
    The default hyperparameters:
    ```python       
    regression_default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }
    ```

## CatBoost

The AutoML is using [`CatBoost`](https://catboost.ai/) package.

### Classification

!!! note "`CatBoost` hyperparameters for classification"
    The allowed hyperparameters values:
    ```python
    classification_params = {
        "learning_rate": [0.05, 0.1, 0.2],
        "depth": [2, 3, 4, 5, 6],
        "rsm": [0.7, 0.8, 0.9, 1],  # random subspace method
        "subsample": [0.7, 0.8, 0.9, 1],  # random subspace method
        "min_data_in_leaf": [1, 5, 10, 15, 20, 30, 50],
    }
    ```
    The default hyperparameters:
    ```python
    classification_default_params = {
        "learning_rate": 0.1,
        "depth": 6,
        "rsm": 0.9,
        "subsample": 1.0,
        "min_data_in_leaf": 15,
    }
    ```

- for binary classification `loss_function=Logloss`,
- for mutliclass classification `loss_function=MultiClass`.

## Regression

!!! note "`CatBoost` hyperparameters for regression"
    The allowed hyperparameters values:
    ```python
    regression_params = {
        "learning_rate": [0.05, 0.1, 0.2],
        "depth": [2, 3, 4, 5, 6],
        "rsm": [0.7, 0.8, 0.9, 1],  # random subspace method
        "subsample": [0.7, 0.8, 0.9, 1],  # random subspace method
        "min_data_in_leaf": [1, 5, 10, 15, 20, 30, 50],
    }
    ```
    The default hyperparameters:
    ```python       
    regression_default_params = {
        "learning_rate": 0.1,
        "depth": 6,
        "rsm": 0.9,
        "subsample": 1.0,
        "min_data_in_leaf": 15,
    }
    ```

For regression `loss_function=RMSE`.

## LightGBM


The AutoML is using [`LightGBM`](https://lightgbm.readthedocs.io/en/latest/) package.

### Binary Classification

!!! note "`LightGBM` hyperparameters for binary classification"
    The allowed hyperparameters values:
    ```python
    lgbm_bin_params = {
        "objective": ["binary"],
        "metric": ["binary_logloss"],
        "num_leaves": [3, 7, 15, 31],  
        "learning_rate": [0.05, 0.075, 0.1, 0.15],
        "feature_fraction": [0.8, 0.9, 1.0],
        "bagging_fraction": [0.8, 0.9, 1.0],
        "min_data_in_leaf": [5, 10, 15, 20, 30, 50],
    }
    ```
    The default hyperparameters:
    ```python
    classification_bin_default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "min_data_in_leaf": 10,
    }
    ```


### Multi-class Classification

!!! note "`LightGBM` hyperparameters for multi-class classification"
    The allowed hyperparameters values:
    ```python
    lgbm_bin_params = {
        "objective": ["multiclass"],
        "metric": ["multi_logloss"],
        "num_leaves": [3, 7, 15, 31], 
        "learning_rate": [0.05, 0.075, 0.1, 0.15],
        "feature_fraction": [0.8, 0.9, 1.0],
        "bagging_fraction": [0.8, 0.9, 1.0],
        "min_data_in_leaf": [5, 10, 15, 20, 30, 50],
    }
    ```
    The default hyperparameters:
    ```python
    classification_multi_default_params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "min_data_in_leaf": 10,
    }
    ```

### Regression

!!! note "`LightGBM` hyperparameters for regression"
    The allowed hyperparameters values:
    ```python
    lgbm_bin_params = {
        "objective": ["regression"],
        "metric": ["l2"],
        "num_leaves": [3, 7, 15, 31], 
        "learning_rate": [0.05, 0.075, 0.1, 0.15],
        "feature_fraction": [0.8, 0.9, 1.0],
        "bagging_fraction": [0.8, 0.9, 1.0],
        "min_data_in_leaf": [5, 10, 15, 20, 30, 50],
    }
    ```
    The default hyperparameters:
    ```python
    regression_default_params = {
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "min_data_in_leaf": 10,
    }
    ```

## Neural Network

For `Neural Network` algorithm the [Keras](https://keras.io) and [Tensorflow](https://www.tensorflow.org/) are used. The same set of hyperparameters are used for all Machine Learning tasks (classification and regression). There is difference in output neurons type and loss function depending on ML task. 

!!! note "`Neural Network` hyperparameters"
    The allowed hyperparameters values:
    ```python
    nn_params = {
        "dense_layers": [2],
        "dense_1_size": [16, 32, 64],
        "dense_2_size": [4, 8, 16, 32],
        "dropout": [0, 0.1, 0.25],
        "learning_rate": [0.01, 0.05, 0.08, 0.1],
        "momentum": [0.85, 0.9, 0.95],
        "decay": [0.0001, 0.001, 0.01],
    }
    ```
    The default hyperparameters:
    ```python
    default_nn_params = {
        "dense_layers": 2,
        "dense_1_size": 32,
        "dense_2_size": 16,
        "dropout": 0,
        "learning_rate": 0.05,
        "momentum": 0.9,
        "decay": 0.001,
    }
    ```

### Binary Classification

- There is single output neuron with `sigmoid` activation.
- The loss function: `binary_crossentropy`.

### Multi-class Classification

- The number of output neurons is equal to the number of unique classes in the target. The activation in the output layer is `softmax`.
- The loss function: `categorical_crossentropy`.

### Regression

- There is single output neuron with `linear` activation.
- The loss function: `mean_squared_error`.

## Nearest Neighbor

The `Nearest Neighbor` algorithm is using scikit-learn: 

- the [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) for classification,
- the [`KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor) for regression.

!!! note "`Nearest Neighbor` hyperparameters"
    The allowed hyperparameters values:
    ```python
    knn_params = {
        "n_neighbors": [3, 5, 7], 
        "weights": ["uniform", "distance"]
    }
    ```
    The default hyperparameters:
    ```python
    default_params = {
        "n_neighbors": 5, 
        "weights": "uniform"
    }
    ```

## Stacked Algorithm

The stacked algorithms are built with predictions from previous (unstacked) models. The stacked algorithms are reusing hyperparameters of already found good models.

- During the stacking up to 10 best models from each algorithm are used, except `Baseline`.
- The **out-of-folds** predictions are used to construct extended training data. The stacking only works for `validation_strategy="kfold"` (k-fold cross-validation).
- The stacked model can be only: `Xgboost`, `LightGBM`, `CatBoost`. The AutoML algorithm selects the best models from **unstacked** `Xgboost`, `LightGBM`, `CatBoost` and reuses its hyperparameters to train **stacked** models.

## Ensemble

The `Ensemble` algorithm is implemented based on [Caruana article](http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf). The `Ensemble` is using average method, which does a greedy search over all models and try to add (with repetition) a model to the ensemble to improve ensemble's performance. The ensemble performance is computed based on out-of-folds predictions of used models
