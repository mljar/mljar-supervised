import os
import sys
import json
import copy
import time
import numpy as np
import pandas as pd
import logging
from tabulate import tabulate
from abc import ABC
from copy import deepcopy

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from sklearn.metrics import r2_score, accuracy_score

from supervised.base_automl import BaseAutoML
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION
from supervised.callbacks.early_stopping import EarlyStopping
from supervised.callbacks.metric_logger import MetricLogger
from supervised.callbacks.learner_time_constraint import LearnerTimeConstraint
from supervised.callbacks.total_time_constraint import TotalTimeConstraint
from supervised.ensemble import Ensemble
from supervised.exceptions import AutoMLException
from supervised.model_framework import ModelFramework
from supervised.preprocessing.exclude_missing_target import ExcludeRowsMissingTarget
from supervised.tuner.data_info import DataInfo
from supervised.tuner.mljar_tuner import MljarTuner
from supervised.utils.additional_metrics import AdditionalMetrics
from supervised.utils.config import mem
from supervised.utils.config import LOG_LEVEL
from supervised.utils.leaderboard_plots import LeaderboardPlots
from supervised.utils.metric import Metric
from supervised.preprocessing.eda import EDA
from supervised.tuner.time_controller import TimeController
from supervised.utils.data_validation import (
    check_positive_integer,
    check_greater_than_zero_integer,
    check_bool,
)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.ERROR
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class AutoML(BaseAutoML):

    """
    Automated Machine Learning for supervised tasks (binary classification, multiclass classification, regression).

    Provides scikit-learn API compatibily

    Parameters
    ----------

    mode : str {"Explain", "Perform", "Compete"}, optional, default="Explain"
        Defines the goal and how intensive the AutoML search will be.
        There are three available modes:
        - `Explain` : To to be used when the user wants to explain and understand the data.
            - Uses 75%/25% train/test split.
            - Uses the following models: `Baseline`, `Linear`, `Decision Tree`, `Random Forest`, `XGBoost`, `Artificial Neural Network`, and `Ensemble`.
            - Has full explanations in reports: learning curves, importance plots, and SHAP plots.
        - `Perform` : To be used when the user wants to train a model that will be used in real-life use cases.
            - Uses 5-fold CV (Cross-Validation).
            - Uses the following models: `Linear`, `Random Forest`, `LightGBM`, `XGBoost`, `CatBoost`, `Artificial Neural Network`, and `Ensemble`.
            - Has learning curves and importance plots in reports.
        - `Compete` : To be used for machine learning competitions (maximum performance).
            - Uses 10-fold CV (Cross-Validation).
            - Uses the following models: `Linear`, `DecisionTree`, `Random Forest`, `Extra Trees`, `XGBoost`, `CatBoost`, `Artificial Neural Network`,
                `Artificial Neural Network`, `Nearest Neighbors`, `Ensemble`, and `Stacking`.
            - It has only learning curves in the reports.

    ml_task : str {"auto","binary_classification", "multiclass_classification", "regression"} , optional, default="auto"
        If left `None` AutoML will try to guess the task based on target values.
        If there will be only 2 values in the target, then task will be set to `"binary_classification"`.
        If number of values in the target will be between 2 and 20 (included), then task will be set to `"multiclass_classification"`.
        In all other casses, the task is set to `"regression"`.

    tuning_mode :  str {"Normal", "Sport", "Insane", "Perfect"}, optional, default="Normal"
        The mode for tuning. The names are kept the same as `MLjar web application <https://mljar.com>`
        Each mode describe how many models will be checked:
        - `Normal` : about 5-10 models of each algorithm will be trained,
        - `Sport` : about 10-15 models of each algorithm will be trained,
        - `Insane` : about 15-20 models of each algorithm will be trained,
        - `Perfect` : about 25-35 models of each algorithm will be trained.

    results_path : str, optional, default=None
        The path with results. If None, then the name of directory will be generated with the template: AutoML_{number},
        where the number can be from 1 to 1,000 - depends which direcory name will be available.
        If the `results_path` will point to directory with AutoML results (`params.json` must be present),
        then all models will be loaded.

    total_time_limit : int or None, optional, default=1800
        The time limit in seconds for AutoML training. If None, then
        `model_time_limit` is not used.

    model_time_limit : int, optional, default=None
        The time limit for training a single model, in seconds.
        If `model_time_limit` is set, the `total_time_limit` is not respected.
        The single model can contain several learners. The time limit for subsequent learners is computed based on `model_time_limit`.
        For example, in the case of 10-fold cross-validation, one model will have 10 learners.
        The `model_time_limit` is the time for all 10 learners.

    algorithms : list of str, optional
        The list of algorithms that will be used in the training. The algorithms can be:
        [
            "Baseline",
            "Linear",
            "Decision Tree",
            "Random Forest",
            "Extra Trees",
            "LightGBM",
            "Xgboost",
            "CatBoost",
            "Neural Network",
            "Nearest Neighbors",
        ]

    train_ensemble : bool, optional, default=True
        Whether an ensemble gets created at the end of the training.

    stack_models : bool, optional, default=True
        Whether a models stack gets created at the end of the training. Stack level is 1.

    eval_metric : str, optional, default="auto"
        The metric to be optimized.
        If "auto", then:
            - `logloss` is used for classifications taks.
            - `rmse` is used for regression taks.
        .. note:: Still not implemented, please left `None`

    validation_strategy : dict, optional,  default="auto"
        Dictionary with validation type. Right now only Cross-Validation is supported.
        Example::
        {"validation_type": "kfold", "k_folds": 5, "shuffle": True, "stratify": True, "random_seed": 123}

    explain_level : "auto" or int {0,1,2}, optional, default="auto"
        The level of explanations included to each model:
        - if `explain_level` is `0` no explanations are produced.
        - if `explain_level` is `1` the following explanations are produced: importance plot (with permutation method), for decision trees produce tree plots, for linear models save coefficients.
        - if `explain_level` is `2` the following explanations are produced: the same as `1` plus SHAP explanations.
        If left `auto` AutoML will produce explanations based on the selected `mode`.

    golden_features : "auto" or bool, optional, default="auto"
        Whether to use golden features
        If left `auto` AutoML will use golden features based on the selected `mode`:
            - If `mode` is "Explain", `golden_features` = False.
            - If `mode` is "Perform", `golden_features` = True.
            - If `mode` is "Compete", `golden_features` = True.

    feature_selection : "auto" or bool, optional, default="auto"
        Whether to do feature_selection
         If left `auto` AutoML will do feature selection based on the selected `mode`:
            - If `mode` is "Explain", `feature_selection` = False.
            - If `mode` is "Perform", `feature_selection` = True.
            - If `mode` is "Compete", `feature_selection` = True.

    start_random_models : "auto" or int (> 0)
        Number of starting random models to try.
        If left `auto` AutoML will select it based on the selected `mode`:
            - If `mode` is "Explain", `start_random_models` = 1.
            - If `mode` is "Perform", `start_random_models` = 5.
            - If `mode` is "Compete", `start_random_models` = 10.

    hill_climbing_steps : "auto" or int (>= 0)
        Number of steps to perform during hill climbing.
        If left `auto` AutoML will select it based on the selected `mode`:
            - If `mode` is "Explain", `hill_climbing_steps` = 0.
            - If `mode` is "Perform", `hill_climbing_steps` = 2.
            - If `mode` is "Compete", `hill_climbing_steps` = 2.

    top_models_to_improve : "auto" or int (>= 0)
        Number of best models to improve.
        If left `auto` AutoML will select it based on the selected `mode`:
            - If `mode` is "Explain", `top_models_to_improve` = 0.
            - If `mode` is "Perform", `top_models_to_improve` = 2.
            - If `mode` is "Compete", `top_models_to_improve` = 3.

    verbose : int, optional, default=1
        Controls the verbosity when fitting and predicting.
        .. note:: Still not implemented, please left `1`

    random_state : int, default=None
        Controls the randomness of the MLjar Tuner


    Examples
    --------
    Binary Classification Example:
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import roc_auc_score
    >>> from supervised import AutoML
    >>> df = pd.read_csv(
    ...        "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv",
    ...       skipinitialspace=True
    ...    )
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ... df[df.columns[:-1]], df["income"], test_size=0.25
    ... )
    >>> automl = AutoML()
    >>> automl.fit(X_train, y_train)
    >>> y_pred_prob = automl.predict_proba(X_test)
    >>> print(f"AUROC: {roc_auc_score(y_test, y_pred_prob):.2f}%")

    Multi-Class Classification Example:
    >>> import pandas as pd
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.model_selection import train_test_split
    >>> from supervised import AutoML
    >>> digits = load_digits()
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     digits.data, digits.target, stratify=digits.target, test_size=0.25,
    ...     random_state=123
    ... )
    >>> automl = AutoML(mode="Perform")
    >>> automl.fit(X_train, y_train)
    >>> y_pred = automl.predict(X_test)
    >>> print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}%")

    Regression Example:
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import mean_squared_error
    >>> from supervised import AutoML
    >>> housing = load_boston()
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...       pd.DataFrame(housing.data, columns=housing.feature_names),
    ...       housing.target,
    ...       test_size=0.25,
    ...       random_state=123,
    ... )
    >>> automl = AutoML(mode="Compete")
    >>> automl.fit(X_train, y_train)
    >>> print("Test R^2:", automl.score(X_test, y_test))

    Scikit-learn Pipeline Integration Example:
    >>> from imblearn.over_sampling import RandomOverSampler
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from supervised import AutoML
    >>> X, y = make_classification()
    >>> X_train, X_test, y_train, y_test = train_test_split(X,y)
    >>> pipeline = make_pipeline(RandomOverSampler(), AutoML())
    >>> print(pipeline.fit(X_train, y_train).score(X_test, y_test))
    """

    def __init__(
        self,
        mode="Explain",
        ml_task="auto",
        tuning_mode="Normal",
        results_path=None,
        total_time_limit=30 * 60,
        model_time_limit=None,
        algorithms="auto",
        train_ensemble=True,
        stack_models="auto",
        eval_metric="auto",
        validation_strategy="auto",
        explain_level="auto",
        golden_features="auto",
        feature_selection="auto",
        start_random_models="auto",
        hill_climbing_steps="auto",
        top_models_to_improve="auto",
        verbose=1,
        random_state=1234,
    ):
        super(AutoML, self).__init__()
        # Set user arguments
        self.mode = mode
        self.ml_task = ml_task
        self.tuning_mode = tuning_mode
        self.results_path = results_path
        self.total_time_limit = total_time_limit
        self.model_time_limit = model_time_limit
        self.algorithms = algorithms
        self.train_ensemble = train_ensemble
        self.stack_models = stack_models
        self.eval_metric = eval_metric
        self.validation_strategy = validation_strategy
        self.verbose = verbose
        self.explain_level = explain_level
        self.golden_features = golden_features
        self.feature_selection = feature_selection
        self.start_random_models = start_random_models
        self.hill_climbing_steps = hill_climbing_steps
        self.top_models_to_improve = top_models_to_improve
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the AutoML model.
        Parameters
        ----------
        X : list or numpy.ndarray or pandas.DataFrame
            Training data
        y : list or numpy.ndarray or pandas.DataFrame
            Training targets

        Returns
        -------
        self : AutoML object
        """
        return self._fit(X, y)

    def predict(self, X):
        """
        Computes predictions from AutoML best model.
        Parameters
        ----------
        X : list or numpy.ndarray or pandas.DataFrame
            Input values to make predictions on.

        Returns
        -------
        predictions : numpy.ndarray
            One-dimensional array of class label for each object.

        Raises
        ------
        AutoMLException
            Model has not yet been fitted.
        """
        return self._predict(X)

    def predict_proba(self, X):
        """
        Computes class probabilities from AutoML best model. This method can only be used for classification tasks.

        Parameters
        ----------
        X : list or numpy.ndarray or pandas.DataFrame
            Input values to make predictions on.

        Returns
        -------
        predictions : numpy.ndarray of shape (n_samples, n_classes)
            Matrix of containing class probabilities of the input samples

        Raises
        ------
        AutoMLException
            Model has not yet been fitted.

        """
        return self._predict_proba(X)

    def predict_all(self, X):
        """
        Computes both class probabilities and class labels from AutoML best model. This method can only be used for classification tasks.

        Parameters
        ----------
        X : list or numpy.ndarray or pandas.DataFrame
            Input values to make predictions on.

        Returns
        -------
        predictions : pandas.Dataframe of shape (n_samples, n_classes + 1 )
            Dataframe containing both class probabilities and class labels of the input samples

        Raises
        ------
        AutoMLException
            Model has not yet been fitted.

        """
        return self._predict_all(X)

    def score(self, X, y=None):
        """
        Calculates a goodness of `fit` for an AutoML instance.

        Parameters
        ----------
        X : list or numpy.ndarray or pandas.DataFrame
            Test values to make predictions on.

        y : list or numpy.ndarray or pandas.DataFrame
            True labels for X.

        Returns
        -------
        score : float
            Returns a goodness of fit measure (higher is better):
                - For classification tasks: returns the mean accuracy on the given test data and labels.
                - For regression tasks: returns the R^2 (coefficient of determination) on the given test data and labels.
        """
        return self._score(X, y)
