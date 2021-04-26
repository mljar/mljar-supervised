import logging

from supervised.base_automl import BaseAutoML

from supervised.utils.config import LOG_LEVEL


logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.ERROR
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class AutoML(BaseAutoML):

    """
    Automated Machine Learning for supervised tasks (binary classification, multiclass classification, regression).
    """

    def __init__(
        self,
        results_path=None,
        total_time_limit=60 * 60,
        mode="Explain",
        ml_task="auto",
        model_time_limit=None,
        algorithms="auto",
        train_ensemble=True,
        stack_models="auto",
        eval_metric="auto",
        validation_strategy="auto",
        explain_level="auto",
        golden_features="auto",
        features_selection="auto",
        start_random_models="auto",
        hill_climbing_steps="auto",
        top_models_to_improve="auto",
        boost_on_errors="auto",
        kmeans_features="auto",
        mix_encoding="auto",
        max_single_prediction_time=None,
        optuna_time_budget=None,
        optuna_init_params={},
        optuna_verbose=True,
        n_jobs=-1,
        verbose=1,
        random_state=1234,
    ):
        """
        Initialize `AutoML` object.

        Arguments:
            results_path (str): The path with results. If None, then the name of directory will be generated with the template: AutoML_{number},
                where the number can be from 1 to 1,000 - depends which direcory name will be available.
                If the `results_path` will point to directory with AutoML results (`params.json` must be present),
                then all models will be loaded.

            total_time_limit (int): The total time limit in seconds for AutoML training.
                It is not used when `model_time_limit` is not `None`.

            mode (str): Can be {`Explain`, `Perform`, `Compete`, `Optuna`}. This parameter defines the goal of AutoML and how intensive the AutoML search will be.

                - `Explain` : To to be used when the user wants to explain and understand the data.
                    - Uses 75%/25% train/test split.
                    - Uses the following models: `Baseline`, `Linear`, `Decision Tree`, `Random Forest`, `XGBoost`, `Neural Network`, and `Ensemble`.
                    - Has full explanations in reports: learning curves, importance plots, and SHAP plots.
                - `Perform` : To be used when the user wants to train a model that will be used in real-life use cases.
                    - Uses 5-fold CV (Cross-Validation).
                    - Uses the following models: `Linear`, `Random Forest`, `LightGBM`, `XGBoost`, `CatBoost`, `Neural Network`, and `Ensemble`.
                    - Has learning curves and importance plots in reports.
                - `Compete` : To be used for machine learning competitions (maximum performance).
                    - Uses 80/20 train/test split, or 5-fold CV, or 10-fold CV (Cross-Validation) - it depends on `total_time_limit`. If not set directly, AutoML will select validation automatically.
                    - Uses the following models: `Decision Tree`, `Random Forest`, `Extra Trees`, `LightGBM`,  `XGBoost`, `CatBoost`, `Neural Network`,
                        `Nearest Neighbors`, `Ensemble`, and `Stacking`.
                    - It has only learning curves in the reports.
                - `Optuna` : To be used for creating highly-tuned machine learning models.
                    - Uses 10-fold CV (Cross-Validation).
                    - It tunes with Optuna the following algorithms: `Random Forest`, `Extra Trees`, `LightGBM`, `XGBoost`, `CatBoost`, `Neural Network`.
                    - It applies `Ensemble` and `Stacking` for trained models.
                    - It has only learning curves in the reports.

            ml_task (str): Can be {"auto", "binary_classification", "multiclass_classification", "regression"}.

                - If left `auto` AutoML will try to guess the task based on target values.
                - If there will be only 2 values in the target, then task will be set to `"binary_classification"`.
                - If number of values in the target will be between 2 and 20 (included), then task will be set to `"multiclass_classification"`.
                - In all other casses, the task is set to `"regression"`.

            model_time_limit (int): The time limit for training a single model, in seconds.
                If `model_time_limit` is set, the `total_time_limit` is not respected.
                The single model can contain several learners. The time limit for subsequent learners is computed based on `model_time_limit`.

                For example, in the case of 10-fold cross-validation, one model will have 10 learners.
                The `model_time_limit` is the time for all 10 learners.

            algorithms (list of str): The list of algorithms that will be used in the training.
                The algorithms can be:

                - `Baseline`,
                - `Linear`,
                - `Decision Tree`,
                - `Random Forest`,
                - `Extra Trees`,
                - `LightGBM`,
                - `Xgboost`,
                - `CatBoost`,
                - `Neural Network`,
                - `Nearest Neighbors`,


            train_ensemble (boolean): Whether an ensemble gets created at the end of the training.

            stack_models (boolean): Whether a models stack gets created at the end of the training. Stack level is 1.

            eval_metric (str): The metric to be used in early stopping and to compare models.

                - for binary classification: `logloss`, `auc`, `f1`, `average_precision`, `accuracy` - default is logloss (if left "auto")
                - for mutliclass classification: `logloss`, `f1`, `accuracy` - default is `logloss` (if left "auto")
                - for regression: `rmse`, `mse`, `mae`, `r2`, `mape`, `spearman`, `pearson` - default is `rmse` (if left "auto")

            validation_strategy (dict): Dictionary with validation type. Right now train/test split and cross-validation are supported.

                Example:

                    Cross-validation exmaple:
                    {
                        "validation_type": "kfold",
                        "k_folds": 5,
                        "shuffle": True,
                        "stratify": True,
                        "random_seed": 123
                    }

                    Train/test example:
                    {
                        "validation_type": "split",
                        "train_ratio": 0.75,
                        "shuffle": True,
                        "stratify": True
                    }

            explain_level (int): The level of explanations included to each model:

                - if `explain_level` is `0` no explanations are produced.
                - if `explain_level` is `1` the following explanations are produced: importance plot (with permutation method), for decision trees produce tree plots, for linear models save coefficients.
                - if `explain_level` is `2` the following explanations are produced: the same as `1` plus SHAP explanations.

                If left `auto` AutoML will produce explanations based on the selected `mode`.

            golden_features (boolean or int): Whether to use golden features (and how many should be added)
                If left `auto` AutoML will use golden features based on the selected `mode`:

                - If `mode` is "Explain", `golden_features` = False.
                - If `mode` is "Perform", `golden_features` = True.
                - If `mode` is "Compete", `golden_features` = True.

                If `boolean` value is set then the number of Golden Features is set automatically.
                It is set to min(100, max(10, 0.1*number_of_input_features)).

                If `int` value is set, the number of Golden Features is set to this value.

            features_selection (boolean): Whether to do features_selection
                If left `auto` AutoML will do feature selection based on the selected `mode`:

                - If `mode` is "Explain", `features_selection` = False.
                - If `mode` is "Perform", `features_selection` = True.
                - If `mode` is "Compete", `features_selection` = True.

            start_random_models (int): Number of starting random models to try.
                If left `auto` AutoML will select it based on the selected `mode`:

                - If `mode` is "Explain", `start_random_models` = 1.
                - If `mode` is "Perform", `start_random_models` = 5.
                - If `mode` is "Compete", `start_random_models` = 10.

            hill_climbing_steps (int): Number of steps to perform during hill climbing.
                If left `auto` AutoML will select it based on the selected `mode`:

                - If `mode` is "Explain", `hill_climbing_steps` = 0.
                - If `mode` is "Perform", `hill_climbing_steps` = 2.
                - If `mode` is "Compete", `hill_climbing_steps` = 2.

            top_models_to_improve (int): Number of best models to improve in `hill_climbing` steps.
                If left `auto` AutoML will select it based on the selected `mode`:

                - If `mode` is "Explain", `top_models_to_improve` = 0.
                - If `mode` is "Perform", `top_models_to_improve` = 2.
                - If `mode` is "Compete", `top_models_to_improve` = 3.

            boost_on_errors (boolean): Whether a model with boost on errors from previous best model should be trained. By default available in the `Compete` mode.

            kmeans_features (boolean): Whether a model with k-means generated features should be trained. By default available in the `Compete` mode.

            mix_encoding (boolean): Whether a model with mixed encoding should be trained. Mixed encoding is the encoding that uses label encoding
                for categoricals with more than 25 categories, and one-hot binary encoding for other categoricals. It is only applied if there are
                categorical features with cardinality smaller than 25. By default it is available in the `Compete` mode.

            max_single_prediction_time (int or float): The limit for prediction time for single sample. Use it if you want to have a model with fast predictions.
                Ideal for creating ML pipelines used as REST API. Time is in seconds. By default (`max_single_prediction_time=None`) models are not optimized for fast predictions,
                except the mode `Perform`. For the mode `Perform` the default is `0.5` seconds.

            optuna_time_budget (int): The time in seconds which should be used by Optuna to tune each algorithm. It is time for tuning single algorithm.
                If you select two algorithms: Xgboost and CatBoost, and set optuna_time_budget=1000, then Xgboost will be tuned for 1000 seconds and CatBoost will be tuned for 1000 seconds.
                What is more, the tuning is made for each data type, for example for raw data and for data with inserted Golden Features.
                This parameter is only used when `mode="Optuna"`. If you set `mode="Optuna"` and forget to set this parameter, it will be set to 3600 seconds.

            optuna_init_params (dict): If you have already tuned parameters from Optuna you can reuse them by setting this parameter.
                This parameter is only used when `mode="Optuna"`. The dict should have structure and params as specified in the MLJAR AutoML .

            optuna_verbose (boolean): If true the Optuna tuning details are displayed. Set to `True` by default.

            n_jobs (int): Number of CPU cores to be used. By default is set to `-1` which means using  all processors.

            verbose (int): Controls the verbosity when fitting and predicting.

                Note:
                    Still not implemented, please left `1`

            random_state (int): Controls the randomness of the `AutoML`


        Examples:

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
        super(AutoML, self).__init__()
        # Set user arguments
        self.mode = mode
        self.ml_task = ml_task
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
        self.features_selection = features_selection
        self.start_random_models = start_random_models
        self.hill_climbing_steps = hill_climbing_steps
        self.top_models_to_improve = top_models_to_improve
        self.boost_on_errors = boost_on_errors
        self.kmeans_features = kmeans_features
        self.mix_encoding = mix_encoding
        self.max_single_prediction_time = max_single_prediction_time
        self.optuna_time_budget = optuna_time_budget
        self.optuna_init_params = optuna_init_params
        self.optuna_verbose = optuna_verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, cv=None):
        """Fit the AutoML model.

        Arguments:
            X (numpy.ndarray or pandas.DataFrame): Training data

            y (numpy.ndarray or pandas.Series): Training targets

            sample_weight (numpy.ndarray or pandas.Series): Training sample weights

            cv (iterable or list): List or iterable with (train, validation) splits representing array of indices.
            It is used only with custom validation (`validation_strategy={'validation_type': 'custom'}`).

        Returns:
            AutoML object: Returns `self`
        """
        return self._fit(X, y, sample_weight, cv)

    def predict(self, X):
        """
        Computes predictions from AutoML best model.

        Arguments:
            X (list or numpy.ndarray or pandas.DataFrame):
                Input values to make predictions on.

        Returns:
            numpy.ndarray:

            - One-dimensional array of class labels for classification.
            - One-dimensional array of predictions for regression.

        Raises:
            AutoMLException: Model has not yet been fitted.
        """
        return self._predict(X)

    def predict_proba(self, X):
        """
        Computes class probabilities from AutoML best model.
        This method can only be used for classification tasks.

        Arguments:
            X (list or numpy.ndarray or pandas.DataFrame):
                Input values to make predictions on.

        Returns:
            numpy.ndarray of shape (n_samples, n_classes):
                Matrix of containing class probabilities of the input samples

        Raises:
            AutoMLException: Model has not yet been fitted.

        """
        return self._predict_proba(X)

    def predict_all(self, X):
        """
        Computes both class probabilities and class labels for classification tasks.
        Computes predictions for regression tasks.

        Arguments:
            X (list or numpy.ndarray or pandas.DataFrame):
                Input values to make predictions on.

        Returns:
            pandas.Dataframe:
                Dataframe (n_samples, n_classes + 1) containing both class probabilities and class
                labels of the input samples for classification tasks.
                Dataframe with predictions for regression tasks.

        Raises:
            AutoMLException: Model has not yet been fitted.

        """
        return self._predict_all(X)

    def score(self, X, y=None, sample_weight=None):
        """Calculates a goodness of `fit` for an AutoML instance.

        Arguments:
            X (numpy.ndarray or pandas.DataFrame):
                Test values to make predictions on.

            y (numpy.ndarray or pandas.Series):
                True labels for X.

            sample_weight (numpy.ndarray or pandas.Series):
                Sample weights.
        Returns:
            float: Returns a goodness of fit measure (higher is better):

            - For classification tasks: returns the mean accuracy on the given test data and labels.
            - For regression tasks: returns the R^2 (coefficient of determination) on the given test data and labels.
        """
        return self._score(X, y, sample_weight)

    def report(self, width=900, height=1200):
        return self._report(width, height)

    def need_retrain(self, X, y, sample_weight=None, decrease=0.1):
        """Decides about model retraining based on new data.

        Arguments:
            X (numpy.ndarray or pandas.DataFrame):
                New data.

            y (numpy.ndarray or pandas.Series):
                True labels for X.

            sample_weight (numpy.ndarray or pandas.Series):
                Sample weights.

            decrease (float): The ratio of change in the performance used as a threshold for retraining decision.
                By default, it is set to `0.1` which means that if the performance of AutoML will decrease by 10%
                on new data then there is a need to retrain. This value should be set depending on your project needs.
                Sometimes, 10% is enough, but for some projects, it can be even lower than 1%.

            Returns:
                boolean: Decides if there is a need to retrain the AutoML.
        """
        return self._need_retrain(X, y, sample_weight, decrease)
