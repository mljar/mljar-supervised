from functools import partial

import copy
import json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import ClassifierMixin, RegressorMixin

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.exceptions import CandidateValidationError
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
    AlgorithmsRegistry,
)
from supervised.utils.config import LOG_LEVEL
from supervised.utils.metric import (
    F1_METRICS,
    xgboost_eval_metric_accuracy,
    xgboost_eval_metric_average_precision,
    xgboost_eval_metric_f1,
    xgboost_eval_metric_mse,
    xgboost_eval_metric_pearson,
    xgboost_eval_metric_r2,
    xgboost_eval_metric_spearman,
    xgboost_eval_metric_user_defined,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class XgbAlgorithmException(Exception):
    def __init__(self, message):
        super(XgbAlgorithmException, self).__init__(message)
        logger.error(message)


def time_constraint(env):
    # print("time constraint")
    pass


def xgboost_eval_metric(ml_task, automl_eval_metric):
    # the mapping is almost the same
    eval_metric_name = automl_eval_metric
    if ml_task == MULTICLASS_CLASSIFICATION:
        if automl_eval_metric == "logloss":
            eval_metric_name = "mlogloss"
    return eval_metric_name


def xgboost_objective(ml_task, automl_eval_metric):
    objective = "reg:squarederror"
    if ml_task == BINARY_CLASSIFICATION:
        objective = "binary:logistic"
    elif ml_task == MULTICLASS_CLASSIFICATION:
        objective = "multi:softprob"
    else:  # ml_task == REGRESSION
        objective = "reg:squarederror"
    return objective


class XgbAlgorithm(BaseAlgorithm):
    """
    This is a wrapper over xgboost algorithm.
    """

    algorithm_name = "Extreme Gradient Boosting"
    algorithm_short_name = "Xgboost"

    def __init__(self, params):
        super(XgbAlgorithm, self).__init__(params)
        self.library_version = xgb.__version__

        self.explain_level = params.get("explain_level", 0)
        self.boosting_rounds = additional.get("max_rounds", 10000)
        self.max_iters = 1
        self.early_stopping_rounds = additional.get("early_stopping_rounds", 50)
        self.learner_params = {
            "tree_method": "hist",
            "device": "cpu",
            "booster": "gbtree",
            "objective": self.params.get("objective"),
            "eval_metric": self.params.get("eval_metric"),
            "eta": self.params.get("eta", 0.01),
            "max_depth": self.params.get("max_depth", 1),
            "min_child_weight": self.params.get("min_child_weight", 1),
            "subsample": self.params.get("subsample", 0.8),
            "colsample_bytree": self.params.get("colsample_bytree", 0.8),
            "n_jobs": self.params.get("n_jobs", -1),
            # "silent": self.params.get("silent", 1),
            "seed": self.params.get("seed", 1),
            "verbosity": 0,
        }

        if "lambda" in self.params:
            self.learner_params["lambda"] = self.params["lambda"]
        if "alpha" in self.params:
            self.learner_params["alpha"] = self.params["alpha"]

        # check https://github.com/dmlc/xgboost/issues/5637
        if self.learner_params["seed"] > 2147483647:
            self.learner_params["seed"] = self.learner_params["seed"] % 2147483647
        if "num_class" in self.params:  # multiclass classification
            self.learner_params["num_class"] = self.params.get("num_class")

        if "max_rounds" in self.params:
            self.boosting_rounds = self.params["max_rounds"]

        self.custom_eval_metric = None
        if self.params.get("eval_metric", "") == "r2":
            self.custom_eval_metric = xgboost_eval_metric_r2
        elif self.params.get("eval_metric", "") == "spearman":
            self.custom_eval_metric = xgboost_eval_metric_spearman
        elif self.params.get("eval_metric", "") == "pearson":
            self.custom_eval_metric = xgboost_eval_metric_pearson
        elif self.params.get("eval_metric", "") in F1_METRICS:
            self.custom_eval_metric = partial(
                xgboost_eval_metric_f1, metric_name=self.params.get("eval_metric", "")
            )
        elif self.params.get("eval_metric", "") == "average_precision":
            self.custom_eval_metric = xgboost_eval_metric_average_precision
        elif self.params.get("eval_metric", "") == "accuracy":
            self.custom_eval_metric = xgboost_eval_metric_accuracy
        elif self.params.get("eval_metric", "") == "mse":
            self.custom_eval_metric = xgboost_eval_metric_mse
        elif self.params.get("eval_metric", "") == "user_defined_metric":
            self.custom_eval_metric = xgboost_eval_metric_user_defined

        logger.debug("XgbLearner __init__")

    """
    def get_boosting_rounds(self, dtrain, evals, esr, max_time):
        if max_time is None:
            return self.boosting_rounds

        start_time = time.time()
        evals_result = {}
        model = xgb.train(
            self.learner_params,
            dtrain,
            2,
            evals=evals,
            early_stopping_rounds=esr,
            evals_result=evals_result,
            verbose_eval=False,
        )
        time_1_iter = (time.time() - start_time) / 2.0

        # 2.0 is just a scaling factor
        # purely heuristic
        iters = int(max_time / time_1_iter * 2.0)
        iters = max(iters, 100)
        iters = min(iters, 10000)
        return iters
    """

    def _binary_min_child_weight(self):
        if self.learner_params.get("objective") != "binary:logistic":
            return None
        minimum = float(self.learner_params.get("min_child_weight", 1))
        return minimum if np.isfinite(minimum) and minimum > 0 else None

    @staticmethod
    def _insufficient_hessian(total, minimum):
        # DMatrix and tree statistics use floating-point weights. Do not reject
        # a configuration at (or numerically close to) the admissible boundary.
        required = 2 * minimum
        return total + 1e-6 * max(1.0, total, required) < required

    def _validate_split_capacity(self, y, sample_weight):
        minimum = self._binary_min_child_weight()
        if minimum is None:
            return
        targets = np.asarray(y).ravel()
        weights = (np.ones(len(targets), dtype=float) if sample_weight is None
                   else np.asarray(sample_weight, dtype=float).ravel())
        # Leave malformed labels/weights to XGBoost's own input validation.
        if (not len(targets) or len(weights) != len(targets)
                or not np.isin(targets, [0, 1]).all()
                or not np.isfinite(weights).all() or (weights < 0).any()):
            return
        positive_weight = float(self.learner_params.get("scale_pos_weight", 1))
        if not np.isfinite(positive_weight) or positive_weight < 0:
            return
        weights = weights * np.where(targets == 1, positive_weight, 1.0)
        # Logistic Hessian w*p*(1-p) <= w/4 at EVERY boosting round.
        # Do not multiply by subsample: random sampling can exceed its expectation.
        upper_bound = float(weights.sum()) / 4
        if np.isfinite(upper_bound) and self._insufficient_hessian(upper_bound, minimum):
            raise CandidateValidationError(
                f"Xgboost min_child_weight={minimum:g} requires at least "
                f"{2 * minimum:g} Hessian weight for two children, but this "
                f"training fold can provide at most {upper_bound:g}; skipping candidate.",
                reason="insufficient_split_hessian", phase="before_fit",
                min_child_weight=minimum, hessian_upper_bound=upper_bound,
                training_rows=len(targets),
            )

    def _validate_trained_split_capacity(self):
        minimum = self._binary_min_child_weight()
        if minimum is None or self.model.get_score(importance_type="weight"):
            return
        # The global bound can pass while imbalance/subsampling prevents actual
        # splits. Reject only when ALL trained trees are leaves and ALL observed
        # root Hessians are insufficient, not merely when the best prefix is small.
        trees = [json.loads(t) for t in self.model.get_dump(dump_format="json", with_stats=True)]
        if not trees or any("leaf" not in t for t in trees):
            return
        covers = [float(t["cover"]) for t in trees]
        if not np.isfinite(covers).all():
            return
        maximum = max(covers)
        if self._insufficient_hessian(maximum, minimum):
            self.model = None
            raise CandidateValidationError(
                f"Xgboost trained {len(trees)} unsplit trees: min_child_weight={minimum:g} "
                f"requires {2 * minimum:g} Hessian weight, but the largest observed "
                f"root weight was {maximum:g}; skipping candidate.",
                reason="insufficient_split_hessian", phase="after_fit",
                min_child_weight=minimum, max_observed_root_hessian=maximum,
                trained_trees=len(trees),
            )

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_validation=None,
        y_validation=None,
        sample_weight_validation=None,
        log_to_file=None,
        max_time=None,
    ):
        self.model = None
        self._validate_split_capacity(y, sample_weight)
        dtrain = xgb.DMatrix(
            X.values if isinstance(X, pd.DataFrame) else X,
            label=y,
            missing=np.nan,
            weight=sample_weight,
        )

        if X_validation is not None and y_validation is not None:       
            dvalidation = xgb.DMatrix(
                X_validation.values
                if isinstance(X_validation, pd.DataFrame)
                else X_validation,
                label=y_validation,
                missing=np.nan,
                weight=sample_weight_validation,
            )
        else:
            dvalidation = None
            
        evals_result = {}

        evals = []
        esr = None
        if X_validation is not None and y_validation is not None:
            evals = [(dtrain, "train"), (dvalidation, "validation")]
            esr = self.early_stopping_rounds

        # disable for now, dont have better idea how to handle time limit ...
        # looks like there is better not to limit the algorithm
        # just wait till they converge ...
        # boosting_rounds = self.get_boosting_rounds(dtrain, evals, esr, max_time)

        if self.custom_eval_metric is not None:
            del self.learner_params["eval_metric"]

        self.model = xgb.train(
            self.learner_params,
            dtrain,
            self.boosting_rounds,
            evals=evals,
            early_stopping_rounds=esr,
            evals_result=evals_result,
            verbose_eval=False,
            custom_metric=self.custom_eval_metric,
            # callbacks=[time_constraint] # callback slows down by factor ~8
        )

        del dtrain
        del dvalidation
        self._validate_trained_split_capacity()

        if log_to_file is not None:
            metric_name = list(evals_result["train"].keys())[-1]

            result = pd.DataFrame(
                {
                    "iteration": range(len(evals_result["train"][metric_name])),
                    "train": evals_result["train"][metric_name],
                    "validation": evals_result["validation"][metric_name],
                }
            )
            # it a is custom metric
            # that is always minimized
            # we need to revert it
            if metric_name in [
                "r2",
                "spearman",
                "pearson",
                *F1_METRICS,
                "average_precision",
                "accuracy",
            ]:
                result["train"] *= -1.0
                result["validation"] *= -1.0

            result.to_csv(log_to_file, index=False, header=False)

        if self.params["ml_task"] != REGRESSION:
            self.classes_ = np.unique(y)

        # fix high memory consumption in xgboost,
        # waiting for release with fix
        # https://github.com/dmlc/xgboost/issues/5474
        """
        # disable, for now all learners are saved to hard disk and then deleted from RAM
        with tempfile.NamedTemporaryFile() as tmp:
            self.model.save_model(tmp.name)
            del self.model
            self.model = xgb.Booster()
            self.model.load_model(tmp.name)
        """

    def is_fitted(self):
        return self.model is not None

    def predict(self, X):
        self.reload()

        if self.model is None:
            raise XgbAlgorithmException("Xgboost model is None")

        dtrain = xgb.DMatrix(
            X.values if isinstance(X, pd.DataFrame) else X, missing=np.nan
        )
        # xgboost > 2.0.0 version
        if hasattr(self.model, "best_iteration"):
            a = self.model.predict(
                dtrain, iteration_range=(0, self.model.best_iteration + 1)
            )
        else:
            a = self.model.predict(dtrain)

        return a

    def copy(self):
        return copy.deepcopy(self)

    def save(self, model_file_path):
        self.model.save_model(model_file_path)
        self.model_file_path = model_file_path
        logger.debug("XgbAlgorithm save model to %s" % model_file_path)

    def load(self, model_file_path):
        logger.debug("XgbLearner load model from %s" % model_file_path)
        self.model = xgb.Booster()  # init model
        self.model.load_model(model_file_path)
        self.model_file_path = model_file_path

    def file_extension(self):
        # we need to keep models as json files
        # to keep information about best_iteration
        return "xgboost.json"

    def get_metric_name(self):
        metric = self.params.get("eval_metric")
        if metric is None:
            return None
        if metric == "mlogloss":
            return "logloss"
        return metric


# For binary classification target should be 0, 1. There should be no NaNs in target.
xgb_bin_class_params = {
    "objective": ["binary:logistic"],
    "eta": [0.05, 0.075, 0.1, 0.15],
    "max_depth": [4, 5, 6, 7, 8, 9],
    "min_child_weight": [1, 5, 10, 25, 50],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

classification_bin_default_params = {
    "objective": "binary:logistic",
    "eta": 0.075,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}

xgb_regression_params = dict(xgb_bin_class_params)
xgb_regression_params["objective"] = ["reg:squarederror"]
# xgb_regression_params["eval_metric"] = ["rmse", "mae", "mape"]
xgb_regression_params["max_depth"] = [4, 5, 6, 7, 8, 9]


xgb_multi_class_params = dict(xgb_bin_class_params)
xgb_multi_class_params["objective"] = ["multi:softprob"]
# xgb_multi_class_params["eval_metric"] = ["mlogloss"]

classification_multi_default_params = {
    "objective": "multi:softprob",
    "eta": 0.075,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}


regression_default_params = {
    "objective": "reg:squarederror",
    "eta": 0.075,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}

additional = {
    "max_rounds": 10000,
    "early_stopping_rounds": 50,
    "max_rows_limit": None,
    "max_cols_limit": None,
}
required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "ip_transform",
    "target_as_integer",
]


class XgbClassifier(ClassifierMixin, XgbAlgorithm):
    pass


AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    XgbClassifier,
    xgb_bin_class_params,
    required_preprocessing,
    additional,
    classification_bin_default_params,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    XgbClassifier,
    xgb_multi_class_params,
    required_preprocessing,
    additional,
    classification_multi_default_params,
)

regression_required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "ip_transform",
    "target_scale",
]


class XgbRegressor(RegressorMixin, XgbAlgorithm):
    pass


AlgorithmsRegistry.add(
    REGRESSION,
    XgbRegressor,
    xgb_regression_params,
    regression_required_preprocessing,
    additional,
    regression_default_params,
)
