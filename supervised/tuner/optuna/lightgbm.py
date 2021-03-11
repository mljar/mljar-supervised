import lightgbm as lgb
import optuna

from supervised.utils.metric import Metric
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION

EPS = 1e-8


class LightgbmObjective:
    def __init__(
        self,
        ml_task,
        X_train,
        y_train,
        sample_weight,
        X_validation,
        y_validation,
        sample_weight_validation,
        eval_metric,
        cat_features_indices,
        n_jobs,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.sample_weight_validation = sample_weight_validation
        self.dtrain = lgb.Dataset(
            self.X_train, label=self.y_train, weight=self.sample_weight
        )
        self.dvalid = lgb.Dataset(
            self.X_validation,
            label=self.y_validation,
            weight=self.sample_weight_validation,
        )

        self.cat_features_indices = cat_features_indices
        self.eval_metric = eval_metric
        self.learning_rate = 0.05
        self.rounds = 1000
        self.early_stopping_rounds = 50
        self.seed = 123

        self.n_jobs = n_jobs
        if n_jobs == -1:
            self.n_jobs = 0

        self.objective = ""
        self.eval_metric_name = ""
        # MLJAR -> LightGBM
        metric_name_mapping = {
            BINARY_CLASSIFICATION: {"auc": "auc", "logloss": "binary_logloss"},
            MULTICLASS_CLASSIFICATION: {"logloss": "multi_logloss"},
            REGRESSION: {"rmse": "rmse", "mae": "mae", "mape": "mape"},
        }
        self.eval_metric_name = metric_name_mapping[ml_task][self.eval_metric.name]
        if ml_task == BINARY_CLASSIFICATION:
            self.objective = "binary"
        elif ml_task == MULTICLASS_CLASSIFICATION:
            self.objective = "binary:logistic"
        else:  # ml_task == REGRESSION
            self.objective = "reg:squarederror"

    def __call__(self, trial):
        # max_bin = trial.suggest_int("max_bin", 2, 1024)
        param = {
            "objective": self.objective,
            "metric": self.eval_metric_name,
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": self.learning_rate,
            "num_leaves": trial.suggest_int("num_leaves", 2, 2048),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": min(
                trial.suggest_float("feature_fraction", 0.3, 1.0 + EPS), 1.0
            ),
            "bagging_fraction": min(
                trial.suggest_float("bagging_fraction", 0.3, 1.0 + EPS), 1.0
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "feature_pre_filter": False,
            "seed": self.seed,
            "num_threads": self.n_jobs,
        }

        if self.cat_features_indices:
            param["cat_feature"] = self.cat_features_indices
            param["cat_l2"] = trial.suggest_float("cat_l2", EPS, 100.0)
            param["cat_smooth"] = trial.suggest_float("cat_smooth", EPS, 100.0)

        try:

            pruning_callback = optuna.integration.LightGBMPruningCallback(
                trial, "auc", "validation"
            )

            gbm = lgb.train(
                param,
                self.dtrain,
                valid_sets=[self.dvalid],
                valid_names=["validation"],
                verbose_eval=False,
                callbacks=[pruning_callback],
                num_boost_round=self.rounds,
                early_stopping_rounds=self.early_stopping_rounds,
            )

            preds = gbm.predict(self.X_validation)
            score = self.eval_metric(self.y_validation, preds)
            if Metric.optimize_negative(self.eval_metric.name):
                score *= -1.0
        except optuna.exceptions.TrialPruned as e:
            raise e
        except Exception as e:
            print("Exception in LightgbmObjective", str(e))
            return None

        return score
