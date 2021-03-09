import lightgbm as lgb
import optuna

from supervised.utils.metric import Metric

EPS = 1e-8


class LightgbmObjective:
    def __init__(
        self,
        X_train,
        y_train,
        sample_weight,
        X_validation,
        y_validation,
        sample_weight_validation,
        eval_metric,
        cat_features_indices,
    ):
        print("LightgbmObjective", eval_metric.name)

        self.dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        self.dvalid = lgb.Dataset(
            X_validation, label=y_validation, weight=sample_weight_validation
        )
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.cat_features_indices = cat_features_indices
        self.eval_metric = eval_metric

    def __call__(self, trial):
        param = {
            "objective": "binary",
            "metric": self.eval_metric.name,
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "num_leaves": trial.suggest_int("num_leaves", 2, 512),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 20.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 20.0, log=True),
            "feature_fraction": min(
                trial.suggest_float("feature_fraction", 0.4, 1.0 + EPS), 1.0
            ),
            "bagging_fraction": min(
                trial.suggest_float("bagging_fraction", 0.4, 1.0 + EPS), 1.0
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "feature_pre_filter": False,
        }

        if self.cat_features_indices:
            param["cat_feature"] = self.cat_features_indices
            param["cat_l2"] = trial.suggest_float("cat_l2", EPS, 100.0)
            param["cat_smooth"] = trial.suggest_float("cat_smooth", EPS, 100.0)

        try:
            pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
            gbm = lgb.train(
                param,
                self.dtrain,
                valid_sets=[self.dvalid],
                verbose_eval=[100],
                callbacks=[pruning_callback],
                num_boost_round=1000,
                early_stopping_rounds=50,
            )

            preds = gbm.predict(self.X_validation)
            score = self.eval_metric(self.y_validation, preds)
            if Metric.optimize_negative(self.eval_metric.name):
                score *= -1.0
        except Exception as e:
            print("Exception in LightgbmObjective", str(e))
            return None

        return score
