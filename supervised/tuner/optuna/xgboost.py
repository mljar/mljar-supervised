import xgboost as xgb
import optuna

from supervised.utils.metric import Metric

EPS = 1e-8

class XgboostObjective:
    def __init__(
        self,
        X_train,
        y_train,
        sample_weight,
        X_validation,
        y_validation,
        sample_weight_validation,
        eval_metric,
    ):
        self.dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
        self.dvalidation = xgb.DMatrix(
            X_validation, label=y_validation, weight=sample_weight_validation
        )
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.eval_metric = eval_metric

    def __call__(self, trial):
        param = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "booster": "gbtree",
            "eta": 0.1,
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "lambda": trial.suggest_float("lambda", EPS, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", EPS, 10.0, log=True),
            "colsample_bytree": min(trial.suggest_float("colsample_bytree", 0.3, 1.0+EPS), 1.0),
            "subsample": min(trial.suggest_float("subsample", 0.3, 1.0+EPS), 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "n_jobs": -1,
            "seed": 1,
        }
        try:
            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "validation-auc"
            )
            bst = xgb.train(
                param,
                self.dtrain,
                1000,
                evals=[(self.dvalidation, "validation")],
                early_stopping_rounds=50,
                callbacks=[pruning_callback],
                verbose_eval=False,
            )

            preds = bst.predict(self.dvalidation, ntree_limit=bst.best_ntree_limit)
            score = self.eval_metric(self.y_validation, preds)
            if Metric.optimize_negative(self.eval_metric.name):
                score *= -1.0

        except Exception as e:
            print("Exception in XgboostObjective", str(e))
            return None

        return score

