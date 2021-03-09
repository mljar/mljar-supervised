

from catboost import CatBoostClassifier, CatBoostRegressor, CatBoost, Pool
import catboost
import optuna

from supervised.utils.metric import Metric

EPS = 1e-8


class CatBoostObjective:
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
        print("CatBoostObjective")

        self.X_train = X_train
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.eval_metric = eval_metric
        self.cat_features = cat_features_indices
        self.eval_set = Pool(
            data=X_validation,
            label=y_validation,
            cat_features=self.cat_features,
            weight=sample_weight_validation,
        )

    def __call__(self, trial):
        try:
            model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.1,
                depth=trial.suggest_int("depth", 2, 12),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", EPS, 10.0, log=True),
                random_strength=trial.suggest_float(
                    "random_strength", EPS, 10.0, log=True
                ),
                rsm=trial.suggest_float("rsm", EPS, 1),  # colsample_bylevel=rsm
                eval_metric="AUC",
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(
                self.X_train,
                self.y_train,
                sample_weight=self.sample_weight,
                early_stopping_rounds=50,
                eval_set=self.eval_set,
                verbose_eval=False,
                cat_features=self.cat_features,
            )
        except Exception as e:
            print("Exception in CatBoostObjective", str(e))
            return None
        preds = model.predict_proba(
            self.X_validation, ntree_end=model.best_iteration_ + 1
        )[:, 1]
        
        score = self.eval_metric(self.y_validation, preds)
        if Metric.optimize_negative(self.eval_metric.name):
            score *= -1.0

        return score

