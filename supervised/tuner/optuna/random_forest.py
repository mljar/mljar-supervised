

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import catboost
import optuna

from supervised.utils.metric import Metric

EPS = 1e-8


class RandomForestObjective:
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
        print("RandomForestObjective")

        self.X_train = X_train
        self.y_train = y_train
        self.sample_weight = sample_weight
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.eval_metric = eval_metric

    def __call__(self, trial):
        try:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=trial.suggest_int('max_depth', 2, 24),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 100),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 100),
                max_features=trial.suggest_float("max_features", 1e-8, 1),
                n_jobs=-1
            )
            model.fit(
                self.X_train,
                self.y_train,
                sample_weight=self.sample_weight
            )
        except Exception as e:
            print("Exception in RandomForestObjective", str(e))
            return None
        preds = model.predict_proba(
            self.X_validation
        )[:, 1]
        
        score = self.eval_metric(self.y_validation, preds)
        if Metric.optimize_negative(self.eval_metric.name):
            score *= -1.0

        return score

