import os
import json
import joblib
import optuna

from supervised.utils.metric import Metric
from supervised.tuner.optuna.lightgbm import LightgbmObjective
from supervised.tuner.optuna.xgboost import XgboostObjective
from supervised.tuner.optuna.catboost import CatBoostObjective
from supervised.tuner.optuna.random_forest import RandomForestObjective
from supervised.tuner.optuna.extra_trees import ExtraTreesObjective

class OptunaTuner:
    def __init__(
        self,
        results_path,
        ml_task,
        eval_metric,
        time_budget=1800,
        init_params={},
        verbose=True,
        random_state=42,
    ):
        if eval_metric.name not in ["auc"]:
            print(f"Metric {eval_metric.name} is not supported")

        self.study_dir = os.path.join(results_path, "optuna")
        if not os.path.exists(self.study_dir):
            try:
                os.mkdir(self.study_dir)
            except Exception as e:
                print("Problem while creating directory for optuna studies.", str(e))
        self.tuning_fname = os.path.join(self.study_dir, "optuna.json")
        self.tuning = init_params
        self.eval_metric = eval_metric

        self.direction = (
            "maximize" if Metric.optimize_negative(eval_metric.name) else "minimize"
        )
        self.time_budget = time_budget
        self.random_state = random_state

        self.cat_features_indices = []
        data_info_fname = os.path.join(results_path, "data_info.json")
        if os.path.exists(data_info_fname):
            data_info = json.loads(open(data_info_fname).read())
            for i, (k, v) in enumerate(data_info["columns_info"].items()):
                if "categorical" in v:
                    self.cat_features_indices += [i]
        print("Cat features->", self.cat_features_indices)

    def optimize(
        self,
        algorithm,
        data_type,
        X_train,
        y_train,
        sample_weight,
        X_validation,
        y_validation,
        sample_weight_validation,
        learner_params,
    ):
        print("optimize::check")
        key = f"{data_type}_{algorithm}"
        if key in self.tuning:
            return self.update_learner_params(learner_params, self.tuning[key])

        print("optimize::create_study", algorithm, data_type)
        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
        )
        obejctive = None
        if algorithm == "LightGBM":
            objective = LightgbmObjective(
                X_train,
                y_train,
                sample_weight,
                X_validation,
                y_validation,
                sample_weight_validation,
                self.eval_metric,
                self.cat_features_indices,
            )
        elif algorithm == "Xgboost":
            objective = XgboostObjective(
                X_train,
                y_train,
                sample_weight,
                X_validation,
                y_validation,
                sample_weight_validation,
                self.eval_metric,
            )
        elif algorithm == "CatBoost":
            objective = CatBoostObjective(
                X_train,
                y_train,
                sample_weight,
                X_validation,
                y_validation,
                sample_weight_validation,
                self.eval_metric,
                self.cat_features_indices,
            )
        elif algorithm == "Random Forest":
            objective = RandomForestObjective(
                X_train,
                y_train,
                sample_weight,
                X_validation,
                y_validation,
                sample_weight_validation,
                self.eval_metric
            )
        elif algorithm == "Extra Trees":
            objective = ExtraTreesObjective(
                X_train,
                y_train,
                sample_weight,
                X_validation,
                y_validation,
                sample_weight_validation,
                self.eval_metric
            )

        study.optimize(objective, n_trials=5000, timeout=self.time_budget)

        best = study.best_params

        joblib.dump(study, os.path.join(self.study_dir, key + ".joblib"))

        if algorithm == "LightGBM":
            best["metric"] = self.eval_metric.name
            best["num_boost_round"] = 1000
            best["early_stopping_rounds"] = 50
            best["learning_rate"] = 0.1
        elif algorithm == "CatBoost":
            best["eval_metric"] = self.eval_metric.name
            if best["eval_metric"] == "auc":
                best["eval_metric"] = "AUC"
            best["num_boost_round"] = 1000
            best["early_stopping_rounds"] = 50
            best["learning_rate"] = 0.1
        elif algorithm == "Xgboost":
            best["eval_metric"] = self.eval_metric.name
            best["eta"] = 0.1
            best["max_rounds"] = 1000
            best["early_stopping_rounds"] = 50

        self.tuning[key] = best
        self.save()

        return self.update_learner_params(learner_params, best)

    def update_learner_params(self, learner_params, best):
        for k, v in best.items():
            learner_params[k] = v
        return learner_params

    def save(self):
        with open(self.tuning_fname, "w") as fout:
            fout.write(json.dumps(self.tuning, indent=4))

    def load(self):
        if os.path.exists(self.tuning_fname):
            self.tuning == json.loads(open(self.tuning_fname).read())
