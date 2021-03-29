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
from supervised.tuner.optuna.knn import KNNObjective
from supervised.tuner.optuna.nn import NeuralNetworkObjective
from supervised.exceptions import AutoMLException



class OptunaTuner:
    def __init__(
        self,
        results_path,
        ml_task,
        eval_metric,
        time_budget=3600,
        init_params={},
        verbose=True,
        n_jobs=-1,
        random_state=42,
    ):
        if eval_metric.name not in [
            "auc",
            "logloss",
            "rmse",
            "mae",
            "mape",
            "r2",
            "spearman",
            "pearson",
            "f1",
            "average_precision",
        ]:
            raise AutoMLException(f"Metric {eval_metric.name} is not supported")

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
        self.n_warmup_steps = (
            500  # set large enough to give small learning rates a chance
        )
        self.time_budget = time_budget
        self.verbose = verbose
        self.ml_task = ml_task
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.cat_features_indices = []
        data_info_fname = os.path.join(results_path, "data_info.json")
        if os.path.exists(data_info_fname):
            data_info = json.loads(open(data_info_fname).read())
            for i, (k, v) in enumerate(data_info["columns_info"].items()):
                if "categorical" in v:
                    self.cat_features_indices += [i]

        self.load()
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    @staticmethod
    def is_optimizable(algorithm_name):
        return algorithm_name in [
            "Extra Trees",
            "Random Forest",
            "CatBoost",
            "Xgboost",
            "LightGBM",
            "Nearest Neighbors",
            "Neural Network"
        ]

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
        # only tune models with original data type
        if data_type != "original":
            return learner_params

        key = f"{data_type}_{algorithm}"
        if key in self.tuning:
            return self.update_learner_params(learner_params, self.tuning[key])

        if self.verbose:
            print(
                f"Optuna optimizes {algorithm} with time budget {self.time_budget} seconds "
                f"eval_metric {self.eval_metric.name} ({self.direction})"
            )

        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=self.n_warmup_steps),
        )
        obejctive = None
        if algorithm == "LightGBM":
            objective = LightgbmObjective(
                self.ml_task,
                X_train,
                y_train,
                sample_weight,
                X_validation,
                y_validation,
                sample_weight_validation,
                self.eval_metric,
                self.cat_features_indices,
                self.n_jobs,
                self.random_state,
            )
        elif algorithm == "Xgboost":
            objective = XgboostObjective(
                self.ml_task,
                X_train,
                y_train,
                sample_weight,
                X_validation,
                y_validation,
                sample_weight_validation,
                self.eval_metric,
                self.n_jobs,
                self.random_state,
            )
        elif algorithm == "CatBoost":
            objective = CatBoostObjective(
                self.ml_task,
                X_train,
                y_train,
                sample_weight,
                X_validation,
                y_validation,
                sample_weight_validation,
                self.eval_metric,
                self.cat_features_indices,
                self.n_jobs,
                self.random_state,
            )
        elif algorithm == "Random Forest":
            objective = RandomForestObjective(
                self.ml_task,
                X_train,
                y_train,
                sample_weight,
                X_validation,
                y_validation,
                sample_weight_validation,
                self.eval_metric,
                self.n_jobs,
                self.random_state,
            )
        elif algorithm == "Extra Trees":
            objective = ExtraTreesObjective(
                self.ml_task,
                X_train,
                y_train,
                sample_weight,
                X_validation,
                y_validation,
                sample_weight_validation,
                self.eval_metric,
                self.n_jobs,
                self.random_state,
            )
        elif algorithm == "Nearest Neighbors":
            objective = KNNObjective(
                self.ml_task,
                X_train,
                y_train,
                sample_weight,
                X_validation,
                y_validation,
                sample_weight_validation,
                self.eval_metric,
                self.n_jobs,
                self.random_state,
            )
        elif algorithm == "Neural Network":
            objective = NeuralNetworkObjective(
                self.ml_task,
                X_train,
                y_train,
                sample_weight,
                X_validation,
                y_validation,
                sample_weight_validation,
                self.eval_metric,
                self.n_jobs,
                self.random_state,
            )

        study.optimize(objective, n_trials=5000, timeout=self.time_budget)

        joblib.dump(study, os.path.join(self.study_dir, key + ".joblib"))

        best = study.best_params

        if algorithm == "LightGBM":
            best["metric"] = objective.eval_metric_name
            best["custom_eval_metric_name"] = objective.custom_eval_metric_name
            best["num_boost_round"] = objective.rounds
            best["early_stopping_rounds"] = objective.early_stopping_rounds
            # best["learning_rate"] = objective.learning_rate
            best["cat_feature"] = self.cat_features_indices
            best["feature_pre_filter"] = False
            best["seed"] = objective.seed
        elif algorithm == "CatBoost":
            best["eval_metric"] = objective.eval_metric_name
            best["num_boost_round"] = objective.rounds
            best["early_stopping_rounds"] = objective.early_stopping_rounds
            # best["bootstrap_type"] = "Bernoulli"
            # best["learning_rate"] = objective.learning_rate
            best["seed"] = objective.seed
        elif algorithm == "Xgboost":
            best["objective"] = objective.objective
            best["eval_metric"] = objective.eval_metric_name
            # best["eta"] = objective.learning_rate
            best["max_rounds"] = objective.rounds
            best["early_stopping_rounds"] = objective.early_stopping_rounds
            best["seed"] = objective.seed
        elif algorithm == "Extra Trees":
            # Extra Trees are not using early stopping
            best["max_steps"] = objective.max_steps  # each step has 100 trees
            best["seed"] = objective.seed
            best["eval_metric_name"] = self.eval_metric.name
        elif algorithm == "Random Forest":
            # Random Forest is not using early stopping
            best["max_steps"] = objective.max_steps  # each step has 100 trees
            best["seed"] = objective.seed
            best["eval_metric_name"] = self.eval_metric.name
        elif algorithm == "Nearest Neighbors":
            best["rows_limit"] = 100000
        elif algorithm == "Neural Network":
            pass

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
            params = json.loads(open(self.tuning_fname).read())
            for k, v in params.items():
                self.tuning[k] = v
