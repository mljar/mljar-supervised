import os
import json
import joblib
import optuna
import warnings
import matplotlib
from matplotlib import pyplot as plt

from supervised.utils.metric import Metric
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
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
            "mse",
            "mae",
            "mape",
            "r2",
            "spearman",
            "pearson",
            "f1",
            "average_precision",
            "accuracy",
            "user_defined_metric",
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
            "Neural Network",
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

        self.cat_features_indices = []
        for i in range(X_train.shape[1]):
            if PreprocessingUtils.is_categorical(X_train.iloc[:, i]):
                self.cat_features_indices += [i]

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

        self.plot_study(algorithm, data_type, study)

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

    def plot_study(self, algorithm, data_type, study):

        key = f"{data_type}_{algorithm}"

        plots = [
            (
                optuna.visualization.matplotlib.plot_optimization_history,
                "optimization_history",
            ),
            (
                optuna.visualization.matplotlib.plot_parallel_coordinate,
                "parallel_coordinate",
            ),
            (
                optuna.visualization.matplotlib.plot_param_importances,
                "param_importances",
            ),
            # (optuna.visualization.matplotlib.plot_slice, "slice"),
        ]

        matplotlib_default_figsize = matplotlib.rcParams["figure.figsize"]
        matplotlib.rcParams["figure.figsize"] = (11, 7)

        md = f"# Optuna tuning for {algorithm} on {data_type} data\n\n"
        for plot, title in plots:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plt.figure()
                    plt.rcParams["axes.grid"] = title != "parallel_coordinate"
                    plot(study)
                    plt.tight_layout(pad=2.0)
                    fname = f"{key}_{title}.png"
                    plt.savefig(os.path.join(self.study_dir, fname))
                    plt.close("all")

                    md += f'## {algorithm} {title.replace("_", " ").title()}\n\n'
                    md += f"![{algorithm} {data_type} {title}]({fname})\n\n"

            except Exception as e:
                print(str(e))

        matplotlib.rcParams["figure.figsize"] = matplotlib_default_figsize
        plt.style.use("default")

        with open(os.path.join(self.study_dir, "README.md"), "a") as fout:
            fout.write(md)
            fout.write("\n\n[<< Go back](../README.md)\n")
