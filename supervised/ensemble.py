import os
import logging
import copy
import numpy as np
import pandas as pd
import time
import uuid
import json
import operator

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.model_framework import ModelFramework
from supervised.utils.metric import Metric
from supervised.utils.config import LOG_LEVEL
from supervised.utils.additional_metrics import AdditionalMetrics

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

import matplotlib.pyplot as plt
from tabulate import tabulate

from supervised.utils.learning_curves import LearningCurves


class Ensemble:

    algorithm_name = "Greedy Ensemble"
    algorithm_short_name = "Ensemble"

    def __init__(
        self, optimize_metric="logloss", ml_task=BINARY_CLASSIFICATION, is_stacked=False
    ):
        self.library_version = "0.1"
        self.uid = str(uuid.uuid4())

        self.metric = Metric({"name": optimize_metric})
        self.best_loss = self.metric.get_maximum()  # the best loss obtained by ensemble
        self.models_map = None
        self.selected_models = []
        self.train_time = None
        self.total_best_sum = None  # total sum of predictions, the oof of ensemble
        self.target = None
        self.target_columns = None
        self.sample_weight = None
        self._ml_task = ml_task
        self._optimize_metric = optimize_metric
        self._is_stacked = is_stacked

        self._additional_metrics = None
        self._threshold = None
        self._name = "Ensemble_Stacked" if is_stacked else "Ensemble"
        self._scores = []
        self.oof_predictions = None

    def get_train_time(self):
        return self.train_time

    def get_final_loss(self):
        return self.best_loss

    def is_valid(self):
        return len(self.selected_models) > 1

    def get_type(self):
        prefix = ""  # "Stacked" if self._is_stacked else ""
        return prefix + self.algorithm_short_name

    def get_name(self):
        return self._name

    def get_metric_name(self):
        return self.metric.name

    def get_metric(self):
        return self.metric

    def get_out_of_folds(self):
        """ Needed when ensemble is treated as model and we want to compute additional metrics for it """
        # single prediction (in case of binary classification and regression)
        if self.oof_predictions is not None:
            return self.oof_predictions.copy(deep=True)

        if self.total_best_sum.shape[1] == 1:
            tmp_df = pd.DataFrame({"prediction": self.total_best_sum["prediction"]})
            tmp_df["target"] = self.target[self.target_columns]
            return tmp_df

        ensemble_oof = pd.DataFrame(
            data=self.total_best_sum,
            columns=self.total_best_sum.columns
            # [
            # "prediction_{}".format(i) for i in range(self.total_best_sum.shape[1])
            # ]
        )
        ensemble_oof["target"] = self.target
        if self.sample_weight is not None:
            ensemble_oof["sample_weight"] = self.sample_weight

        self.oof_predictions = ensemble_oof
        return ensemble_oof

    def _get_mean(self, oof_selected, best_sum, best_count):
        resp = copy.deepcopy(oof_selected)
        if best_count > 1:
            resp += best_sum
            resp /= float(best_count)
        return resp

    def get_oof_matrix(self, models):
        # remember models, will be needed in predictions
        self.models_map = {m.get_name(): m for m in models}

        oofs = {}
        for m in models:
            # do not use model with RandomFeature
            if "RandomFeature" in m.get_name():
                continue

            # ensemble only the same level of stack
            # if m._is_stacked != self._is_stacked:
            #    continue
            oof = m.get_out_of_folds()
            prediction_cols = [c for c in oof.columns if "prediction" in c]
            oofs[m.get_name()] = oof[prediction_cols]  # oof["prediction"]
            if self.target is None:

                self.target_columns = [c for c in oof.columns if "target" in c]
                self.target = oof[
                    self.target_columns
                ]  # it will be needed for computing advance model statistics

            if self.sample_weight is None and "sample_weight" in oof.columns:
                self.sample_weight = oof["sample_weight"]

        return oofs, self.target, self.sample_weight

    def get_additional_metrics(self):
        if self._additional_metrics is None:
            logger.debug("Get additional metrics for Ensemble")
            # 'target' - the target after processing used for model training
            # 'prediction' - out of folds predictions of the model
            oof_predictions = self.get_out_of_folds()
            prediction_cols = [c for c in oof_predictions.columns if "prediction" in c]
            target_cols = [c for c in oof_predictions.columns if "target" in c]

            oof_preds = oof_predictions[prediction_cols]
            if self._ml_task == MULTICLASS_CLASSIFICATION:
                cols = oof_preds.columns.tolist()
                # prediction_
                labels = {i: v[11:] for i, v in enumerate(cols)}

                oof_preds["label"] = np.argmax(
                    np.array(oof_preds[prediction_cols]), axis=1
                )
                oof_preds["label"] = oof_preds["label"].map(labels)

            sample_weight = None
            if "sample_weight" in oof_predictions.columns:
                sample_weight = oof_predictions["sample_weight"]

            self._additional_metrics = AdditionalMetrics.compute(
                oof_predictions[target_cols], oof_preds, sample_weight, self._ml_task
            )
            if self._ml_task == BINARY_CLASSIFICATION:
                self._threshold = float(self._additional_metrics["threshold"])

        return self._additional_metrics

    def fit(self, oofs, y, sample_weight=None):
        logger.debug("Ensemble.fit")
        start_time = time.time()
        selected_algs_cnt = 0  # number of selected algorithms
        self.best_algs = []  # selected algoritms indices from each loop

        best_sum = None  # sum of best algorihtms
        for j in range(len(oofs)):  # iterate over all solutions
            min_score = self.metric.get_maximum()
            best_model = None
            # try to add some algorithm to the best_sum to minimize metric
            for model_name in oofs.keys():
                y_ens = self._get_mean(oofs[model_name], best_sum, j + 1)
                score = self.metric(y, y_ens, sample_weight)

                if self.metric.improvement(previous=min_score, current=score):
                    min_score = score
                    best_model = model_name

            # there is improvement, save it
            self._scores += [min_score]

            if self.metric.improvement(previous=self.best_loss, current=min_score):
                self.best_loss = min_score
                selected_algs_cnt = j

            self.best_algs.append(best_model)  # save the best algoritm
            # update best_sum value
            best_sum = (
                oofs[best_model] if best_sum is None else best_sum + oofs[best_model]
            )
            if j == selected_algs_cnt:
                self.total_best_sum = copy.deepcopy(best_sum)
        # end of main loop #

        # keep oof predictions of ensemble
        self.total_best_sum /= float(selected_algs_cnt + 1)
        self.best_algs = self.best_algs[: (selected_algs_cnt + 1)]

        logger.debug("Selected models for ensemble:")
        for model_name in np.unique(self.best_algs):
            self.selected_models += [
                {
                    "model": self.models_map[model_name],
                    "repeat": float(self.best_algs.count(model_name)),
                }
            ]
            logger.debug(f"{model_name} {self.best_algs.count(model_name)}")

        self._additional_metrics = self.get_additional_metrics()

        self.train_time = time.time() - start_time

    def predict(self, X, X_stacked=None):
        logger.debug(
            "Ensemble.predict with {} models".format(len(self.selected_models))
        )
        y_predicted_ensemble = None
        total_repeat = 0.0

        for selected in self.selected_models:
            model = selected["model"]
            repeat = selected["repeat"]
            total_repeat += repeat

            if model._is_stacked:
                y_predicted_from_model = model.predict(X_stacked)
            else:
                y_predicted_from_model = model.predict(X)

            prediction_cols = []
            if self._ml_task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                prediction_cols = [
                    c for c in y_predicted_from_model.columns if "prediction_" in c
                ]
            else:  # REGRESSION
                prediction_cols = ["prediction"]
            y_predicted_from_model = y_predicted_from_model[prediction_cols]
            y_predicted_ensemble = (
                y_predicted_from_model * repeat
                if y_predicted_ensemble is None
                else y_predicted_ensemble + y_predicted_from_model * repeat
            )

        y_predicted_ensemble /= total_repeat

        if self._ml_task == MULTICLASS_CLASSIFICATION:
            cols = y_predicted_ensemble.columns.tolist()
            # prediction_
            labels = {i: v[11:] for i, v in enumerate(cols)}

            y_predicted_ensemble["label"] = np.argmax(
                np.array(y_predicted_ensemble[prediction_cols]), axis=1
            )
            y_predicted_ensemble["label"] = y_predicted_ensemble["label"].map(labels)

        return y_predicted_ensemble

    def to_json(self):
        models_json = []
        for selected in self.selected_models:
            model = selected["model"]
            repeat = selected["repeat"]
            models_json += [{"model": model.to_json(), "repeat": repeat}]

        json_desc = {
            "library_version": self.library_version,
            "algorithm_name": self.algorithm_name,
            "algorithm_short_name": self.algorithm_short_name,
            "uid": self.uid,
            "models": models_json,
        }
        return json_desc

    def from_json(self, json_desc):
        self.library_version = json_desc.get("library_version", self.library_version)
        self.algorithm_name = json_desc.get("algorithm_name", self.algorithm_name)
        self.algorithm_short_name = json_desc.get(
            "algorithm_short_name", self.algorithm_short_name
        )
        self.uid = json_desc.get("uid", self.uid)
        self.selected_models = []
        models_json = json_desc.get("models")
        for selected in models_json:
            model = selected["model"]
            repeat = selected["repeat"]

            il = ModelFramework(model.get("params"))
            il.from_json(model)
            self.selected_models += [
                # {"model": LearnerFactory.load(model), "repeat": repeat}
                {"model": il, "repeat": repeat}
            ]

    def save(self, model_path):
        logger.info(f"Save the ensemble to {model_path}")

        predictions = self.get_out_of_folds()
        predictions_fname = os.path.join(model_path, f"predictions_ensemble.csv")
        predictions.to_csv(predictions_fname, index=False)

        with open(os.path.join(model_path, "ensemble.json"), "w") as fout:
            ms = []
            for selected in self.selected_models:
                ms += [{"model": selected["model"]._name, "repeat": selected["repeat"]}]

            desc = {
                "name": self._name,
                "ml_task": self._ml_task,
                "optimize_metric": self._optimize_metric,
                "selected_models": ms,
                "predictions_fname": predictions_fname,
                "metric_name": self.get_metric_name(),
                "final_loss": self.get_final_loss(),
                "train_time": self.get_train_time(),
                "is_stacked": self._is_stacked,
            }

            if self._threshold is not None:
                desc["threshold"] = self._threshold
            fout.write(json.dumps(desc, indent=4))

        LearningCurves.plot_for_ensemble(self._scores, self.metric.name, model_path)

        # call additional metics just to be sure they are computed
        self._additional_metrics = self.get_additional_metrics()

        AdditionalMetrics.save(
            self._additional_metrics, self._ml_task, self.model_markdown(), model_path
        )

        with open(os.path.join(model_path, "status.txt"), "w") as fout:
            fout.write("ALL OK!")

    def model_markdown(self):
        select_models_desc = []
        for selected in self.selected_models:
            select_models_desc += [
                {"model": selected["model"]._name, "repeat": selected["repeat"]}
            ]
        desc = f"# Summary of {self.get_name()}\n\n"
        desc += "[<< Go back](../README.md)\n\n"
        desc += "\n## Ensemble structure\n"
        selected = pd.DataFrame(select_models_desc)
        desc += tabulate(selected.values, ["Model", "Weight"], tablefmt="pipe")
        desc += "\n"
        return desc

    @staticmethod
    def load(model_path, models_map):
        logger.info(f"Loading ensemble from {model_path}")

        json_desc = json.load(open(os.path.join(model_path, "ensemble.json")))

        ensemble = Ensemble(json_desc.get("optimize_metric"), json_desc.get("ml_task"))
        ensemble._name = json_desc.get("name", ensemble._name)
        ensemble._threshold = json_desc.get("threshold", ensemble._threshold)
        for m in json_desc.get("selected_models", []):
            ensemble.selected_models += [
                {"model": models_map[m["model"]], "repeat": m["repeat"]}
            ]

        ensemble.best_loss = json_desc.get("final_loss", ensemble.best_loss)
        ensemble.train_time = json_desc.get("train_time", ensemble.train_time)
        ensemble._is_stacked = json_desc.get("is_stacked", ensemble._is_stacked)
        predictions_fname = json_desc.get("predictions_fname")
        if predictions_fname is not None:
            ensemble.oof_predictions = pd.read_csv(predictions_fname)
        return ensemble
