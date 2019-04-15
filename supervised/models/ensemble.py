import os
import logging
import copy
import numpy as np
import pandas as pd
import time
import uuid

from supervised.config import storage_path
from supervised.models.learner import Learner
from supervised.tuner.registry import ModelsRegistry
from supervised.tuner.registry import BINARY_CLASSIFICATION
from supervised.models.learner_factory import LearnerFactory
import operator

log = logging.getLogger(__name__)

from supervised.metric import Metric


class Ensemble:

    algorithm_name = "Greedy Ensemble"
    algorithm_short_name = "Ensemble"

    def __init__(self):
        self.library_version = "0.1"
        self.uid = str(uuid.uuid4())
        self.model_file = self.uid + ".ensemble.model"
        self.model_file_path = os.path.join(storage_path, self.model_file)
        # right now only logloss can be optimized by ensemble
        self.metric = Metric({"name": "logloss"})
        self.best_loss = 10e12  # the best loss obtained by ensemble
        self.models = None
        self.selected_models = []
        self.train_time = None
        self.total_best_sum = None  # total sum of predictions, the oof of ensemble
        self.target = None

    def get_train_time(self):
        return self.train_time

    def get_final_loss(self):
        return self.best_loss

    def get_name(self):
        return self.algorithm_short_name

    def get_out_of_folds(self):
        return pd.DataFrame({"prediction": self.total_best_sum, "target": self.target})

    def _get_mean(self, X, best_sum, best_count, selected):
        resp = copy.deepcopy(X[selected])
        if best_count > 1:
            resp += best_sum
            resp /= float(best_count)
        return resp

    def get_oof_matrix(self, models):
        oofs = {}
        for i, m in enumerate(models):
            oof = m.get_out_of_folds()
            oofs["model_{}".format(i)] = oof["prediction"]
            if self.target is None:
                self.target = oof["target"] # it will be needed for computing advance model statistics
                                            # it can be a mess in the future when target will be transformed depending on each model

        X = pd.DataFrame(oofs)
        self.models = models  # remeber models, will be needed in predictions
        return X

    def fit(self, X, y):
        start_time = time.time()
        selected_algs_cnt = 0  # number of selected algorithms
        self.best_algs = []  # selected algoritms indices from each loop

        best_sum = None  # sum of best algorihtms
        for j in range(X.shape[1]):  # iterate over all solutions
            min_score = 10e12
            best_index = -1
            # try to add some algorithm to the best_sum to minimize metric
            for i in range(X.shape[1]):
                y_ens = self._get_mean(X, best_sum, j + 1, "model_{}".format(i))
                score = self.metric(y, y_ens)
                if score < min_score:
                    min_score = score
                    best_index = i

            # there is improvement, save it
            if min_score + 10e-6 < self.best_loss:
                self.best_loss = min_score
                selected_algs_cnt = j

            self.best_algs.append(best_index)  # save the best algoritm index
            # update best_sum value
            best_sum = (
                X["model_{}".format(best_index)]
                if best_sum is None
                else best_sum + X["model_{}".format(best_index)]
            )
            if j == selected_algs_cnt:
                self.total_best_sum = copy.deepcopy(best_sum)

        # keep oof predictions of ensemble
        self.total_best_sum /= float(selected_algs_cnt + 1)
        self.best_algs = self.best_algs[: (selected_algs_cnt + 1)]
        for i in np.unique(self.best_algs):
            self.selected_models += [
                {"model": self.models[i], "repeat": np.sum(self.best_algs == i)}
            ]
        self.train_time = time.time() - start_time

    def predict(self, X):
        y_predicted = None
        total_repeat = 0.0
        for selected in self.selected_models:
            model = selected["model"]
            repeat = selected["repeat"]
            total_repeat += repeat
            y_predicted = (
                model.predict(X) * repeat
                if y_predicted is None
                else y_predicted + model.predict(X) * repeat
            )
        return y_predicted / total_repeat

    def to_json(self):
        models_json = []
        for selected in self.selected_models:
            model = selected["model"]
            repeat = selected["repeat"]
            models_json += [{"model": model.save(), "repeat": repeat}]

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
            self.selected_models += [
                {"model": LearnerFactory.load(model), "repeat": repeat}
            ]
