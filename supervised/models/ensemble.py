import logging
import copy
import numpy as np
import pandas as pd
import uuid
from supervised.models.learner import Learner
from supervised.tuner.registry import ModelsRegistry
from supervised.tuner.registry import BINARY_CLASSIFICATION

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
        self.model_file_path = "/tmp/" + self.model_file
        self.metric = Metric({"name": "logloss"})
        log.debug("EnsembleLearner __init__")

    def get_final_loss(self):
        pass

    def _get_mean(self, X, best_sum, best_count, selected):
        resp = copy.deepcopy(X[selected])
        if best_count > 1:
            resp += best_sum
            resp /= float(best_count)
        return resp

    def fit(self, models, y):
        log.debug("EnsembleLearner.fit")
        oofs = {}
        for i, m in enumerate(models):
            print(
                "ensemble",
                m.learners[0].algorithm_name,
                m.callbacks.callbacks[0].final_loss,
            )
            oof = m.get_out_of_folds()
            oofs["model_{}".format(i)] = oof["prediction"]
        X = pd.DataFrame(oofs)
        print(X.shape)
        print(X.head())

        total_min = 10e12  # total minimum value
        total_j = 0  # total minimum index
        total_best_sum = 0  # total sum of predictions
        self.models = models
        self.best_algs = []  # selected algoritms indices from each loop
        best_sum = None  # sum of best algorihtms
        cost_in_iters = []  # track cost in all iterations
        for j in range(X.shape[1]):  # iterate over all solutions
            min_score = 10e12
            best_index = -1
            # try to add some algorithm to the best_sum to minimize metric
            for i in range(X.shape[1]):
                y_ens = self._get_mean(X, best_sum, j + 1, "model_{}".format(i))

                print(y_ens.shape, y.shape)
                # score = get_score_for_opt(metric_type, y_train, y_ens, w_train)
                score = self.metric(y, y_ens)

                # logger.info('EnsembleAvg: _get_score time = {}'.format(str(time.time()-score_start_time)))
                # print 'score', score, min_score
                if score < min_score:
                    min_score = score
                    best_index = i

            print("j", j, "min_score", min_score, best_index, total_min)

            if min_score + 10e-6 < total_min:
                total_min = min_score
                total_j = j

            self.best_algs.append(best_index)  # save the best algoritm index
            # update best_sum value
            best_sum = (
                X["model_{}".format(best_index)]
                if best_sum is None
                else best_sum + X["model_{}".format(best_index)]
            )
            if j == total_j:
                total_best_sum = best_sum
            print("loop end step")

        total_best_sum /= float(total_j + 1)
        print(total_best_sum.shape)
        # total_best_sum = total_best_sum.reshape((total_best_sum.shape[0],))
        # print(total_best_sum.shape)
        self.best_algs = self.best_algs[: (total_j + 1)]
        # return [all_ids[i] for i in best_algs[:(total_j+1)]], cost_in_iters, total_best_sum, get_score_value(metric_type, total_min)

    def predict(self, X):
        print("Ensemble predict")
        y_predicted = None
        for best in self.best_algs:
            print("best", best)
            y_predicted = (
                self.models[best].predict(X)
                if y_predicted is None
                else y_predicted + self.models[best].predict(X)
            )

        return y_predicted / float(len(self.best_algs))

    def copy(self):
        return copy.deepcopy(self)

    def save(self):
        pass

    def load(self, json_desc):
        pass
