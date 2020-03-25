import os
import logging
import copy
import numpy as np
import pandas as pd
import time
import uuid
import json
import operator

from supervised.utils.config import storage_path
from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.algorithm_factory import AlgorithmFactory
from supervised.model_framework import ModelFramework
from supervised.utils.metric import Metric
from supervised.utils.config import LOG_LEVEL
from supervised.utils.additional_metrics import AdditionalMetrics

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class Ensemble:

    algorithm_name = "Greedy Ensemble"
    algorithm_short_name = "Ensemble"

    def __init__(self, optimize_metric="logloss", ml_task=BINARY_CLASSIFICATION):
        self.library_version = "0.1"
        self.uid = str(uuid.uuid4())
        self.model_file = self.uid + ".ensemble.model"
        self.model_file_path = os.path.join(storage_path, self.model_file)
        self.metric = Metric({"name": optimize_metric})
        self.best_loss = self.metric.get_maximum()  # the best loss obtained by ensemble
        self.models = None
        self.selected_models = []
        self.train_time = None
        self.total_best_sum = None  # total sum of predictions, the oof of ensemble
        self.target = None
        self.target_columns = None
        self._ml_task = ml_task

        self._additional_metrics = None 
        self._threshold = None
        self._name = None

    def get_train_time(self):
        return self.train_time

    def get_final_loss(self):
        return self.best_loss

    def get_name(self):
        return self.algorithm_short_name

    def get_out_of_folds(self):
        """ Needed when ensemble is treated as model and we want to compute additional metrics for it """
        # single prediction (in case of binary classification and regression)
        logger.debug(self.total_best_sum.shape)
        logger.debug(self.total_best_sum.head())
        
        logger.debug(self.target.shape)
        logger.debug(self.target.head())

        if self.total_best_sum.shape[1] == 1:
            tmp_df = pd.DataFrame({"prediction": self.total_best_sum["prediction"]})
            tmp_df["target"] = self.target[self.target_columns]
            return tmp_df

        ensemble_oof = pd.DataFrame(
            data=self.total_best_sum,
            columns=[
                "prediction_{}".format(i) for i in range(self.total_best_sum.shape[1])
            ],
        )
        ensemble_oof["target"] = self.target
        return ensemble_oof

    def _get_mean(self, oofs, best_sum, best_count, selected):
        resp = copy.deepcopy(oofs[selected])
        if best_count > 1:
            resp += best_sum
            resp /= float(best_count)
        return resp

    def get_oof_matrix(self, models):
        # remeber models, will be needed in predictions
        self.models = models

        oofs = {}
        for i, m in enumerate(models):
            oof = m.get_out_of_folds()
            prediction_cols = [c for c in oof.columns if "prediction" in c]
            oofs["model_{}".format(i)] = oof[prediction_cols]  # oof["prediction"]
            if self.target is None:

                self.target_columns = [c for c in oof.columns if "target" in c]
                self.target = oof[
                    self.target_columns
                ]  # it will be needed for computing advance model statistics
                
        return oofs, self.target

    def get_additional_metrics(self):
        if self._additional_metrics is None:
            # 'target' - the target after processing used for model training
            # 'prediction' - out of folds predictions of the model
            oof_predictions = self.get_out_of_folds()
            prediction_cols = [c for c in oof_predictions.columns if "prediction" in c]
            target_cols = [c for c in oof_predictions.columns if "target" in c]
            self._additional_metrics = AdditionalMetrics.compute(
                oof_predictions[target_cols],
                oof_predictions[prediction_cols],
                self._ml_task,
            )
            if self._ml_task == BINARY_CLASSIFICATION:
                self._threshold = float(self._additional_metrics["threshold"])
                print(self._additional_metrics["max_metrics"])
                print(self._threshold)
        return self._additional_metrics

    def fit(self, oofs, y):
        logger.debug("Ensemble.fit")
        start_time = time.time()
        selected_algs_cnt = 0  # number of selected algorithms
        self.best_algs = []  # selected algoritms indices from each loop

        best_sum = None  # sum of best algorihtms
        for j in range(len(oofs)):  # iterate over all solutions
            min_score = self.metric.get_maximum()
            best_index = -1
            # try to add some algorithm to the best_sum to minimize metric
            for i in range(len(oofs)):
                y_ens = self._get_mean(oofs, best_sum, j + 1, "model_{}".format(i))
                score = self.metric(y, y_ens)

                if self.metric.improvement(previous=min_score, current=score):
                    min_score = score
                    best_index = i

            # there is improvement, save it
            print(j, self.best_loss, min_score)
            if self.metric.improvement(previous=self.best_loss, current=min_score):
                self.best_loss = min_score
                selected_algs_cnt = j

            self.best_algs.append(best_index)  # save the best algoritm index
            # update best_sum value
            best_sum = (
                oofs["model_{}".format(best_index)]
                if best_sum is None
                else best_sum + oofs["model_{}".format(best_index)]
            )
            if j == selected_algs_cnt:
                self.total_best_sum = copy.deepcopy(best_sum)
        # end of main loop #

        # keep oof predictions of ensemble
        self.total_best_sum /= float(selected_algs_cnt + 1)
        self.best_algs = self.best_algs[: (selected_algs_cnt + 1)]
        logger.debug("Selected models for ensemble:")
        for i in np.unique(self.best_algs):
            self.selected_models += [
                {"model": self.models[i], "repeat": float(np.sum(self.best_algs == i))}
            ]
            logger.debug(f"model_{i}")
        self.get_additional_metrics()
        self.train_time = time.time() - start_time

    def predict(self, X):
        logger.debug(
            "Ensemble.predict with {} models".format(len(self.selected_models))
        )
        y_predicted_ensemble = None
        total_repeat = 0.0

        for selected in self.selected_models:
            model = selected["model"]
            repeat = selected["repeat"]
            total_repeat += repeat

            y_predicted_from_model = model.predict(X)
            prediction_cols = [c for c in y_predicted_from_model.columns if "p_" in c]
            y_predicted_from_model = y_predicted_from_model[prediction_cols]
            y_predicted_ensemble = (
                y_predicted_from_model * repeat
                if y_predicted_ensemble is None
                else y_predicted_ensemble + y_predicted_from_model * repeat
            )

        y_predicted_ensemble /= total_repeat
        '''
        # Ensemble needs to apply reverse transformation of target !!!
        if self._ml_task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:

            label = np.argmax(np.array(y_predicted_ensemble), axis=1)
            prediction_labels = [c[2:] for c in y_predicted_ensemble if "p_" in c]

            prediction_labels = dict((i, l) for i, l in enumerate(prediction_labels))
            # df["label"] = df["label"]
            y_predicted_ensemble["label"] = label
            y_predicted_ensemble["label"] = y_predicted_ensemble["label"].map(
                prediction_labels
            )
        '''
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
        logger.info(f"Save the ensemble {model_path}")

        with open(os.path.join(model_path, "ensemble.json"), "w") as fout:
            desc = []
            for selected in self.selected_models:
                desc += [{"model": selected["model"]._name, "repeat": selected["repeat"]}]
            fout.write(json.dumps(desc, indent=4))

        
        predictions = self.get_out_of_folds()
        predictions.to_csv(
            os.path.join(model_path, f"predictions_ensemble.csv"),
            index=False,
        )

        self._additional_metrics = self.get_additional_metrics()

        with open(os.path.join(model_path, "ensemble_metrics.txt"), "w") as fout:
            if self._ml_task == BINARY_CLASSIFICATION:
                max_metrics = self._additional_metrics["max_metrics"]
                confusion_matrix = self._additional_metrics["confusion_matrix"]
                threshold = self._additional_metrics["threshold"]

                fout.write("Metric details:\n{}\n\n".format(max_metrics.transpose()))
                fout.write(
                    "Confusion matrix (at threshold={}):\n{}".format(
                        np.round(threshold, 6), confusion_matrix
                    )
                )
            elif self._ml_task == MULTICLASS_CLASSIFICATION:
                fout.write("TODO")

        with open(os.path.join(model_path, "status.txt"), "w") as fout:
            fout.write("ALL OK!")

    @staticmethod
    def load(model_path):
        logger.info("Loading model framework")

        json_desc = json.load(open(os.path.join(model_path, "framework.json")))
        mf = ModelFramework(json_desc["params"])
        mf.uid = json_desc.get("uid", mf.uid)
        mf._name = json_desc.get("name", mf._name)
        mf._threshold = json_desc.get("threshold")
        mf.learners = []
        for learner_desc, learner_path in zip(json_desc.get("learners"), json_desc.get("saved")):
            
            l = AlgorithmFactory.load(learner_desc, learner_path)     
            mf.learners += [l]
        
        mf.preprocessings = []
        for p in json_desc.get("preprocessing"):
            ps = PreprocessingStep()
            ps.from_json(p) 
            mf.preprocessings += [ps]

        return mf