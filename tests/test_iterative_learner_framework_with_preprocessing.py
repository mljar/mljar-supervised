import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import copy

from numpy.testing import assert_almost_equal
from sklearn import datasets
from supervised.models.learner_xgboost import XgbLearner
from supervised.iterative_learner_framework import IterativeLearner
from supervised.callbacks.early_stopping import EarlyStopping
from supervised.callbacks.metric_logger import MetricLogger
from supervised.metric import Metric
from supervised.tuner.random_parameters import RandomParameters
from supervised.tuner.registry import ModelsRegistry
from supervised.tuner.registry import BINARY_CLASSIFICATION
from supervised.tuner.preprocessing_tuner import PreprocessingTuner


class IterativeLearnerWithPreprocessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(None)
        df = pd.read_csv("tests/data/adult_missing_values_missing_target_500rows.csv")
        cls.data = {"train": {"X": df[df.columns[:-1]], "y": df["income"]}}

        available_models = list(ModelsRegistry.registry[BINARY_CLASSIFICATION].keys())
        model_type = np.random.permutation(available_models)[0]
        model_info = ModelsRegistry.registry[BINARY_CLASSIFICATION][model_type]
        model_params = RandomParameters.get(model_info["params"])
        required_preprocessing = model_info["required_preprocessing"]
        model_additional = model_info["additional"]
        preprocessing_params = PreprocessingTuner.get(
            required_preprocessing, cls.data, BINARY_CLASSIFICATION
        )

        cls.train_params = {
            "additional": model_additional,
            "preprocessing": preprocessing_params,
            "validation": {
                "validation_type": "split",
                "train_ratio": 0.8,
                # "validation_type": "kfold",
                # "k_folds": 5,
                "shuffle": True,
            },
            "learner": {
                "model_type": model_info["class"].algorithm_short_name,
                **model_params,
            },
        }

    def test_fit_and_predict_split(self):
        print(self.data["train"]["X"].head())
        self.assertTrue("Private" in list(self.data["train"]["X"]["workclass"]))

        early_stop = EarlyStopping({"metric": {"name": "logloss"}})
        metric_logger = MetricLogger({"metric_names": ["logloss", "auc"]})
        il = IterativeLearner(self.train_params, callbacks=[early_stop, metric_logger])
        il.train(self.data)

        print(self.data["train"]["X"].head())
        self.assertTrue("Private" in list(self.data["train"]["X"]["workclass"]))

        y_predicted = il.predict(self.data["train"]["X"])
        print(self.data["train"]["X"].head())
        self.assertTrue("Private" in list(self.data["train"]["X"]["workclass"]))

        metric = Metric({"name": "logloss"})
        loss = metric(self.data["train"]["y"], y_predicted)
        self.assertTrue(loss < 0.6)


    def test_fit_and_predict_kfold(self):
        print(self.data["train"]["X"].head())
        self.assertTrue("Private" in list(self.data["train"]["X"]["workclass"]))

        early_stop = EarlyStopping({"metric": {"name": "logloss"}})
        metric_logger = MetricLogger({"metric_names": ["logloss", "auc"]})

        params = copy.deepcopy(self.train_params)
        params["validation"] = {
                "validation_type": "kfold",
                "k_folds": 4,
                "shuffle": True,
        }
        il = IterativeLearner(params, callbacks=[early_stop, metric_logger])
        il.train(self.data)

        print(self.data["train"]["X"].head())
        self.assertTrue("Private" in list(self.data["train"]["X"]["workclass"]))

        y_predicted = il.predict(self.data["train"]["X"])
        print(self.data["train"]["X"].head())
        self.assertTrue("Private" in list(self.data["train"]["X"]["workclass"]))

        metric = Metric({"name": "logloss"})
        loss = metric(self.data["train"]["y"], y_predicted)
        self.assertTrue(loss < 0.6)

    def test_save_and_load(self):
        print(self.data["train"]["X"].head())
        self.assertTrue("Private" in list(self.data["train"]["X"]["workclass"]))
        early_stop = EarlyStopping({"metric": {"name": "logloss"}})
        metric_logger = MetricLogger({"metric_names": ["logloss", "auc"]})
        print("train_params",self.train_params)
        il = IterativeLearner(self.train_params, callbacks=[early_stop, metric_logger])
        il.train(self.data)
        y_predicted = il.predict(self.data["train"]["X"])
        metric = Metric({"name": "logloss"})
        loss_1 = metric(self.data["train"]["y"], y_predicted)

        json_desc = il.save()

        il2 = IterativeLearner(self.train_params, callbacks=[])
        self.assertTrue(il.uid != il2.uid)
        il2.load(json_desc)
        self.assertTrue(il.uid == il2.uid)
        y_predicted_2 = il2.predict(self.data["train"]["X"])
        loss_2 = metric(self.data["train"]["y"], y_predicted_2)
        print(loss_1, loss_2)
        assert_almost_equal(loss_1, loss_2)

        uids = [i.uid for i in il.learners]
        uids2 = [i.uid for i in il2.learners]
        for u in uids:
            self.assertTrue(u in uids2)


if __name__ == "__main__":
    unittest.main()
