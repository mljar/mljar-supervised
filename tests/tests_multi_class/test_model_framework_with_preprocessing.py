import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import copy

from numpy.testing import assert_almost_equal
from sklearn import datasets
from supervised.algorithms.xgboost import XgbAlgorithm
from supervised.model_framework import ModelFramework
from supervised.callbacks.early_stopping import EarlyStopping
from supervised.callbacks.metric_logger import MetricLogger
from supervised.utils.metric import Metric
from supervised.tuner.random_parameters import RandomParameters
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.tuner.preprocessing_tuner import PreprocessingTuner


class ModelFrameworkWithPreprocessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(None)
        df = pd.read_csv("./tests/data/iris_missing_values_missing_target.csv")
        cls.data = {"train": {"X": df[df.columns[:-1]], "y": df["class"]}}

        model_type = "Xgboost" 
        model_info = AlgorithmsRegistry.registry[MULTICLASS_CLASSIFICATION][model_type]
        model_params = RandomParameters.get(model_info["params"])
        required_preprocessing = model_info["required_preprocessing"]
        model_additional = model_info["additional"]
        preprocessing_params = PreprocessingTuner.get(
            required_preprocessing, cls.data, MULTICLASS_CLASSIFICATION
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
        
        early_stop = EarlyStopping({"metric": {"name": "logloss"}})
        metric_logger = MetricLogger({"metric_names": ["logloss"]})
        il = ModelFramework(self.train_params, callbacks=[early_stop, metric_logger])
        il.train(self.data)

        response = il.predict(self.data["train"]["X"])
        
        metric = Metric({"name": "logloss"})
        y = self.data["train"]["y"]
        not_null_index = ~pd.isnull(y)
        ll = metric(y[not_null_index], response[["p_Iris-setosa", "p_Iris-versicolor", "p_Iris-virginica"]][list(not_null_index)])
        self.assertTrue(ll < 0.6)

    def test_fit_and_predict_kfold(self):
        
        early_stop = EarlyStopping({"metric": {"name": "logloss"}})
        metric_logger = MetricLogger({"metric_names": ["logloss"]})

        params = copy.deepcopy(self.train_params)
        params["validation"] = {
            "validation_type": "kfold",
            "k_folds": 5,
            "shuffle": True,
        }
        il = ModelFramework(params, callbacks=[early_stop, metric_logger])
        il.train(self.data)
        oof = il.get_out_of_folds()

        self.assertEqual(len(np.unique(oof.index)), oof.shape[0])
        self.assertTrue(oof.shape[0], self.data["train"]["X"].shape[0])
        
        response = il.predict(self.data["train"]["X"])
        metric = Metric({"name": "logloss"})
        y = self.data["train"]["y"]
        not_null_index = ~pd.isnull(y)
        ll = metric(y[not_null_index], response[["p_Iris-setosa", "p_Iris-versicolor", "p_Iris-virginica"]][list(not_null_index)])
        self.assertTrue(ll < 0.6)
        
    def test_save_and_load(self):
        
        early_stop = EarlyStopping({"metric": {"name": "logloss"}})
        metric_logger = MetricLogger({"metric_names": ["logloss"]})

        il = ModelFramework(self.train_params, callbacks=[early_stop, metric_logger])
        il.train(self.data)

        response = il.predict(self.data["train"]["X"])
        metric = Metric({"name": "logloss"})
        y = self.data["train"]["y"]
        not_null_index = ~pd.isnull(y)
        loss_1 = metric(y[not_null_index], response[["p_Iris-setosa", "p_Iris-versicolor", "p_Iris-virginica"]][list(not_null_index)])
        self.assertTrue(loss_1 < 0.6)

        json_desc = il.to_json()

        il2 = ModelFramework(self.train_params, callbacks=[])
        self.assertTrue(il.uid != il2.uid)
        il2.from_json(json_desc)
        self.assertTrue(il.uid == il2.uid)
        
        response = il2.predict(self.data["train"]["X"])
        metric = Metric({"name": "logloss"})
        y = self.data["train"]["y"]
        not_null_index = ~pd.isnull(y)
        loss_2 = metric(y[not_null_index], response[["p_Iris-setosa", "p_Iris-versicolor", "p_Iris-virginica"]][list(not_null_index)])
        self.assertTrue(loss_2 < 0.6)


        assert_almost_equal(loss_1, loss_2)

        uids = [i.uid for i in il.learners]
        uids2 = [i.uid for i in il2.learners]
        for u in uids:
            self.assertTrue(u in uids2)


if __name__ == "__main__":
    unittest.main()
