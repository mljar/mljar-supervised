import unittest
import tempfile
import json
import numpy as np
import pandas as pd

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
                "shuffle": True,
            },
            "learner": {
                "model_type": model_info["class"].algorithm_short_name,
                **model_params,
            },
        }

    def test_fit_and_predict(self):
        print(json.dumps(self.train_params, indent=4))
        early_stop = EarlyStopping({"metric": {"name": "logloss"}})
        metric_logger = MetricLogger({"metric_names": ["logloss", "auc"]})
        il = IterativeLearner(self.train_params, callbacks=[early_stop, metric_logger])
        il.train(self.data)


if __name__ == "__main__":
    unittest.main()
