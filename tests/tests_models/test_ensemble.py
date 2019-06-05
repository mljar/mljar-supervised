import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.models.ensemble import Ensemble
from supervised.metric import Metric
from supervised.models.learner_factory import LearnerFactory


class SimpleFramework:
    def __init__(self, params):
        pass

    def predict(self, X):
        y = np.array([0.1, 0.2, 0.8, 0.9])
        return pd.DataFrame({"p_0": 1 - y, "p_1": y})

    def to_json(self):
        return {
            "params": {
                "model_type": "simple",
                "learner": {"model_type": "simple"},
                "validation": {
                    "validation_type": "kfold",
                    "k_folds": 5,
                    "shuffle": True,
                },
            }
        }

    def from_json(self, json_desc):
        pass

    def load(self, json_desc):
        pass


class EnsembleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.X = pd.DataFrame(
            {
                "model_0": [0.1, 0.2, 0.8, 0.9],
                "model_1": [0.2, 0.1, 0.9, 0.8],
                "model_2": [0.8, 0.8, 0.1, 0.1],
                "model_3": [0.8, 0.8, 0.1, 0.1],
                "model_4": [0.8, 0.8, 0.1, 0.1],
                "model_5": [0.8, 0.8, 0.1, 0.1],
            }
        )
        cls.y = np.array([0, 0, 1, 1])

    def test_fit_predict(self):
        ensemble = Ensemble()
        ensemble.models = [SimpleFramework({})] * 5
        ensemble.fit(self.X, self.y)
        self.assertEqual(1, ensemble.selected_models[1]["repeat"])
        self.assertEqual(1, ensemble.selected_models[1]["repeat"])
        self.assertTrue(len(ensemble.selected_models) == 2)
        y = ensemble.predict(self.X)

        assert_almost_equal(y["p_1"][0], 0.1)
        assert_almost_equal(y["p_1"][1], 0.2)
        assert_almost_equal(y["p_1"][2], 0.8)
        assert_almost_equal(y["p_1"][3], 0.9)

    """
    def test_save_load(self):

        ensemble = Ensemble()
        ensemble.models = [SimpleFramework({})] * 5
        ensemble.fit(self.X, self.y)
        y = ensemble.predict(self.X)
        assert_almost_equal(y[0], 0.1)
        ensemble_json = ensemble.to_json()
        ensemble2 = Ensemble()
        ensemble2.from_json(ensemble_json)
        y2 = ensemble2.predict(self.X)
        assert_almost_equal(y2[0], 0.1)
    """


if __name__ == "__main__":
    unittest.main()
