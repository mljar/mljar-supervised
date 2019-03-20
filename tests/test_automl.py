import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from sklearn import datasets
from supervised.automl import AutoML


class AutoMLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = datasets.make_classification(
            n_samples=200,
            n_features=5,
            n_informative=5,
            n_redundant=0,
            n_classes=2,
            n_clusters_per_class=1,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )
        cls.X = pd.DataFrame(cls.X, columns=["f0", "f1", "f2", "f3", "f4"])
        cls.y = pd.DataFrame(cls.y)

    def test_fit_and_predict(self):
        automl = AutoML()
        automl.fit(self.X, self.y)
        y_predicted = automl.predict(self.y)
        print(y_predicted)
        #self.assertTrue(y_predicted is not None)

if __name__ == "__main__":
    unittest.main()
