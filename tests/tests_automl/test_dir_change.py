import os
import shutil
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised import AutoML


class AutoMLDirChangeTest(unittest.TestCase):
    automl_dir_a = "automl_testing_A"
    automl_dir_b = "automl_testing_B"
    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir_a, ignore_errors=True)
        shutil.rmtree(self.automl_dir_b, ignore_errors=True)

    def create_dir(self, dir_path):
        if not os.path.exists(dir_path):
            try:
                os.mkdir(dir_path)
            except Exception as e:
                pass

    def test_create_report_after_dir_change(self):
        #
        # test for https://github.com/mljar/mljar-supervised/issues/384
        #
        self.create_dir(self.automl_dir_a)
        self.create_dir(self.automl_dir_b)

        path_a = os.path.join(self.automl_dir_a, self.automl_dir)
        path_b = os.path.join(self.automl_dir_b, self.automl_dir)

        X = np.random.uniform(size=(30, 2))
        y = np.random.randint(0, 2, size=(30,))

        automl = AutoML(results_path=path_a, algorithms=["Baseline"], explain_level=0)
        automl.fit(X, y)

        shutil.move(path_a, path_b)

        automl2 = AutoML(
            results_path=path_b,
        )
        automl2.report()

    def test_compute_predictions_after_dir_change(self):
        #
        # test for https://github.com/mljar/mljar-supervised/issues/384
        #
        self.create_dir(self.automl_dir_a)
        self.create_dir(self.automl_dir_b)

        path_a = os.path.join(self.automl_dir_a, self.automl_dir)
        path_b = os.path.join(self.automl_dir_b, self.automl_dir)

        X, y = datasets.make_regression(
            n_samples=100,
            n_features=5,
            n_informative=4,
            n_targets=1,
            shuffle=False,
            random_state=0,
        )

        automl = AutoML(
            results_path=path_a,
            explain_level=0,
            ml_task="regression",
            total_time_limit=10,
        )
        automl.fit(X, y)
        p = automl.predict(X[:3])

        shutil.move(path_a, path_b)

        automl2 = AutoML(
            results_path=path_b,
        )
        p2 = automl2.predict(X[:3])

        for i in range(3):
            assert_almost_equal(p[i], p2[i])
