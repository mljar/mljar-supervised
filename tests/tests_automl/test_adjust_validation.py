import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil

from supervised import AutoML
from supervised.exceptions import AutoMLException


class AutoMLAdjustValidationTest(unittest.TestCase):
    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_custom_init(self):
        X = np.random.uniform(size=(60, 2))
        y = np.random.randint(0, 2, size=(60,))

        automl = AutoML(
            results_path=self.automl_dir,
            model_time_limit=10,
            algorithms=["Xgboost"],
            mode="Compete",
            explain_level=0,
            start_random_models=1,
            hill_climbing_steps=0,
            top_models_to_improve=0,
            kmeans_features=False,
            golden_features=False,
            features_selection=False,
            boost_on_errors=False,
        )
        automl.fit(X, y)

        self.assertFalse(
            os.path.exists(os.path.join(self.automl_dir, "1_DecisionTree"))
        )
