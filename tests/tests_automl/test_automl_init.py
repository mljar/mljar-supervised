import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil

from supervised import AutoML
from supervised.exceptions import AutoMLException


class AutoMLInitTest(unittest.TestCase):

    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_custom_init(self):

        X = np.random.uniform(size=(30, 2))
        y = np.random.randint(0, 2, size=(30,))

        automl = AutoML(
            results_path=self.automl_dir,
            model_time_limit=1,
            algorithms=["Xgboost"],
            explain_level=0,
            train_ensemble=False,
            stack_models=False,
            validation_strategy={"validation_type": "split"},
            start_random_models=3,
            hill_climbing_steps=1,
            top_models_to_improve=1,
        )

        automl.fit(X, y)
        self.assertGreater(len(automl._models), 4)
