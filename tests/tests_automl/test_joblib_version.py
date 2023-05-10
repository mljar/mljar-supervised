import unittest
import joblib
import numpy as np
import json
import shutil
import os

from supervised import AutoML


class TestJoblibVersion(unittest.TestCase):

    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_joblib_version(self):

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

        # Test equl version 
        jb_version = "1.2.0"
        joblib.__version__ = jb_version
        
        json_path = os.path.join(self.automl_dir, "1_Default_Xgboost", "framework.json") 

        with open(json_path) as file:
            frame = json.load(file)

        json_version = frame['joblib_version']
        expected_result = jb_version

        self.assertEqual(expected_result, json_version)


        # Test version 2.0.0
        jb_version = "0.0.1"
        joblib.__version__ = jb_version

        with open(json_path) as file:
            frame = json.load(file)

        json_version = frame['joblib_version']
        expected_result = jb_version

        self.assertNotEqual(expected_result, json_version)


        # Test version None
        
        joblib.__version__ = None
        expected_result = jb_version

        with open(json_path) as file:
            frame = json.load(file)

        json_version = frame['joblib_version']

        self.assertNotEqual(expected_result, json_version)


if __name__ == '__main__':
    unittest.main()