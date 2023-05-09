import unittest
import shutil
import json
import numpy as np
import joblib
from supervised import AutoML


class TestModelFramework(unittest.TestCase):

    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_joblib_versions(self):
        X = np.random.uniform(size=(10, 2))
        y = np.random.randint(0, 2, size=(10,))

        versions_to_test = [
            f"{joblib.__version__}",  # Test equal version
            "0.0.1",  # Test version 0.0.1
        ]

        for jb_version in versions_to_test:
            with self.subTest(jb_version):
                joblib.__version__ = jb_version

                automl = AutoML(
                    results_path=self.automl_dir,
                    model_time_limit=5,
                    algorithms=["Baseline"],
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

                json_path = f"./{self.automl_dir}/1_Baseline/framework.json"

                with open(json_path) as file:
                    frame = json.load(file)

                json_version_frame = frame['joblib_version']
                expected_result = jb_version


                if jb_version == "0.0.1":
                    self.assertNotEqual(expected_result, json_version_frame)
                elif jb_version == joblib.__version__:
                    self.assertEqual(expected_result, json_version_frame)


if __name__ == '__main__':
    unittest.main()
