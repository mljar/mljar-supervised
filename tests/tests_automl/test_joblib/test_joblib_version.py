import unittest
from supervised.model_framework import ModelFramework
import joblib
import os



class TestModelFramework(unittest.TestCase):
    
    results_path = "tests/tests_algorithms/test_joblib"
    model_subpath = "joblib_testing"

#   create framework.json from model_framework



    def test_load_joblib_version(self, results_path, model_subpath):
        # test version
        mock_version = "1.0.0"
        joblib.__version__ = mock_version
        model_path = os.path.join(self.results_path, self.model_subpath)

        # Check if true
        expected_result = mock_version
        actual_result = ModelFramework.load(self.results_path, self.model_subpath)
        self.assertEqual(expected_result, actual_result)

        # Check if false 
        mock_version = "2.0.0"
        joblib.__version__ = mock_version
        expected_result = "Different version"
        actual_result = ModelFramework.load(self.results_path, self.model_subpath)
        self.assertEqual(expected_result, actual_result)

        # check if none
        joblib.__version__ = None
        expected_result = "No version found"
        actual_result = ModelFramework.load(self.results_path, self.model_subpath)
        self.assertEqual(expected_result, actual_result)

if __name__ == '__main__':
    unittest.main()