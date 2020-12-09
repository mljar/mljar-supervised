import unittest
import tempfile
import numpy as np
import pandas as pd
from supervised.preprocessing.exclude_missing_target import ExcludeRowsMissingTarget


class ExcludeRowsMissingTargetTest(unittest.TestCase):
    def test_transform(self):
        d_test = {
            "col1": [1, 1, np.nan, 3],
            "col2": ["a", "a", np.nan, "a"],
            "col3": [1, 1, 1, 3],
            "col4": ["a", "a", "b", "c"],
            "y": [np.nan, 1, np.nan, 2],
        }
        df_test = pd.DataFrame(data=d_test)
        X = df_test.loc[:, ["col1", "col2", "col3", "col4"]]
        y = df_test.loc[:, "y"]

        self.assertEqual(X.shape[0], 4)
        self.assertEqual(y.shape[0], 4)
        X, y, _ = ExcludeRowsMissingTarget.transform(X, y)
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(y.shape[0], 2)
        
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], 2)

    def test_transform_with_sample_weight(self):
        d_test = {
            "col1": [1, 1, np.nan, 3],
            "col2": ["a", "a", np.nan, "a"],
            "col3": [1, 1, 1, 3],
            "col4": ["a", "a", "b", "c"],
            "sample_weight": [1,2,3,4],
            "y": [np.nan, 1, np.nan, 2],
        }
        df_test = pd.DataFrame(data=d_test)
        X = df_test.loc[:, ["col1", "col2", "col3", "col4"]]
        y = df_test.loc[:, "y"]
        sample_weight = df_test.loc[:, "sample_weight"]


        self.assertEqual(X.shape[0], 4)
        self.assertEqual(y.shape[0], 4)
        X, y, sw = ExcludeRowsMissingTarget.transform(X, y, sample_weight)
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(sw.shape[0], 2)

        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], 2)
        self.assertEqual(sw[0], 2)
        self.assertEqual(sw[1], 4)
        