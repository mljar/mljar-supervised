import unittest
import tempfile
import numpy as np
import pandas as pd
from supervised.preprocessing.exclude_missing_target import (
    ExcludeRowsMissingTarget,
)


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
        X, y = ExcludeRowsMissingTarget.transform(X, y)
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], 2)

