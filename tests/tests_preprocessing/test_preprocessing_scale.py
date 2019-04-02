import unittest
import tempfile
import json
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from supervised.preprocessing.preprocessing_scale import PreprocessingScale


class PreprocessingScaleTest(unittest.TestCase):
    def test_fit(self):
        # training data
        d = {
            "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0],
            "col2": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30.0],
        }
        df = pd.DataFrame(data=d)

        scale = PreprocessingScale(["col1"])
        scale.fit(df)
        df = scale.transform(df)

        assert_almost_equal(np.mean(df["col1"]), 0)
        assert_almost_equal(np.mean(df["col2"]), 25.5)

    def test_to_and_from_json(self):
        # training data
        d = {
            "col1": [1, 2, 3, 4, 5, 6, 7, 8.0, 9, 10],
            "col2": [21, 22.0, 23, 24, 25, 26, 27, 28, 29, 30],
        }
        df = pd.DataFrame(data=d)

        scale = PreprocessingScale(["col1"])
        scale.fit(df)
        # do not transform
        assert_almost_equal(np.mean(df["col1"]), 5.5)
        assert_almost_equal(np.mean(df["col2"]), 25.5)
        # to and from json

        json_data = scale.to_json()
        print(json_data)
        scale2 = PreprocessingScale()
        scale2.from_json(json_data)
        # transform with loaded scaler
        df = scale2.transform(df)
        assert_almost_equal(np.mean(df["col1"]), 0)
        assert_almost_equal(np.mean(df["col2"]), 25.5)


if __name__ == "__main__":
    unittest.main()
