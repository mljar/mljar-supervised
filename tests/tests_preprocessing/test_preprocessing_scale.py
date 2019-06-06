import unittest
import tempfile
import json
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from supervised.preprocessing.preprocessing_scale import PreprocessingScale


class PreprocessingScaleTest(unittest.TestCase):

    def test_fit_log_and_normal(self):
        # training data
        d = {
            "col1": [12, 13, 3, 4, 5, 6, 7, 8000, 9000, 10000.0],
            "col2": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30.0],
            "col3": [12, 2, 3, 4, 5, 6, 7, 8000, 9000, 10000.0],
        }
        df = pd.DataFrame(data=d)

        scale = PreprocessingScale(["col1", "col3"], scale_method=PreprocessingScale.SCALE_LOG_AND_NORMAL)
        scale.fit(df)
        df = scale.transform(df)
        val = float(df["col1"][0])

        assert_almost_equal(np.mean(df["col1"]), 0)
        self.assertTrue(df["col1"][0] + 0.01 < df["col1"][1]) # in case of wrong scaling the small values will be squeezed

        df = scale.inverse_transform(df)

        scale2 = PreprocessingScale()
        scale_params = scale.to_json()

        scale2.from_json(scale_params)
        df = scale2.transform(df)
        assert_almost_equal(df["col1"][0], val)


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

        df = scale.inverse_transform(df)
        assert_almost_equal(df["col1"][0],1)
        assert_almost_equal(df["col1"][1],2)

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
        scale2 = PreprocessingScale()
        scale2.from_json(json_data)
        # transform with loaded scaler
        df = scale2.transform(df)
        assert_almost_equal(np.mean(df["col1"]), 0)
        assert_almost_equal(np.mean(df["col2"]), 25.5)


if __name__ == "__main__":
    unittest.main()
