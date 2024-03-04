import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from supervised.preprocessing.loo_encoder import LooEncoder


class LabelEncoderTest(unittest.TestCase):
    def test_fit(self):
        # training data
        d = {"col1": ["a", "a", "c"], "col2": ["w", "e", "d"], "y": [1, 2, 0]}
        df = pd.DataFrame(data=d)
        le = LooEncoder(cols=["col1"])
        le.fit(df[["col1", "col2"]], df["y"])

        self.assertTrue(le.enc is not None)
        self.assertTrue(le.enc._dim == 2)
        assert_almost_equal(le.enc._mean, 1.0)
        self.assertTrue("col1" in le.enc.mapping)
        self.assertTrue("col2" not in le.enc.mapping)

    def test_transform(self):
        # training data
        d = {"col1": ["a", "a", "c"]}
        y = [1, 1, 0]
        df = pd.DataFrame(data=d)
        # fit encoder
        le = LooEncoder(cols=["col1"])
        le.fit(df, y)
        t1 = le.transform(df)

        # test data
        d_test = {"col1": ["c", "c", "a"]}
        df_test = pd.DataFrame(data=d_test)
        # transform
        t2 = le.transform(df_test)
        assert_almost_equal(t1["col1"][0], t2["col1"][2])
        assert_almost_equal(t1["col1"][2], t2["col1"][1])

    def test_transform_with_new_and_missing_values(self):
        # training data
        d = {"col1": ["a", "a", "c"]}
        y = [1, 1, 1]
        df = pd.DataFrame(data=d)
        # fit encoder
        le = LooEncoder(cols=["col1"])
        le.fit(df, y)
        # test data
        d_test = {"col1": ["c", "a", "d", "f", np.nan]}
        df_test = pd.DataFrame(data=d_test)
        # transform
        t = le.transform(df_test)
        assert_almost_equal(t["col1"][2], 1)
        assert_almost_equal(t["col1"][3], 1)
        assert_almost_equal(t["col1"][4], 1)

    def test_to_and_from_json(self):
        # training data
        d = {"col1": ["a", "a", "c"]}
        y = [1, 1, 1]
        df = pd.DataFrame(data=d)
        # fit encoder
        le = LooEncoder()
        le.fit(df, y)

        # new encoder
        new_le = LooEncoder()
        new_le.from_json(le.to_json())

        # test data
        d_test = {"col1": ["c", "c", "a", "e"]}
        df_test = pd.DataFrame(data=d_test)
        # transform
        t = new_le.transform(df_test)
        self.assertEqual(t["col1"][0], 1)
        self.assertEqual(t["col1"][1], 1)
        self.assertEqual(t["col1"][2], 1)
        self.assertEqual(t["col1"][3], 1)
