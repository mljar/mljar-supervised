import unittest
import tempfile
import json
import numpy as np
import pandas as pd
from preprocessing.preprocessing_utils import PreprocessingUtils
from label_encoder import LabelEncoder


class LabelEncoderTest(unittest.TestCase):
    def test_fit(self):
        # training data
        d = {"col1": ["a", "a", "c"], "col2": ["w", "e", "d"]}
        df = pd.DataFrame(data=d)
        le = LabelEncoder()
        # check first column
        le.fit(df["col1"])
        data_json = le.to_json()
        # values from column should be in data json
        self.assertTrue("a" in data_json)
        self.assertTrue("c" in data_json)
        self.assertTrue("b" not in data_json)
        # there is alphabetical order for values
        self.assertEqual(0, data_json["a"])
        self.assertEqual(1, data_json["c"])

        # check next column
        le.fit(df["col2"])
        data_json = le.to_json()
        self.assertEqual(0, data_json["d"])
        self.assertEqual(1, data_json["e"])
        self.assertEqual(2, data_json["w"])

    def test_transform(self):
        # training data
        d = {"col1": ["a", "a", "c"]}
        df = pd.DataFrame(data=d)
        # fit encoder
        le = LabelEncoder()
        le.fit(df["col1"])
        # test data
        d_test = {"col2": ["c", "c", "a"]}
        df_test = pd.DataFrame(data=d_test)
        # transform
        y = le.transform(df_test["col2"])
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], 1)
        self.assertEqual(y[2], 0)

    def test_transform_with_new_values(self):
        # training data
        d = {"col1": ["a", "a", "c"]}
        df = pd.DataFrame(data=d)
        # fit encoder
        le = LabelEncoder()
        le.fit(df["col1"])
        # test data
        d_test = {"col2": ["c", "a", "d", "f"]}
        df_test = pd.DataFrame(data=d_test)
        # transform
        y = le.transform(df_test["col2"])
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], 0)
        self.assertEqual(y[2], 2)
        self.assertEqual(y[3], 3)

    def test_to_and_from_json(self):
        # training data
        d = {"col1": ["a", "a", "c"]}
        df = pd.DataFrame(data=d)
        # fit encoder
        le = LabelEncoder()
        le.fit(df["col1"])

        # new encoder
        new_le = LabelEncoder()
        new_le.from_json(le.to_json())

        # test data
        d_test = {"col2": ["c", "c", "a"]}
        df_test = pd.DataFrame(data=d_test)
        # transform
        y = new_le.transform(df_test["col2"])
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], 1)
        self.assertEqual(y[2], 0)

    def test_to_and_from_json_booleans(self):
        # training data
        d = {"col1": [True, False, True]}
        df = pd.DataFrame(data=d)
        # fit encoder
        le = LabelEncoder()
        le.fit(df["col1"])

        # new encoder
        new_le = LabelEncoder()
        new_le.from_json(json.loads(json.dumps(le.to_json(), indent=4)))

        # test data
        d_test = {"col2": [True, False, True]}
        df_test = pd.DataFrame(data=d_test)
        # transform
        y = new_le.transform(df_test["col2"])

        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], 0)
        self.assertEqual(y[2], 1)


if __name__ == "__main__":
    unittest.main()
