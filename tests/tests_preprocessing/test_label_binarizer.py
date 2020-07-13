import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from supervised.preprocessing.label_binarizer import LabelBinarizer


class LabelBinarizerTest(unittest.TestCase):
    def test_fit(self):
        # training data
        d = {"col1": ["a", "a", "c"], "col2": ["w", "e", "d"]}
        df = pd.DataFrame(data=d)
        lb = LabelBinarizer()
        # check first column
        lb.fit(df, "col1")
        data_json = lb.to_json()
        self.assertTrue("new_columns" in data_json)
        # we take alphabetical order
        self.assertTrue("col1_c" in data_json["new_columns"])
        self.assertTrue("col1_a" not in data_json["new_columns"])
        self.assertTrue("unique_values" in data_json)
        self.assertTrue("a" in data_json["unique_values"])
        self.assertTrue("c" in data_json["unique_values"])

        lb = LabelBinarizer()
        # check second column
        lb.fit(df, "col2")
        data_json = lb.to_json()
        self.assertTrue("new_columns" in data_json)
        self.assertTrue("col2_w" in data_json["new_columns"])
        self.assertTrue("col2_e" in data_json["new_columns"])
        self.assertTrue("col2_d" in data_json["new_columns"])
        self.assertTrue("unique_values" in data_json)
        self.assertTrue("w" in data_json["unique_values"])
        self.assertTrue("e" in data_json["unique_values"])
        self.assertTrue("d" in data_json["unique_values"])

    def test_transform(self):
        # training data
        d = {"col1": ["a", "a", "c"], "col2": ["w", "e", "d"]}
        df = pd.DataFrame(data=d)
        # fit binarizer
        lb1 = LabelBinarizer()
        lb1.fit(df, "col1")
        lb2 = LabelBinarizer()
        lb2.fit(df, "col2")
        # test data
        d_test = {"col1": ["c", "c", "a"], "col2": ["e", "d", "w"], "col3": [2, 3, 4]}
        df_test = pd.DataFrame(data=d_test)
        # transform
        df_test = lb1.transform(df_test, "col1")
        df_test = lb2.transform(df_test, "col2")
        # for binary column, only one value is left, old column should be deleted
        self.assertTrue("col1_c" in df_test.columns)
        self.assertTrue("col1" not in df_test.columns)
        self.assertEqual(2, np.sum(df_test["col1_c"]))
        # for multiple value colum, all columns should be added
        self.assertTrue("col2_w" in df_test.columns)
        self.assertTrue("col2_e" in df_test.columns)
        self.assertTrue("col2_d" in df_test.columns)
        self.assertTrue("col2" not in df_test.columns)
        self.assertEqual(1, np.sum(df_test["col2_w"]))
        self.assertEqual(1, np.sum(df_test["col2_e"]))
        self.assertEqual(1, np.sum(df_test["col2_d"]))
        # do not touch continuous attribute
        self.assertTrue("col3" in df_test.columns)

    def test_transform_with_new_values(self):
        # training data
        d = {"col1": ["a", "a", "c"], "col2": ["w", "e", "d"]}
        df = pd.DataFrame(data=d)
        # fit binarizer
        lb1 = LabelBinarizer()
        lb1.fit(df, "col1")
        lb2 = LabelBinarizer()
        lb2.fit(df, "col2")
        # test data
        d_test = {"col1": ["c", "d", "d"], "col2": ["g", "e", "f"], "col3": [2, 3, 4]}
        df_test = pd.DataFrame(data=d_test)
        # transform
        df_test = lb1.transform(df_test, "col1")
        df_test = lb2.transform(df_test, "col2")
        self.assertTrue("col1_c" in df_test.columns)
        self.assertTrue("col1_d" not in df_test.columns)
        self.assertTrue("col2_w" in df_test.columns)
        self.assertTrue("col2_e" in df_test.columns)
        self.assertTrue("col2_d" in df_test.columns)
        self.assertTrue("col2_g" not in df_test.columns)
        self.assertTrue("col2_f" not in df_test.columns)
        self.assertEqual(df_test["col1_c"][0], 1)
        self.assertEqual(df_test["col1_c"][1], 0)
        self.assertEqual(df_test["col1_c"][2], 0)
        self.assertEqual(np.sum(df_test["col2_w"]), 0)
        self.assertEqual(np.sum(df_test["col2_d"]), 0)
        self.assertEqual(df_test["col2_e"][0], 0)
        self.assertEqual(df_test["col2_e"][1], 1)
        self.assertEqual(df_test["col2_e"][2], 0)

    def test_to_and_from_json(self):
        # training data
        d = {"col1": ["a", "a", "c"], "col2": ["w", "e", "d"]}
        df = pd.DataFrame(data=d)
        # fit binarizer
        lb1 = LabelBinarizer()
        lb1.fit(df, "col1")
        lb2 = LabelBinarizer()
        lb2.fit(df, "col2")
        # test data
        d_test = {"col1": ["c", "c", "a"], "col2": ["e", "d", "w"], "col3": [2, 3, 4]}
        df_test = pd.DataFrame(data=d_test)
        # to json and from json
        new_lb1 = LabelBinarizer()
        new_lb2 = LabelBinarizer()
        new_lb1.from_json(lb1.to_json())
        new_lb2.from_json(lb2.to_json())
        # transform
        df_test = new_lb1.transform(df_test, "col1")
        df_test = new_lb2.transform(df_test, "col2")
        # for binary column, only one value is left, old column should be deleted
        self.assertTrue("col1_c" in df_test.columns)
        self.assertTrue("col1" not in df_test.columns)
        self.assertEqual(2, np.sum(df_test["col1_c"]))
        # for multiple value colum, all columns should be added
        self.assertTrue("col2_w" in df_test.columns)
        self.assertTrue("col2_e" in df_test.columns)
        self.assertTrue("col2_d" in df_test.columns)
        self.assertTrue("col2" not in df_test.columns)
        self.assertEqual(1, np.sum(df_test["col2_w"]))
        self.assertEqual(1, np.sum(df_test["col2_e"]))
        self.assertEqual(1, np.sum(df_test["col2_d"]))
        # do not touch continuous attribute
        self.assertTrue("col3" in df_test.columns)

    def test_to_and_from_json_booleans(self):
        # training data
        d = {"col1": ["a", "a", "c"], "col2": [True, True, False]}
        df = pd.DataFrame(data=d)
        # fit binarizer
        lb1 = LabelBinarizer()
        lb1.fit(df, "col1")
        lb2 = LabelBinarizer()
        lb2.fit(df, "col2")
        # test data
        d_test = {
            "col1": ["c", "c", "a"],
            "col2": [False, False, True],
            "col3": [2, 3, 4],
        }
        df_test = pd.DataFrame(data=d_test)
        # to json and from json
        new_lb1 = LabelBinarizer()
        new_lb2 = LabelBinarizer()
        new_lb1.from_json(lb1.to_json())
        new_lb2.from_json(json.loads(json.dumps(lb2.to_json(), indent=4)))

        # transform
        df_test = new_lb1.transform(df_test, "col1")
        df_test = new_lb2.transform(df_test, "col2")
        # for binary column, only one value is left, old column should be deleted
        self.assertTrue("col1_c" in df_test.columns)
        self.assertTrue("col1" not in df_test.columns)
        self.assertEqual(2, np.sum(df_test["col1_c"]))
        # for multiple value colum, all columns should be added
        self.assertTrue("col2_True" in df_test.columns)
        self.assertTrue("col2" not in df_test.columns)
        self.assertEqual(1, np.sum(df_test["col2_True"]))
        # do not touch continuous attribute
        self.assertTrue("col3" in df_test.columns)

    def test_inverse_transform(self):
        d = {"col1": ["a", "a", "c"], "col2": ["w", "e", "d"]}
        df = pd.DataFrame(data=d)
        lb = LabelBinarizer()
        # check first column
        lb.fit(df, "col1")
        bb = lb.transform(df, "col1")
        self.assertTrue("col1_c" in bb.columns)
        self.assertTrue(np.sum(bb["col1_c"]) == 1)
        bb = lb.inverse_transform(bb)
        self.assertTrue("col1_c" not in bb.columns)
        # check second column
        lb = LabelBinarizer()
        lb.fit(df, "col2")
        bb = lb.transform(df, "col2")
        self.assertTrue("col2_w" in bb.columns)
        self.assertTrue("col2_e" in bb.columns)
        self.assertTrue("col2_d" in bb.columns)
        self.assertTrue(np.sum(bb["col2_w"]) == 1)
        bb = lb.inverse_transform(bb)
        self.assertTrue("col2_w" not in bb.columns)


if __name__ == "__main__":
    unittest.main()
