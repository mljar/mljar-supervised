import unittest
import tempfile
import numpy as np
import pandas as pd
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical


class PreprocessingCategoricalOneHotTest(unittest.TestCase):
    def test_constructor_preprocessing_categorical(self):
        """
            Check if PreprocessingCategorical object is properly initialized
        """
        categorical = PreprocessingCategorical(PreprocessingCategorical.CONVERT_ONE_HOT)
        self.assertEqual(
            categorical._convert_method, PreprocessingCategorical.CONVERT_ONE_HOT
        )
        self.assertEqual(categorical._convert_params, {})

    def test_fit_one_hot(self):
        # training data
        d = {
            "col1": [1, 2, 3],
            "col2": ["a", "a", "c"],
            "col3": [1, 1, 3],
            "col4": ["a", "b", "c"],
        }
        df = pd.DataFrame(data=d)
        categorical = PreprocessingCategorical(PreprocessingCategorical.CONVERT_ONE_HOT)
        categorical.fit(df)
        self.assertTrue("col2" in categorical._convert_params)
        self.assertTrue("col4" in categorical._convert_params)

        for col in ["col2", "col4"]:
            self.assertTrue("unique_values" in categorical._convert_params[col])
            self.assertTrue("new_columns" in categorical._convert_params[col])

    def test_fit_transform_one_hot(self):
        # training data
        d = {
            "col1": [1, 2, 3],
            "col2": ["a", "a", "c"],
            "col3": [1, 1, 3],
            "col4": ["a", "b", "c"],
        }
        df = pd.DataFrame(data=d)
        categorical = PreprocessingCategorical(PreprocessingCategorical.CONVERT_ONE_HOT)
        categorical.fit(df)
        df = categorical.transform(df)
        for col in ["col1", "col2_c", "col3", "col4_a", "col4_b", "col4_c"]:
            self.assertTrue(col in df.columns)
        self.assertEqual(df["col2_c"][0], 0)
        self.assertEqual(df["col2_c"][1], 0)
        self.assertEqual(df["col2_c"][2], 1)
        self.assertEqual(df["col4_a"][0], 1)
        self.assertEqual(df["col4_a"][1], 0)
        self.assertEqual(df["col4_a"][2], 0)
        self.assertEqual(df["col4_b"][1], 1)
        self.assertEqual(df["col4_c"][2], 1)

    def test_fit_transform_one_hot_with_new_values(self):
        # training data
        d_train = {
            "col1": [1, 2, 3],
            "col2": ["a", "a", "c"],
            "col3": [1, 1, 3],
            "col4": ["a", "b", "c"],
        }
        df_train = pd.DataFrame(data=d_train)
        categorical = PreprocessingCategorical(PreprocessingCategorical.CONVERT_ONE_HOT)
        categorical.fit(df_train)
        # testing data
        d = {
            "col1": [1, 2, 3],
            "col2": ["a", "d", "f"],
            "col3": [1, 1, 3],
            "col4": ["e", "b", "z"],
        }
        df = pd.DataFrame(data=d)
        df = categorical.transform(df)
        for col in ["col1", "col2_c", "col3", "col4_a", "col4_b", "col4_c"]:
            self.assertTrue(col in df.columns)
        self.assertEqual(df["col2_c"][0], 0)
        self.assertEqual(df["col2_c"][1], 0)
        self.assertEqual(df["col2_c"][2], 0)
        self.assertEqual(np.sum(df["col4_a"]), 0)
        self.assertEqual(np.sum(df["col4_b"]), 1)
        self.assertEqual(np.sum(df["col4_c"]), 0)
        self.assertEqual(df["col4_b"][1], 1)

    def test_to_and_from_json_convert_one_hot(self):
        d = {
            "col1": [1, 2, 3],
            "col2": ["a", "a", "c"],
            "col3": [1, 1, 3],
            "col4": ["a", "b", "c"],
        }
        df = pd.DataFrame(data=d)
        cat1 = PreprocessingCategorical(PreprocessingCategorical.CONVERT_ONE_HOT)
        cat1.fit(df)

        cat2 = PreprocessingCategorical(PreprocessingCategorical.CONVERT_ONE_HOT)
        cat2.from_json(cat1.to_json())
        df = cat2.transform(df)
        for col in ["col1", "col2_c", "col3", "col4_a", "col4_b", "col4_c"]:
            self.assertTrue(col in df.columns)
        self.assertEqual(df["col2_c"][0], 0)
        self.assertEqual(df["col2_c"][1], 0)
        self.assertEqual(df["col2_c"][2], 1)
        self.assertEqual(df["col4_a"][0], 1)
        self.assertEqual(df["col4_a"][1], 0)
        self.assertEqual(df["col4_a"][2], 0)
        self.assertEqual(df["col4_b"][1], 1)
        self.assertEqual(df["col4_c"][2], 1)


if __name__ == "__main__":
    unittest.main()
