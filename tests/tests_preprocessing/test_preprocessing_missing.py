import unittest

import numpy as np
import pandas as pd

from supervised.preprocessing.preprocessing_missing import PreprocessingMissingValues


class PreprocessingMissingValuesTest(unittest.TestCase):
    def test_preprocessing_constructor(self):
        """
        Check if PreprocessingMissingValues object is properly initialized
        """
        preprocess_missing = PreprocessingMissingValues(
            PreprocessingMissingValues.FILL_NA_MEDIAN
        )
        self.assertEqual(
            preprocess_missing._na_fill_method,
            PreprocessingMissingValues.FILL_NA_MEDIAN,
        )
        self.assertEqual(preprocess_missing._na_fill_params, {})

    def test_get_fill_value(self):
        """
        Check if correct value is returned for filling in case of different
        column type and fill method
        """
        d = {"col1": [1, 2, 3, np.nan, np.nan], "col2": ["a", "a", np.nan, "b", "c"]}
        df = pd.DataFrame(data=d)
        # fill with median
        preprocess_missing = PreprocessingMissingValues(
            df.columns, PreprocessingMissingValues.FILL_NA_MEDIAN
        )
        self.assertEqual(preprocess_missing._get_fill_value(df["col1"]), 2)
        self.assertEqual(preprocess_missing._get_fill_value(df["col2"]), "a")
        # fill with mean
        preprocess_missing = PreprocessingMissingValues(
            df.columns, PreprocessingMissingValues.FILL_NA_MEDIAN
        )
        self.assertEqual(preprocess_missing._get_fill_value(df["col1"]), 2)
        self.assertEqual(preprocess_missing._get_fill_value(df["col2"]), "a")
        # fill with min
        preprocess_missing = PreprocessingMissingValues(
            df.columns, PreprocessingMissingValues.FILL_NA_MIN
        )
        self.assertEqual(preprocess_missing._get_fill_value(df["col1"]), 0)
        self.assertEqual(
            preprocess_missing._get_fill_value(df["col2"]), "_missing_value_"
        )  # added new value

    def test_fit_na_fill(self):
        """
        Check fit private method
        """
        d = {
            "col1": [1, 2, 3, np.nan, np.nan],
            "col2": ["a", "a", np.nan, "b", "c"],
            "col3": ["a", "a", "d", "b", "c"],
        }
        df = pd.DataFrame(data=d)
        # fill with median
        preprocess_missing = PreprocessingMissingValues(
            df.columns, PreprocessingMissingValues.FILL_NA_MEDIAN
        )
        preprocess_missing._fit_na_fill(df)
        self.assertTrue("col1" in preprocess_missing._na_fill_params)
        self.assertTrue("col2" in preprocess_missing._na_fill_params)
        self.assertTrue("col3" not in preprocess_missing._na_fill_params)
        self.assertEqual(2, preprocess_missing._na_fill_params["col1"])
        self.assertEqual("a", preprocess_missing._na_fill_params["col2"])
        # fill with mean
        preprocess_missing = PreprocessingMissingValues(
            df.columns, PreprocessingMissingValues.FILL_NA_MEAN
        )
        preprocess_missing._fit_na_fill(df)
        self.assertTrue("col1" in preprocess_missing._na_fill_params)
        self.assertTrue("col2" in preprocess_missing._na_fill_params)
        self.assertTrue("col3" not in preprocess_missing._na_fill_params)
        self.assertEqual(2, preprocess_missing._na_fill_params["col1"])
        self.assertEqual("a", preprocess_missing._na_fill_params["col2"])
        # fill with min
        preprocess_missing = PreprocessingMissingValues(
            df.columns, PreprocessingMissingValues.FILL_NA_MIN
        )
        preprocess_missing._fit_na_fill(df)
        self.assertTrue("col1" in preprocess_missing._na_fill_params)
        self.assertTrue("col2" in preprocess_missing._na_fill_params)
        self.assertTrue("col3" not in preprocess_missing._na_fill_params)
        self.assertEqual(0, preprocess_missing._na_fill_params["col1"])
        self.assertEqual("_missing_value_", preprocess_missing._na_fill_params["col2"])

    def test_transform(self):
        """
        Check transform
        """
        # training data
        d = {
            "col1": [1, 2, 3, np.nan, np.nan],
            "col2": ["a", "a", np.nan, "a", "c"],
            "col3": [1, 1, 3, 1, 1],
            "col4": ["a", "a", "a", "c", "a"],
        }
        df = pd.DataFrame(data=d)
        # test data
        d_test = {
            "col1": [1, 2, 3, np.nan, np.nan],
            "col2": ["b", "b", np.nan, "b", "c"],
            "col3": [1, 2, 2, np.nan, 2],
            "col4": ["b", "b", np.nan, "b", "c"],
        }
        df_test = pd.DataFrame(data=d_test)
        # fill with median
        preprocess_missing = PreprocessingMissingValues(
            df.columns, PreprocessingMissingValues.FILL_NA_MEDIAN
        )
        preprocess_missing.fit(df)
        self.assertEqual(
            2, len(preprocess_missing._na_fill_params)
        )  # there should be only two columns
        df_transformed = preprocess_missing.transform(df_test)
        self.assertTrue(
            np.isnan(df.loc[3, "col1"])
        )  # training data frame is not filled
        self.assertEqual(
            2, df_test.loc[3, "col1"]
        )  # data frame is filled after transform
        self.assertEqual("a", df_test.loc[2, "col2"])

        # it is disabled, should be treated separately at the end of preprocessing
        # columns without missing values in training set are also filled
        # but they are filled based on their own values
        # self.assertEqual(2, df_test.loc[3, "col3"])
        # self.assertEqual("b", df_test.loc[3, "col4"])

    def test_transform_on_new_data(self):
        # training data
        d = {
            "col1": [1, 1, np.nan, 3],
            "col2": ["a", "a", np.nan, "a"],
            "col3": [1, 1, 1, 3],
            "col4": ["a", "a", "b", "c"],
            "y": [0, 1, 1, 1],
        }
        df = pd.DataFrame(data=d)
        X_train = df.loc[:, ["col1", "col2", "col3", "col4"]]
        y_train = df.loc[:, "y"]

        d_test = {
            "col1": [1, 1, np.nan, 3],
            "col2": ["a", "a", np.nan, "a"],
            "col3": [1, 1, 1, 3],
            "col4": ["a", "a", "b", "c"],
            "y": [np.nan, 1, np.nan, 1],
        }
        df_test = pd.DataFrame(data=d_test)
        X_test = df_test.loc[:, ["col1", "col2", "col3", "col4"]]
        y_test = df_test.loc[:, "y"]

        pm = PreprocessingMissingValues(
            X_train.columns, PreprocessingMissingValues.FILL_NA_MEDIAN
        )
        pm.fit(X_train)
        X_train = pm.transform(X_train)
        X_test = pm.transform(X_test)

        self.assertEqual(1, X_test.loc[2, "col1"])
        self.assertEqual("a", X_test.loc[2, "col2"])


if __name__ == "__main__":
    unittest.main()
