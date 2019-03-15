import unittest
import tempfile
import numpy as np
import pandas as pd
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
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
        self.assertTrue(preprocess_missing._na_fill_params is None)

    def test_get_fill_value(self):
        """
            Check if correct value is returned for filling in case of different
            column type and fill method
        """
        d = {"col1": [1, 2, 3, np.nan, np.nan], "col2": ["a", "a", np.nan, "b", "c"]}
        df = pd.DataFrame(data=d)
        # fill with median
        preprocess_missing = PreprocessingMissingValues(
            PreprocessingMissingValues.FILL_NA_MEDIAN
        )
        self.assertEqual(preprocess_missing._get_fill_value(df["col1"]), 2)
        self.assertEqual(preprocess_missing._get_fill_value(df["col2"]), "a")
        # fill with mean
        preprocess_missing = PreprocessingMissingValues(
            PreprocessingMissingValues.FILL_NA_MEDIAN
        )
        self.assertEqual(preprocess_missing._get_fill_value(df["col1"]), 2)
        self.assertEqual(preprocess_missing._get_fill_value(df["col2"]), "a")
        # fill with min
        preprocess_missing = PreprocessingMissingValues(
            PreprocessingMissingValues.FILL_NA_MIN
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

        for col in df.columns:
            # fill with median
            preprocess_missing = PreprocessingMissingValues(
                PreprocessingMissingValues.FILL_NA_MEDIAN
            )
            preprocess_missing._fit_na_fill(df[col])
            if col == "col1":
                self.assertEqual(2, preprocess_missing._na_fill_params)
            if col == "col2":
                self.assertEqual("a", preprocess_missing._na_fill_params)
            if col == "col3":
                self.assertTrue(preprocess_missing._na_fill_params is None)

        for col in df.columns:
            # fill with mean
            preprocess_missing = PreprocessingMissingValues(
                PreprocessingMissingValues.FILL_NA_MEAN
            )
            preprocess_missing._fit_na_fill(df[col])
            if col == "col1":
                self.assertEqual(2, preprocess_missing._na_fill_params)
            if col == "col2":
                self.assertEqual("a", preprocess_missing._na_fill_params)
            if col == "col3":
                self.assertTrue(preprocess_missing._na_fill_params is None)

        for col in df.columns:
            # fill with min
            preprocess_missing = PreprocessingMissingValues(
                PreprocessingMissingValues.FILL_NA_MIN
            )
            preprocess_missing._fit_na_fill(df[col])
            if col == "col1":
                self.assertEqual(0, preprocess_missing._na_fill_params)
            if col == "col2":
                self.assertEqual("_missing_value_", preprocess_missing._na_fill_params)
            if col == "col3":
                self.assertTrue(preprocess_missing._na_fill_params is None)

    def test_transform(self):
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

        for col in df.columns:
            # fill with median
            preprocess_missing = PreprocessingMissingValues(
                PreprocessingMissingValues.FILL_NA_MEDIAN
            )
            preprocess_missing.fit(df[col])
            df_transformed = preprocess_missing.transform(df_test.loc[:,col])
            if col == "col1":
                self.assertTrue(
                    np.isnan(df.loc[3, "col1"])
                )  # training data frame is not filled
                self.assertEqual(
                    2, df_test.loc[3, "col1"]
                )  # data frame is filled after transform
            if col == "col2":
                self.assertEqual("a", df_test.loc[2, "col2"])
            # columns without missing values in training set are also filled
            # but they are filled based on their own values
            if col == "col3":
                self.assertEqual(2, df_test.loc[3, "col3"])
            if col == "col4":
                self.assertEqual("b", df_test.loc[3, "col4"])

if __name__ == "__main__":
    unittest.main()
