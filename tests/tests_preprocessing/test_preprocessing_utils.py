import unittest
import numpy as np
import pandas as pd
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.automl import AutoML
from supervised.exceptions import AutoMLException


class PreprocessingUtilsTest(unittest.TestCase):
    def test_get_type_numpy_number(self):
        tmp = np.array([1, 2, 3])
        tmp_type = PreprocessingUtils.get_type(tmp)
        self.assertNotEqual(tmp_type, PreprocessingUtils.CATEGORICAL)

    def test_get_type_numpy_categorical(self):
        tmp = np.array(["a", "b", "c"])
        tmp_type = PreprocessingUtils.get_type(tmp)
        self.assertEqual(tmp_type, PreprocessingUtils.CATEGORICAL)

    def test_get_type_pandas_bug(self):
        d = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        df = pd.DataFrame(data=d)
        col1_type = PreprocessingUtils.get_type(df.loc[:, "col2"])
        self.assertEqual(col1_type, PreprocessingUtils.CATEGORICAL)

    def test_get_type_pandas(self):
        d = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        df = pd.DataFrame(data=d)
        col1_type = PreprocessingUtils.get_type(df["col1"])
        self.assertNotEqual(col1_type, PreprocessingUtils.CATEGORICAL)
        col2_type = PreprocessingUtils.get_type(df["col2"])
        self.assertNotEqual(col1_type, PreprocessingUtils.CATEGORICAL)

    def test_get_stats(self):
        tmp = np.array([1, np.nan, 2, 3, np.nan, np.nan])
        self.assertEqual(1, PreprocessingUtils.get_min(tmp))
        self.assertEqual(2, PreprocessingUtils.get_mean(tmp))
        self.assertEqual(2, PreprocessingUtils.get_median(tmp))
        d = {"col1": [1, 2, 1, 3, 1, np.nan], "col2": ["a", np.nan, "b", "a", "c", "a"]}
        df = pd.DataFrame(data=d)
        self.assertEqual(1, PreprocessingUtils.get_min(df["col1"]))
        self.assertEqual(8.0 / 5.0, PreprocessingUtils.get_mean(df["col1"]))
        self.assertEqual(1, PreprocessingUtils.get_median(df["col1"]))

        self.assertEqual(1, PreprocessingUtils.get_most_frequent(df["col1"]))
        self.assertEqual("a", PreprocessingUtils.get_most_frequent(df["col2"]))


    def test_object_datatype_input(self):
        """ Checks an Exception is thrown  
            if X or y have the type of object."""
        obj_array = np.array([1, 2, "A"], dtype=object)
        y = pd.DataFrame(obj_array, columns=["target"])
        X = y.copy()

        a = AutoML(total_time=30,tuning_mode="Normal")

        with self.assertRaises(AutoMLException) as context:
            a.fit(X, y)
        self.assertTrue("object" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
