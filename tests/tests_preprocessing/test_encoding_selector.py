import unittest
import tempfile
import numpy as np
import pandas as pd
from supervised.preprocessing.encoding_selector import EncodingSelector
from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical


class CategoricalIntegersTest(unittest.TestCase):
    def test_selector(self):

        d = {"col1": ["a", "a", "c"], "col2": ["a", "b", "c"]}
        df = pd.DataFrame(data=d)

        self.assertEqual(
            EncodingSelector.get(df, None, "col1"),
            PreprocessingCategorical.CONVERT_INTEGER,
        )
        self.assertEqual(
            EncodingSelector.get(df, None, "col2"),
            PreprocessingCategorical.CONVERT_ONE_HOT,
        )
