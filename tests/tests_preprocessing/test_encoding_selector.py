import unittest
import tempfile
import numpy as np
import pandas as pd
from supervised.preprocessing.encoding_selector import EncodingSelector
from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical


class CategoricalIntegersTest(unittest.TestCase):
    def test_selector(self):
        d = {"col1": [f"{i}" for i in range(31)], "col2": ["a"] * 31}
        df = pd.DataFrame(data=d)

        self.assertEqual(
            EncodingSelector.get(df, None, "col1"),
            PreprocessingCategorical.MANY_CATEGORIES,
        )
        self.assertEqual(
            EncodingSelector.get(df, None, "col2"),
            PreprocessingCategorical.FEW_CATEGORIES,
        )
