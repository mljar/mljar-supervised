import unittest

import pandas as pd
from numpy.testing import assert_almost_equal

from supervised.preprocessing.text_transformer import TextTransformer


class TextTransformerTest(unittest.TestCase):
    def test_transformer(self):
        d = {
            "col1": [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                None,
                "Is this the first document?",
            ]
        }
        df = pd.DataFrame(data=d)
        df_org = df.copy()

        transf = TextTransformer()
        transf.fit(df, "col1")
        df = transf.transform(df)
        self.assertTrue(df.shape[0] == 5)
        self.assertTrue("col1" not in df.columns)

        transf2 = TextTransformer()
        transf2.from_json(transf.to_json())
        df2 = transf2.transform(df_org)
        self.assertTrue("col1" not in df2.columns)

        assert_almost_equal(df.iloc[0, 0], df2.iloc[0, 0])
