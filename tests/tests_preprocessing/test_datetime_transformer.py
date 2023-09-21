import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from supervised.preprocessing.datetime_transformer import DateTimeTransformer


class DateTimeTransformerTest(unittest.TestCase):
    def test_transformer(self):
        d = {
            "col1": [
                "2020/06/01",
                "2020/06/02",
                "2020/06/03",
                "2021/06/01",
                "2022/06/01",
            ]
        }
        df = pd.DataFrame(data=d)
        df["col1"] = pd.to_datetime(df["col1"])
        df_org = df.copy()

        transf = DateTimeTransformer()
        transf.fit(df, "col1")
        df = transf.transform(df)

        self.assertTrue(df.shape[0] == 5)
        self.assertTrue("col1" not in df.columns)
        self.assertTrue("col1_Year" in df.columns)

        transf2 = DateTimeTransformer()
        transf2.from_json(transf.to_json())
        df2 = transf2.transform(df_org)
        self.assertTrue("col1" not in df2.columns)
        self.assertTrue("col1_Year" in df2.columns)
