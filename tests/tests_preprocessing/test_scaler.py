import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from supervised.preprocessing.scaler import Scaler


class ScalerTest(unittest.TestCase):
    def test_fit(self):
        # training data
        d = {"col1": [1,2,3,4,5,6,7,8,9,10], "col2": [21,22,23,24,25,26,27,28,29,30]}
        df = pd.DataFrame(data=d)

        scale = Scaler()
        scale.fit(df)
        print(scale.to_json())

    def test_to_and_from_json(self):
        pass


if __name__ == "__main__":
    unittest.main()
