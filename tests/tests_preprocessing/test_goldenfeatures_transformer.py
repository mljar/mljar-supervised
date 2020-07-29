import unittest
import tempfile
import json
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from sklearn import datasets
from supervised.preprocessing.goldenfeatures_transformer import (
    GoldenFeaturesTransformer,
)


class GoldenFeaturesTransformerTest(unittest.TestCase):
    def test_transformer(self):

        X, y = datasets.make_classification(
            n_samples=100,
            n_features=10,
            n_informative=6,
            n_redundant=1,
            n_classes=2,
            n_clusters_per_class=3,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        print(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            gft = GoldenFeaturesTransformer(tmpdir, "binary_classification")
            gft.fit(df, y)

            df = gft.transform(df)
            print(df)

            print(gft.to_json())

            gft3 = GoldenFeaturesTransformer(tmpdir, "binary_classification")
            gft3.from_json(gft.to_json())
