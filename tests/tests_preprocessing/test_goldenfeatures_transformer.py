import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn import datasets

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.preprocessing.goldenfeatures_transformer import (
    GoldenFeaturesTransformer,
)


class GoldenFeaturesTransformerTest(unittest.TestCase):
    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

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

        with tempfile.TemporaryDirectory() as tmpdir:
            gft = GoldenFeaturesTransformer(tmpdir, "binary_classification")
            gft.fit(df, y)

            df = gft.transform(df)

            gft3 = GoldenFeaturesTransformer(tmpdir, "binary_classification")
            gft3.from_json(gft.to_json(), tmpdir)

    def test_subsample_regression_10k(self):
        rows = 10000
        X = np.random.rand(rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = pd.Series(np.random.rand(rows), name="target")

        gft3 = GoldenFeaturesTransformer(self.automl_dir, REGRESSION)
        X_train, X_test, y_train, y_test = gft3._subsample(X, y)

        self.assertTrue(X_train.shape[0], 2500)
        self.assertTrue(X_test.shape[0], 2500)
        self.assertTrue(y_train.shape[0], 2500)
        self.assertTrue(y_test.shape[0], 2500)

    def test_subsample_regression_4k(self):
        rows = 4000
        X = np.random.rand(rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = pd.Series(np.random.rand(rows), name="target")

        gft3 = GoldenFeaturesTransformer(self.automl_dir, REGRESSION)
        X_train, X_test, y_train, y_test = gft3._subsample(X, y)

        self.assertTrue(X_train.shape[0], 2000)
        self.assertTrue(X_test.shape[0], 2000)
        self.assertTrue(y_train.shape[0], 2000)
        self.assertTrue(y_test.shape[0], 2000)

    def test_subsample_multiclass_10k(self):
        rows = 10000
        X = np.random.rand(rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = pd.Series(np.random.randint(0, 4, rows), name="target")

        gft3 = GoldenFeaturesTransformer(self.automl_dir, MULTICLASS_CLASSIFICATION)
        X_train, X_test, y_train, y_test = gft3._subsample(X, y)

        self.assertTrue(X_train.shape[0], 2500)
        self.assertTrue(X_test.shape[0], 2500)
        self.assertTrue(y_train.shape[0], 2500)
        self.assertTrue(y_test.shape[0], 2500)

        for uni in [np.unique(y_train), np.unique(y_test)]:
            for i in range(4):
                self.assertTrue(i in uni)

    def test_subsample_multiclass_4k(self):
        rows = 4000
        X = np.random.rand(rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = pd.Series(np.random.randint(0, 4, rows), name="target")

        gft3 = GoldenFeaturesTransformer(self.automl_dir, MULTICLASS_CLASSIFICATION)
        X_train, X_test, y_train, y_test = gft3._subsample(X, y)

        self.assertTrue(X_train.shape[0], 2000)
        self.assertTrue(X_test.shape[0], 2000)
        self.assertTrue(y_train.shape[0], 2000)
        self.assertTrue(y_test.shape[0], 2000)

        for uni in [np.unique(y_train), np.unique(y_test)]:
            for i in range(4):
                self.assertTrue(i in uni)

    def test_subsample_binclass_4k(self):
        rows = 4000
        X = np.random.rand(rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = pd.Series(np.random.randint(0, 2, rows), name="target")

        gft3 = GoldenFeaturesTransformer(self.automl_dir, BINARY_CLASSIFICATION)
        X_train, X_test, y_train, y_test = gft3._subsample(X, y)

        self.assertTrue(X_train.shape[0], 2000)
        self.assertTrue(X_test.shape[0], 2000)
        self.assertTrue(y_train.shape[0], 2000)
        self.assertTrue(y_test.shape[0], 2000)

        for uni in [np.unique(y_train), np.unique(y_test)]:
            for i in range(2):
                self.assertTrue(i in uni)

    def test_features_count(self):
        N_COLS = 10
        X, y = datasets.make_classification(
            n_samples=100,
            n_features=N_COLS,
            n_informative=6,
            n_redundant=1,
            n_classes=2,
            n_clusters_per_class=3,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        with tempfile.TemporaryDirectory() as tmpdir:
            FEATURES_COUNT = 42
            gft = GoldenFeaturesTransformer(
                tmpdir, "binary_classification", features_count=FEATURES_COUNT
            )
            gft.fit(df, y)

            self.assertEqual(len(gft._new_features), FEATURES_COUNT)

            gft3 = GoldenFeaturesTransformer(tmpdir, "binary_classification")
            gft3.from_json(gft.to_json(), tmpdir)

            df = gft3.transform(df)
            self.assertEqual(df.shape[1], N_COLS + FEATURES_COUNT)
