import unittest
import numpy as np
import pandas as pd
from supervised.validation.validator_split import SplitValidator

import os
import shutil


class SplitValidatorTest(unittest.TestCase):
    def setUp(self):
        self._results_path = "/tmp/split_test"
        os.mkdir(self._results_path)

    def tearDown(self):
        shutil.rmtree(self._results_path, ignore_errors=True)

    def test_create(self):

        data = {
            "X": pd.DataFrame(
                np.array(
                    [[0, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [1, 1]]
                ),
                columns=["a", "b"],
            ),
            "y": pd.DataFrame(np.array([0, 0, 1, 0, 1, 0, 1, 1]), columns=["target"]),
        }

        X_path = os.path.join(self._results_path, "X.parquet")
        y_path = os.path.join(self._results_path, "y.parquet")

        data["X"].to_parquet(X_path, index=False)
        data["y"].to_parquet(y_path, index=False)

        params = {
            "shuffle": False,
            "stratify": False,
            "train_ratio": 0.5,
            "results_path": self._results_path,
            "X_path": X_path,
            "y_path": y_path,
        }
        vl = SplitValidator(params)

        self.assertEqual(1, vl.get_n_splits())
        # for train, validation in vl.split():
        for k_fold in range(vl.get_n_splits()):
            train, validation = vl.get_split(k_fold)

            X_train, y_train = train.get("X"), train.get("y")
            X_validation, y_validation = validation.get("X"), validation.get("y")

            self.assertEqual(X_train.shape[0], 4)
            self.assertEqual(y_train.shape[0], 4)
            self.assertEqual(X_validation.shape[0], 4)
            self.assertEqual(y_validation.shape[0], 4)

    def test_missing_target_values(self):

        data = {
            "X": pd.DataFrame(
                np.array([[1, 0], [2, 1], [3, 0], [4, 1], [5, 1], [6, 1]]),
                columns=["a", "b"],
            ),
            "y": pd.DataFrame(
                np.array(["a", "b", np.nan, "a", "b", np.nan]), columns=["target"]
            ),
        }

        X_path = os.path.join(self._results_path, "X.parquet")
        y_path = os.path.join(self._results_path, "y.parquet")

        data["X"].to_parquet(X_path, index=False)
        data["y"].to_parquet(y_path, index=False)

        params = {
            "shuffle": False,
            "stratify": False,
            "train_ratio": 0.5,
            "results_path": self._results_path,
            "X_path": X_path,
            "y_path": y_path,
        }
        vl = SplitValidator(params)

        self.assertEqual(1, vl.get_n_splits())

        for k_fold in range(vl.get_n_splits()):
            train, validation = vl.get_split(k_fold)
            X_train, y_train = train.get("X"), train.get("y")
            X_validation, y_validation = validation.get("X"), validation.get("y")

            self.assertEqual(X_train.shape[0], 3)
            self.assertEqual(y_train.shape[0], 3)
            self.assertEqual(X_validation.shape[0], 3)
            self.assertEqual(y_validation.shape[0], 3)

    def test_create_with_target_as_labels(self):

        data = {
            "X": pd.DataFrame(
                np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), columns=["a", "b"]
            ),
            "y": pd.DataFrame(np.array(["a", "b", "a", "b"]), columns=["target"]),
        }

        X_path = os.path.join(self._results_path, "X.parquet")
        y_path = os.path.join(self._results_path, "y.parquet")

        data["X"].to_parquet(X_path, index=False)
        data["y"].to_parquet(y_path, index=False)

        params = {
            "shuffle": True,
            "stratify": True,
            "train_ratio": 0.5,
            "results_path": self._results_path,
            "X_path": X_path,
            "y_path": y_path,
        }
        vl = SplitValidator(params)

        self.assertEqual(1, vl.get_n_splits())

        for k_fold in range(vl.get_n_splits()):
            train, validation = vl.get_split(k_fold)
            X_train, y_train = train.get("X"), train.get("y")
            X_validation, y_validation = validation.get("X"), validation.get("y")

            self.assertEqual(X_train.shape[0], 2)
            self.assertEqual(y_train.shape[0], 2)
            self.assertEqual(X_validation.shape[0], 2)
            self.assertEqual(y_validation.shape[0], 2)

    def test_repeats(self):

        data = {
            "X": pd.DataFrame(
                np.array(
                    [[0, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [1, 1]]
                ),
                columns=["a", "b"],
            ),
            "y": pd.DataFrame(np.array([0, 0, 1, 0, 1, 0, 1, 1]), columns=["target"]),
        }

        X_path = os.path.join(self._results_path, "X.parquet")
        y_path = os.path.join(self._results_path, "y.parquet")

        data["X"].to_parquet(X_path, index=False)
        data["y"].to_parquet(y_path, index=False)

        params = {
            "shuffle": True,
            "stratify": False,
            "train_ratio": 0.5,
            "results_path": self._results_path,
            "X_path": X_path,
            "y_path": y_path,
            "repeats": 3,
        }
        vl = SplitValidator(params)

        self.assertEqual(1, vl.get_n_splits())
        self.assertEqual(3, vl.get_repeats())

        cnt = 0
        for repeat in range(vl.get_repeats()):
            for k_fold in range(vl.get_n_splits()):
                train, validation = vl.get_split(k_fold, repeat)

                X_train, y_train = train.get("X"), train.get("y")
                X_validation, y_validation = validation.get("X"), validation.get("y")

                self.assertEqual(X_train.shape[0], 4)
                self.assertEqual(y_train.shape[0], 4)
                self.assertEqual(X_validation.shape[0], 4)
                self.assertEqual(y_validation.shape[0], 4)
                cnt += 1

        self.assertEqual(cnt, 3)

    def test_disable_repeats_when_disabled_shuffle(self):

        data = {
            "X": pd.DataFrame(
                np.array(
                    [[0, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [1, 1]]
                ),
                columns=["a", "b"],
            ),
            "y": pd.DataFrame(np.array([0, 0, 1, 0, 1, 0, 1, 1]), columns=["target"]),
        }

        X_path = os.path.join(self._results_path, "X.parquet")
        y_path = os.path.join(self._results_path, "y.parquet")

        data["X"].to_parquet(X_path, index=False)
        data["y"].to_parquet(y_path, index=False)

        params = {
            "shuffle": False,
            "stratify": False,
            "train_ratio": 0.5,
            "results_path": self._results_path,
            "X_path": X_path,
            "y_path": y_path,
            "repeats": 3,
        }
        vl = SplitValidator(params)

        self.assertEqual(1, vl.get_n_splits())
        self.assertEqual(1, vl.get_repeats())
