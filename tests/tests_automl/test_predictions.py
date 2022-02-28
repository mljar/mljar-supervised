import unittest
import shutil
import numpy as np
import pandas as pd
import json
import os

from supervised import AutoML


class PredictionsText(unittest.TestCase):

    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_predictions(self):
        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=20,
        )
        train = pd.read_csv(os.path.join("tests/data/Titanic/train.csv"))
        X = train[train.columns[2:]]
        y = train["Survived"]
        automl.fit(X, y)
        test = pd.read_csv(os.path.join("tests/data/Titanic/train.csv"))
        automl2 = AutoML(
            results_path=self.automl_dir,
        )
        load_on_predict = json.load(
            open(os.path.join(self.automl_dir, "params.json"), "r")
        )["load_on_predict"]
        for m in load_on_predict:
            automl.predict(test, m)
            automl.predict_proba(test,m)
            automl.predict_all(test,m)
