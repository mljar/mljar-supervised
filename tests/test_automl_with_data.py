import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from sklearn import datasets
from supervised.automl import AutoML
from supervised.metric import Metric

import sklearn.model_selection
from sklearn.metrics import log_loss


class AutoMLTestWithData(unittest.TestCase):
    def test_fit_and_predict(self):
        seed = 1706
        for dataset_id in [44]:  # 31,44,737
            df = pd.read_csv("./tests/data/data/{0}.csv".format(dataset_id))
            x_cols = [c for c in df.columns if c != "target"]
            X = df[x_cols]
            y = df["target"]

            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                X, y, test_size=0.3, random_state=seed
            )
            automl = AutoML()

            automl.fit(X_train, y_train)

            response = automl.predict(X_test)
            # Compute the logloss on test dataset
            ll = log_loss(y_test, response)
            print("(*) Dataset id {} logloss {}".format(dataset_id, ll))

            for m in automl._models:
                response = m.predict(X_test)
                ll = log_loss(y_test, response)
                print("Dataset id {} logloss {}".format(dataset_id, ll))

        # y_predicted = automl.predict(self.X)
        # print(y_predicted)
        # metric = Metric({"name": "logloss"})
        # loss = metric(self.y, y_predicted)
        # print("Loss", loss)
        # self.assertTrue(y_predicted is not None)


if __name__ == "__main__":
    unittest.main()
