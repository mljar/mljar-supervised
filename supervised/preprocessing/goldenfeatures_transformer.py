import os
import numpy as np
import pandas as pd
import datetime
import json
import time
import itertools
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import log_loss, mean_squared_error
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.exceptions import AutoMLException


def get_binary_score(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]
    ll = log_loss(y_test, pred)
    return ll


def get_regression_score(X_train, y_train, X_test, y_test):
    clf = DecisionTreeRegressor(max_depth=3)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    ll = mean_squared_error(y_test, pred)
    return ll


def get_multiclass_score(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)
    ll = log_loss(y_test, pred)
    return ll


def get_score(item):
    col1 = item[0]
    col2 = item[1]
    X_train = item[2]
    y_train = item[3]
    X_test = item[4]
    y_test = item[5]
    scorer = item[6]

    try:
        x_train = np.array(X_train[col1] - X_train[col2]).reshape(-1, 1)
        x_test = np.array(X_test[col1] - X_test[col2]).reshape(-1, 1)
        diff_score = scorer(x_train, y_train, x_test, y_test)
    except Exception as e:
        diff_score = None
        print(str(e))

    try:
        a, b = (
            np.array(X_train[col1], dtype=float),
            np.array(X_train[col2], dtype=float),
        )
        x_train = np.divide(a, b, out=np.zeros_like(a), where=b != 0).reshape(-1, 1)
        a, b = np.array(X_test[col1], dtype=float), np.array(X_test[col2], dtype=float)
        x_test = np.divide(a, b, out=np.zeros_like(a), where=b != 0).reshape(-1, 1)
        ratio_1_score = scorer(x_train, y_train, x_test, y_test)
    except Exception as e:
        print(str(e))
        ratio_1_score = None

    try:
        b, a = (
            np.array(X_train[col1], dtype=float),
            np.array(X_train[col2], dtype=float),
        )
        x_train = np.divide(a, b, out=np.zeros_like(a), where=b != 0).reshape(-1, 1)
        b, a = np.array(X_test[col1], dtype=float), np.array(X_test[col2], dtype=float)
        x_test = np.divide(a, b, out=np.zeros_like(a), where=b != 0).reshape(-1, 1)
        ratio_2_score = scorer(x_train, y_train, x_test, y_test)
    except Exception as e:
        print(str(e))
        ratio_2_score = None

    return (diff_score, ratio_1_score, ratio_2_score)


class GoldenFeaturesTransformer(object):
    def __init__(self, results_path=None, ml_task=None):
        self._new_features = []
        self._new_columns = []
        self._ml_task = ml_task
        self._scorer = None
        if self._ml_task == BINARY_CLASSIFICATION:
            self._scorer = get_binary_score
        elif self._ml_task == MULTICLASS_CLASSIFICATION:
            self._scorer = get_multiclass_score
        else:
            self._scorer = get_regression_score

        self._error = None

        if results_path is not None:
            self._result_file = os.path.join(results_path, "golden_features.json")
            self.try_load()

    def fit(self, X, y):
        if self._new_features:
            return
        if self._error is not None and self._error:
            raise AutoMLException(
                "Golden Features not created due to error (please check errors.md)."
            )
            return
        if X.shape[1] == 0:
            self._error = f"Golden Features not created. No continous features. Input data shape: {X.shape}, {y.shape}"
            self.save(self._result_file)
            raise AutoMLException("Golden Features not created. No continous features.")

        start_time = time.time()
        combinations = itertools.combinations(X.columns, r=2)
        items = [i for i in combinations]
        if len(items) > 250000:
            si = np.random.choice(len(items), 250000, replace=False)
            items = [items[i] for i in si]

        X_train, X_test, y_train, y_test = self._subsample(X, y)

        for i in range(len(items)):
            items[i] += (X_train, y_train, X_test, y_test, self._scorer)

        scores = []
        # parallel version
        with Pool() as p:
            scores = p.map(get_score, items)
        # single process version
        # for item in items:
        #    scores += [get_score(item)]

        if not scores:
            self._error = f"Golden Features not created. Empty scores. Input data shape: {X.shape}, {y.shape}"
            self.save(self._result_file)
            raise AutoMLException("Golden Features not created. Empty scores.")

        result = []
        for i in range(len(items)):
            if scores[i][0] is not None:
                result += [(items[i][0], items[i][1], "diff", scores[i][0])]
            if scores[i][1] is not None:
                result += [(items[i][0], items[i][1], "ratio", scores[i][1])]
            if scores[i][2] is not None:
                result += [(items[i][1], items[i][0], "ratio", scores[i][2])]

        df = pd.DataFrame(
            result, columns=["feature1", "feature2", "operation", "score"]
        )
        df.sort_values(by="score", inplace=True)

        new_cols_cnt = np.min([50, np.max([5, int(0.05 * X.shape[1])])])

        self._new_features = json.loads(df.head(new_cols_cnt).to_json(orient="records"))

        for new_feature in self._new_features:
            new_col = "_".join(
                [
                    new_feature["feature1"],
                    new_feature["operation"],
                    new_feature["feature2"],
                ]
            )
            self._new_columns += [new_col]
            print(f"Add Golden Feature: {new_col}")

        self.save(self._result_file)

        print(
            f"Created {len(self._new_features)} Golden Features in {np.round(time.time() - start_time,2)} seconds."
        )

    def transform(self, X):
        for new_feature in self._new_features:
            new_col = "_".join(
                [
                    new_feature["feature1"],
                    new_feature["operation"],
                    new_feature["feature2"],
                ]
            )
            X[new_col] = X[new_feature["feature1"]] - X[new_feature["feature2"]]

        return X

    def to_json(self):
        data_json = {
            "new_features": json.dumps(self._new_features, indent=4),
            "new_columns": json.dumps(self._new_columns, indent=4),
            "result_file": self._result_file,
            "ml_task": self._ml_task,
        }
        if self._error is not None and self._error:
            data_json["error"] = self._error
        return data_json

    def from_json(self, data_json):
        self._new_features = json.loads(data_json.get("new_features", []))
        self._new_columns = json.loads(data_json.get("new_columns", []))
        self._result_file = data_json.get("result_file")
        self._ml_task = data_json.get("ml_task")
        self._error = data_json.get("error")

    def save(self, destination_file):
        with open(destination_file, "w") as fout:
            fout.write(json.dumps(self.to_json(), indent=4))

    def try_load(self):
        if os.path.exists(self._result_file):
            self.from_json(json.load(open(self._result_file, "r")))

    def _subsample(self, X, y):

        MAX_SIZE = 5000
        TRAIN_SIZE = 2500

        shuffle = True
        stratify = None

        if X.shape[0] > 5000:
            if self._ml_task != REGRESSION:
                stratify = y
            X_train, _, y_train, _ = train_test_split(
                X, y, train_size=MAX_SIZE, shuffle=shuffle, stratify=stratify
            )
            if self._ml_task != REGRESSION:
                stratify = y_train

            X_train, X_test, y_train, y_test = train_test_split(
                X_train,
                y_train,
                train_size=TRAIN_SIZE,
                shuffle=shuffle,
                stratify=stratify,
            )
        else:
            if self._ml_task != REGRESSION:
                stratify = y
            train_size = X.shape[0] // 2
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, shuffle=shuffle, stratify=stratify
            )

        return X_train, X_test, y_train, y_test
