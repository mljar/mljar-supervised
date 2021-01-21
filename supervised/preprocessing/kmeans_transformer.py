import os
import numpy as np
import pandas as pd
import datetime
import json
import time
import joblib

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from supervised.exceptions import AutoMLException


class KMeansTransformer(object):
    def __init__(self, results_path=None, model_name = None, k_fold=None):
        self._new_features = []
        self._input_columns = []
        self._error = None
        self._kmeans = None
        self._scale = None
        self._model_name = model_name
        self._k_fold = k_fold

        if results_path is not None:
            self._result_file = os.path.join(results_path, self._model_name, f"kmeans_fold_{k_fold}.joblib")
            #self.try_load()

    def fit(self, X, y):
        if self._new_features:
            return
        if self._error is not None and self._error:
            raise AutoMLException(
                "KMeans Features not created due to error (please check errors.md). "
                + self._error
            )
            return
        if X.shape[1] == 0:
            self._error = f"KMeans not created. No continous features. Input data shape: {X.shape}, {y.shape}"
            raise AutoMLException("KMeans Features not created. No continous features.")

        start_time = time.time()

        n_clusters = int(np.log10(X.shape[0]) * 8)
        n_clusters = max(8, n_clusters)
        n_clusters = min(n_clusters, X.shape[1])

        self._input_columns = X.columns.tolist()
        # scale data
        self._scale = StandardScaler(
            copy=True, with_mean=True, with_std=True
        )
        X = self._scale.fit_transform(X)

        # Kmeans
        self._kmeans = kmeans = MiniBatchKMeans(n_clusters=n_clusters, init="k-means++")
        self._kmeans.fit(X)
        self._create_new_features_names()
        
        print(
            f"Created {len(self._new_features)} KMeans Features in {np.round(time.time() - start_time,2)} seconds."
        )

    def _create_new_features_names(self):
        n_clusters = self._kmeans.cluster_centers_.shape[0]
        self._new_features = [f"Dist_Cluster_{i}" for i in range(n_clusters)]
        self._new_features += ["Cluster"]

    def transform(self, X):
        if self._kmeans is None:
            raise AutoMLException("KMeans not fitted")

        # scale
        X_scaled = self._scale.transform(X[self._input_columns])

        # kmeans
        distances = self._kmeans.transform(X_scaled)
        clusters = self._kmeans.predict(X_scaled)

        X[self._new_features[:-1]] = distances
        X[self._new_features[-1]] = clusters

        return X

    def to_json(self):
        self.save()
        data_json = {
            "new_features": self._new_features,
            "result_file": self._result_file,
            "input_columns": self._input_columns
        }
        if self._error is not None and self._error:
            data_json["error"] = self._error
        return data_json

    def from_json(self, data_json):
        self._new_features = data_json.get("new_features", [])
        self._input_columns = data_json.get("input_columns", [])
        self._result_file = data_json.get("result_file")
        self._error = data_json.get("error")
        self.try_load()

    def save(self):
        joblib.dump({"kmeans": self._kmeans, "scale": self._scale}, self._result_file, compress=True)

    def try_load(self):
        if os.path.exists(self._result_file):
            data = joblib.load(self._result_file)
            self._kmeans = data["kmeans"]
            self._scale = data["scale"]
            
            self._create_new_features_names()
