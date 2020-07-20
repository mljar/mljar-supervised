import numpy as np
import pandas as pd
import datetime
import json
from sklearn.feature_extraction.text import TfidfVectorizer


class TextTransformer(object):
    def __init__(self):
        self._new_columns = []
        self._old_column = None
        self._max_features = 100
        self._vectorizer = None

    def fit(self, X, column):
        self._old_column = column
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            stop_words="english",
            lowercase=True,
            max_features=self._max_features,
        )

        x = X[column][~pd.isnull(X[column])]
        self._vectorizer.fit(x)
        for f in self._vectorizer.get_feature_names():
            new_col = self._old_column + "_" + f
            self._new_columns += [new_col]

    def transform(self, X):

        ii = ~pd.isnull(X[self._old_column])
        x = X[self._old_column][ii]
        vect = self._vectorizer.transform(x)

        for f in self._new_columns:
            X[f] = 0.0

        X.loc[ii, self._new_columns] = vect.toarray()
        X.drop(self._old_column, axis=1, inplace=True)
        return X

    def to_json(self):
        for k in self._vectorizer.vocabulary_.keys():
            self._vectorizer.vocabulary_[k] = int(self._vectorizer.vocabulary_[k])

        data_json = {
            "new_columns": list(self._new_columns),
            "old_column": self._old_column,
            "vocabulary": self._vectorizer.vocabulary_,
            "fixed_vocabulary": self._vectorizer.fixed_vocabulary_,
            "idf": list(self._vectorizer.idf_),
        }
        return data_json

    def from_json(self, data_json):
        self._new_columns = data_json.get("new_columns", None)
        self._old_column = data_json.get("old_column", None)
        vocabulary = data_json.get("vocabulary")
        fixed_vocabulary = data_json.get("fixed_vocabulary")
        idf = data_json.get("idf")
        if vocabulary is not None and fixed_vocabulary is not None and idf is not None:
            self._vectorizer = TfidfVectorizer(
                analyzer="word",
                stop_words="english",
                lowercase=True,
                max_features=self._max_features,
            )
            self._vectorizer.vocabulary_ = vocabulary
            self._vectorizer.fixed_vocabulary_ = fixed_vocabulary
            self._vectorizer.idf_ = idf
