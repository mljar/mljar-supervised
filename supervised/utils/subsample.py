import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from supervised.algorithms.registry import REGRESSION


def subsample(X, y, ml_task, train_size):

    shuffle = True
    stratify = None

    if ml_task != REGRESSION:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, shuffle=shuffle, stratify=stratify
    )

    return X_train, X_test, y_train, y_test
