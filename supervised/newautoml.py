from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class AutoMl(ABC, BaseEstimator):
    """
    AutoML model. Contains training, prediction and evaluation methods.
    """

    def __init__(self):
        super().__init__()
