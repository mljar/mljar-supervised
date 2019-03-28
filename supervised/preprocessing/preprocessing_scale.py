import pandas as pd
import numpy as np


from supervised.preprocessing.preprocessing_utils import PreprocessingUtils


class PreprocessingScale(object):

    SCALE_MIN_MAX = "scale_min_max"
    SCALE_NORMAL = "scale_normal"
    SCALE_LOG = "scale_log"
    SCALE_LOG_AND_NORMAL = "scale_log_and_normal"

    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def to_json(self):
        pass

    def from_json(self, json_data):
        pass
