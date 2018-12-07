import pandas as pd
import numpy as np

from utils.jsonable import Jsonable
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils


class PreprocessingScale(Jsonable):

    SCALE_MIN_MAX = "scale_min_max"
    SCALE_NORMAL = "scale_normal"

    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass
