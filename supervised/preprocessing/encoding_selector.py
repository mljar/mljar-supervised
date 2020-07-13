import pandas as pd
import numpy as np

from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical


class EncodingSelector:

    """
    EncodingSelector object decides which method should be used for categorical encoding.

    Please keep it fast and simple. Thank you.
    """

    @staticmethod
    def get(X, y, column):
        unique_cnt = len(np.unique(X.loc[~pd.isnull(X[column]), column]))
        if unique_cnt <= 2 or unique_cnt > 25:
            return PreprocessingCategorical.CONVERT_INTEGER

        return PreprocessingCategorical.CONVERT_ONE_HOT
