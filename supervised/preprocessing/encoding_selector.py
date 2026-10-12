import numpy as np
import pandas as pd

from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical


class EncodingSelector:
    """
    EncodingSelector object decides which method should be used for categorical encoding.

    Please keep it fast and simple. Thank you.
    """

    @staticmethod
    def get(X, y, column, *, unique_count=None):
        try:
            unique_cnt = unique_count
            if unique_cnt is None:
                unique_cnt = len(np.unique(X.loc[~pd.isnull(X[column]), column]))
            if unique_cnt <= 20:
                return PreprocessingCategorical.FEW_CATEGORIES
        except Exception as e:
            pass

        return PreprocessingCategorical.MANY_CATEGORIES
        """
        if unique_cnt <= 2 or unique_cnt > 25:
            return PreprocessingCategorical.CONVERT_INTEGER

        return PreprocessingCategorical.CONVERT_ONE_HOT
        """
