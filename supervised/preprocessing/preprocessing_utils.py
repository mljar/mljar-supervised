import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing


class PreprocessingUtilsException(Exception):
    pass


class PreprocessingUtils(object):
    CATEGORICAL = "categorical"
    CONTINOUS = "continous"
    DISCRETE = "discrete"
    DATETIME = "datetime"
    TEXT = "text"

    @staticmethod
    def get_type(x):
        if len(x.shape) > 1:
            if x.shape[1] != 1:
                raise PreprocessingUtilsException(
                    "Please select one column to get its type"
                )
        col_type = str(x.dtype)

        data_type = PreprocessingUtils.CATEGORICAL
        if col_type.startswith("float"):
            data_type = PreprocessingUtils.CONTINOUS
        elif col_type.startswith("int") or col_type.startswith("uint"):
            data_type = PreprocessingUtils.DISCRETE
        elif col_type.startswith("datetime"):
            data_type = PreprocessingUtils.DATETIME
        elif col_type.startswith("category"):
            # do not check the additional condition for text feature
            # treat it as categorical
            return PreprocessingUtils.CATEGORICAL

        if data_type == PreprocessingUtils.CATEGORICAL:
            # check maybe this categorical is a text
            # it is a text, if:
            # has more than 200 unique values
            # more than half of rows is unique
            unique_cnt = len(np.unique(x[~pd.isnull(x)]))
            if unique_cnt > 200 and unique_cnt > int(0.5 * x.shape[0]):
                data_type = PreprocessingUtils.TEXT

        return data_type

    @staticmethod
    def is_categorical(x_org):
        x = x_org[~pd.isnull(x_org)]
        return PreprocessingUtils.get_type(x) == PreprocessingUtils.CATEGORICAL

    @staticmethod
    def is_datetime(x_org):
        x = x_org[~pd.isnull(x_org)]
        return PreprocessingUtils.get_type(x) == PreprocessingUtils.DATETIME

    @staticmethod
    def is_text(x_org):
        x = x_org[~pd.isnull(x_org)]
        return PreprocessingUtils.get_type(x) == PreprocessingUtils.TEXT

    @staticmethod
    def is_0_1(x_org):
        x = x_org[~pd.isnull(x_org)]
        u = np.unique(x)
        if len(u) != 2:
            return False
        return 0 in u and 1 in u

    @staticmethod
    def num_class(x_org):
        x = x_org[~pd.isnull(x_org)]
        u = np.unique(x)
        return len(u)

    @staticmethod
    def is_scale_needed(x_org):
        x = x_org[~pd.isnull(x_org)]
        abs_avg = np.abs(np.mean(x))
        stddev = np.std(x)
        if abs_avg > 0.5 or stddev > 1.5:
            return True
        return False

    @staticmethod
    def is_log_scale_needed(x_org):
        x_full = np.array(x_org[~pd.isnull(x_org)])
        # first scale on raw data
        x = preprocessing.scale(x_full)
        # second scale on log data
        x_log = preprocessing.scale(np.log(x_full - np.min(x_full) + 1))

        # the old approach, let's check how new approach will work
        # original_skew = np.abs(stats.skew(x))
        # log_skew = np.abs(stats.skew(x_log))
        # return log_skew < original_skew
        ########################################################################
        # p is probability of being normal distributions
        k2, p1 = stats.normaltest(x)
        k2, p2 = stats.normaltest(x_log)

        return p2 > p1

    @staticmethod
    def is_na(x):
        return np.sum(pd.isnull(x) == True) > 0

    @staticmethod
    def get_most_frequent(x):
        a = x.value_counts()
        first = sorted(dict(a).items(), key=lambda x: -x[1])[0]
        return first[0]

    @staticmethod
    def get_min(x):
        v = np.amin(np.nanmin(x))
        if pd.isnull(v):
            return 0
        return float(v)

    @staticmethod
    def get_mean(x):
        v = np.nanmean(x)
        if pd.isnull(v):
            return 0
        return float(v)

    @staticmethod
    def get_median(x):
        v = np.nanmedian(x)
        if pd.isnull(v):
            return 0
        return float(v)
