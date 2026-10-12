import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing

from supervised.preprocessing.ip_transformer import parse_ip


class PreprocessingUtilsException(Exception):
    pass


class PreprocessingUtils(object):
    CATEGORICAL = "categorical"
    CONTINOUS = "continous"
    DISCRETE = "discrete"
    DATETIME = "datetime"
    TEXT = "text"
    IP = "ip"

    @staticmethod
    def timedelta_columns(X):
        if X is None:
            return []
        return X.select_dtypes(include="timedelta").columns.tolist()

    @staticmethod
    def convert_timedeltas(X, columns):
        """Represent durations in seconds, preserving NaT as NaN."""
        if X is None or not columns:
            return X
        X = X.copy(deep=False)
        for column in columns:
            if column in X and pd.api.types.is_timedelta64_dtype(X[column].dtype):
                X[column] = X[column].dt.total_seconds()
        return X

    @staticmethod
    def get_type(x, *, unique_count=None):
        if len(x.shape) > 1:
            if x.shape[1] != 1:
                raise PreprocessingUtilsException(
                    "Please select one column to get its type"
                )
        data_type = PreprocessingUtils._type_from_dtype(x.dtype)
        if str(x.dtype).startswith("category"):
            return PreprocessingUtils.CATEGORICAL

        if data_type == PreprocessingUtils.CATEGORICAL:
            if PreprocessingUtils.is_ip(x):
                return PreprocessingUtils.IP
            # check maybe this categorical is a text
            # it is a text, if:
            # has more than 200 unique values
            # more than half of rows is unique
            unique_cnt = unique_count
            if unique_cnt is None:
                unique_cnt = len(np.unique(x[~pd.isnull(x)]))
            if unique_cnt > 200 and unique_cnt > int(0.5 * x.shape[0]):
                data_type = PreprocessingUtils.TEXT

        return data_type

    @staticmethod
    def _type_from_dtype(dtype):
        # Keep the existing dtype-name rules, including extension dtypes. Changing
        # their classifications is separate from avoiding repeated data scans.
        name = str(dtype)
        if name.startswith("float"):
            return PreprocessingUtils.CONTINOUS
        if name.startswith("int") or name.startswith("uint"):
            return PreprocessingUtils.DISCRETE
        if name.startswith("datetime"):
            return PreprocessingUtils.DATETIME
        return PreprocessingUtils.CATEGORICAL

    @staticmethod
    def _column_type(x):
        data_type = PreprocessingUtils._type_from_dtype(x.dtype)
        if data_type != PreprocessingUtils.CATEGORICAL:
            return data_type
        if str(x.dtype).startswith("category"):
            return data_type
        # Text detection historically uses the non-null row count here.
        return PreprocessingUtils.get_type(x[~pd.isnull(x)])

    @staticmethod
    def is_categorical(x_org):
        return PreprocessingUtils._column_type(x_org) == PreprocessingUtils.CATEGORICAL

    @staticmethod
    def is_datetime(x_org):
        return PreprocessingUtils._column_type(x_org) == PreprocessingUtils.DATETIME

    @staticmethod
    def is_text(x_org):
        return PreprocessingUtils._column_type(x_org) == PreprocessingUtils.TEXT

    @staticmethod
    def is_ip(x):
        if (
            PreprocessingUtils._type_from_dtype(x.dtype)
            != PreprocessingUtils.CATEGORICAL
        ):
            return False
        if str(x.dtype).startswith("category"):
            return False
        found = False
        seen = set()
        for value in x:
            if pd.isna(value):
                continue
            if not isinstance(value, str) or ("." not in value and ":" not in value):
                return False
            if value in seen:
                continue
            try:
                parse_ip(value)
            except ValueError:
                return False
            seen.add(value)
            found = True
        return found

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
        return PreprocessingUtils._is_scale_needed(x)

    @staticmethod
    def _is_scale_needed(x):
        """Inspect already non-null values without another mask or copy."""
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
