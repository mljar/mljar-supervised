import numpy as np


class PreprocessingUtilsException(Exception):
    pass


class PreprocessingUtils(object):
    CATEGORICAL = "categorical"
    CONTINOUS = "continous"
    DISCRETE = "discrete"

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
        elif col_type.startswith("int"):
            data_type = PreprocessingUtils.DISCRETE
        return data_type

    @staticmethod
    def get_most_frequent(x):
        a = x.value_counts()
        first = sorted(dict(a).items(), key=lambda x: -x[1])[0]
        return first[0]

    @staticmethod
    def get_min(x):
        return np.amin(np.nanmin(x))

    @staticmethod
    def get_mean(x):
        return np.nanmean(x)

    @staticmethod
    def get_median(x):
        return np.nanmedian(x)
