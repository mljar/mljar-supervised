import numpy as np
import pandas as pd

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.preprocessing.encoding_selector import EncodingSelector
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.utils.column_scan import iter_columns


class DataInfo:
    @staticmethod
    def compute(X, y, machinelearning_task):
        X = PreprocessingUtils.convert_timedeltas(
            X, PreprocessingUtils.timedelta_columns(X)
        )
        columns_info = {}
        for col, series in iter_columns(X):
            info = []
            columns_info[col] = info
            missing = pd.isnull(series)
            if missing.all():
                info.append("empty_column")
                continue
            has_missing = missing.any()
            values = series[~missing] if has_missing else series
            dtype_type = PreprocessingUtils._type_from_dtype(series.dtype)
            unique_count = None
            if dtype_type == PreprocessingUtils.CATEGORICAL:
                # Preserve object/category comparison semantics, but reuse the
                # count for constant, text, and encoding decisions.
                unique_count = len(np.unique(values))
                constant = unique_count == 1
            else:
                # Most numeric columns vary immediately. Check a bounded prefix
                # first, and scan the full column only if it could be constant.
                first = values.iloc[0]
                constant = values.iloc[:32].eq(first).all() and values.eq(first).all()
            if constant:
                info.append("constant_column")
                continue
            if has_missing:
                info.append("missing_values")
            data_type = PreprocessingUtils.get_type(values, unique_count=unique_count)
            if data_type == PreprocessingUtils.CATEGORICAL:
                info.append("categorical")
                info.append(EncodingSelector.get(X, y, col, unique_count=unique_count))
            elif data_type == PreprocessingUtils.DATETIME:
                info.append("datetime_transform")
            elif data_type == PreprocessingUtils.TEXT:
                columns_info[col] = ["text_transform"]
            elif data_type == PreprocessingUtils.IP:
                info.append("ip_transform")
            elif PreprocessingUtils._is_scale_needed(values):
                info.append("scale")

        target_info = []
        if machinelearning_task == BINARY_CLASSIFICATION:
            if not PreprocessingUtils.is_0_1(y):
                target_info += ["convert_0_1"]

        if machinelearning_task == REGRESSION:
            if PreprocessingUtils.is_log_scale_needed(y):
                target_info += ["scale_log"]
            elif PreprocessingUtils.is_scale_needed(y):
                target_info += ["scale"]

        num_class = None
        if machinelearning_task == MULTICLASS_CLASSIFICATION:
            num_class = PreprocessingUtils.num_class(y)

        return {
            "columns_info": columns_info,
            "target_info": target_info,
            "num_class": num_class,
        }
