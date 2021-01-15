import numpy as np
import pandas as pd
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical
from supervised.preprocessing.preprocessing_missing import PreprocessingMissingValues
from supervised.preprocessing.scale import Scale

from supervised.algorithms.registry import (
    REGRESSION,
    MULTICLASS_CLASSIFICATION,
    BINARY_CLASSIFICATION,
)


class PreprocessingTuner:

    """
        This class prepare configuration for data preprocessing
    """

    CATEGORICALS_MIX = "categorical_mix"  # mix int and one-hot
    CATEGORICALS_ALL_INT = "categoricals_all_integers"
    CATEGORICALS_LOO = "categoricals_loo"

    @staticmethod
    def get(
        required_preprocessing,
        data_info,
        machinelearning_task,
        categorical_strategy=CATEGORICALS_ALL_INT,
    ):

        columns_preprocessing = {}
        columns_info = data_info["columns_info"]

        for col, preprocessing_needed in columns_info.items():
            preprocessing_to_apply = []

            # remove empty columns and columns with only one variable
            if (
                "empty_column" in preprocessing_needed
                or "constant_column" in preprocessing_needed
            ):
                preprocessing_to_apply += ["remove_column"]
                columns_preprocessing[col] = preprocessing_to_apply
                continue

            # always check for missing values
            if (
                "missing_values_inputation" in required_preprocessing
                and "missing_values" in preprocessing_needed
            ):
                preprocessing_to_apply += [PreprocessingMissingValues.FILL_NA_MEDIAN]
            # convert to categorical only for categorical types
            convert_to_integer_will_be_applied = False
            if (
                "convert_categorical"
                in required_preprocessing  # the algorithm needs converted categoricals
                and "categorical" in preprocessing_needed  # the feature is categorical
            ):

                if categorical_strategy == PreprocessingTuner.CATEGORICALS_MIX:
                    if PreprocessingCategorical.MANY_CATEGORIES in preprocessing_needed:
                        preprocessing_to_apply += [
                            PreprocessingCategorical.CONVERT_INTEGER
                        ]
                        convert_to_integer_will_be_applied = True  # maybe scale needed
                    else:
                        preprocessing_to_apply += [
                            PreprocessingCategorical.CONVERT_ONE_HOT
                        ]

                elif categorical_strategy == PreprocessingTuner.CATEGORICALS_LOO:
                    preprocessing_to_apply += [PreprocessingCategorical.CONVERT_LOO]
                    convert_to_integer_will_be_applied = True  # maybe scale needed
                else:  # all integers
                    preprocessing_to_apply += [PreprocessingCategorical.CONVERT_INTEGER]
                    convert_to_integer_will_be_applied = True  # maybe scale needed

                """
                if PreprocessingCategorical.CONVERT_ONE_HOT in preprocessing_needed:
                    preprocessing_to_apply += [PreprocessingCategorical.CONVERT_ONE_HOT]
                elif PreprocessingCategorical.CONVERT_LOO in preprocessing_needed:
                    preprocessing_to_apply += [PreprocessingCategorical.CONVERT_LOO]
                    convert_to_integer_will_be_applied = True  # maybe scale needed
                else:
                    preprocessing_to_apply += [PreprocessingCategorical.CONVERT_INTEGER]
                    convert_to_integer_will_be_applied = True  # maybe scale needed
                """

            if (
                "datetime_transform" in required_preprocessing
                and "datetime_transform" in preprocessing_needed
            ):
                preprocessing_to_apply += ["datetime_transform"]
            if (
                "text_transform" in required_preprocessing
                and "text_transform" in preprocessing_needed
            ):
                preprocessing_to_apply += ["text_transform"]

            if "scale" in required_preprocessing:
                if (
                    convert_to_integer_will_be_applied
                    or "scale" in preprocessing_needed
                ):
                    preprocessing_to_apply += [Scale.SCALE_NORMAL]

            # remeber which preprocessing we need to apply
            if preprocessing_to_apply:
                columns_preprocessing[col] = preprocessing_to_apply

        target_info = data_info["target_info"]
        target_preprocessing = []
        # always remove missing values from target,
        # target with missing values might be in the train and in the validation datasets
        target_preprocessing += [PreprocessingMissingValues.NA_EXCLUDE]

        if "target_as_integer" in required_preprocessing:
            if machinelearning_task == BINARY_CLASSIFICATION:
                if "convert_0_1" in target_info:
                    target_preprocessing += [PreprocessingCategorical.CONVERT_INTEGER]

            if machinelearning_task == MULTICLASS_CLASSIFICATION:
                # if PreprocessingUtils.is_categorical(y):
                # always convert to integer, there can be many situations that can break
                # for example, classes starting from 1, ...
                # or classes not for every number, for example 0,2,3,4
                # just always convert
                target_preprocessing += [PreprocessingCategorical.CONVERT_INTEGER]

        elif "target_as_one_hot" in required_preprocessing:
            target_preprocessing += [PreprocessingCategorical.CONVERT_ONE_HOT]

        if (
            machinelearning_task == REGRESSION
            and "target_scale" in required_preprocessing
        ):
            if "scale_log" in target_info:
                target_preprocessing += [Scale.SCALE_LOG_AND_NORMAL]
            elif "scale" in target_info:
                target_preprocessing += [Scale.SCALE_NORMAL]

        return {
            "columns_preprocessing": columns_preprocessing,
            "target_preprocessing": target_preprocessing,
            "ml_task": machinelearning_task,
        }
