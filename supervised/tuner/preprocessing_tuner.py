
from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical
from supervised.preprocessing.preprocessing_missing import PreprocessingMissingValues
from supervised.preprocessing.preprocessing_scale import PreprocessingScale

class PreprocessingTuner:
    @staticmethod
    def get(required_preprocessing):
        preprocessing_params = {}
        if "missing_values_inputation" in required_preprocessing:
            preprocessing_params["missing_values"] = PreprocessingMissingValues.FILL_NA_MEDIAN
        if "convert_categorical" in required_preprocessing:
            preprocessing_params["categorical"] = PreprocessingCategorical.CONVERT_INTEGER
        if "scale" in required_preprocessing:
            preprocessing_params["scale"] = PreprocessingScale.SCALE_NORMAL
        if "target_preprocessing" in required_preprocessing:
            preprocessing_params["target_preprocessing"] = True
        return preprocessing_params
