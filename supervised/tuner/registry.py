# tasks that can be handled by the package
BINARY_CLASSIFICATION = "binary_classification"
MULTICLASS_CLASSIFICATION = "multiclass_classification"
REGRESSION = "regression"


class ModelsRegistry:

    registry = {
        BINARY_CLASSIFICATION: {},
        MULTICLASS_CLASSIFICATION: {},
        REGRESSION: {},
    }

    @staticmethod
    def add(task_name, model_class, model_params, required_preprocessing, additional):
        model_information = {
            "class": model_class,
            "params": model_params,
            "required_preprocessing": required_preprocessing,
            "additional": additional,
        }
        ModelsRegistry.registry[task_name][
            model_class.algorithm_short_name
        ] = model_information
