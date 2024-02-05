# tasks that can be handled by the package

from typing import List, Type

BINARY_CLASSIFICATION = "binary_classification"
MULTICLASS_CLASSIFICATION = "multiclass_classification"
REGRESSION = "regression"


class AlgorithmsRegistry:
    from supervised.algorithms.algorithm import BaseAlgorithm
    registry = {
        BINARY_CLASSIFICATION: {},
        MULTICLASS_CLASSIFICATION: {},
        REGRESSION: {},
    }

    @staticmethod
    def add(
        task_name: str,
        model_class: Type[BaseAlgorithm],
        model_params: dict,
        required_preprocessing: list,
        additional: dict,
        default_params: dict,
    ) -> None:
        model_information = {
            "class": model_class,
            "params": model_params,
            "required_preprocessing": required_preprocessing,
            "additional": additional,
            "default_params": default_params,
        }
        AlgorithmsRegistry.registry[task_name][
            model_class.algorithm_short_name
        ] = model_information

    @staticmethod
    def get_supported_ml_tasks() -> List[str]:
        return AlgorithmsRegistry.registry.keys()

    @staticmethod
    def get_algorithm_class(ml_task: str, algorithm_name: str) -> Type[BaseAlgorithm]:
        return AlgorithmsRegistry.registry[ml_task][algorithm_name]["class"]

    @staticmethod
    def get_long_name(ml_task: str, algorithm_name: str) -> str:
        return AlgorithmsRegistry.registry[ml_task][algorithm_name][
            "class"
        ].algorithm_name

    @staticmethod
    def get_max_rows_limit(ml_task: str, algorithm_name: str) -> int:
        return AlgorithmsRegistry.registry[ml_task][algorithm_name]["additional"][
            "max_rows_limit"
        ]

    @staticmethod
    def get_max_cols_limit(ml_task: str, algorithm_name: str) -> int:
        return AlgorithmsRegistry.registry[ml_task][algorithm_name]["additional"][
            "max_cols_limit"
        ]

    @staticmethod
    def get_eval_metric(ml_task: str, algorithm_name: str, automl_eval_metric: str):
        if algorithm_name == "Xgboost":
            return xgboost_eval_metric(ml_task, automl_eval_metric)

        return automl_eval_metric

# Import algorithm to be registered
