import logging
import os
import sklearn
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.sklearn import SklearnTreesClassifierAlgorithm
from supervised.algorithms.sklearn import SklearnTreesRegressorAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


from dtreeviz.trees import dtreeviz


class DecisionTreeAlgorithm(SklearnTreesClassifierAlgorithm):

    algorithm_name = "Decision Tree"
    algorithm_short_name = "Decision Tree"

    def __init__(self, params):
        super(DecisionTreeAlgorithm, self).__init__(params)
        logger.debug("DecisionTreeAlgorithm.__init__")
        self.library_version = sklearn.__version__
        self.max_iters = additional.get("max_steps", 1)
        self.model = DecisionTreeClassifier(
            criterion=params.get("criterion", "gini"),
            max_depth=params.get("max_depth", 3),
            random_state=params.get("seed", 1),
        )

    def file_extenstion(self):
        return "decision_tree"

    def interpret(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_file_path,
        learner_name,
        target_name=None,
        class_names=None,
        metric_name=None,
        ml_task=None,
    ):
        super(DecisionTreeAlgorithm, self).interpret(
            X_train,
            y_train,
            X_validation,
            y_validation,
            model_file_path,
            learner_name,
            target_name,
            class_names,
            metric_name,
            ml_task,
        )
        try:
            if len(class_names) > 10:
                # dtreeviz does not support more than 10 classes
                return
            viz = dtreeviz(
                self.model,
                X_train,
                y_train,
                target_name="target",
                feature_names=X_train.columns,
                class_names=class_names,
            )
            tree_file_plot = os.path.join(
                model_file_path, learner_name + "_tree.svg"
            )
            viz.save(tree_file_plot)
        except Exception as e:
            logger.info(f"Problem when visuzalizin decision tree. {str(e)}")
            

class DecisionTreeRegressorAlgorithm(SklearnTreesRegressorAlgorithm):

    algorithm_name = "Decision Tree"
    algorithm_short_name = "Decision Tree"

    def __init__(self, params):
        super(DecisionTreeRegressorAlgorithm, self).__init__(params)
        logger.debug("DecisionTreeRegressorAlgorithm.__init__")
        self.library_version = sklearn.__version__
        self.max_iters = additional.get("max_steps", 1)
        self.model = DecisionTreeRegressor(
            criterion=params.get("criterion", "mse"),
            max_depth=params.get("max_depth", 3),
            random_state=params.get("seed", 1),
        )
        

    def file_extenstion(self):
        return "decision_tree"

    def interpret(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_file_path,
        learner_name,
        target_name=None,
        class_names=None,
        metric_name=None,
        ml_task=None,
    ):
        super(DecisionTreeRegressorAlgorithm, self).interpret(
            X_train,
            y_train,
            X_validation,
            y_validation,
            model_file_path,
            learner_name,
            target_name,
            class_names,
            metric_name,
            ml_task,
        )
        try:
            viz = dtreeviz(
                self.model,
                X_train,
                y_train,
                target_name="target",
                feature_names=X_train.columns,
            )
            tree_file_plot = os.path.join(
                model_file_path, learner_name + "_tree.svg"
            )
            viz.save(tree_file_plot)
        except Exception as e:
            logger.info(f"Problem when visuzalizin decision tree regressor. {str(e)}")


dt_params = {"criterion": ["gini", "entropy"], "max_depth": [1, 2, 3, 4]}

additional = {
    "trees_in_step": 1,
    "train_cant_improve_limit": 0,
    "max_steps": 1,
    "max_rows_limit": None,
    "max_cols_limit": None,
}
required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "target_as_integer",
]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    DecisionTreeAlgorithm,
    dt_params,
    required_preprocessing,
    additional,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    DecisionTreeAlgorithm,
    dt_params,
    required_preprocessing,
    additional,
)

dt_regression_params = {
    "criterion": ["mse", "friedman_mse", "mae"],
    "max_depth": [1, 2, 3, 4],
}
regression_required_preprocessing = ["missing_values_inputation", "convert_categorical"]

AlgorithmsRegistry.add(
    REGRESSION,
    DecisionTreeRegressorAlgorithm,
    dt_regression_params,
    regression_required_preprocessing,
    additional,
)
