import uuid
import numpy as np
from supervised.utils.importance import PermutationImportance
from supervised.utils.shap import PlotSHAP


class BaseAlgorithm:
    """
    This is an abstract class.
    All algorithms inherit from BaseAlgorithm.
    """

    algorithm_name = "Unknown"
    algorithm_short_name = "Unknown"

    def __init__(self, params):
        self.params = params
        self.stop_training = False
        self.library_version = None
        self.model = None
        self.uid = params.get("uid", str(uuid.uuid4()))

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    # needed for feature importance
    def predict_proba(self, X):
        y = self.predict(X)
        if "num_class" in self.params:
            return y
        return np.column_stack((1 - y, y))

    def update(self, update_params):
        pass

    def copy(self):
        pass

    def save(self, model_file_path):
        pass

    def load(self, model_file_path):
        pass

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
        # do not produce feature importance for Baseline
        if self.algorithm_short_name == "Baseline":
            return
        PermutationImportance.compute_and_plot(
            self,
            X_validation,
            y_validation,
            model_file_path,
            learner_name,
            metric_name,
            ml_task,
        )

        PlotSHAP.compute(
            self,
            X_train,
            y_train,
            X_validation,
            y_validation,
            model_file_path,
            learner_name,
            class_names,
            ml_task,
        )
