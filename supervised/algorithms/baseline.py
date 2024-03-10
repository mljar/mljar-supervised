import logging

import sklearn
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
    AlgorithmsRegistry,
)
from supervised.algorithms.sklearn import SklearnAlgorithm
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class BaselineClassifierAlgorithm(SklearnAlgorithm, ClassifierMixin):
    algorithm_name = "Baseline Classifier"
    algorithm_short_name = "Baseline"

    def __init__(self, params: dict):
        super(BaselineClassifierAlgorithm, self).__init__(params)
        logger.debug("BaselineClassifierAlgorithm.__init__")

        self.library_version: str = sklearn.__version__
        self.max_iters: int = additional.get("max_steps", 1)
        self.model = DummyClassifier(
            strategy="prior", random_state=params.get("seed", 1)
        )

    def file_extension(self):
        return "baseline"

    def is_fitted(self):
        return (
            hasattr(self.model, "n_outputs_")
            and self.model.n_outputs_ is not None
            and self.model.n_outputs_ > 0
        )


class BaselineRegressorAlgorithm(SklearnAlgorithm, RegressorMixin):
    algorithm_name = "Baseline Regressor"
    algorithm_short_name = "Baseline"

    def __init__(self, params):
        super(BaselineRegressorAlgorithm, self).__init__(params)
        logger.debug("BaselineRegressorAlgorithm.__init__")

        self.library_version = sklearn.__version__
        self.max_iters = additional.get("max_steps", 1)
        self.model = DummyRegressor(strategy="mean")

    def file_extension(self):
        return "baseline"

    def is_fitted(self):
        return (
            hasattr(self.model, "n_outputs_")
            and self.model.n_outputs_ is not None
            and self.model.n_outputs_ > 0
        )


additional = {"max_steps": 1, "max_rows_limit": None, "max_cols_limit": None}
required_preprocessing = ["target_as_integer"]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    BaselineClassifierAlgorithm,
    {},
    required_preprocessing,
    additional,
    {},
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    BaselineClassifierAlgorithm,
    {},
    required_preprocessing,
    additional,
    {},
)


AlgorithmsRegistry.add(REGRESSION, BaselineRegressorAlgorithm, {}, {}, additional, {})
