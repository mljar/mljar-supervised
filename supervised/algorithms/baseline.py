import logging
import os
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor

from supervised.algorithms.sklearn import SklearnAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class BaselineClassifierAlgorithm(SklearnAlgorithm):

    algorithm_name = "Baseline Classifier"
    algorithm_short_name = "Baseline"

    def __init__(self, params):
        super(BaselineClassifierAlgorithm, self).__init__(params)
        logger.debug("BaselineClassifierAlgorithm.__init__")

        self.library_version = sklearn.__version__
        self.max_iters = additional.get("max_steps", 1)
        self.model = DummyClassifier(
            strategy="prior", random_state=params.get("seed", 1)
        )

    def file_extension(self):
        return "baseline"


class BaselineRegressorAlgorithm(SklearnAlgorithm):

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
