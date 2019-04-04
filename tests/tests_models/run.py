import os
import sys
import unittest

# import logging
# logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

from tests.tests_models.test_registry import ModelsRegistryTest
from tests.tests_models.test_learner_factory import LearnerFactoryTest
from tests.tests_models.test_learner_xgboost import XgboostLearnerTest
from tests.tests_models.test_learner_random_forest import RandomForestLearnerTest
from tests.tests_models.test_learner_lightgbm import LightgbmLearnerTest
from tests.tests_models.test_ensemble import EnsembleTest

from tests.tests_models.test_registry import ModelsRegistryTest

if __name__ == "__main__":
    unittest.main()
