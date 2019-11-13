import os
import sys
import unittest

# import logging
# logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

from tests.tests_algorithms.test_registry import AlgorithmsRegistryTest
from tests.tests_algorithms.test_algorithm_factory import AlgorithmFactoryTest
from tests.tests_algorithms.test_xgboost import XgboostAlgorithmTest
from tests.tests_algorithms.test_random_forest import RandomForestAlgorithmTest
from tests.tests_algorithms.test_lightgbm import LightgbmAlgorithmTest
from tests.tests_algorithms.test_ensemble import EnsembleTest
from tests.tests_algorithms.test_compute_additional_metrics import (
    ComputeAdditionalMetricsTest,
)

if __name__ == "__main__":
    unittest.main()
