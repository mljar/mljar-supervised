import os
import sys
import unittest

# import logging
# logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

from tests_algorithms.test_registry import AlgorithmsRegistryTest
from tests_algorithms.test_algorithm_factory import AlgorithmFactoryTest
from tests_algorithms.test_xgboost import XgboostAlgorithmTest
from tests_algorithms.test_random_forest import RandomForestAlgorithmTest
from tests_algorithms.test_lightgbm import LightgbmAlgorithmTest
from tests_algorithms.test_ensemble import EnsembleTest


if __name__ == "__main__":
    unittest.main()
