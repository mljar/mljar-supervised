
import os
import sys
import unittest

#import logging
#logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from .test_registry import ModelsRegistryTest
from .test_learner_factory import LearnerFactoryTest
from .test_learner_xgboost import XgboostLearnerTest

if __name__ == '__main__':
    unittest.main()
