
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

#from .test_registry import ModelsRegistryTest
from .test_learner_xgboost import XgboostLearnerTest

if __name__ == '__main__':
    unittest.main()
