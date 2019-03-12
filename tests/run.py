import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from .test_iterative_learner_framework import IterativeLearnerTest

# from .test_metric import MetricTest

if __name__ == "__main__":
    unittest.main()
