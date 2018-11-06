import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'supervised'))

#from preprocessing.tests.run import *

from .test_iterative_learner_framework import IterativeLearnerTest

if __name__ == '__main__':
    unittest.main()
