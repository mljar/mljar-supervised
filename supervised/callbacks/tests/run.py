import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from .test_early_stopping import EarlyStoppingTest

if __name__ == '__main__':
    unittest.main()
