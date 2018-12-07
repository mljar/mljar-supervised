import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from .test_validator_kfold import KFoldValidatorTest
from .test_validator_split import SplitValidatorTest
from .test_validator_with_dataset import WithDatasetValidatorTest
from .test_validation_step import ValidationStepTest

if __name__ == "__main__":
    unittest.main()
