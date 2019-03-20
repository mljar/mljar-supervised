import os
import sys
import unittest

from tests.tests_preprocessing.test_preprocessing_missing import PreprocessingMissingValuesTest
from tests.tests_preprocessing.test_preprocessing_categorical_integers import (
    PreprocessingCategoricalIntegersTest,
)
#from tests.tests_preprocessing.test_preprocessing_categorical_one_hot import PreprocessingCategoricalOneHotTest
from tests.tests_preprocessing.test_label_encoder import LabelEncoderTest
from tests.tests_preprocessing.test_label_binarizer import LabelBinarizerTest
#from tests.tests_preprocessing.test_preprocessing_step import PreprocessingStepTest


if __name__ == "__main__":
    unittest.main()
