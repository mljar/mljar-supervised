import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from .test_preprocessing_utils import PreprocessingUtilsTest
#from test_preprocessing_missing import PreprocessingMissingValuesTest
#from test_preprocessing_categorical_integers import PreprocessingCategoricalIntegersTest
#from test_preprocessing_categorical_one_hot import PreprocessingCategoricalOneHotTest
#from test_label_encoder import LabelEncoderTest
#from test_label_binarizer import LabelBinarizerTest
#from test_preprocessing_box import PreprocessingBoxTest
#from test_preprocessing_step import PreprocessingStepTest


if __name__ == '__main__':
    unittest.main()
