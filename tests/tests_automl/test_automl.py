import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from supervised import AutoML


class AutoMLTest(unittest.TestCase):
    def test_set_directory(self):
        
        automl_dir = "automl_1"
        self.assertTrue(not os.path.exists(automl_dir))
        a = AutoML(results_path = automl_dir)
        self.assertTrue(os.path.exists(automl_dir))

        shutil.rmtree(automl_dir)