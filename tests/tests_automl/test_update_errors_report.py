import os
import shutil
import unittest

import numpy as np

from supervised import AutoML


class AutoMLUpdateErrorsReportTest(unittest.TestCase):
    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_custom_init(self):
        X = np.random.uniform(size=(30, 2))
        y = np.random.randint(0, 2, size=(30,))

        automl = AutoML(results_path=self.automl_dir)
        automl._update_errors_report("model_1", "bad error")

        errors_filename = os.path.join(self.automl_dir, "errors.md")
        self.assertTrue(os.path.exists(errors_filename))
        with open(errors_filename) as file:
            self.assertTrue("bad error" in file.read())
