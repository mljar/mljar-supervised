import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from supervised import AutoML

from supervised.exceptions import AutoMLException


class AutoMLTest(unittest.TestCase):

    automl_dir = "automl_1"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_set_directory(self):
        """ Directory does not exist, create it """
        self.assertTrue(not os.path.exists(self.automl_dir))
        a = AutoML(results_path=self.automl_dir)
        a.fit(
            [2, 3, 4, 5], [1, 2, 3, 4]
        )  # AutoML only validates constructor params on `fit()` call
        self.assertTrue(os.path.exists(self.automl_dir))

    def test_use_directory_if_empty_exists(self):
        """ Directory exists and is empty, use it """
        os.mkdir(self.automl_dir)
        self.assertTrue(os.path.exists(self.automl_dir))
        a = AutoML(results_path=self.automl_dir)
        a.fit(
            [2, 3, 4, 5], [1, 2, 3, 4]
        )  # AutoML only validates constructor params on `fit()` call
        self.assertTrue(os.path.exists(self.automl_dir))

    def test_dont_use_directory_if_non_empty_exists_without_params_json(self):
        """
        Directory exists and is not empty,
        there is no params.json file in it, dont use it, raise exception
        """
        os.mkdir(self.automl_dir)
        open(os.path.join(self.automl_dir, "test.file"), "w").close()
        self.assertTrue(os.path.exists(self.automl_dir))
        with self.assertRaises(AutoMLException) as context:
            a = AutoML(results_path=self.automl_dir)
            a.fit(
                [2, 3, 4, 5], [1, 2, 3, 4]
            )  # AutoML only validates constructor params on `fit()` call

        self.assertTrue("not empty" in str(context.exception))

    def test_use_directory_if_non_empty_exists_with_params_json(self):
        """
        Directory exists and is not empty,
        there is params.json in it, try to load it,
        raise exception because of fake params.json
        """
        os.mkdir(self.automl_dir)
        open(os.path.join(self.automl_dir, "params.json"), "w").close()
        self.assertTrue(os.path.exists(self.automl_dir))
        with self.assertRaises(AutoMLException) as context:
            a = AutoML(results_path=self.automl_dir)
            a.fit(
                [2, 3, 4, 5], [1, 2, 3, 4]
            )  # AutoML only validates constructor params on `fit()` call
        self.assertTrue("Cannot load" in str(context.exception))
