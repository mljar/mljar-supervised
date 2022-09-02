import os
import unittest
import pytest
import json
import shutil

import supervised.exceptions
from supervised import AutoML
from sklearn import datasets

iris = datasets.load_iris()

class ModelSelectionTest(unittest.TestCase):

    automl_dir = "model_selection_tests"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_choose_model(self):
        model = AutoML(
            explain_level=0, verbose=1, random_state=1, results_path=self.automl_dir
        )
        model.fit(iris.data, iris.target)
        params = json.load(open(os.path.join(self.automl_dir, "params.json")))
        for model_name in params['saved']:
            model.predict(iris.data,model_name)
            model.predict_all(iris.data, model_name)
            model.predict_proba(iris.data, model_name)

    def test_raise_with_wrong_model(self):
        model = AutoML(
            explain_level=0, verbose=1, random_state=1, results_path=self.automl_dir
        )
        model.fit(iris.data, iris.target)
        msg = "Cannot load AutoML directory. model name random_name does not exist in file"
        with pytest.raises(supervised.exceptions.AutoMLException, match=msg):
            model.predict(iris.data, "random_name")


