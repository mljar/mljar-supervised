import json
import os
import tempfile
import unittest

from supervised import AutoML
from supervised.exceptions import AutoMLException


class AutoMLModelsNeededForPredictTest(unittest.TestCase):
    # models_needed_on_predict

    def test_models_needed_on_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {
                "saved": [
                    "model_1",
                    "model_2",
                    "model_3",
                    "unused_model",
                    "Ensemble",
                    "model_4_Stacked",
                    "Stacked_Ensemble",
                ],
                "stacked": ["Ensemble", "model_1", "model_2"],
            }
            with open(os.path.join(tmpdir, "params.json"), "w") as fout:
                fout.write(json.dumps(params))
            os.mkdir(os.path.join(tmpdir, "Ensemble"))
            with open(os.path.join(tmpdir, "Ensemble", "ensemble.json"), "w") as fout:
                params = {
                    "selected_models": [
                        {"model": "model_2"},
                        {"model": "model_3"},
                    ]
                }
                fout.write(json.dumps(params))
            os.mkdir(os.path.join(tmpdir, "Stacked_Ensemble"))
            with open(
                os.path.join(tmpdir, "Stacked_Ensemble", "ensemble.json"), "w"
            ) as fout:
                params = {
                    "selected_models": [
                        {"model": "Ensemble"},
                        {"model": "model_4_Stacked"},
                    ]
                }
                fout.write(json.dumps(params))

            automl = AutoML(results_path=tmpdir)
            with self.assertRaises(AutoMLException) as context:
                l = automl.models_needed_on_predict("missing_model")
            l = automl.models_needed_on_predict("model_1")
            self.assertTrue("model_1" in l)
            self.assertTrue(len(l) == 1)
            l = automl.models_needed_on_predict("model_3")
            self.assertTrue("model_3" in l)
            self.assertTrue(len(l) == 1)
            l = automl.models_needed_on_predict("Ensemble")
            self.assertTrue("model_2" in l)
            self.assertTrue("model_3" in l)
            self.assertTrue("Ensemble" in l)
            self.assertTrue(len(l) == 3)
            l = automl.models_needed_on_predict("model_4_Stacked")
            self.assertTrue("model_1" in l)
            self.assertTrue("model_2" in l)
            self.assertTrue("model_3" in l)
            self.assertTrue("Ensemble" in l)
            self.assertTrue("model_4_Stacked" in l)
            self.assertTrue(len(l) == 5)
            l = automl.models_needed_on_predict("Stacked_Ensemble")
            self.assertTrue("model_1" in l)
            self.assertTrue("model_2" in l)
            self.assertTrue("model_3" in l)
            self.assertTrue("Ensemble" in l)
            self.assertTrue("model_4_Stacked" in l)
            self.assertTrue("Stacked_Ensemble" in l)
            self.assertTrue(len(l) == 6)
