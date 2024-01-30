import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from supervised.utils.importance import PermutationImportance


class PermutationImportanceTest(unittest.TestCase):
    def test_compute_and_plot(self):
        rows = 20
        X = np.random.rand(rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.randint(0, 2, rows)

        model = XGBClassifier(n_estimators=1, max_depth=2)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            PermutationImportance.compute_and_plot(
                model,
                X_validation=X,
                y_validation=y,
                model_file_path=tmpdir,
                learner_name="learner_test",
                metric_name=None,
                ml_task="binary_classification",
            )
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir, "learner_test_importance.csv"))
            )
