import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from supervised.algorithms.registry import AlgorithmsRegistry


class AlgorithmsRegistryTest(unittest.TestCase):
    def test_add_to_registry(self):
        class Model1:
            algorithm_short_name = ""

        model1 = {
            "task_name": "binary_classification",
            "model_class": Model1,
            "model_params": {},
            "required_preprocessing": {},
            "additional": {},
            "default_params": {}
        }
        AlgorithmsRegistry.add(**model1)


if __name__ == "__main__":
    unittest.main()
