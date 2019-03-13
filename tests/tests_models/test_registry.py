import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from supervised.models.registry import ModelsRegistry


class ModelsRegistryTest(unittest.TestCase):
    def test_add_to_registry(self):
        class Model1:
            algorithm_short_name = ""


        model1 = {
            "task_name": "binary_classification",
            "model_class": Model1,
            "model_params": {},
            "additional": {}
        }
        ModelsRegistry.add(**model1)

if __name__ == "__main__":
    unittest.main()
