import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from supervised.models.registry import ModelsRegistry


class ModelsRegistryTest(unittest.TestCase):
    def test_add_to_registry(self):
        class Model1:
            pass

        model1 = {
            "task_name": "binary_classification",
            "model_name": "Model 1",
            "model_code": "M 1",
            "model_class": Model1,
        }
        ModelsRegistry(**model1)
        
