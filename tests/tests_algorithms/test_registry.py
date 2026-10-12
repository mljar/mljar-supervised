import unittest
from unittest.mock import patch

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
            "default_params": {},
        }
        # Do not leak the synthetic algorithm into subsequent registry-wide tests.
        with patch.dict(AlgorithmsRegistry.registry["binary_classification"]):
            AlgorithmsRegistry.add(**model1)


if __name__ == "__main__":
    unittest.main()
