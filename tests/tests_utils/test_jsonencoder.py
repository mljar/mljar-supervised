import json
import unittest
from datetime import date, datetime

import numpy as np

from supervised.utils.jsonencoder import MLJSONEncoder


class MLJSONEncoderTest(unittest.TestCase):
    def dumps(self, value):
        return json.loads(json.dumps(value, cls=MLJSONEncoder))

    def test_numpy_integers(self):
        self.assertEqual(self.dumps({"n": np.int64(3), "u": np.uint8(7)}), {"n": 3, "u": 7})

    def test_numpy_floats(self):
        out = self.dumps({"a": np.float32(0.5), "b": np.float64(1.25), "c": np.float16(2.0)})
        self.assertEqual(out, {"a": 0.5, "b": 1.25, "c": 2.0})

    def test_numpy_array(self):
        self.assertEqual(self.dumps(np.array([1, 2, 3])), [1, 2, 3])

    def test_dates(self):
        self.assertEqual(self.dumps({"d": date(2024, 5, 17)}), {"d": "2024-05-17"})
        self.assertEqual(self.dumps({"d": datetime(2024, 5, 17, 13, 45)}), {"d": "2024-05-17"})

    def test_unknown_type_still_raises(self):
        with self.assertRaises(TypeError):
            json.dumps({"x": object()}, cls=MLJSONEncoder)
