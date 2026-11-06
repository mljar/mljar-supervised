import json
from datetime import date

import numpy as np


class MLJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(
            o,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(o)
        elif isinstance(o, (np.float16, np.float32, np.float64)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, date):
            return o.strftime("%Y-%m-%d")

        return super(MLJSONEncoder, self).default(o)
