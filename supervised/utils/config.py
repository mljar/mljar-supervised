import logging

LOG_LEVEL = logging.ERROR

import numpy as np


def mem():
    """ Memory usage in MB """

    with open("/proc/self/status") as f:
        memusage = f.read().split("VmRSS:")[1].split("\n")[0][:-3]

    print("Memory:", np.round(float(memusage.strip()) / 1024.0), "MB")
