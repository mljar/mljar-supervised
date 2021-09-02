import logging

LOG_LEVEL = logging.ERROR

import numpy as np

# from guppy import hpy
# from pympler import summary
# from pympler import muppy
import time


def mem(msg=""):
    """Memory usage in MB"""

    time.sleep(5)

    with open("/proc/self/status") as f:
        memusage = f.read().split("VmRSS:")[1].split("\n")[0][:-3]

    print(msg, "- memory:", np.round(float(memusage.strip()) / 1024.0), "MB")

    # all_objects = muppy.get_objects()
    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)
