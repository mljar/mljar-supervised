import tempfile

# this set a storage to temp dir
# you can overwrite this by setting your own dir
storage_path = tempfile.gettempdir()

import logging

LOG_LEVEL = logging.DEBUG

import numpy as np
def mem():
    ''' Memory usage in MB '''

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

    print("Memory:", np.round(float(memusage.strip())/1024.0), "MB")