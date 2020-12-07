import logging

log = logging.getLogger(__name__)


class BaseValidator(object):
    def __init__(self, params):
        self.params = params

    def split(self):
        pass

    def get_n_splits(self):
        pass

    def get_repeats(self):
        return 1
