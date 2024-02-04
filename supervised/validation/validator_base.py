import logging

log = logging.getLogger(__name__)


class BaseValidator(object):
    def __init__(self, params: dict):
        self.params: dict = params

    def split(self):
        pass

    def get_n_splits(self) -> int:
        pass

    def get_repeats(self) -> int:
        return 1
