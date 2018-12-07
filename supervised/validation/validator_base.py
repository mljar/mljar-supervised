import logging

log = logging.getLogger(__name__)


class BaseValidatorException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)


class BaseValidator(object):
    def __init__(self, params, data):
        self.data = data
        self.params = params
        self.validate()

    def validate(self):
        if self.data.get("train") is None:
            msg = "Missing train data"
            raise BaseValidatorException(msg)
        for i in ["X", "y"]:
            if self.data["train"].get(i) is None:
                msg = "Missing {0} in train data".format(i)
                raise BaseValidatorException(msg)

    def split(self):
        pass

    def get_n_splits(self):
        pass
