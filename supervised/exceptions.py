import logging

from supervised.utils.config import LOG_LEVEL

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.ERROR
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class AutoMLException(Exception):
    def __init__(self, message):
        super(AutoMLException, self).__init__(message)


class NativeCompatibilityError(AutoMLException):
    """Native execution was declined before risking the caller's process."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


class CandidateValidationError(AutoMLException):
    """A candidate configuration cannot produce a usable fit on this fold."""

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


class ResourceAdmissionError(AutoMLException):
    """A candidate exceeded an estimated budget before allocating its model.

    Actual allocation failures remain ordinary errors. Prediction callers also
    receive this exception: a trained model is never silently replaced.
    """

    def __init__(self, message, resources=None):
        super().__init__(message)
        self.resources = dict(resources or {})


class NotTrainedException(Exception):
    def __init__(self, message):
        super(NotTrainedException, self).__init__(message)
        logger.debug(message)


class BudgetExceededError(AutoMLException, NotTrainedException):
    """An explicit training deadline or completion estimate stopped a trial."""

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details
