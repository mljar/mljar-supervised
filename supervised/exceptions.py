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
        logger.error(message)


class NotTrainedException(Exception):
    def __init__(self, message):
        super(NotTrainedException, self).__init__(message)
        logger.debug(message)
