import logging
from typing import Tuple

import pandas as pd

log = logging.getLogger(__name__)

from supervised.exceptions import AutoMLException
from supervised.validation.validator_custom import CustomValidator
from supervised.validation.validator_kfold import KFoldValidator
from supervised.validation.validator_split import SplitValidator


class ValidationStep:
    def __init__(self, params):
        # kfold is default validation technique
        self.validation_type = params.get("validation_type", "kfold")

        if self.validation_type == "kfold":
            self.validator = KFoldValidator(params)
        elif self.validation_type == "split":
            self.validator = SplitValidator(params)
        elif self.validation_type == "custom":
            self.validator = CustomValidator(params)
        else:
            raise AutoMLException(
                f"The validation type ({self.validation_type}) is not implemented."
            )

    def get_split(self, k: int, repeat: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.validator.get_split(k, repeat)

    def split(self):
        return self.validator.split()

    def get_n_splits(self):
        return self.validator.get_n_splits()

    def get_repeats(self) -> int:
        return self.validator.get_repeats()
