import logging

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.ERROR)
import uuid
import os

from supervised.config import storage_path

log = logging.getLogger(__name__)

from supervised.validation.validation_step import ValidationStep


from supervised.callbacks.early_stopping import EarlyStopping
from supervised.callbacks.time_constraint import TimeConstraint


from supervised.callbacks.callback_list import CallbackList


class LearnerFrameworkParametersException(Exception):
    pass


class LearnerFramework:
    def __init__(self, params, callbacks=[]):
        log.debug("LearnerFramework __init__")
        self.uid = str(uuid.uuid4())

        self.framework_file = self.uid + ".framework"
        self.framework_file_path = os.path.join(storage_path, self.framework_file)

        for i in ["learner", "validation"]:  # mandatory parameters
            if i not in params:
                msg = "Missing {0} parameter in LearnerFramework params".format(i)
                log.error(msg)
                raise ValueError(msg)

        self.params = params
        self.callbacks = CallbackList(callbacks)

        self.additional_params = params.get("additional")
        self.preprocessing_params = params.get("preprocessing")
        self.validation_params = params.get("validation")
        self.learner_params = params.get("learner")

        self.validation = None
        self.preprocessings = []
        self.learners = []

    def get_params_key(self):
        key = "key_"
        for main_key in ["additional", "preprocessing", "validation", "learner"]:
            key += main_key
            for k, v in self.params[main_key].items():
                key += "_{}_{}".format(k, v)
        return key

    def train(self, data):
        pass

    def predict(self, X):
        pass

    def to_json(self):
        pass

    def from_json(self, json_desc):
        pass
