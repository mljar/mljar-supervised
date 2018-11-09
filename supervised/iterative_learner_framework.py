
import numpy as np
import pandas as pd

from learner_framework import LearnerFramework
from validation.validation_step import ValidationStep
from models.learner_factory import LearnerFactory

import logging
log = logging.getLogger(__name__)

class IterativeLearnerError(Exception):
    pass


class IterativeLearner(LearnerFramework):

    def __init__(self, params, callbacks = []):
        LearnerFramework.__init__(self, params, callbacks)
        log.debug('IterativeLearner __init__')

    def predictions(self, learner, train_data, validation_data):
        return {
            'y_train_true': train_data.get('y'),
            'y_train_predicted': learner.predict(train_data.get('X')),
            'y_validation_true': validation_data.get('y'),
            'y_validation_predicted': learner.predict(validation_data.get('X')),
        }

    def train(self, data):

        self.validation = ValidationStep(self.validation_params, data)

        for train_data, validation_data in self.validation.split():

            #self.preprocessings += [PreprocessingStep(self.preprocessing_params)]
            #self.preprocessings[-1].fit_and_transform(train_data, validation_data)

            self.learners += [LearnerFactory.get_learner(self.learner_params)]
            learner = self.learners[-1]

            self.callbacks.add_and_set_learner(learner)
            self.callbacks.on_learner_train_start()

            for i in range(learner.max_iters):
                self.callbacks.on_iteration_start()
                learner.fit(train_data)
                self.callbacks.on_iteration_end({'iter_cnt': i},
                        self.predictions(learner, train_data, validation_data))
                if learner.stop_training: break
            # end of learner iters loop
            self.callbacks.on_learner_train_end()
        # end of validation loop
