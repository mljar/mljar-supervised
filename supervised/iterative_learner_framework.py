
import numpy as np
import pandas as pd
import zipfile
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

            log.debug('Train data, X: {0} y: {1}'.format(train_data.get('X').shape,
                                                            train_data.get('y').shape))
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
                                                self.predictions(learner,
                                                    train_data, validation_data))
                if learner.stop_training: break
            # end of learner iters loop
            self.callbacks.on_learner_train_end()
        # end of validation loop

    def predict(self, X):
        # run predict on all learners and return the average
        y_predicted = np.zeros((X.shape[0],))
        for learner in self.learners:
            print('learner->', learner.algorithm_short_name)
            y_predicted += learner.predict(X)
        return y_predicted / float(len(self.learners))

    def save(self):
        learner_file_paths = []
        for learner in self.learners:
            learner_file_paths += [learner.save()]

        zf = zipfile.ZipFile(self.framwork_file_path, mode='w')
        try:
            for lf in learner_file_paths:
                zf.write(lf)
        finally:
            zf.close()
        desc = {
            'file_path': self.framwork_file_path,
            'learner_type': self.learners[0].algorithm_short_name,
            'library_version': self.learners[0].library_version,
            'model_uids': [i.uid for i in self.learners]
        }
        return desc

    def load(self, json_desc):

        with zipfile.ZipFile(json_desc.get('file_path'), 'r') as zip_ref:
            zip_ref.extractall('/tmp')
