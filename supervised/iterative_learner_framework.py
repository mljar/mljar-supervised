
import numpy as np
import pandas as pd

from learner_framework import LearnerFramework


class IterativeLearnerError(Exception):
    pass


class IterativeLearner(LearnerFramework):
    '''
    Framework for all algorithms that are iterative - the more iterations the more
    accurate learner is on train set, example algorithms (Random Forest, GBM, Neural Networks)
    '''
    def _init(self, validation_schema):
        logger.info('IterativeLearner %s initialize parameters' % self.learner_type)

    def train_cv(self):
        '''
        skf - cross validation indices, for now only: StratifiedKFold
        '''
        logger.info('IterativeLearner %s train_cv method' % self.learner_type)

    def train_val(self):
        logger.info('IterativeLearner %s train with validation set' % self.learner_type)
