import sys, time
import csv
import os
import cPickle
import numpy as np
import pandas as pd

from models.learner_xgboost import XgbLearner
from iterative_learner_framework import IterativeLearner

from learner_factory import LearnerFactory

class LearnerFrameworkFactoryError(Exception):
    def __init__(self, message):
        super(LearnerFrameworkFactoryError, self).__init__(message)
        logger.error(message)

class LearnerFrameworkFactory():
    @staticmethod
    def create_framework(job_params):
        learner_type = job_params['model_type']

        elif learner_type in ['xgb', 'xgbr']:
            return IterativeLearner(job_params, learner = XgbLearner)

        raise LearnerFrameworkFactoryError('Learner Framework not constructed, \
                                        learner %s not defiend' % learner_type)
