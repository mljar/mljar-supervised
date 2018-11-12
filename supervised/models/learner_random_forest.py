import logging
from supervised.models.learner import Learner
from sklearn.externals import joblib
import copy
logger = logging.getLogger(__name__)

import sklearn
from sklearn.ensemble import RandomForestClassifier
from supervised.models.learner_sklearn import SklearnTreesClassifierLearner

class RandomForestLearner(SklearnTreesClassifierLearner):

    def __init__(self, params):
        super(RandomForestLearner, self).__init__(params)

        self.library_version = sklearn.__version__
        self.algorithm_name = 'Random Forest'
        self.algorithm_short_name = 'RF'
        self.model_file = self.uid + '.rf.model'
        self.model_file_path = '/tmp/' + self.model_file

        self.trees_in_step = params.get('trees_in_step', 10)
        self.model = RandomForestClassifier(n_estimators = self.trees_in_step,
                                criterion = params.get('criterion', 'gini'),
                                max_features = params.get('max_features', 0.8),
                                min_samples_split = params.get('min_samples_split', 4),
                                min_samples_leaf= params.get('min_samples_leaf', 4),
                                warm_start=True,
                                n_jobs=-1,
                                random_state=params.get('random_seed', 1706))
        logger.debug('RandomForestLearner __init__')
