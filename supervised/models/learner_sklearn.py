import logging
from supervised.models.learner import Learner
from sklearn.externals import joblib
import copy
logger = logging.getLogger(__name__)

class SklearnLearner(Learner):

    def __init__(self, params):
        super(SklearnLearner, self).__init__(params)

    def fit(self, data):
        X = data.get('X')
        y = data.get('y')
        self.model.fit(X, y)

    def copy(self):
        return copy.deepcopy(self)

    def save(self):
        joblib.dump(self.model, self.model_file_path, compress=True)
        logger.debug('SklearnLearner save to {0}'.format(self.model_file_path))
        return self.model_file_path

    def load(self, model_path):
        logger.debug('SklearnLearner loading model from {0}'.format(model_path))
        self.model = joblib.load(model_path)


class SklearnTreesClassifierLearner(SklearnLearner):

    def __init__(self, params):
        SklearnLearner.__init__(self, params)

    def fit(self, data):
        X = data.get('X')
        y = data.get('y')
        self.model.fit(X, y)
        self.model.n_estimators += self.trees_in_step

    def predict(self, X):
        return self.model.predict_proba(X)[:,1]
