import logging
import copy
import numpy as np
import pandas as pd

from models.learner import Learner
import xgboost as xgb
import operator

log = logging.getLogger(__name__)

class XgbLearnerException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)


def reg_log_obj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x =preds-labels
    grad =con*x / (np.abs(x)+con)
    hess =con**2 / (np.abs(x)+con)**2
    return grad, hess

def reg_log_evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))

class XgbLearner(Learner):
    '''
    This is a wrapper over xgboost algorithm.
    '''
    def __init__(self, params):
        Learner.__init__(self, params)
        self.model_base_fname = self.uid + '.xgb.model'
        self.model_fname = '/tmp/' + self.model_base_fname
        self.rounds = 1
        log.debug('XgbLearner __init__')

        self.max_iters = params.get('max_iters', 3)

        self.learner_params = {
            'booster': self.params.get('booster', 'gbtree'),
            'objective': self.params.get('objective'),
            'eval_metric': self.params.get('eval_metric'),
            'eta': self.params.get('eta', 0.01),
            'max_depth':self.params.get('max_depth', 1),
            'min_child_weight':self.params.get('min_child_weight', 1),
            'subsample':self.params.get('subsample', 0.8),
            'colsample_bytree': self.params.get('colsample_bytree', 0.8),
            'silent': self.params.get('silent', 1)
        }
        mandatory_params = {
            'objective': ['binary:logistic'],
            'eval_metric': ['auc', 'logloss']
        }
        for p, v in mandatory_params.items():
            if self.learner_params[p] is None:
                msg = 'Please specify the {0}, should be from {1}'.format(p, v)
                raise XgbLearnerException(msg)


    def update(self, update_params):
        self.rounds = update_params['iters']


    def fit(self, data):
        X = data.get('X')
        y = data.get('y')
        dtrain = xgb.DMatrix(X, label = y, missing = np.NaN)
        self._set_params()
        self.model = xgb.train(self.learner_params, dtrain, self.rounds, xgb_model=self.model)
        log.debug('XgbLearner.fit')

    def predict(self, X):
        dtrain=xgb.DMatrix(X, missing=np.NaN)
        return self.model.predict(dtrain)

    def save(self):
        return 'saved'
        #model_fname = self.save_locally()
        #log.debug('XgbLearner save model to %s' % model_fname)
        #return model_fname

    def load(self, model_path):
        log.debug('Xgboost load model from %s' % model_path)
        self.model = xgb.Booster() #init model
        self.model.load_model(model_path)

    def importance(self, column_names, normalize = True):
        return None
        '''
        # add tmp files here TODO
        tmp_fmap =  '/tmp/xgb.fmap.' + self.uid
        print tmp_fmap
        with open(tmp_fmap, 'w') as fout:
            for i, feat in enumerate(column_names):
                fout.write('{0}\t{1}\tq\n'.format(i, feat.encode('utf-8').strip().replace(' ', '_')))

        if self.fit_params['booster'] == 'gbtree':
            self.model.dump_model('/tmp/' + self.uid + '-xgb.dump',fmap = tmp_fmap, with_stats=True)

        imp = self.model.get_fscore(fmap=tmp_fmap)

        if normalize:
            total = 0.01*float(sum(imp.values())) # in percents
            for k,v in imp.iteritems():
                imp[k] /= total
                imp[k] = round(imp[k],4)

        imp = dict(sorted(imp.items(), key=operator.itemgetter(1), reverse=True))
        return imp
        '''
