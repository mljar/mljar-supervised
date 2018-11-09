import logging
import copy
import numpy as np
import pandas as pd
from .learner import Learner
import xgboost as xgb
import operator

log = logging.getLogger(__name__)


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
        self.max_iters = params.get('max_iters', 3)
        log.debug('XgbLearner __init__')

    def update(self, update_params):
        self.rounds = update_params['iters']

    def _set_params(self):
        pass
        '''
        self.params = {
            'booster': self.fit_params['booster'],
            "objective": self.fit_params['objective'],
            "eval_metric": self.fit_params['eval_metric'],
            'eta':self.fit_params['eta'],
            'max_depth':self.fit_params['max_depth'],
            'min_child_weight':self.fit_params['min_child_weight'],
            'subsample':self.fit_params['subsample'],
            'colsample_bytree':self.fit_params['colsample_bytree'],
            'silent':1
        }
        '''

    def fit(self, data):
        X = data.get('X')
        y = data.get('y')
        dtrain = xgb.DMatrix(X, label = y, missing = np.NaN)
        self._set_params()
        self.model = xgb.train(self.params, dtrain, self.rounds, xgb_model=self.model)
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
