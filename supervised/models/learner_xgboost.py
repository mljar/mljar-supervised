import logging
import copy
import numpy as np
import pandas as pd

from supervised.models.learner import Learner
import xgboost as xgb
import operator

log = logging.getLogger(__name__)


class XgbLearnerException(Exception):
    def __init__(self, message):
        super(XgbLearnerException, self).__init__(message)
        log.error(message)


class XgbLearner(Learner):
    """
    This is a wrapper over xgboost algorithm.
    """

    def __init__(self, params):
        super(XgbLearner, self).__init__(params)
        self.library_version = xgb.__version__
        self.algorithm_name = "Extreme Gradient Boosting"
        self.algorithm_short_name = "Xgboost"
        self.model_file = self.uid + ".xgb.model"
        self.model_file_path = "/tmp/" + self.model_file

        self.boosting_rounds = params.get("boosting_rounds", 50)
        self.max_iters = params.get("max_iters", 3)
        self.learner_params = {
            "booster": self.params.get("booster", "gbtree"),
            "objective": self.params.get("objective"),
            "eval_metric": self.params.get("eval_metric"),
            "eta": self.params.get("eta", 0.01),
            "max_depth": self.params.get("max_depth", 1),
            "min_child_weight": self.params.get("min_child_weight", 1),
            "subsample": self.params.get("subsample", 0.8),
            "colsample_bytree": self.params.get("colsample_bytree", 0.8),
            "silent": self.params.get("silent", 1),
        }
        mandatory_params = {
            "objective": ["binary:logistic"],
            "eval_metric": ["auc", "logloss"],
        }
        for p, v in mandatory_params.items():
            if self.learner_params[p] is None:
                msg = "Please specify the {0}, it should be one from {1}".format(p, v)
                raise XgbLearnerException(msg)
        log.debug("XgbLearner __init__")

    def update(self, update_params):
        # Dont need to update boosting rounds, it is adding rounds incrementally
        pass

    def fit(self, data):
        log.debug("XgbLearner.fit")
        X = data.get("X")
        y = data.get("y")
        # print('rounds', self.boosting_rounds)
        # print('model', self.model)
        dtrain = xgb.DMatrix(X, label=y, missing=np.NaN)
        self.model = xgb.train(
            self.learner_params, dtrain, self.boosting_rounds, xgb_model=self.model
        )

    def predict(self, X):
        if self.model is None:
            raise XgbLearnerException("Xgboost model is None")
        dtrain = xgb.DMatrix(X, missing=np.NaN)
        return self.model.predict(dtrain)

    def copy(self):
        return copy.deepcopy(self)

    def save(self):
        self.model.save_model(self.model_file_path)

        json_desc = {
            "library_version": self.library_version,
            "algorithm_name": self.algorithm_name,
            "algorithm_short_name": self.algorithm_short_name,
            "uid": self.uid,
            "model_file": self.model_file,
            "model_file_path": self.model_file_path,
            "params": self.params,
        }

        log.debug("XgbLearner save model to %s" % self.model_file_path)
        return json_desc

    def load(self, json_desc):

        self.library_version = json_desc.get("library_version", self.library_version)
        self.algorithm_name = json_desc.get("algorithm_name", self.algorithm_name)
        self.algorithm_short_name = json_desc.get(
            "algorithm_short_name", self.algorithm_short_name
        )
        self.uid = json_desc.get("uid", self.uid)
        self.model_file = json_desc.get("model_file", self.model_file)
        self.model_file_path = json_desc.get("model_file_path", self.model_file_path)
        self.params = json_desc.get("params", self.params)

        log.debug("XgbLearner load model from %s" % self.model_file_path)
        self.model = xgb.Booster()  # init model
        self.model.load_model(self.model_file_path)

    def importance(self, column_names, normalize=True):
        return None
        """
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
        """


# For binary classification target should be 0, 1. There should be no NaNs in target.
