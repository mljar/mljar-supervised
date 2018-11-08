
class LearnerFactory(object):

    @staticmethod
    def get_learner(params):
        learner_type = params.get('learner_type', 'xgb')
        if learner_type == 'xbg':
            return XgbLearner(params)
        else:
            msg = 'Learner not defined'
            raise ValueError(msg)
