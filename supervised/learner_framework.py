
class LearnerFrameworkParametersException(Exception):
    pass

class LearnerFramework():

    def __init__(self, data, train_params, callbacks = None):
        print('LearnerFramework __init__')

        for i in ['model', 'metrics']:
            if i not in train_params:
                msg = 'Missing {0} parameter in train_params'.format(i)
                raise LearnerFrameworkParametersException(msg)
        if data is None or 'train' not in data:
            raise LearnerFrameworkParametersException('Missing training data')

        self.preprocessing = train_params.get('preprocessing')
        self.validation = train_params.get('validation')
        self.metrics = train_params.get('metrics')
        self.model = train_params.get('model')

    def train(self):
        print('--- LearnerFramework start train ---')
