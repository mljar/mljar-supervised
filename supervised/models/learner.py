import uuid

class Learner():
    '''
    This is a Learner base class.
    All algorithms inherit from Learner.
    '''
    def __init__(self, params):
        self.params = params
        self.stop_training = False
        self.version = '0.0.0'
        self.model = None
        self.uid = str(uuid.uuid4())

    def update(self, update_params):
        pass
    def fit(self, X, y, sample_weight = None):
        pass
    def predict(self, X):
        pass
    def copy(self):
        pass
    def save(self):
        pass
    def load(self, model_path):
        pass
    def importance(self, column_names, normalize = True):
        pass
