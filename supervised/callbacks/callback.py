
class Callback(object):

    def __init__(self, params):
        self.params = params
        self.learner = None

    def add_and_set_learner(self, learner):
        self.learners += [learner]
        self.learner = learner
        
    def on_learner_training_start(self, logs):
        pass

    def on_learner_training_end(self, logs):
        pass

    def on_iteration_start(self, logs):
        pass

    def on_iteration_end(self, logs, predictions):
        pass
