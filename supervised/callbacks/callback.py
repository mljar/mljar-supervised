class Callback(object):
    def __init__(self, params):
        self.params = params
        self.learners = []
        self.learner = None  # current learner
        self.name = "callback"

    def add_and_set_learner(self, learner):
        self.learners += [learner]
        self.learner = learner

    def on_learner_train_start(self, logs):
        pass

    def on_learner_train_end(self, logs):
        pass

    def on_iteration_start(self, logs):
        pass

    def on_iteration_end(self, logs, predictions):
        pass

    def on_framework_train_end(self, logs):
        pass
