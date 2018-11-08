
class Callback(object):

    def __init__(self, learner, params):
        self.learner = learner
        self.params = params

    def on_training_start(self):
        pass

    def on_training_end(self):
        pass

    def on_iteration_start(self, iter_cnt, data):
        pass

    def on_iteration_end(self, iter_cnt, data):
        pass
