
class CallbackList(object):

    def __init__(self, callbacks = []):
        self.callbacks = callbacks

    def on_training_start(self):
        for cb in callbacks:
            cb.on_training_start()

    def on_training_end(self):
        for cb in callbacks:
            cb.on_training_end()

    def on_iteration_start(self, iter_cnt, data):
        for cb in callbacks:
            cb.on_iteration_start(iter_cnt)

    def on_iteration_end(self, iter_cnt, data):
        for cb in callbacks:
            cb.on_iteration_end(iter_cnt)
