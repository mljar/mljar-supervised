import time
import logging
from .callback import Callback
log = logging.getLogger(__name__)

class TimeConstraint(Callback):

    def __init__(self, learner, params):
        super(TimeConstraint, self).__init__(learner, params)
        self.train_time_limit = params.get('train_time_limit', 60) # in seconds

    def on_training_start(self):
        self.train_start_time = time.time()

    def on_iteration_start(self, iter_cnt, data):
        self.iter_start_time = time.time()

    def on_iteration_end(self, iter_cnt, data):
        log.debug('Iteration took {0} seconds'.format(self.iter_start_time-time.time()))
        if time.time() - self.train_start_time > self.train_time_limit:
            self.learner.stop_training = True
            log.info('Terminating learning, time limit reached')
