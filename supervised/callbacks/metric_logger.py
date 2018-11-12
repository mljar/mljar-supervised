import logging
log = logging.getLogger(__name__)

import numpy as np

from .callback import Callback
from metric import Metric

class MetricLogger(Callback):

    def __init__(self,params):
        super(MetricLogger, self).__init__(params)
        self.metrics = []
        for metric_name in params.get('metric_names'):
            self.metrics += [Metric({'name': metric_name})]


    def on_iteration_end(self, logs, predictions):
        for m in self.metrics:
            loss = m(predictions.get('y_validation_true'),
                                predictions.get('y_validation_predicted'))
            log.info('Iteration {0} {1}: {2}'.format(logs['iter_cnt'], m.name, loss))
