import logging
log = logging.getLogger(__name__)

import numpy as np

from callback import Callback
from metric import Metric

class MetricLogger(Callback):

    def __init__(self, learner, params):
        super(MetricLogger, self).__init__(learner, params, data)
        self.metrics = []
        self.loss_values = {}
        for metric_name in params.get('metric_names'):
            self.metrics += [Metric({'metric_name': metric_name})]
            self.loss_values[metric_name] = {'train': [],
                                            'validation': [],
                                            'iter_cnts': []}

    def on_iteration_end(self, iter_cnt, data):
        for m in self.metrics:
            self.loss_values[m.metric_name]['iter_cnts'] += [iter_cnt]
            for t in ['train', 'validation']:
                loss = m(data.get('y_{0}_true'.format(t)),
                            data.get('y_{0}_predicted'.format(t)))
                self.loss_values[m.metric_name][t] += [loss]
