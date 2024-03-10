from typing import List

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.callbacks.callback import Callback


class CallbackList(object):

    def __init__(self, callbacks: List[Callback] = []):
        self.callbacks: List[Callback] = callbacks

    def add_and_set_learner(self, learner: BaseAlgorithm) -> None:
        for cb in self.callbacks:
            cb.add_and_set_learner(learner)

    def on_learner_train_start(self, logs: dict = None) -> None:
        for cb in self.callbacks:
            cb.on_learner_train_start(logs)

    def on_learner_train_end(self, logs: dict = None) -> None:
        for cb in self.callbacks:
            cb.on_learner_train_end(logs)

    def on_iteration_start(self, logs: dict = None) -> None:
        for cb in self.callbacks:
            cb.on_iteration_start(logs)

    def on_iteration_end(self, logs: dict = None, predictions: dict = None) -> None:
        for cb in self.callbacks:
            cb.on_iteration_end(logs, predictions)

    def on_framework_train_end(self, logs: dict = None) -> None:
        for cb in self.callbacks:
            cb.on_framework_train_end(logs)

    def get(self, callback_name: str) -> Callback:
        for cb in self.callbacks:
            if cb.name == callback_name:
                return cb
        return None
