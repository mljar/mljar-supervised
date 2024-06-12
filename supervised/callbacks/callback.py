from typing import List

from supervised.algorithms.algorithm import BaseAlgorithm


class Callback(object):

    def __init__(self, params: dict):
        self.params: dict = params
        self.learners: List[BaseAlgorithm] = []
        self.learner: BaseAlgorithm = None  # current learner
        self.name: str = "callback"

    def add_and_set_learner(self, learner: BaseAlgorithm):
        self.learners += [learner]
        self.learner = learner

    def on_learner_train_start(self, logs: dict) -> None:
        pass

    def on_learner_train_end(self, logs: dict) -> None:
        pass

    def on_iteration_start(self, logs: dict) -> None:
        pass

    def on_iteration_end(self, logs: dict, predictions: dict) -> None:
        pass

    def on_framework_train_end(self, logs: dict) -> None:
        pass
