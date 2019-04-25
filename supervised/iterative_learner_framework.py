import numpy as np
import pandas as pd
import time
import zipfile
import os

from supervised.config import storage_path
from supervised.learner_framework import LearnerFramework
from supervised.validation.validation_step import ValidationStep
from supervised.models.learner_factory import LearnerFactory

from supervised.preprocessing.preprocessing_step import PreprocessingStep
import logging

log = logging.getLogger(__name__)
from supervised.preprocessing.preprocessing_exclude_missing import (
    PreprocessingExcludeMissingValues,
)


class IterativeLearnerException(Exception):
    def __init__(self, message):
        super(IterativeLearnerException, self).__init__(message)
        log.error(message)


class IterativeLearner(LearnerFramework):
    def __init__(self, params, callbacks=[]):
        LearnerFramework.__init__(self, params, callbacks)
        self.train_time = None
        log.debug("IterativeLearner.__init__")

    def get_train_time(self):
        return self.train_time

    def predictions(self, learner, train_data, validation_data):
        return {
            "y_train_true": train_data.get("y"),
            "y_train_predicted": learner.predict(train_data.get("X")),
            "y_validation_true": validation_data.get("y"),
            "y_validation_predicted": learner.predict(validation_data.get("X")),
            "validation_index": validation_data.get("X").index,
        }

    def train(self, data):
        start_time = time.time()
        log.debug("IterativeLearner.train")
        np.random.seed(self.learner_params["seed"])
        data = PreprocessingExcludeMissingValues.remove_rows_without_target(data)
        self.validation = ValidationStep(self.validation_params, data)

        for train_data, validation_data in self.validation.split():
            # the proprocessing is done at every validation step
            self.preprocessings += [PreprocessingStep(self.preprocessing_params)]
            train_data, _ = self.preprocessings[-1].run(
                train_data
            )
            validation_data = self.preprocessings[-1].transform(
                validation_data
            )


            self.learners += [LearnerFactory.get_learner(self.learner_params)]
            learner = self.learners[-1]

            self.callbacks.add_and_set_learner(learner)
            self.callbacks.on_learner_train_start()

            for i in range(learner.max_iters):
                self.callbacks.on_iteration_start()
                learner.fit(train_data.get("X"), train_data.get("y"))
                # do a target postprocessing here
                self.callbacks.on_iteration_end(
                    {"iter_cnt": i},
                    self.predictions(learner, train_data, validation_data),
                )
                if learner.stop_training:
                    break
                learner.update({"step": i})
            # end of learner iters loop
            self.callbacks.on_learner_train_end()
        # end of validation loop
        self.callbacks.on_framework_train_end()
        self.train_time = time.time() - start_time

    def get_out_of_folds(self):
        early_stopping = self.callbacks.get("early_stopping")
        if early_stopping is None:
            return None
        return early_stopping.best_y_oof

    def get_final_loss(self):
        early_stopping = self.callbacks.get("early_stopping")
        if early_stopping is None:
            return None
        return early_stopping.final_loss

    def get_metric_logs(self):
        metric_logger = self.callbacks.get("metric_logger")
        if metric_logger is None:
            return None
        return metric_logger.loss_values

    def get_name(self):
        return self.learner_params.get("model_type")

    def predict(self, X):
        if self.learners is None or len(self.learners) == 0:
            raise IterativeLearnerException("Learnes are not initialized")
        # run predict on all learners and return the average
        y_predicted = np.zeros((X.shape[0],))
        for ind, learner in enumerate(self.learners):
            # preprocessing goes here
            validation_data = self.preprocessings[ind].transform({"X": X})
            y_predicted += learner.predict(validation_data.get("X"))
        y_predicted_average = y_predicted / float(len(self.learners))
        # get first preprocessing and reverse transform target
        # we can use the preprocessing of the first model, because for target they are all the same
        y_predicted_final = self.preprocessings[0].reverse_transform_target(
            y_predicted_average
        )
        return y_predicted_final

    def to_json(self):
        preprocessing = []
        for p in self.preprocessings:
            preprocessing += [p.to_json()]

        learners_desc = []
        for learner in self.learners:
            learners_desc += [learner.save()]

        zf = zipfile.ZipFile(self.framework_file_path, mode="w")
        try:
            for lf in learners_desc:
                zf.write(lf["model_file_path"])
        finally:
            zf.close()
        desc = {
            "uid": self.uid,
            "algorithm_short_name": self.get_name(),
            "framework_file": self.framework_file,
            "framework_file_path": self.framework_file_path,
            "preprocessing": preprocessing,
            "learners": learners_desc,
            "params": self.params,  # this is needed while constructing new Iterative Learner Framework
        }
        return desc

    def from_json(self, json_desc):
        self.uid = json_desc.get("uid", self.uid)
        self.framework_file = json_desc.get("framework_file", self.framework_file)
        self.framework_file_path = json_desc.get(
            "framework_file_path", self.framework_file_path
        )

        with zipfile.ZipFile(json_desc.get("framework_file_path"), "r") as zip_ref:
            zip_ref.extractall(storage_path)
        self.learners = []
        for learner_desc in json_desc.get("learners"):
            self.learners += [LearnerFactory.load(learner_desc)]
        preprocessing = json_desc.get("preprocessing", [])

        for p in preprocessing:
            preproc = PreprocessingStep()
            preproc.from_json(p)
            self.preprocessings += [preproc]
