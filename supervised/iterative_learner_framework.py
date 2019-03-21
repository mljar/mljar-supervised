import numpy as np
import pandas as pd
import zipfile
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
        log.debug("IterativeLearner __init__")

    def predictions(self, learner, train_data, validation_data):
        return {
            "y_train_true": train_data.get("y"),
            "y_train_predicted": learner.predict(train_data.get("X")),
            "y_validation_true": validation_data.get("y"),
            "y_validation_predicted": learner.predict(validation_data.get("X")),
        }

    def train(self, data):

        data = PreprocessingExcludeMissingValues.remove_rows_without_target(data)
        self.validation = ValidationStep(self.validation_params, data)

        for train_data, validation_data in self.validation.split():

            for d in [train_data, validation_data]:
                log.debug(
                    "Data, X: {0} y: {1}".format(d.get("X").shape, d.get("y").shape)
                )
            print("before preprocessing", train_data["X"].head())
            # the proprocessing is done at every validation step
            self.preprocessings += [PreprocessingStep(self.preprocessing_params)]
            train_data, validation_data = self.preprocessings[-1].run(
                train_data, validation_data
            )
            print("after preprocessing", train_data["X"].head())

            self.learners += [LearnerFactory.get_learner(self.learner_params)]
            learner = self.learners[-1]

            self.callbacks.add_and_set_learner(learner)
            self.callbacks.on_learner_train_start()

            for i in range(learner.max_iters):
                self.callbacks.on_iteration_start()
                learner.fit(train_data)
                # do a target postprocessing here
                self.callbacks.on_iteration_end(
                    {"iter_cnt": i},
                    self.predictions(learner, train_data, validation_data),
                )
                if learner.stop_training:
                    break
            # end of learner iters loop
            self.callbacks.on_learner_train_end()
        # end of validation loop

    def predict(self, X):
        if self.learners is None or len(self.learners) == 0:
            raise IterativeLearnerException("Learnes are not initialized")
        # run predict on all learners and return the average
        y_predicted = np.zeros((X.shape[0],))
        for ind, learner in enumerate(self.learners):
            # preprocessing goes here
            validation_data = self.preprocessings[ind].transform({"X":X})
            y_predicted += learner.predict(validation_data.get("X"))
        return y_predicted / float(len(self.learners))

    def save(self):
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
            "framework_file": self.framework_file,
            "framework_file_path": self.framework_file_path,
            "preprocessing": preprocessing,
            "learners": learners_desc,
        }
        return desc

    def load(self, json_desc):
        self.uid = json_desc.get("uid", self.uid)
        self.framework_file = json_desc.get("framework_file", self.framework_file)
        self.framework_file_path = json_desc.get(
            "framework_file_path", self.framework_file_path
        )

        destination_dir = "/tmp"
        with zipfile.ZipFile(json_desc.get("framework_file_path"), "r") as zip_ref:
            zip_ref.extractall("/tmp")
        self.learners = []
        for learner_desc in json_desc.get("learners"):
            self.learners += [LearnerFactory.load(learner_desc)]
        preprocessing = json_desc.get("preprocessing", [])

        for p in preprocessing:
            preproc = PreprocessingStep()
            preproc.from_json(p)
            self.preprocessings += [preproc]
