import uuid
import numpy as np
import pandas as pd
import time
import zipfile
import os
import logging
import json

from supervised.callbacks.callback_list import CallbackList
from supervised.validation.validation_step import ValidationStep
from supervised.algorithms.algorithm_factory import AlgorithmFactory
from supervised.preprocessing.preprocessing_step import PreprocessingStep
from supervised.preprocessing.preprocessing_exclude_missing import (
    PreprocessingExcludeMissingValues,
)

from supervised.exceptions import AutoMLException
from supervised.utils.config import storage_path
from supervised.utils.config import LOG_LEVEL
from supervised.utils.additional_metrics import AdditionalMetrics

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class ModelFramework:
    def __init__(self, params, callbacks=[]):
        logger.debug("ModelFramework.__init__")
        self.uid = str(uuid.uuid4())

        for i in ["learner", "validation"]:  # mandatory parameters
            if i not in params:
                msg = "Missing {0} parameter in ModelFramework params".format(i)
                logger.error(msg)
                raise ValueError(msg)

        self.params = params
        self.callbacks = CallbackList(callbacks)

        self._name = params.get("name", "model")
        self.additional_params = params.get("additional")
        self.preprocessing_params = params.get("preprocessing")
        self.validation_params = params.get("validation")
        self.learner_params = params.get("learner")

        self._ml_task = params.get("ml_task")

        self.validation = None
        self.preprocessings = []
        self.learners = []

        self.train_time = None
        self._additional_metrics = None
        self._threshold = None # used only for binary classifiers

    def get_train_time(self):
        return self.train_time

    def predictions(self, learner, X_train, y_train, X_validation, y_validation):

        y_train_true = y_train
        y_train_predicted = learner.predict(X_train)
        y_validation_true = y_validation
        y_validation_predicted = learner.predict(X_validation)

        if self.preprocessings[-1]._scale_y is not None:
            y_train_true = self.preprocessings[-1].inverse_scale_target(y_train_true)
            y_train_predicted = self.preprocessings[-1].inverse_scale_target(
                y_train_predicted
            )
            y_validation_true = self.preprocessings[-1].inverse_scale_target(
                y_validation_true
            )
            y_validation_predicted = self.preprocessings[-1].inverse_scale_target(
                y_validation_predicted
            )

        return {
            "y_train_true": y_train_true,
            "y_train_predicted": y_train_predicted,
            "y_validation_true": y_validation_true,
            "y_validation_predicted": y_validation_predicted,
            "validation_index": X_validation.index,
        }

    def train(self, data):
        logger.debug("ModelFramework.train")
        start_time = time.time()
        np.random.seed(self.learner_params["seed"])

        # data = PreprocessingExcludeMissingValues.remove_rows_without_target(data)

        self.validation = ValidationStep(self.validation_params, data)

        for train_data, validation_data in self.validation.split():
            logger.debug("-" * 51)
            logger.debug(
                "Data split, train X:{} y:{}, validation X:{}, y:{}".format(
                    train_data["X"].shape,
                    train_data["y"].shape,
                    validation_data["X"].shape,
                    validation_data["y"].shape,
                )
            )
            # the proprocessing is done at every validation step
            self.preprocessings += [PreprocessingStep(self.preprocessing_params)]

            X_train, y_train = self.preprocessings[-1].fit_and_transform(
                train_data["X"], train_data["y"]
            )
            X_validation, y_validation = self.preprocessings[-1].transform(
                validation_data["X"], validation_data["y"]
            )

            self.learners += [AlgorithmFactory.get_algorithm(self.learner_params)]
            learner = self.learners[-1]

            self.callbacks.add_and_set_learner(learner)
            self.callbacks.on_learner_train_start()

            for i in range(learner.max_iters):
                self.callbacks.on_iteration_start()
                learner.fit(X_train, y_train)
                # do a target postprocessing here
                self.callbacks.on_iteration_end(
                    {"iter_cnt": i},
                    self.predictions(
                        learner, X_train, y_train, X_validation, y_validation
                    ),
                )
                if learner.stop_training:
                    break
                learner.update({"step": i})
            # end of learner iters loop
            self.callbacks.on_learner_train_end()
        # end of validation loop
        self.callbacks.on_framework_train_end()
        self.train_time = time.time() - start_time
        self.get_additional_metrics()
        logger.debug("ModelFramework end of training")

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

    def get_type(self):
        return self.learner_params.get("model_type")

    def get_name(self):
        return self._name

    def predict(self, X):
        logger.debug("ModelFramework.predict")
        if self.learners is None or len(self.learners) == 0:
            raise Exception("Learnes are not initialized")
        # run predict on all learners and return the average
        y_predicted = None  # np.zeros((X.shape[0],))
        for ind, learner in enumerate(self.learners):
            # preprocessing goes here
            X_data, _ = self.preprocessings[ind].transform(X, None)
            y_p = learner.predict(X_data)
            y_predicted = y_p if y_predicted is None else y_predicted + y_p
            # y_predicted += learner.predict(validation_data.get("X"))
        y_predicted_average = y_predicted / float(len(self.learners))
        # get first preprocessing and reverse transform target
        # we can use the preprocessing of the first model, because for target they are all the same
        y_predicted_final = self.preprocessings[0].reverse_transform_target(
            y_predicted_average
        )
        return y_predicted_final
    '''
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
            "algorithm_short_name": self.get_type(),
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
            self.learners += [AlgorithmFactory.load(learner_desc)]
        preprocessing = json_desc.get("preprocessing", [])

        for p in preprocessing:
            preproc = PreprocessingStep()
            preproc.from_json(p)
            self.preprocessings += [preproc]
    '''
    def get_additional_metrics(self):
        if self._additional_metrics is None:
            # 'target' - the target after processing used for model training
            # 'prediction' - out of folds predictions of the model
            oof_predictions = self.get_out_of_folds()
            prediction_cols = [c for c in oof_predictions.columns if "prediction" in c]
            target_cols = [c for c in oof_predictions.columns if "target" in c]
            self._additional_metrics = AdditionalMetrics.compute(
                oof_predictions[target_cols],
                oof_predictions[prediction_cols],
                self._ml_task,
            )
            if self._ml_task == BINARY_CLASSIFICATION:
                self._threshold = float(self._additional_metrics["threshold"])
        return self._additional_metrics

    def save(self, model_path):
        logger.info(f"Save the model {model_path}")

        saved = []
        for i, l in enumerate(self.learners):
            p = os.path.join(model_path, f"learner_{i+1}.{l.file_extenstion()}")
            l.save(p)
            saved += [p]

        with open(os.path.join(model_path, "framework.json"), "w") as fout:
            preprocessing = [p.to_json() for p in self.preprocessings]
            learners_params = [learner.get_params() for learner in self.learners]
            desc = {
                "uid": self.uid,
                "name": self._name,
                "preprocessing": preprocessing,
                "learners": learners_params,
                "params": self.params,
                "saved": saved
            }
            if self._threshold is not None:
                desc["threshold"] = self._threshold
            fout.write(json.dumps(desc, indent=4))

        

        type_of_predictions = (
            "validation" if "k_folds" not in self.validation_params else "out_of_folds"
        )
        predictions = self.get_out_of_folds()
        predictions.to_csv(
            os.path.join(model_path, f"predictions_{type_of_predictions}.csv"),
            index=False,
        )

        self._additional_metrics = self.get_additional_metrics()

        with open(os.path.join(model_path, "metrics.txt"), "w") as fout:
            if self._ml_task == BINARY_CLASSIFICATION:
                max_metrics = self._additional_metrics["max_metrics"]
                confusion_matrix = self._additional_metrics["confusion_matrix"]
                threshold = self._additional_metrics["threshold"]

                fout.write("Metric details:\n{}\n\n".format(max_metrics.transpose()))
                fout.write(
                    "Confusion matrix (at threshold={}):\n{}".format(
                        np.round(threshold, 6), confusion_matrix
                    )
                )
            elif self._ml_task == MULTICLASS_CLASSIFICATION:
                fout.write("TODO")

        with open(os.path.join(model_path, "status.txt"), "w") as fout:
            fout.write("ALL OK!")

    @staticmethod
    def load(model_path):
        logger.info(f"Loading model framework from {model_path}")

        json_desc = json.load(open(os.path.join(model_path, "framework.json")))
        mf = ModelFramework(json_desc["params"])
        mf.uid = json_desc.get("uid", mf.uid)
        mf._name = json_desc.get("name", mf._name)
        mf._threshold = json_desc.get("threshold")
        mf.learners = []
        for learner_desc, learner_path in zip(json_desc.get("learners"), json_desc.get("saved")):
            
            l = AlgorithmFactory.load(learner_desc, learner_path)     
            mf.learners += [l]
        
        mf.preprocessings = []
        for p in json_desc.get("preprocessing"):
            ps = PreprocessingStep()
            ps.from_json(p) 
            mf.preprocessings += [ps]

        return mf