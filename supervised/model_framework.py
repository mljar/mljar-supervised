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
from supervised.algorithms.factory import AlgorithmFactory
from supervised.preprocessing.preprocessing import Preprocessing
from supervised.preprocessing.exclude_missing_target import ExcludeRowsMissingTarget

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

from supervised.utils.config import mem
from supervised.utils.learning_curves import LearningCurves


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
        self._threshold = None  # used only for binary classifiers

    def get_train_time(self):
        return self.train_time

    def predictions(
        self, learner, preproces, X_train, y_train, X_validation, y_validation
    ):

        y_train_true = y_train
        y_train_predicted = learner.predict(X_train)
        y_validation_true = y_validation
        y_validation_predicted = learner.predict(X_validation)

        y_train_true = preproces.inverse_scale_target(y_train_true)
        y_train_predicted = preproces.inverse_scale_target(y_train_predicted)
        y_validation_true = preproces.inverse_scale_target(y_validation_true)
        y_validation_predicted = preproces.inverse_scale_target(y_validation_predicted)

        y_validation_columns = []
        if self._ml_task == MULTICLASS_CLASSIFICATION:
            # y_train_true = preproces.inverse_categorical_target(y_train_true)
            # y_validation_true = preproces.inverse_categorical_target(y_validation_true)
            # get columns, omit the last one (it is label)
            y_validation_columns = preproces.prepare_target_labels(
                y_validation_predicted
            ).columns.tolist()[:-1]

        return {
            "y_train_true": y_train_true,
            "y_train_predicted": y_train_predicted,
            "y_validation_true": y_validation_true,
            "y_validation_predicted": y_validation_predicted,
            "validation_index": X_validation.index,
            "validation_columns": y_validation_columns,
        }

    def train(self):  
        logger.debug(f"ModelFramework.train {self.learner_params.get('model_type')}")

        start_time = time.time()
        np.random.seed(self.learner_params["seed"])

        self.validation = ValidationStep(self.validation_params)

        for k_fold in range(self.validation.get_n_splits()):
            train_data, validation_data = self.validation.get_split(k_fold)
            logger.debug(
                "Data split, train X:{} y:{}, validation X:{}, y:{}".format(
                    train_data["X"].shape,
                    train_data["y"].shape,
                    validation_data["X"].shape,
                    validation_data["y"].shape,
                )
            )
            # the proprocessing is done at every validation step
            self.preprocessings += [Preprocessing(self.preprocessing_params)]

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

                self.callbacks.on_iteration_end(
                    {"iter_cnt": i},
                    self.predictions(
                        learner,
                        self.preprocessings[-1],
                        X_train,
                        y_train,
                        X_validation,
                        y_validation,
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

    def get_metric_name(self):
        early_stopping = self.callbacks.get("early_stopping")
        if early_stopping is None:
            return None
        return early_stopping.metric.name

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

            y_p = self.preprocessings[ind].inverse_scale_target(y_p)

            y_predicted = y_p if y_predicted is None else y_predicted + y_p

        y_predicted_average = y_predicted / float(len(self.learners))

        y_predicted_final = self.preprocessings[0].prepare_target_labels(
            y_predicted_average
        )

        return y_predicted_final

    def get_additional_metrics(self):
        if self._additional_metrics is None:
            # 'target' - the target after processing used for model training
            # 'prediction' - out of folds predictions of the model
            oof_predictions = self.get_out_of_folds()
            prediction_cols = [c for c in oof_predictions.columns if "prediction" in c]
            target_cols = [c for c in oof_predictions.columns if "target" in c]

            target = oof_predictions[target_cols]

            oof_preds = None
            if self._ml_task == MULTICLASS_CLASSIFICATION:
                oof_preds = self.preprocessings[0].prepare_target_labels(
                    oof_predictions[prediction_cols].values
                )

            else:
                oof_preds = oof_predictions[prediction_cols]

            self._additional_metrics = AdditionalMetrics.compute(
                target, oof_preds, self._ml_task
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
                "saved": saved,
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

        LearningCurves.plot(
            self.validation.get_n_splits(), self.get_metric_name(), model_path
        )

        self._additional_metrics = self.get_additional_metrics()

        AdditionalMetrics.save(
            self._additional_metrics, self._ml_task, self.model_markdown(), model_path
        )

        with open(os.path.join(model_path, "status.txt"), "w") as fout:
            fout.write("ALL OK!")

    def model_markdown(self):
        desc = f"# Summary of {self.get_name()}\n"
        desc += f"\n ## {self.learner_params['model_type']}\n"
        for k, v in self.learner_params.items():
            if k in ["model_type", "ml_task", "seed"]:
                continue
            desc += f"- **{k}**: {v}\n"
        desc += "\n## Validation\n"
        for k, v in self.validation_params.items():
            if "path" not in k:
                desc += f" - **{k}**: {v}\n"
        desc += "\n## Optimized metric\n"
        desc += f"{self.get_metric_name()}\n"
        desc += "\n## Training time\n"
        desc += f"\n{np.round(self.train_time,1)} seconds\n"
        return desc

    @staticmethod
    def load(model_path):
        logger.info(f"Loading model framework from {model_path}")

        json_desc = json.load(open(os.path.join(model_path, "framework.json")))
        mf = ModelFramework(json_desc["params"])
        mf.uid = json_desc.get("uid", mf.uid)
        mf._name = json_desc.get("name", mf._name)
        mf._threshold = json_desc.get("threshold")
        mf.learners = []
        for learner_desc, learner_path in zip(
            json_desc.get("learners"), json_desc.get("saved")
        ):

            l = AlgorithmFactory.load(learner_desc, learner_path)
            mf.learners += [l]

        mf.preprocessings = []
        for p in json_desc.get("preprocessing"):
            ps = Preprocessing()
            ps.from_json(p)
            mf.preprocessings += [ps]

        return mf
