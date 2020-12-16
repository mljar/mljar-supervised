import os
import logging
import numpy as np
import pandas as pd

from supervised.callbacks.callback import Callback
from supervised.utils.metric import Metric
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

from supervised.utils.config import mem


class EarlyStopping(Callback):
    def __init__(self, params):
        super(EarlyStopping, self).__init__(params)
        self.name = params.get("name", "early_stopping")
        self.metric = Metric(params.get("metric"))
        self.max_no_improvement_cnt = params.get("max_no_improvement_cnt", 5)
        self.log_to_dir = params.get("log_to_dir")

        self.keep_best_model = params.get("keep_best_model", True)
        self.best_iter = {}
        self.best_loss = {}
        self.loss_values = {}
        self.best_models = {}
        self.best_y_predicted = {}
        self.best_y_oof = (
            None
        )  # predictions computed on out of folds or on validation set
        self.final_loss = (
            None
        )  # final score computed on combined predictions from all learners
        # path to best model local copy, only used if cannot deep copy
        self.best_model_paths = {}
        self.multiple_target = False
        self.target_columns = None

    def add_and_set_learner(self, learner):
        self.learners += [learner]
        self.learner = learner
        self.best_iter[learner.uid] = None
        self.best_loss[learner.uid] = self.metric.worst_value()
        self.loss_values[learner.uid] = {"train": [], "validation": [], "iters": []}
        self.best_models[learner.uid] = None
        self.best_model_paths[learner.uid] = None
        self.best_y_predicted[learner.uid] = None

    def on_learner_train_start(self, logs):
        self.no_improvement_cnt = 0

    def on_framework_train_end(self, logs):
        # aggregate predictions from all learners
        # it has two columns: 'prediction', 'target'
        logger.debug("early stopping on framework train end")
        self.best_y_oof = pd.concat(list(self.best_y_predicted.values()))
        self.best_y_oof.sort_index(inplace=True)
        # check for duplicates in index -> repeats of validation
        if np.sum(self.best_y_oof.index.duplicated()):
            # we need to aggregate predictions from multiple repeats
            target_cols = [c for c in self.best_y_oof.columns if "prediction" not in c]
            prediction_cols = [c for c in self.best_y_oof.columns if "prediction" in c]

            aggs = {}
            for t in target_cols:
                aggs[t] = "first"
            for p in prediction_cols:
                aggs[p] = "mean"
            # aggregate predictions from repeats
            self.best_y_oof = self.best_y_oof.groupby(
                target_cols + prediction_cols, level=0
            ).agg(aggs)

        sample_weight = None
        if "sample_weight" in self.best_y_oof.columns:
            sample_weight = self.best_y_oof["sample_weight"]

        if "prediction" in self.best_y_oof:
            self.final_loss = self.metric(
                self.best_y_oof[self.target_columns],
                self.best_y_oof["prediction"],
                sample_weight=sample_weight,
            )
        else:
            prediction_cols = [c for c in self.best_y_oof.columns if "prediction" in c]
            self.final_loss = self.metric(
                self.best_y_oof[self.target_columns],
                self.best_y_oof[prediction_cols],
                sample_weight=sample_weight,
            )

    def on_iteration_end(self, logs, predictions):
        train_loss = 0
        if predictions.get("y_train_predicted") is not None:
            train_loss = self.metric(
                predictions.get("y_train_true"),
                predictions.get("y_train_predicted"),
                predictions.get("sample_weight"),
            )

        validation_loss = self.metric(
            predictions.get("y_validation_true"),
            predictions.get("y_validation_predicted"),
            predictions.get("sample_weight_validation"),
        )
        self.loss_values[self.learner.uid]["train"] += [train_loss]
        self.loss_values[self.learner.uid]["validation"] += [validation_loss]
        self.loss_values[self.learner.uid]["iters"] += [logs.get("iter_cnt")]

        if self.metric.improvement(
            previous=self.best_loss[self.learner.uid], current=validation_loss
        ):

            y_validation_true = predictions.get("y_validation_true")
            self.no_improvement_cnt = 0
            self.best_iter[self.learner.uid] = logs.get("iter_cnt")
            self.best_loss[self.learner.uid] = validation_loss

            if len(y_validation_true.shape) == 1 or y_validation_true.shape[1] == 1:
                self.best_y_predicted[self.learner.uid] = pd.DataFrame(
                    {
                        "target": np.array(y_validation_true)
                        # y_validation_true.values.reshape(
                        #    y_validation_true.shape[0]
                        # )
                    },
                    index=predictions.get("validation_index"),
                )
                self.multiple_target = False
                self.target_columns = "target"
            else:
                # in case of Neural Networks and multi-class classification with one-hot encoding
                self.best_y_predicted[self.learner.uid] = pd.DataFrame(
                    y_validation_true, index=predictions.get("validation_index")
                )
                self.multiple_target = True
                self.target_columns = y_validation_true.columns

            y_validation_predicted = predictions.get("y_validation_predicted")

            if len(y_validation_predicted.shape) == 1:
                # only one prediction column (binary classification or regression)
                self.best_y_predicted[self.learner.uid]["prediction"] = np.array(
                    y_validation_predicted
                )
            else:
                # several columns in multiclass classification
                cols = predictions.get("validation_columns")
                for i_col in range(y_validation_predicted.shape[1]):
                    self.best_y_predicted[self.learner.uid][
                        # "prediction_{}".format(i_col)
                        cols[i_col]
                    ] = y_validation_predicted[:, i_col]

            # store sample_weight
            sample_weight_validation = predictions.get("sample_weight_validation")
            if sample_weight_validation is not None:
                self.best_y_predicted[self.learner.uid]["sample_weight"] = np.array(
                    sample_weight_validation
                )

            self.best_models[self.learner.uid] = self.learner.copy()
            # if local copy is not available, save model and keep path
            if self.best_models[self.learner.uid] is None:
                self.best_model_paths[self.learner.uid] = self.learner.save()
        else:
            self.no_improvement_cnt += 1

        if self.no_improvement_cnt > self.max_no_improvement_cnt:
            self.learner.stop_training = True

        logger.info(
            "EarlyStopping.on_iteration_end, train loss: {}, validation loss: {}, "
            "no improvement cnt {}, iters {}".format(
                train_loss,
                validation_loss,
                self.no_improvement_cnt,
                len(self.loss_values[self.learner.uid]["iters"]),
            )
        )

        if self.log_to_dir is not None and self.learner.algorithm_short_name not in [
            "Xgboost",
            "Random Forest",
            "Extra Trees",
            "LightGBM",
            "CatBoost",
            "Neural Network",
        ]:

            with open(
                os.path.join(self.log_to_dir, f"{self.learner.name}_training.log"), "a"
            ) as fout:
                iteration = len(self.loss_values[self.learner.uid]["iters"])
                fout.write(f"{iteration},{train_loss},{validation_loss}\n")

    def get_status(self):
        return "Train loss: {}, Validation loss: {} @ iteration {}".format(
            self.loss_values[self.learner.uid]["train"][-1],
            self.loss_values[self.learner.uid]["validation"][-1],
            len(self.loss_values[self.learner.uid]["iters"]),
        )
