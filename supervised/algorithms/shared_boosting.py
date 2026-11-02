"""Native boosting with checkpoint patience and explicit round-count prefixes.

Counts are boosting rounds, not zero-based indices or multiclass tree counts.
The full booster is retained until the framework selects a common prefix.
"""

import time

import numpy as np
import pandas as pd

from supervised.algorithms.registry import REGRESSION

CHECKPOINT_INTERVAL = 10
PATIENCE_ROUNDS = 300
SUPPORTED = {"Xgboost", "LightGBM", "CatBoost"}


def checkpoints(rounds):
    # Round 1 also supports boosters that exhaust possible splits before tick 10.
    return [1] + list(range(CHECKPOINT_INTERVAL, rounds + 1, CHECKPOINT_INTERVAL))


class CheckpointStopping:
    def __init__(self, max_time=None):
        self.started = time.monotonic()
        self.max_time = max_time
        self.best_round = 0
        self.best_loss = np.inf
        self.reason = "maximum_rounds"

    def update(self, count, value, higher_is_better):
        if count == 1 or count % CHECKPOINT_INTERVAL == 0:
            loss = -value if higher_is_better else value
            if np.isfinite(loss) and loss < self.best_loss:
                self.best_loss, self.best_round = loss, count
            if count - self.best_round >= PATIENCE_ROUNDS:
                self.reason = "patience"
                return True
        if (
            self.max_time is not None
            and time.monotonic() - self.started >= self.max_time
        ):
            self.reason = "time_limit"
            return True
        return False


def _values(X):
    return X.values if isinstance(X, pd.DataFrame) else X


def fit(
    learner,
    X,
    y,
    sample_weight,
    X_validation,
    y_validation,
    sample_weight_validation,
    metric,
    max_time,
):
    """Train a continuous native booster; checkpoints never restart training."""
    stopping = CheckpointStopping(max_time)
    kind = learner.algorithm_short_name
    if kind == "Xgboost":
        import xgboost as xgb

        learner._validate_split_capacity(y, sample_weight)
        train = xgb.DMatrix(
            _values(X),
            label=y,
            weight=sample_weight,
            nthread=learner.params.get("n_jobs", -1),
        )
        valid = xgb.DMatrix(
            _values(X_validation),
            label=y_validation,
            weight=sample_weight_validation,
            nthread=learner.params.get("n_jobs", -1),
        )
        params = dict(learner.learner_params)
        custom = learner.custom_eval_metric
        if custom is not None:
            params.pop("eval_metric", None)

        class Stop(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                value = list(evals_log["validation"].values())[-1][-1]
                return stopping.update(
                    epoch + 1, value, custom is None and metric.is_negative()
                )

        learner.model = xgb.train(
            params,
            train,
            num_boost_round=learner.boosting_rounds,
            evals=[(valid, "validation")],
            custom_metric=custom,
            callbacks=[Stop()],
            verbose_eval=False,
        )
        learner._validate_trained_split_capacity()
    elif kind == "LightGBM":
        import lightgbm as lgb

        train = lgb.Dataset(_values(X), label=y, weight=sample_weight)
        valid = lgb.Dataset(
            _values(X_validation),
            label=y_validation,
            weight=sample_weight_validation,
            reference=train,
        )

        def stop(env):
            # A custom evaluation metric follows the objective's default metric.
            result = [r for r in env.evaluation_result_list if r[0] == "validation"][-1]
            if stopping.update(env.iteration + 1, result[2], result[3]):
                raise lgb.callback.EarlyStopException(
                    env.iteration, env.evaluation_result_list
                )

        learner.model = lgb.train(
            dict(learner.learner_params),
            train,
            num_boost_round=learner.rounds,
            valid_sets=[valid],
            valid_names=["validation"],
            feval=learner.custom_eval_metric,
            callbacks=[stop],
            keep_training_booster=True,
        )
        # Neither predict() nor save_model() may silently use a local optimum.
        learner.model.best_iteration = -1
        learner.model.free_dataset()
    else:
        from catboost import Pool
        from supervised.preprocessing.preprocessing_utils import PreprocessingUtils

        learner.cat_features = []
        for i in range(X.shape[1]):
            if PreprocessingUtils.is_categorical(X.iloc[:, i]):
                learner.cat_features.append(i)
                X[X.columns[i]] = X.iloc[:, i].astype(str)
                X_validation[X.columns[i]] = X_validation.iloc[:, i].astype(str)
        train = Pool(
            X, label=y, weight=sample_weight, cat_features=learner.cat_features
        )
        valid = Pool(
            X_validation,
            label=y_validation,
            weight=sample_weight_validation,
            cat_features=learner.cat_features,
        )
        native_metric = learner.model.get_params()["eval_metric"]
        higher = (
            native_metric.is_max_optimal()
            if hasattr(native_metric, "is_max_optimal")
            else metric.is_negative()
        )

        class Stop:
            def after_iteration(self, info):
                values = info.metrics["validation"]
                key = next(
                    k
                    for k in values
                    if k.split(":")[0] == learner.log_metric_name.split(":")[0]
                )
                return not stopping.update(info.iteration, values[key][-1], higher)

        # A single fit avoids warm-up offsets and random-state resets from init_model.
        learner.model.fit(
            train,
            eval_set=valid,
            use_best_model=False,
            callbacks=[Stop()],
            verbose=False,
        )
        learner.best_ntree_limit = learner.model.tree_count_
    if learner.params["ml_task"] != REGRESSION:
        learner.classes_ = np.unique(y)
    return stopping


def round_count(learner):
    if learner.algorithm_short_name == "Xgboost":
        return learner.model.num_boosted_rounds()
    if learner.algorithm_short_name == "LightGBM":
        return learner.model.current_iteration()
    return learner.model.tree_count_


def prefix_predictions(learner, X, counts):
    """Reuse one native prediction dataset across all prefixes."""
    kind = learner.algorithm_short_name
    if kind == "Xgboost":
        import xgboost as xgb

        data = xgb.DMatrix(_values(X), nthread=learner.params.get("n_jobs", -1))
        for count in counts:
            yield count, learner.model.predict(data, iteration_range=(0, count))
    elif kind == "LightGBM":
        for count in counts:
            yield count, learner.model.predict(_values(X), num_iteration=count)
    else:
        method = (
            learner.model.predict
            if learner.ml_task == REGRESSION
            else learner.model.predict_proba
        )
        for count in counts:
            prediction = method(X, ntree_end=count)
            if learner.ml_task == "binary_classification":
                prediction = prediction[:, 1]
            yield count, prediction


def select_rounds(learner, count):
    """Physically keep the selected prefix so standalone reloads are consistent."""
    if not 1 <= count <= round_count(learner):
        raise ValueError("Selected count is not available in this fold")
    kind = learner.algorithm_short_name
    if kind == "Xgboost":
        learner.model = learner.model[:count]
        # XGBoost stores an index; prediction's exclusive endpoint is index + 1.
        learner.model.set_attr(best_iteration=str(count - 1), best_score=None)
    elif kind == "LightGBM":
        import lightgbm as lgb

        learner.model = lgb.Booster(
            model_str=learner.model.model_to_string(num_iteration=count)
        )
        learner.model.best_iteration = count
    else:
        learner.model.shrink(ntree_end=count)
        learner.best_ntree_limit = count
    learner.params["selected_boosting_rounds"] = count
