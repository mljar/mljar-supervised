"""Disk-backed, post-CV selection of one boosting-round count per model."""

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from supervised.algorithms import shared_boosting as backend
from supervised.exceptions import AutoMLException, BudgetExceededError


def aggregate_oof(frames):
    frame = pd.concat(frames).sort_index()
    if frame.index.duplicated().any():
        frame = frame.groupby(level=0, sort=True).agg(
            {col: "mean" if "prediction" in col else "first" for col in frame.columns}
        )
    return frame


class SharedBoostingSelection:
    def __init__(self, model_path, metric, deadline=None):
        self.path = Path(model_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix=".boosting-", dir=self.path)
        self.cache = Path(self.temporary.name)
        self.metric = metric
        self.deadline = deadline
        self.folds = []

    def close(self):
        self.temporary.cleanup()

    def check_budget(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise BudgetExceededError(
                "Shared boosting selection exceeded candidate allowance.",
                phase="shared_boosting_selection",
            )

    def fit(
        self,
        learner,
        preprocessing,
        X,
        y,
        sample_weight,
        X_validation,
        y_validation,
        sample_weight_validation,
        log_to_file,
        max_time,
    ):
        self.check_budget()
        if self.deadline is not None:
            remaining = max(0, self.deadline - time.monotonic())
            max_time = remaining if max_time is None else min(max_time, remaining)
        stopping = backend.fit(
            learner,
            X,
            y,
            sample_weight,
            X_validation,
            y_validation,
            sample_weight_validation,
            self.metric,
            None if max_time is None else 0.7 * max_time,
        )
        trained = backend.round_count(learner)
        if stopping.reason == "time_limit":
            raise BudgetExceededError(
                f"{learner.algorithm_short_name} learner {learner.name} reached its "
                f"time limit after {trained} boosting rounds; skipping the incomplete candidate.",
                phase="boosting_fit",
                learner=learner.name,
                stop_reason=stopping.reason,
                trained_rounds=int(trained),
                local_best_rounds=stopping.best_round,
                learner_elapsed=time.monotonic() - stopping.started,
                learner_allowance=stopping.max_time,
            )
        if trained < 1:
            raise AutoMLException("Booster produced no prediction rounds")
        counts = backend.checkpoints(trained)
        # Later folds need not materialize prefixes unavailable in earlier folds.
        if self.folds:
            available = set.intersection(*(set(f["checkpoints"]) for f in self.folds))
            counts = [n for n in counts if n in available]
        target = preprocessing.inverse_scale_target(y_validation)
        logs = []
        for count, pred in backend.prefix_predictions(learner, X_validation, counts):
            self.check_budget()
            pred = np.asarray(preprocessing.inverse_scale_target(pred))
            np.save(self.cache / f"{learner.uid}-{count}.npy", pred, allow_pickle=False)
            weight = (
                None
                if getattr(self, "ignore_sample_weight", False)
                else sample_weight_validation
            )
            loss = self.metric(target, pred, weight)
            logs.append(
                (count - 1, np.nan, -loss if self.metric.is_negative() else loss)
            )
        pd.DataFrame(logs).to_csv(log_to_file, index=False, header=False)
        self.folds.append(
            dict(
                learner=learner.name,
                uid=learner.uid,
                trained_rounds=int(trained),
                local_best_rounds=stopping.best_round,
                stop_reason=stopping.reason,
                checkpoints=counts,
            )
        )

    def _frames(self, count, early_stopping):
        frames = {}
        for fold in self.folds:
            uid = fold["uid"]
            frame = early_stopping.best_y_predicted[uid].copy()
            columns = [c for c in frame if "prediction" in c]
            prediction = np.load(self.cache / f"{uid}-{count}.npy", mmap_mode="r")
            frame[columns] = prediction.reshape(len(frame), len(columns))
            frames[uid] = frame
        return frames

    def _loss(self, frame, target_columns):
        columns = [c for c in frame if "prediction" in c]
        pred = frame[columns[0]] if len(columns) == 1 else frame[columns]
        return float(
            self.metric(frame[target_columns], pred, frame.get("sample_weight"))
        )

    def finalize(self, framework):
        early = framework.callbacks.get("early_stopping")
        if len(self.folds) != len(framework.learners):
            raise AutoMLException("Shared selection requires every completed fold")
        common = sorted(set.intersection(*(set(f["checkpoints"]) for f in self.folds)))
        best_count, best_loss = None, self.metric.worst_value()
        scores = []
        sign = -1 if self.metric.is_negative() else 1
        for count in common:
            self.check_budget()
            frames = self._frames(count, early)
            pooled = self._loss(
                aggregate_oof(list(frames.values())), early.target_columns
            )
            mean = float(
                np.mean([self._loss(f, early.target_columns) for f in frames.values()])
            )
            scores.append(
                dict(
                    rounds=count,
                    pooled_score=sign * pooled,
                    mean_fold_score=sign * mean,
                )
            )
            if not np.isnan(mean) and self.metric.improvement(best_loss, mean):
                best_count, best_loss = count, mean
            del frames
        if best_count is None:
            raise AutoMLException("No valid shared boosting validation score")
        frames = self._frames(best_count, early)
        for learner in framework.learners:
            self.check_budget()
            learner.reload()
            backend.select_rounds(learner, best_count)
            learner.save(learner.model_file_path)
            learner.unload()
            early.best_y_predicted[learner.uid] = frames[learner.uid]
            early.best_loss[learner.uid] = self._loss(
                frames[learner.uid], early.target_columns
            )
            early.best_iter[learner.uid] = best_count - 1
        selected = next(s for s in scores if s["rounds"] == best_count)
        summary = dict(
            selected_rounds=best_count,
            checkpoint_interval=backend.CHECKPOINT_INTERVAL,
            patience_rounds=backend.PATIENCE_ROUNDS,
            selection_metric=self.metric.name,
            selection_aggregation="mean_fold",
            pooled_score=selected["pooled_score"],
            mean_fold_score=selected["mean_fold_score"],
            common_max_rounds=max(common),
        )
        framework.params["shared_boosting"] = summary
        # Publish selection metadata only after every saved fold uses the new count.
        report = dict(summary, folds=self.folds, scores=scores)
        temporary = self.path / "shared_boosting.json.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        os.replace(temporary, self.path / "shared_boosting.json")
        pd.DataFrame(scores).to_csv(
            self.path / "shared_boosting_scores.csv", index=False
        )
        framework.final_loss = None
        framework.oof_predictions = None
        framework._additional_metrics = None
