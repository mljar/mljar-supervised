"""Select one saved RealMLP epoch across every validation fold and repeat."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from supervised.exceptions import AutoMLException


def checkpoint_epochs(epochs, interval=10):
    return sorted(set(range(interval, epochs + 1, interval)) | {epochs})


class SharedRealMLPSelection:
    def __init__(self, model_path, metric, interval=10):
        self.path = Path(model_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix=".realmlp-", dir=self.path)
        self.cache = Path(self.temporary.name)
        self.metric = metric
        self.interval = interval
        self.folds = []

    def close(self):
        self.temporary.cleanup()

    def check_budget(self):
        # Once admitted, complete the candidate's checkpoint finalization.
        pass

    def fit(self, learner, preprocessing, X, y, sample_weight, X_validation,
            y_validation, sample_weight_validation, log_to_file, max_time):
        epochs = learner.params.get("n_epochs", 256)
        counts = checkpoint_epochs(epochs, self.interval)
        directory = self.cache / learner.uid
        directory.mkdir()
        learner._checkpoint_directory = str(directory)
        learner._checkpoint_epochs = counts
        try:
            learner.fit(X, y, sample_weight, X_validation, y_validation,
                        sample_weight_validation, log_to_file, max_time)
        finally:
            del learner._checkpoint_directory
            del learner._checkpoint_epochs
        if learner.training_info.get("stop_reason") == "time_limit":
            # Let ModelFramework record the same budget skip as other learners,
            # before attempting to restore any incomplete checkpoints.
            return
        if (learner.training_info.get("completed_epochs") != epochs
                or learner.training_info.get("stop_reason") != "max_epochs"):
            raise AutoMLException("Shared RealMLP selection requires a complete training schedule")
        from supervised.algorithms._realmlp_backend import restore_checkpoint

        for epoch in counts:
            path = directory / f"epoch_{epoch:04d}.pt"
            if not path.is_file():
                raise AutoMLException(f"Missing RealMLP checkpoint: {learner.name}, epoch {epoch}")
            restore_checkpoint(learner.model, path)
            # Use the public prediction path, including classification probabilities
            # and both levels of target scaling, rather than rounded backend logs.
            prediction = preprocessing.inverse_scale_target(learner.predict(X_validation))
            np.save(self.cache / f"{learner.uid}-{epoch}.npy", prediction, allow_pickle=False)
        self.folds.append(dict(
            learner=learner.name, uid=learner.uid, completed_epochs=epochs,
            individual_best_epoch=learner.training_info["best_epoch"],
            checkpoints=counts, stop_reason=learner.training_info["stop_reason"],
        ))

    def _frames(self, epoch, early):
        frames = {}
        for fold in self.folds:
            uid = fold["uid"]
            frame = early.best_y_predicted[uid].copy()
            columns = [c for c in frame if "prediction" in c]
            prediction = np.load(self.cache / f"{uid}-{epoch}.npy", allow_pickle=False)
            frame[columns] = prediction.reshape(len(frame), len(columns))
            frames[uid] = frame
        return frames

    def _loss(self, frame, target_columns):
        columns = [c for c in frame if "prediction" in c]
        pred = frame[columns[0]] if len(columns) == 1 else frame[columns]
        return float(self.metric(frame[target_columns], pred, frame.get("sample_weight")))

    def finalize(self, framework):
        from supervised.algorithms._realmlp_backend import restore_checkpoint

        early = framework.callbacks.get("early_stopping")
        if len(self.folds) != framework._expected_folds or len(self.folds) != len(framework.learners):
            raise AutoMLException("Shared RealMLP selection requires every completed fold")
        counts = self.folds[0]["checkpoints"]
        if any(f["checkpoints"] != counts for f in self.folds):
            raise AutoMLException("RealMLP folds must share the complete checkpoint schedule")
        # Validate every checkpoint before publishing any replacement model.
        for fold in self.folds:
            for epoch in counts:
                if not (self.cache / fold["uid"] / f"epoch_{epoch:04d}.pt").is_file():
                    raise AutoMLException("Missing RealMLP checkpoint during finalization")
        best_epoch, best_loss = None, self.metric.worst_value()
        scores = []
        sign = -1 if self.metric.is_negative() else 1
        for epoch in counts:
            frames = self._frames(epoch, early)
            losses = [self._loss(f, early.target_columns) for f in frames.values()]
            mean = float(np.mean(losses))
            if not np.isfinite(mean):
                raise AutoMLException("Non-finite shared RealMLP validation score")
            scores.append(dict(epoch=epoch, mean_fold_score=sign * mean))
            if self.metric.improvement(best_loss, mean):
                best_epoch, best_loss = epoch, mean
        if best_epoch is None:
            raise AutoMLException("No valid shared RealMLP checkpoint")
        frames = self._frames(best_epoch, early)
        staged = self.cache / "selected"
        staged.mkdir()
        replacements = []
        for index, learner in enumerate(framework.learners):
            destination = learner.model_file_path
            learner.reload()
            restore_checkpoint(learner.model, self.cache / learner.uid / f"epoch_{best_epoch:04d}.pt")
            learner.training_info.update(
                individual_best_epoch=learner.training_info["best_epoch"],
                best_epoch=best_epoch, selected_epoch=best_epoch,
                selection_aggregation="mean_fold", checkpoint_interval=self.interval,
            )
            interface = learner.model.cv_alg_interface_
            for params in interface.fit_params:
                params["stop_epoch"] = {key: best_epoch for key in params["stop_epoch"]}
            interface.model.fit_params = interface.fit_params
            candidate = staged / Path(destination).name
            learner.save(str(candidate))
            learner.unload()
            # Confirm serialization retained the chosen weights before publishing.
            repeat, fold = divmod(index, framework.validation.get_n_splits())
            _, valid = framework.validation.get_split(fold, repeat)
            prep = framework.preprocessings[index]
            X_valid, _, _ = prep.transform(valid["X"], None)
            prediction = np.asarray(prep.inverse_scale_target(learner.predict(X_valid)))
            frame = frames[learner.uid]
            columns = [c for c in frame if "prediction" in c]
            prediction = prediction.reshape(len(frame), len(columns))
            if not np.allclose(prediction, frame[columns].to_numpy(), rtol=1e-5, atol=1e-5):
                raise AutoMLException("RealMLP shared checkpoint predictions changed after reload")
            frame[columns] = prediction
            (staged / f"{learner.name}_training.log.json").write_text(
                json.dumps(learner.training_info, indent=2)
            )
            learner.unload()
            replacements.append((learner, candidate, destination))
        for learner, candidate, destination in replacements:
            os.replace(candidate, destination)
            log_name = f"{learner.name}_training.log.json"
            os.replace(staged / log_name, self.path / log_name)
            learner.model_file_path = destination
            early.best_y_predicted[learner.uid] = frames[learner.uid]
            early.best_loss[learner.uid] = self._loss(frames[learner.uid], early.target_columns)
            early.best_iter[learner.uid] = best_epoch - 1
        summary = dict(
            selected_epoch=best_epoch, checkpoint_interval=self.interval,
            selection_metric=self.metric.name, selection_aggregation="mean_fold",
            mean_fold_score=sign * float(np.mean(list(early.best_loss.values()))),
        )
        report = dict(summary, folds=self.folds, scores=scores)
        temporary = self.path / "shared_realmlp.json.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        os.replace(temporary, self.path / "shared_realmlp.json")
        pd.DataFrame(scores).to_csv(self.path / "shared_realmlp_scores.csv", index=False)
        framework.params["shared_realmlp"] = summary
        framework.final_loss = None
        framework.oof_predictions = None
        framework._additional_metrics = None
