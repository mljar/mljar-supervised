import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from supervised.utils.config import LOG_LEVEL
from supervised.utils.common import learner_name_to_fold_repeat
from supervised.utils.metric import Metric

logger.setLevel(LOG_LEVEL)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

MY_COLORS = list(mcolors.TABLEAU_COLORS.values())


class LearningCurves:

    output_file_name = "learning_curves.png"

    @staticmethod
    def single_iteration(learner_names, model_path):
        for ln in learner_names:
            df = pd.read_csv(
                os.path.join(model_path, f"{ln}_training.log"),
                names=["iteration", "train", "test"],
            )
            if df.shape[0] > 1:
                return False
        return True

    @staticmethod
    def plot(learner_names, metric_name, model_path, trees_in_iteration=None):
        colors = MY_COLORS
        if len(learner_names) > len(colors):
            repeat_colors = int(np.ceil(len(learner_names) / len(colors)))
            colors = colors * repeat_colors

        if LearningCurves.single_iteration(learner_names, model_path):
            LearningCurves.plot_single_iter(
                learner_names, metric_name, model_path, colors
            )
        else:
            LearningCurves.plot_iterations(
                learner_names, metric_name, model_path, colors, trees_in_iteration
            )

    @staticmethod
    def plot_single_iter(learner_names, metric_name, model_path, colors):
        plt.figure(figsize=(10, 7))
        for ln in learner_names:
            df = pd.read_csv(
                os.path.join(model_path, f"{ln}_training.log"),
                names=["iteration", "train", "test"],
            )

            fold, repeat = learner_name_to_fold_repeat(ln)
            repeat_str = f" Reapeat {repeat+1}," if repeat is not None else ""
            plt.bar(
                f"Fold {fold+1},{repeat_str} train",
                df.train[0],
                color="white",
                edgecolor=colors[fold],
            )
            plt.bar(f"Fold {fold+1},{repeat_str} test", df.test[0], color=colors[fold])

        plt.ylabel(metric_name)
        plt.xticks(rotation=90)
        plt.tight_layout(pad=2.0)
        plot_path = os.path.join(model_path, LearningCurves.output_file_name)
        plt.savefig(plot_path)
        plt.close("all")

    @staticmethod
    def plot_iterations(
        learner_names, metric_name, model_path, colors, trees_in_iteration=None
    ):
        plt.figure(figsize=(10, 7))
        for ln in learner_names:
            df = pd.read_csv(
                os.path.join(model_path, f"{ln}_training.log"),
                names=["iteration", "train", "test"],
            )

            fold, repeat = learner_name_to_fold_repeat(ln)
            repeat_str = f" Reapeat {repeat+1}," if repeat is not None else ""
            # if trees_in_iteration is not None:
            #    df.iteration = df.iteration * trees_in_iteration
            any_none = np.sum(pd.isnull(df.train))
            if any_none == 0:
                plt.plot(
                    df.iteration,
                    df.train,
                    "--",
                    color=colors[fold],
                    label=f"Fold {fold+1},{repeat_str} train",
                )
            any_none = np.sum(pd.isnull(df.test))
            if any_none == 0:
                plt.plot(
                    df.iteration,
                    df.test,
                    color=colors[fold],
                    label=f"Fold {fold+1},{repeat_str} test",
                )

            best_iter = None
            if Metric.optimize_negative(metric_name):
                best_iter = df.test.argmax()
            else:
                best_iter = df.test.argmin()

            if best_iter is not None and best_iter != -1:
                plt.axvline(best_iter, color=colors[fold], alpha=0.3)

        if trees_in_iteration is not None:
            plt.xlabel("#Trees")
        else:
            plt.xlabel("#Iteration")
        plt.ylabel(metric_name)

        # limit number of learners in the legend
        # too many will raise warnings
        if len(learner_names) <= 15:
            plt.legend(loc="best")

        plt.tight_layout(pad=2.0)
        plot_path = os.path.join(model_path, LearningCurves.output_file_name)
        plt.savefig(plot_path)
        plt.close("all")

    @staticmethod
    def plot_for_ensemble(scores, metric_name, model_path):
        plt.figure(figsize=(10, 7))
        plt.plot(range(1, len(scores) + 1), scores, label=f"Ensemble")
        plt.xlabel("#Iteration")
        plt.ylabel(metric_name)
        plt.legend(loc="best")
        plot_path = os.path.join(model_path, LearningCurves.output_file_name)
        plt.savefig(plot_path)
        plt.close("all")
