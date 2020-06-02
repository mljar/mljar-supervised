import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from supervised.utils.config import LOG_LEVEL

logger.setLevel(LOG_LEVEL)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

MY_COLORS = list(mcolors.TABLEAU_COLORS.values())


class LearningCurves:

    output_file_name = "learning_curves.png"

    @staticmethod
    def single_iteration(validation_splits, model_path):
        for l in range(validation_splits):
            df = pd.read_csv(
                os.path.join(model_path, f"learner_{l+1}_training.log"),
                names=["iteration", "train", "test"],
            )
            if df.shape[0] > 1:
                return False
        return True

    @staticmethod
    def plot(validation_splits, metric_name, model_path, trees_in_iteration=None):
        colors = MY_COLORS
        if validation_splits > len(colors):
            repeat_colors = int(np.ceil(validation_splits / len(colors)))
            colors = colors * repeat_colors

        if LearningCurves.single_iteration(validation_splits, model_path):
            LearningCurves.plot_single_iter(
                validation_splits, metric_name, model_path, colors
            )
        else:
            LearningCurves.plot_iterations(
                validation_splits, metric_name, model_path, colors, trees_in_iteration
            )

    @staticmethod
    def plot_single_iter(validation_splits, metric_name, model_path, colors):
        plt.figure(figsize=(10, 7))
        for l in range(validation_splits):
            df = pd.read_csv(
                os.path.join(model_path, f"learner_{l+1}_training.log"),
                names=["iteration", "train", "test"],
            )
            plt.bar(
                f"Fold {l+1}, train", df.train[0], color="white", edgecolor=colors[l]
            )
            plt.bar(f"Fold {l+1}, test", df.test[0], color=colors[l])

        plt.ylabel(metric_name)
        plt.xticks(rotation=90)
        plt.tight_layout(pad=2.0)
        plot_path = os.path.join(model_path, LearningCurves.output_file_name)
        plt.savefig(plot_path)
        plt.close("all")

    @staticmethod
    def plot_iterations(
        validation_splits, metric_name, model_path, colors, trees_in_iteration=None
    ):
        plt.figure(figsize=(10, 7))
        for l in range(validation_splits):
            df = pd.read_csv(
                os.path.join(model_path, f"learner_{l+1}_training.log"),
                names=["iteration", "train", "test"],
            )
            # if trees_in_iteration is not None:
            #    df.iteration = df.iteration * trees_in_iteration
            plt.plot(
                df.iteration,
                df.train,
                "--",
                color=colors[l],
                label=f"Fold {l+1}, train",
            )
            plt.plot(df.iteration, df.test, color=colors[l], label=f"Fold {l+1}, test")
        if trees_in_iteration is not None:
            plt.xlabel("#Trees")
        else:
            plt.xlabel("#Iteration")
        plt.ylabel(metric_name)
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
