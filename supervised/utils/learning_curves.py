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
    def plot(validation_splits, metric_name, model_path):
        colors = MY_COLORS
        if validation_splits > len(colors):
            repeat_colors = int(np.ceil(validation_splits / len(colors)))
            colors = colors * repeat_colors

        plt.figure(figsize=(10, 7))
        for l in range(validation_splits):
            df = pd.read_csv(
                os.path.join(model_path, f"./learner_{l+1}_training.log"),
                names=["iteration", "train", "test", "no_improvement"],
            )
            plt.plot(
                df.iteration,
                df.train,
                "--",
                color=colors[l],
                label=f"Fold {l+1}, train",
            )
            plt.plot(df.iteration, df.test, color=colors[l], label=f"Fold {l+1}, test")
        plt.xlabel("#Iteration")
        plt.ylabel(metric_name)
        plt.legend(loc="best")
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
