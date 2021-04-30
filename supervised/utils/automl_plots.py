import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from supervised.utils.metric import Metric
from supervised.utils.config import LOG_LEVEL

logger.setLevel(LOG_LEVEL)

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class AutoMLPlots:

    features_heatmap_fname = "features_heatmap.png"

    @staticmethod
    def add(results_path, models, fout):

        AutoMLPlots.models_feature_importance(results_path, models)

        features_plot_path = os.path.join(
            results_path, AutoMLPlots.features_heatmap_fname
        )
        if os.path.exists(features_plot_path):
            fout.write("\n\n### Features Importance\n")
            fout.write(
                f"![features importance across models]({AutoMLPlots.features_heatmap_fname})\n\n"
            )

    @staticmethod
    def models_feature_importance(results_path, models):
        try:
            model_feature_imp = {}
            for m in models:

                model_path = os.path.join(results_path, m.get_name())
                imp_data = [
                    f
                    for f in os.listdir(model_path)
                    if "_importance.csv" in f and "shap" not in f
                ]
                if not len(imp_data):
                    continue

                df_all = []
                for fname in imp_data:
                    df = pd.read_csv(os.path.join(model_path, fname), index_col=0)
                    df_all += [df]

                df = pd.concat(df_all, axis=1)

                model_feature_imp[m.get_name()] = df.mean(axis=1)

            if len(model_feature_imp) < 2:
                # too small number of models
                return
            mfi = pd.concat(model_feature_imp, axis=1)

            mfi["m"] = mfi.mean(axis=1)
            mfi = mfi.sort_values(by="m", ascending=False)
            mfi = mfi.drop("m", axis=1)

            title = "Feature importance"
            if mfi.shape[0] > 25:
                mfi = mfi.head(25)
                title = "Top-25 important features"

            fig, ax = plt.subplots(1, 1, figsize=(10, 9))

            image = ax.imshow(
                mfi,
                interpolation="nearest",
                cmap=plt.cm.get_cmap("Blues"),
                aspect="auto",
            )
            plt.colorbar(mappable=image)

            x_tick_marks = np.arange(len(mfi.columns))
            y_tick_marks = np.arange(len(mfi.index))
            ax.set_xticks(x_tick_marks)
            ax.set_xticklabels(mfi.columns, rotation=90)
            ax.set_yticks(y_tick_marks)
            ax.set_yticklabels(mfi.index)
            ax.set_title(title)

            plt.tight_layout(pad=2.0)
            plot_path = os.path.join(results_path, "features_heatmap.png")
            plt.savefig(plot_path)
            plt.close("all")
        except Exception as e:
            pass
