import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import scikitplot as skplt

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)


class AdditionalPlots:
    @staticmethod
    def plots_binary(target, predicted_labels, predicted_probas):
        figures = []
        try:
            #
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            _ = skplt.metrics.plot_confusion_matrix(
                target, predicted_labels, normalize=False, ax=ax1
            )
            figures += [
                {
                    "title": "Confusion Matrix",
                    "fname": "confusion_matrix.png",
                    "figure": fig,
                }
            ]
            #
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            _ = skplt.metrics.plot_confusion_matrix(
                target, predicted_labels, normalize=True, ax=ax1
            )
            figures += [
                {
                    "title": "Normalized Confusion Matrix",
                    "fname": "confusion_matrix_normalized.png",
                    "figure": fig,
                }
            ]
            #
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            _ = skplt.metrics.plot_roc(target, predicted_probas, ax=ax1)
            figures += [{"title": "ROC Curve", "fname": "roc_curve.png", "figure": fig}]
            #
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            _ = skplt.metrics.plot_ks_statistic(target, predicted_probas, ax=ax1)
            figures += [
                {
                    "title": "Kolmogorov-Smirnov Statistic",
                    "fname": "ks_statistic.png",
                    "figure": fig,
                }
            ]
            #
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            _ = skplt.metrics.plot_precision_recall(target, predicted_probas, ax=ax1)
            figures += [
                {
                    "title": "Precision-Recall Curve",
                    "fname": "precision_recall_curve.png",
                    "figure": fig,
                }
            ]
            #
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            _ = skplt.metrics.plot_calibration_curve(
                target.values.ravel(), [predicted_probas], ["Classifier"], ax=ax1
            )
            figures += [
                {
                    "title": "Calibration Curve",
                    "fname": "calibration_curve_curve.png",
                    "figure": fig,
                }
            ]
            #
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            _ = skplt.metrics.plot_cumulative_gain(target, predicted_probas, ax=ax1)
            figures += [
                {
                    "title": "Cumulative Gains Curve",
                    "fname": "cumulative_gains_curve.png",
                    "figure": fig,
                }
            ]
            #
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            _ = skplt.metrics.plot_lift_curve(target, predicted_probas, ax=ax1)
            figures += [
                {"title": "Lift Curve", "fname": "lift_curve.png", "figure": fig}
            ]

        except Exception as e:
            print(str(e))

        return figures

    @staticmethod
    def plots_multiclass(target, predicted_labels, predicted_probas):
        figures = []
        try:
            #
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            _ = skplt.metrics.plot_confusion_matrix(
                target, predicted_labels, normalize=False, ax=ax1
            )
            figures += [
                {
                    "title": "Confusion Matrix",
                    "fname": "confusion_matrix.png",
                    "figure": fig,
                }
            ]
            #
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            _ = skplt.metrics.plot_confusion_matrix(
                target, predicted_labels, normalize=True, ax=ax1
            )
            figures += [
                {
                    "title": "Normalized Confusion Matrix",
                    "fname": "confusion_matrix_normalized.png",
                    "figure": fig,
                }
            ]
            #
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            _ = skplt.metrics.plot_roc(target, predicted_probas, ax=ax1)
            figures += [{"title": "ROC Curve", "fname": "roc_curve.png", "figure": fig}]
            #
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            _ = skplt.metrics.plot_precision_recall(target, predicted_probas, ax=ax1)
            figures += [
                {
                    "title": "Precision Recall Curve",
                    "fname": "precision_recall_curve.png",
                    "figure": fig,
                }
            ]
            plt.close("all")
        except Exception as e:
            print(str(e))

        return figures

    @staticmethod
    def plots_regression(target, predictions):
        figures = []
        try:
            MAX_SAMPLES = 5000
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            samples = target.shape[0]
            if samples > MAX_SAMPLES:
                samples = MAX_SAMPLES
            ax1.scatter(
                target[:samples], predictions[:samples], c="tab:blue", alpha=0.2
            )
            plt.xlabel("True values")
            plt.ylabel("Predicted values")
            plt.title(f"Target values vs Predicted values (samples={samples})")
            plt.tight_layout(pad=5.0)
            figures += [
                {
                    "title": "True vs Predicted",
                    "fname": "true_vs_predicted.png",
                    "figure": fig,
                }
            ]

            # residual plot
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            residuals = target[:samples].values - predictions[:samples].values
            ax1.scatter(predictions[:samples], residuals, c="tab:blue", alpha=0.2)
            plt.xlabel("Predicted values")
            plt.ylabel("Residuals")
            plt.title(f"Predicted values vs Residuals (samples={samples})")
            plt.tight_layout(pad=5.0)
            bb = ax1.get_position()

            ax2 = fig.add_axes((bb.x0 + bb.size[0], bb.y0, 0.05, bb.size[1]))
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.hist(residuals, 50, orientation="horizontal", alpha=0.5)
            ax2.axis("off")

            figures += [
                {
                    "title": "Predicted vs Residuals",
                    "fname": "predicted_vs_residuals.png",
                    "figure": fig,
                }
            ]
            plt.close("all")

        except Exception as e:
            print(str(e))
        return figures

    @staticmethod
    def append(fout, model_path, plots):
        try:
            for plot in plots:
                fname = plot.get("fname")
                fig = plot.get("figure")
                title = plot.get("title", "")
                fig.savefig(os.path.join(model_path, fname))
                fout.write(f"\n## {title}\n\n")
                fout.write(f"![{title}]({fname})\n\n")
        except Exception as e:
            print(str(e))
