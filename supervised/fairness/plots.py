import numpy as np
from matplotlib import pyplot as plt


class FairnessPlots:
    @staticmethod
    def binary_classification(
        fairness_metric,
        col_name,
        metrics,
        selection_rates,
        max_selection_rate,
        fairness_threshold,
    ):

        figures = []
        # selection rate figure
        fair_selection_rate = max_selection_rate * fairness_threshold

        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(1, 1, 1)
        bars = ax1.bar(metrics.index[1:], metrics["Selection Rate"][1:])

        ax1.spines[["right", "top", "left"]].set_visible(False)
        ax1.yaxis.set_visible(False)
        _ = ax1.bar_label(bars, padding=5)

        if fairness_metric == "demographic_parity_ratio":
            ax1.axhline(y=fair_selection_rate, zorder=0, color="grey", ls="--", lw=1.5)
            _ = ax1.text(
                y=fair_selection_rate,
                x=-0.6,
                s="Fairness threshold",
                ha="center",
                fontsize=12,
                bbox=dict(facecolor="white", edgecolor="grey", ls="--"),
            )
            _ = ax1.text(
                y=1.2 * fair_selection_rate,
                x=-0.6,
                s="Fair",
                ha="center",
                fontsize=12,
            )
            _ = ax1.text(
                y=0.8 * fair_selection_rate,
                x=-0.6,
                s="Unfair",
                ha="center",
                fontsize=12,
            )

            ax1.axhspan(
                fairness_threshold * max_selection_rate,
                1.25 * np.max(selection_rates[1:]),
                color="green",
                alpha=0.05,
            )
            ax1.axhspan(
                0, fairness_threshold * max_selection_rate, color="red", alpha=0.05
            )

        figures += [
            {
                "title": f"Selection Rate for {col_name}",
                "fname": f"selection_rate_{col_name}.png",
                "figure": fig,
            }
        ]

        fig, axes = plt.subplots(figsize=(10, 5), ncols=2, sharey=True)
        fig.tight_layout()
        bars = axes[0].barh(
            metrics.index[1:],
            metrics["False Negative Rate"][1:],
            zorder=10,
            color="tab:orange",
        )
        xmax = 1.2 * max(
            metrics["False Negative Rate"][1:].max(),
            metrics["False Positive Rate"][1:].max(),
        )
        axes[0].set_xlim(0, xmax)
        axes[0].invert_xaxis()
        axes[0].set_title("False Negative Rate")
        _ = axes[0].bar_label(bars, padding=5)

        bars = axes[1].barh(
            metrics.index[1:],
            metrics["False Positive Rate"][1:],
            zorder=10,
            color="tab:blue",
        )
        axes[1].tick_params(axis="y", colors="tab:orange")  # tick color
        axes[1].set_xlim(0, xmax)
        axes[1].set_title("False Positive Rate")
        _ = axes[1].bar_label(bars, padding=5)
        _ = plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)

        figures += [
            {
                "title": f"False Rates for {col_name}",
                "fname": f"false_rates_{col_name}.png",
                "figure": fig,
            }
        ]

        return figures

    @staticmethod
    def regression(fairness_metric, col_name, metrics, fairness_metric_name):
        figures = []
        metric_name = fairness_metric.split("@")[1].upper()

        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(1, 1, 1)
        bars = ax1.bar(metrics.index[1:], metrics[metric_name][1:])

        ax1.spines[["right", "top"]].set_visible(False)
        # ax1.yaxis.set_visible(False)
        ax1.set_ylabel(metric_name)
        _ = ax1.bar_label(bars, padding=5)

        figures += [
            {
                "title": f"{metric_name} for {col_name}",
                "fname": f"{metric_name}_{col_name}.png",
                "figure": fig,
            }
        ]

        return figures
