import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def accuracy(t, y):
    return np.round(np.sum(t == y) / t.shape[0], 4)


def selection_rate(y):
    return np.round(
        np.sum((y == 1)) / y.shape[0],
        4,
    )


def true_positive_rate(t, y):
    return np.round(
        np.sum((y == 1) & (t == 1)) / np.sum((t == 1)),
        4,
    )


def false_positive_rate(t, y):
    return np.round(
        np.sum((y == 1) & (t == 0)) / np.sum((t == 0)),
        4,
    )


def true_negative_rate(t, y):
    return np.round(
        np.sum((y == 0) & (t == 0)) / np.sum((t == 0)),
        4,
    )


def false_negative_rate(t, y):
    return np.round(
        np.sum((y == 0) & (t == 1)) / np.sum((t == 1)),
        4,
    )


class FairnessMetrics:
    @staticmethod
    def binary_classification(
        target,
        predicted_labels,
        sensitive_features,
        fairness_metric,
        fairness_threshold,
        protected_groups=[],
    ):

        target = np.array(target).ravel()
        preds = np.array(predicted_labels)

        fairness_metrics = {}

        for col in sensitive_features.columns:

            col_name = col[10:]  # skip 'senstive_'

            accuracies = []
            selection_rates = []
            tprs = []
            fprs = []
            tnrs = []
            fnrs = []
            samples = []
            demographic_parity_diff = None
            demographic_parity_ratio = None
            equalized_odds_diff = None
            equalized_odds_ratio = None

            # overall
            accuracies += [accuracy(target, preds)]
            selection_rates += [selection_rate(preds)]
            tprs += [true_positive_rate(target, preds)]
            fprs += [false_positive_rate(target, preds)]
            tnrs += [true_negative_rate(target, preds)]
            fnrs += [false_negative_rate(target, preds)]
            samples += [target.shape[0]]

            values = sensitive_features[col].unique()

            for value in values:
                accuracies += [
                    accuracy(
                        target[sensitive_features[col] == value],
                        preds[sensitive_features[col] == value],
                    )
                ]
                selection_rates += [
                    selection_rate(preds[sensitive_features[col] == value])
                ]
                tprs += [
                    true_positive_rate(
                        target[sensitive_features[col] == value],
                        preds[sensitive_features[col] == value],
                    )
                ]
                fprs += [
                    false_positive_rate(
                        target[sensitive_features[col] == value],
                        preds[sensitive_features[col] == value],
                    )
                ]
                tnrs += [
                    true_negative_rate(
                        target[sensitive_features[col] == value],
                        preds[sensitive_features[col] == value],
                    )
                ]
                fnrs += [
                    false_negative_rate(
                        target[sensitive_features[col] == value],
                        preds[sensitive_features[col] == value],
                    )
                ]
                samples += [np.sum([sensitive_features[col] == value])]

            metrics = pd.DataFrame(
                {
                    "Samples": samples,
                    "Accuracy": accuracies,
                    "Selection Rate": selection_rates,
                    "True Positive Rate": tprs,
                    "False Negative Rate": fnrs,
                    "False Positive Rate": fprs,
                    "True Negative Rate": tnrs,
                },
                index=["Overall"] + list(values),
            )

            max_selection_rate = np.max(selection_rates[1:])
            min_selection_rate = np.min(selection_rates[1:])

            protected_value = None
            for pg in protected_groups:
                if col_name in pg:
                    protected_value = pg.get(col_name)

            if protected_value is not None:
                for i, v in enumerate(values):
                    if v == protected_value:
                        # starting from 1 because first selection rate is for all samples
                        min_selection_rate = selection_rates[i+1]

            demographic_parity_diff = np.round(max_selection_rate - min_selection_rate, 4)
            demographic_parity_ratio = np.round(
                min_selection_rate / max_selection_rate, 4
            )

            tpr_min = np.min(tprs[1:])
            tpr_max = np.max(tprs[1:])

            fpr_min = np.min(fprs[1:])
            fpr_max = np.max(fprs[1:])

            if protected_value is not None:
                for i, v in enumerate(values):
                    if v == protected_value:
                        # starting from 1 because first value is for all samples
                        tpr_min = tprs[i+1]
                        fpr_min = fprs[i+1]

            equalized_odds_diff = np.round(max(tpr_max - tpr_min, fpr_max - fpr_min), 4)
            equalized_odds_ratio = np.round(
                min(tpr_min / tpr_max, fpr_min / fpr_max), 4
            )

            stats = pd.DataFrame(
                {
                    "": [
                        demographic_parity_diff,
                        demographic_parity_ratio,
                        equalized_odds_diff,
                        equalized_odds_ratio,
                    ]
                },
                index=[
                    "Demographic Parity Difference",
                    "Demographic Parity Ratio",
                    "Equalized Odds Difference",
                    "Equalized Odds Ratio",
                ],
            )

            fairness_metric_name = ""
            fairness_metric_value = 0
            is_fair = False
            if fairness_metric == "demographic_parity_difference":
                fairness_metric_name = "Demographic Parity Difference"
                fairness_metric_value = demographic_parity_diff
                is_fair = demographic_parity_diff < fairness_threshold
            elif fairness_metric == "demographic_parity_ratio":
                fairness_metric_name = "Demographic Parity Ratio"
                fairness_metric_value = demographic_parity_ratio
                is_fair = demographic_parity_ratio > fairness_threshold
            elif fairness_metric == "equalized_odds_difference":
                fairness_metric_name = "Equalized Odds Difference"
                fairness_metric_value = equalized_odds_diff
                is_fair = equalized_odds_diff < fairness_threshold
            elif fairness_metric == "equalized_odds_ratio":
                fairness_metric_name = "Equalized Odds Ratio"
                fairness_metric_value = equalized_odds_ratio
                is_fair = equalized_odds_ratio > fairness_threshold
                
            
            figures = []
            # selection rate figure
            fair_selection_rate = max_selection_rate * fairness_threshold

            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 1, 1)
            bars = ax1.bar(metrics.index[1:], metrics["Selection Rate"][1:])

            ax1.spines[["right", "top", "left"]].set_visible(False)
            ax1.yaxis.set_visible(False)
            _ = ax1.bar_label(bars, padding=5)
            ax1.axhline(y=fair_selection_rate, zorder=0, color="grey", ls="--", lw=1.5)
            _ = ax1.text(
                y=fair_selection_rate,
                x=-0.6,
                s="Fairness threshold",
                ha="center",
                fontsize=12,
                bbox=dict(facecolor="white", edgecolor="grey", ls="--"),
            )
            ax1.axhspan(fairness_threshold*max_selection_rate, 1.25*max_selection_rate, facecolor='0.2', alpha=0.2)

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
            _ = plt.subplots_adjust(
                wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95
            )

            figures += [
                {
                    "title": f"False Rates for {col_name}",
                    "fname": f"false_rates_{col_name}.png",
                    "figure": fig,
                }
            ]

            fairness_metrics[col_name] = {
                "metrics": metrics,
                "stats": stats,
                "figures": figures,
                "fairness_metric_name": fairness_metric_name,
                "fairness_metric_value": fairness_metric_value,
                "is_fair": is_fair
            }
        

        return fairness_metrics

    @staticmethod
    def save_binary_classification(fairness_metrics, fout, model_path):

        for k, v in fairness_metrics.items():
            fout.write(f"\n\n## Fairness metrics for {k} feature\n\n")
            fout.write(v["metrics"].to_markdown())
            fout.write("\n\n")
            fout.write(v["stats"].to_markdown())
            fout.write("\n\n")

            fout.write(f"\n\n## Is model fair for {k} feature?\n")
            fair_str = "fair" if v["is_fair"] else "unfair"
            fout.write(f'Model is {fair_str} for {k} feature because {v["fairness_metric_name"]} is {v["fairness_metric_value"]}.\n\n')

            for figure in v["figures"]:
                fout.write(f"\n\n### {figure['title']}\n\n")
                figure["figure"].savefig(os.path.join(model_path, figure["fname"]))
                fout.write(f"\n![]({figure['fname']})\n\n")
