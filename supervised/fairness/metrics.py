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
        privileged_groups=[],
        underprivileged_groups=[],
        previous_fairness_optimization=None,
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

            privileged_value, underprivileged_value = None, None
            for pg in privileged_groups:
                if col_name in pg:
                    privileged_value = pg.get(col_name)
            for upg in underprivileged_groups:
                if col_name in upg:
                    underprivileged_value = upg.get(col_name)

            if privileged_value is not None:
                for i, v in enumerate(values):
                    if v == privileged_value:
                        # starting from 1 because first selection rate is for all samples
                        max_selection_rate = selection_rates[i + 1]

            if underprivileged_value is not None:
                for i, v in enumerate(values):
                    if v == underprivileged_value:
                        # starting from 1 because first selection rate is for all samples
                        min_selection_rate = selection_rates[i + 1]

            demographic_parity_diff = np.round(
                max_selection_rate - min_selection_rate, 4
            )
            demographic_parity_ratio = np.round(
                min_selection_rate / max_selection_rate, 4
            )

            tpr_min = np.min(tprs[1:])
            tpr_max = np.max(tprs[1:])

            fpr_min = np.min(fprs[1:])
            fpr_max = np.max(fprs[1:])

            if privileged_value is not None:
                for i, v in enumerate(values):
                    if v == privileged_value:
                        # starting from 1 because first value is for all samples
                        tpr_max = tprs[i + 1]
                        fpr_max = fprs[i + 1]

            if underprivileged_value is not None:
                for i, v in enumerate(values):
                    if v == underprivileged_value:
                        # starting from 1 because first value is for all samples
                        tpr_min = tprs[i + 1]
                        fpr_min = fprs[i + 1]

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

            if "parity" in fairness_metric:
                if privileged_value is None:
                    ind = np.argmax(selection_rates[1:])
                    privileged_value = values[ind]
                if underprivileged_value is None:
                    ind = np.argmin(selection_rates[1:])
                    underprivileged_value = values[ind]

            if "odds" in fairness_metric:
                if tpr_max - tpr_min > fpr_max - fpr_min:
                    if privileged_value is None:
                        ind = np.argmax(tprs[1:])
                        privileged_value = values[ind]
                    if underprivileged_value is None:
                        ind = np.argmin(tprs[1:])
                        underprivileged_value = values[ind]
                else:
                    if privileged_value is None:
                        ind = np.argmax(fprs[1:])
                        privileged_value = values[ind]
                    if underprivileged_value is None:
                        ind = np.argmin(fprs[1:])
                        underprivileged_value = values[ind]

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
                "is_fair": is_fair,
                "privileged_value": privileged_value,
                "underprivileged_value": underprivileged_value,
            }

        # fairness optimization stats

        sensitive_values = {}
        for col in sensitive_features.columns:
            col_name = col[10:]  # skip 'senstive_'
            values = list(sensitive_features[col].unique())
            sensitive_values[col] = values

            for v in values:
                ii = sensitive_features[col] == v

            new_sensitive_values = {}
            for k, prev_values in sensitive_values.items():
                if k == col:
                    continue
                new_sensitive_values[f"{k}@{col}"] = []
                for v in values:
                    for pv in prev_values:
                        if isinstance(pv, tuple):
                            new_sensitive_values[f"{k}@{col}"] += [(*pv, v)]
                        else:
                            new_sensitive_values[f"{k}@{col}"] += [(pv, v)]

            sensitive_values = {**sensitive_values, **new_sensitive_values}

        # print(sensitive_values)

        sensitive_indices = {}
        for k, values_list in sensitive_values.items():
            if k.count("@") == sensitive_features.shape[1] - 1:
                # print(k)
                # print("values_list",values_list)
                cols = k.split("@")
                for values in values_list:
                    if not isinstance(values, tuple):
                        values = (values,)
                    # print("values", values)

                    ii = None
                    for i, c in enumerate(cols):
                        if ii is None:
                            ii = sensitive_features[c] == values[i]
                        else:
                            ii &= sensitive_features[c] == values[i]

                    key = "@".join([str(s) for s in values])
                    # print(key, np.sum(ii))
                    sensitive_indices[key] = ii

        total_dp_ratio = min_selection_rate / max_selection_rate
        # print("total dp ratio", total_dp_ratio)

        c0 = np.sum(target == 0)
        c1 = np.sum(target == 1)

        selection_rates = {}
        weights = {}

        for key, indices in sensitive_indices.items():
            selection_rates[key] = np.sum((preds == 1) & indices) / np.sum(indices)
            # print(key, np.sum(indices), selection_rates[key])

            t = np.sum(indices)
            t0 = np.sum(indices & (target == 0))
            t1 = np.sum(indices & (target == 1))

            w0 = t / target.shape[0] * c0 / t0
            w1 = t / target.shape[0] * c1 / t1

            # print("----", key, w0, w1, t, t0, t1)
            weights[key] = [w0, w1]

        max_selection_rate = np.max(list(selection_rates.values()))
        min_selection_rate = np.min(list(selection_rates.values()))

        for k, v in selection_rates.items():
            selection_rates[k] = v / max_selection_rate

        # print("previous fairness optimization")
        # print(previous_fairness_optimization)
        # print("********")

        previous_weights = {}
        if previous_fairness_optimization is not None:

            weights = previous_fairness_optimization.get("weights")
            for key, indices in sensitive_indices.items():
                # print("Previous")
                # print(previous_fairness_optimization["selection_rates"][key], selection_rates[key])

                direction = 0.0
                if (
                    previous_fairness_optimization["selection_rates"][key]
                    < selection_rates[key]
                ):
                    # print("Improvement")
                    direction = 1.0
                elif selection_rates[key] > 0.8:
                    # print("GOOD")
                    direction = 0.0
                else:
                    # print("Decrease")
                    direction = -0.5

                # need to add previous weights instead 1.0
                prev_weights = previous_fairness_optimization.get(
                    "previous_weights", {}
                ).get(key, [1, 1])
                # print("prev_weights", prev_weights)
                delta0 = weights[key][0] - prev_weights[0]
                delta1 = weights[key][1] - prev_weights[1]

                previous_weights[key] = [weights[key][0], weights[key][1]]

                # print("BEFORE")
                # print(weights[key])
                weights[key][0] += direction * delta0
                weights[key][1] += direction * delta1
                # print("AFTER")
                # print(weights[key])
                # print(previous_fairness_optimization["weights"][key])

        step = None
        if previous_fairness_optimization is not None:
            step = previous_fairness_optimization.get("step")

        if step is None:
            step = 0
        else:
            step += 1

        fairness_metrics["fairness_optimization"] = {
            "selection_rates": selection_rates,
            "previous_weights": previous_weights,
            "weights": weights,
            "total_dp_ratio": total_dp_ratio,
            "step": step,
            "fairness_threshold": fairness_threshold,
        }

        return fairness_metrics

    @staticmethod
    def save_binary_classification(fairness_metrics, fout, model_path):

        for k, v in fairness_metrics.items():
            if k == "fairness_optimization":
                continue
            fout.write(f"\n\n## Fairness metrics for {k} feature\n\n")
            fout.write(v["metrics"].to_markdown())
            fout.write("\n\n")
            fout.write(v["stats"].to_markdown())
            fout.write("\n\n")

            fout.write(f"\n\n## Is model fair for {k} feature?\n")
            fair_str = "fair" if v["is_fair"] else "unfair"
            fairness_threshold = fairness_metrics.get("fairness_optimization", {}).get(
                "fairness_threshold"
            )
            fairness_threshold_str = ""
            if fairness_threshold is not None:
                if "ratio" in v["fairness_metric_name"].lower():
                    fairness_threshold_str = (
                        f"It should be higher than {fairness_threshold}."
                    )
                else:
                    fairness_threshold_str = (
                        f"It should be lower than {fairness_threshold}."
                    )

            fout.write(f"Model is {fair_str} for {k} feature.\n")
            fout.write(
                f'The {v["fairness_metric_name"]} is {v["fairness_metric_value"]}. {fairness_threshold_str}\n'
            )
            if not v["is_fair"]:
                # display information about privileged and underprivileged groups
                # for unfair models
                if v.get("underprivileged_value") is not None:
                    fout.write(
                        f'Underprivileged value is {v["underprivileged_value"]}.\n'
                    )
                if v.get("privileged_value") is not None:
                    fout.write(f'Privileged value is {v["privileged_value"]}.\n')

            for figure in v["figures"]:
                fout.write(f"\n\n### {figure['title']}\n\n")
                figure["figure"].savefig(os.path.join(model_path, figure["fname"]))
                fout.write(f"\n![]({figure['fname']})\n\n")
