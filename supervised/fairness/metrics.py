import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)

from supervised.fairness.optimization import FairnessOptimization
from supervised.fairness.plots import FairnessPlots
from supervised.fairness.utils import (accuracy, false_negative_rate,
                                       false_positive_rate, selection_rate,
                                       true_negative_rate, true_positive_rate)
from supervised.utils.metric import pearson, spearman


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

            fairness_metrics[col_name] = {
                "metrics": metrics,
                "stats": stats,
                "figures": FairnessPlots.binary_classification(
                    fairness_metric,
                    col_name,
                    metrics,
                    selection_rates,
                    max_selection_rate,
                    fairness_threshold,
                ),
                "fairness_metric_name": fairness_metric_name,
                "fairness_metric_value": fairness_metric_value,
                "is_fair": is_fair,
                "privileged_value": privileged_value,
                "underprivileged_value": underprivileged_value,
            }

        # fairness optimization stats
        fairness_metrics[
            "fairness_optimization"
        ] = FairnessOptimization.binary_classification(
            target,
            predicted_labels,
            sensitive_features,
            fairness_metric,
            fairness_threshold,
            privileged_groups,
            underprivileged_groups,
            previous_fairness_optimization,
            min_selection_rate,
            max_selection_rate,
        )

        return fairness_metrics

    @staticmethod
    def regression(
        target,
        predictions,
        sensitive_features,
        fairness_metric,
        fairness_threshold,
        privileged_groups=[],
        underprivileged_groups=[],
        previous_fairness_optimization=None,
    ):
        metric_name = fairness_metric.split("@")[1].upper()

        if "ratio" in fairness_metric.lower():
            fairness_metric_name = f"Group Loss Ratio @ {metric_name}"
        else:
            fairness_metric_name = f"Group Loss Difference @ {metric_name}"

        fairness_metrics = {}

        regression_metrics = {
            "SAMPLES": lambda t, p, sw=None: t.shape[0],
            "MAE": mean_absolute_error,
            "MSE": mean_squared_error,
            "RMSE": lambda t, p, sample_weight=None: np.sqrt(
                mean_squared_error(t, p, sample_weight=sample_weight)
            ),
            "R2": r2_score,
            "MAPE": mean_absolute_percentage_error,
            "SPEARMAN": spearman,
            "PEARSON": pearson,
        }
        overall = {}
        for k, v in regression_metrics.items():
            overall[k] = v(target, predictions)

        for col in sensitive_features.columns:
            col_name = col[10:]  # skip 'senstive_'

            values = sensitive_features[col].unique()
            all_metrics = [overall]

            for value in values:
                metrics = {}
                for k, v in regression_metrics.items():
                    metrics[k] = v(
                        target[sensitive_features[col] == value],
                        predictions[sensitive_features[col] == value],
                    )
                all_metrics += [metrics]

            mdf = pd.DataFrame(all_metrics, index=["Overall"] + list(values))

            privileged_value, underprivileged_value = None, None
            for pg in privileged_groups:
                if col_name in pg:
                    privileged_value = pg.get(col_name)
            for upg in underprivileged_groups:
                if col_name in upg:
                    underprivileged_value = upg.get(col_name)

            if privileged_value is None:
                if metric_name in ["R2", "SPEARMAN", "PEARSON"]:
                    # the higher the better
                    privileged_value = mdf.index[
                        mdf[metric_name][1:].argmax() + 1
                    ]  # without overall metrics
                else:
                    # the lower the better
                    privileged_value = mdf.index[
                        mdf[metric_name][1:].argmin() + 1
                    ]  # without overall metrics

            if underprivileged_value is None:
                if metric_name in ["R2", "SPEARMAN", "PEARSON"]:
                    # the higher the better
                    underprivileged_value = mdf.index[
                        mdf[metric_name][1:].argmin() + 1
                    ]  # without overall metrics
                else:
                    # the lower the better
                    underprivileged_value = mdf.index[
                        mdf[metric_name][1:].argmax() + 1
                    ]  # without overall metrics

            metric_min = mdf[metric_name].loc[privileged_value]
            metric_max = mdf[metric_name].loc[underprivileged_value]

            ratio = np.round(metric_min / metric_max, 4)
            diff = np.round(metric_max - metric_min, 4)

            # ratio = np.round(mdf[metric_name][1:].min()/mdf[metric_name][1:].max(), 4)
            # diff = np.round(mdf[metric_name][1:].max()-mdf[metric_name][1:].min(), 4)

            is_fair = False
            if "ratio" in fairness_metric.lower():
                fairness_metric_value = ratio
                if ratio > fairness_threshold:
                    is_fair = True
            else:
                fairness_metric_value = diff
                if diff < fairness_threshold:
                    is_fair = True

            fairness_metrics[col_name] = {
                "metrics": mdf,
                "figures": FairnessPlots.regression(
                    fairness_metric, col_name, mdf, fairness_metric_name
                ),
                "privileged_value": privileged_value,
                "underprivileged_value": underprivileged_value,
                "ratio": ratio,
                "diff": diff,
                "metric_name": metric_name,
                "fairness_metric_name": fairness_metric_name,
                "fairness_metric_value": fairness_metric_value,
                "is_fair": is_fair,
                "fairness_threshold": fairness_threshold,
            }

        fairness_metrics["fairness_optimization"] = FairnessOptimization.regression(
            target,
            predictions,
            sensitive_features,
            fairness_metric,
            fairness_threshold,
            privileged_groups,
            underprivileged_groups,
            previous_fairness_optimization,
            performance_metric=regression_metrics[metric_name],
            performance_metric_name=metric_name,
        )

        return fairness_metrics

    @staticmethod
    def multiclass_classification(
        original_target,
        predicted_labels,
        sensitive_features,
        fairness_metric,
        fairness_threshold,
        privileged_groups=[],
        underprivileged_groups=[],
        previous_fairness_optimization=None,
    ):
        original_target = np.array(original_target).ravel()
        predicted_labels = np.array(predicted_labels)
        target_values = list(np.unique(original_target))

        fairness_metrics = {}

        for col in sensitive_features.columns:
            col_name = col[10:]  # skip 'senstive_'

            for target_value in target_values:
                # we need to reset them for each target value
                privileged_value, underprivileged_value = None, None
                for pg in privileged_groups:
                    if col_name in pg:
                        privileged_value = pg.get(col_name)
                for upg in underprivileged_groups:
                    if col_name in upg:
                        underprivileged_value = upg.get(col_name)

                target = np.copy(original_target)
                target[original_target == target_value] = 1
                target[original_target != target_value] = 0

                preds = np.copy(predicted_labels)
                preds[predicted_labels == target_value] = 1
                preds[predicted_labels != target_value] = 0

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

                equalized_odds_diff = np.round(
                    max(tpr_max - tpr_min, fpr_max - fpr_min), 4
                )
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

                fairness_metrics[f"{col_name}__{target_value}"] = {
                    "metrics": metrics,
                    "stats": stats,
                    "figures": FairnessPlots.binary_classification(
                        fairness_metric,
                        f"{col_name}__{target_value}",
                        metrics,
                        selection_rates,
                        max_selection_rate,
                        fairness_threshold,
                    ),
                    "fairness_metric_name": fairness_metric_name,
                    "fairness_metric_value": fairness_metric_value,
                    "is_fair": is_fair,
                    "privileged_value": privileged_value,
                    "underprivileged_value": underprivileged_value,
                }

        # fairness optimization stats
        fairness_metrics[
            "fairness_optimization"
        ] = FairnessOptimization.multiclass_classification(
            original_target,
            predicted_labels,
            sensitive_features,
            fairness_metric,
            fairness_threshold,
            privileged_groups,
            underprivileged_groups,
            previous_fairness_optimization,
        )

        return fairness_metrics
