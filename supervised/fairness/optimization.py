import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class FairnessOptimization:
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
        min_selection_rate=None,
        max_selection_rate=None,
    ):

        target = np.array(target).ravel()
        preds = np.array(predicted_labels)

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

        return {
            "selection_rates": selection_rates,
            "previous_weights": previous_weights,
            "weights": weights,
            "total_dp_ratio": total_dp_ratio,
            "step": step,
            "fairness_threshold": fairness_threshold,
        }

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
        performance_metric=None,
        performance_metric_name=None,
    ):
        target = np.array(target).ravel()
        preds = np.array(predictions)

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

        sensitive_indices = {}
        least_frequent_key = None
        least_frequency = sensitive_features.shape[0]
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
                    if np.sum(ii) < least_frequency:
                        least_frequency = np.sum(ii)
                        least_frequent_key = key

        weights = {}
        performance = {}

        for key, indices in sensitive_indices.items():
            w = target.shape[0] / len(sensitive_indices) / np.sum(indices)
            weights[key] = w
            performance[key] = performance_metric(target[indices], predictions[indices])

        # try to upscale more the largest weight
        weights[least_frequent_key] *= 1.5

        denominator = np.max(list(performance.values()))
        new_performance = {}
        for k, v in performance.items():
            new_performance[k] = np.round(v / denominator, 4)
        performance = new_performance

        previous_weights = {}
        if previous_fairness_optimization is not None:

            weights = previous_fairness_optimization.get("weights")
            for key, indices in sensitive_indices.items():
                direction = 0.0
                if (
                    previous_fairness_optimization["performance"][key]
                    < performance[key]
                ):
                    direction = 1.0
                elif performance[key] > fairness_threshold:
                    direction = 0.0
                else:
                    direction = -0.5

                # need to add previous weights instead 1.0
                prev_weights = previous_fairness_optimization.get(
                    "previous_weights", {}
                ).get(key, 1)
                delta0 = weights[key] - prev_weights
                previous_weights[key] = weights[key]
                weights[key] = max(weights[key] + direction * delta0, 0.01)

        no_weights_change = False
        if str(previous_weights) == str(weights):
            no_weights_change = True

        step = None
        if previous_fairness_optimization is not None:
            step = previous_fairness_optimization.get("step")

        if step is None:
            step = 0
        else:
            if not no_weights_change:
                step += 1

        return {
            "performance": performance,
            "previous_weights": previous_weights,
            "weights": weights,
            "step": step,
            "fairness_threshold": fairness_threshold,
        }

    @staticmethod
    def multiclass_classification(
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
        target_values = list(np.unique(target))

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

        sensitive_indices = {}
        for k, values_list in sensitive_values.items():
            if k.count("@") == sensitive_features.shape[1] - 1:
                cols = k.split("@")
                for values in values_list:
                    if not isinstance(values, tuple):
                        values = (values,)

                    ii = None
                    for i, c in enumerate(cols):
                        if ii is None:
                            ii = sensitive_features[c] == values[i]
                        else:
                            ii &= sensitive_features[c] == values[i]

                    key = "@".join([str(s) for s in values])
                    sensitive_indices[key] = ii

        cs = {}
        for t in target_values:
            cs[t] = np.sum(target == t)
        selection_rates = {}
        weights = {}

        for key, indices in sensitive_indices.items():
            weights[key] = []
            sv = np.sum(indices)
            selection_rates[key] = {}
            for t in target_values:
                selection_rates[key][t] = np.sum((preds == t) & indices) / np.sum(indices)
                
                t_k = np.sum(indices & (target == t))
                w_k = sv / target.shape[0] * cs[t] / t_k
                weights[key] += [w_k]

        for t in target_values:
            values = []
            for k, v in selection_rates.items():
                values += [v[t]]
            max_selection_rate = np.max(values)
            for k, v in selection_rates.items():
                v[t] /= max_selection_rate

        previous_weights = {}
        if previous_fairness_optimization is not None:
            weights = previous_fairness_optimization.get("weights")
            for key, indices in sensitive_indices.items():
                previous_weights[key] = [1]*len(target_values)
                for i, t in enumerate(target_values):
                    direction = 0.0
                    if (
                        previous_fairness_optimization["selection_rates"][key][t]
                        < selection_rates[key][t]
                    ):
                        direction = 1.0
                    elif selection_rates[key][t] > 0.8:
                        direction = 0.0
                    else:
                        direction = -0.5

                    # need to add previous weights instead 1.0
                    prev_weights = previous_fairness_optimization.get(
                        "previous_weights", {}
                    ).get(key, [1]*len(target_values))

                    delta_i = weights[key][i] - prev_weights[i]

                    previous_weights[key][i] = weights[key][i] 

                    weights[key][i] += direction * delta_i
                    
        step = None
        if previous_fairness_optimization is not None:
            step = previous_fairness_optimization.get("step")

        if step is None:
            step = 0
        else:
            step += 1

        return {
            "selection_rates": selection_rates,
            "previous_weights": previous_weights,
            "weights": weights,
            "step": step,
            "fairness_threshold": fairness_threshold,
            "target_values": target_values
        }

    