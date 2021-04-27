import os
import json
import logging
import copy
import numpy as np
import pandas as pd
import time
import uuid
import warnings

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from supervised.utils.metric import logloss

logger = logging.getLogger(__name__)
from supervised.utils.config import LOG_LEVEL

logger.setLevel(LOG_LEVEL)
from supervised.utils.learning_curves import LearningCurves
from supervised.utils.common import construct_learner_name, get_fold_repeat_cnt
from supervised.utils.additional_plots import AdditionalPlots
from tabulate import tabulate


class AdditionalMetrics:
    @staticmethod
    def binary_classification(target, predictions, sample_weight=None):

        negative_label, positive_label = "0", "1"
        mapping = None
        try:
            pred_col = predictions.columns[0]
            if "_0_for_" in pred_col and "_1_for_" in pred_col:
                t = pred_col.split("_0_for_")[1]
                t = t.split("_1_for_")
                negative_label, positive_label = t[0], t[1]
                mapping = {0: negative_label, 1: positive_label}
        except Exception as e:
            pass

        predictions = np.array(predictions)
        sorted_predictions = np.sort(predictions)
        STEPS = 100  # can go lower for speed increase ???
        details = {
            "threshold": [],
            "f1": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "mcc": [],
        }
        samples_per_step = max(1, np.floor(predictions.shape[0] / STEPS))

        for i in range(STEPS):
            idx = int(i * samples_per_step)
            if idx + 1 >= predictions.shape[0]:
                break
            if i == 0:
                th = 0.9 * np.min(sorted_predictions)
            else:
                th = float(
                    0.5 * (sorted_predictions[idx] + sorted_predictions[idx + 1])
                )

            if np.sum(predictions > th) < 1:
                break
            response = (predictions > th).astype(int)

            details["threshold"] += [th]
            details["f1"] += [f1_score(target, response, sample_weight=sample_weight)]
            details["accuracy"] += [
                accuracy_score(target, response, sample_weight=sample_weight)
            ]
            details["precision"] += [
                precision_score(target, response, sample_weight=sample_weight)
            ]
            details["recall"] += [
                recall_score(target, response, sample_weight=sample_weight)
            ]
            if i == 0:
                details["mcc"] += [0.0]
            else:
                details["mcc"] += [
                    matthews_corrcoef(target, response, sample_weight=sample_weight)
                ]

        # max metrics
        max_metrics = {
            "logloss": {
                "score": logloss(target, predictions, sample_weight=sample_weight),
                "threshold": None,
            },  # there is no threshold for LogLoss
            "auc": {
                "score": roc_auc_score(
                    target, predictions, sample_weight=sample_weight
                ),
                "threshold": None,
            },  # there is no threshold for AUC
            "f1": {
                "score": np.max(details["f1"]),
                "threshold": details["threshold"][np.argmax(details["f1"])],
            },
            "accuracy": {
                "score": np.max(details["accuracy"]),
                "threshold": details["threshold"][np.argmax(details["accuracy"])],
            },
            "precision": {
                "score": np.max(details["precision"]),
                "threshold": details["threshold"][np.argmax(details["precision"])],
            },
            "recall": {
                "score": np.max(details["recall"]),
                "threshold": details["threshold"][np.argmax(details["recall"])],
            },
            "mcc": {
                "score": np.max(details["mcc"]),
                "threshold": details["threshold"][np.argmax(details["mcc"])],
            },
        }

        threshold = float(max_metrics["accuracy"]["threshold"])

        # if sample_weight is not None:
        #    new_max_metrics = {}
        #    for k, v in max_metrics.items():
        #        new_max_metrics["weighted_" + k] = v
        #    max_metrics = new_max_metrics

        # confusion matrix

        conf_matrix = confusion_matrix(
            target, predictions > threshold, sample_weight=sample_weight
        )

        conf_matrix = pd.DataFrame(
            conf_matrix,
            columns=[
                f"Predicted as {negative_label}",
                f"Predicted as {positive_label}",
            ],
            index=[f"Labeled as {negative_label}", f"Labeled as {positive_label}"],
        )

        predicted_labels = pd.Series((predictions.ravel() > threshold).astype(int))
        predicted_probas = pd.DataFrame(
            {
                "proba_0": 1 - predictions.ravel(),
                "proba_1": predictions.ravel(),
            }
        )

        if mapping is not None:
            labeled_target = target["target"].map(mapping)
            predicted_labels = predicted_labels.map(mapping)
        else:
            labeled_target = target

        return {
            "metric_details": pd.DataFrame(details),
            "max_metrics": pd.DataFrame(max_metrics),
            "confusion_matrix": conf_matrix,
            "threshold": threshold,
            "additional_plots": AdditionalPlots.plots_binary(
                labeled_target, predicted_labels, predicted_probas
            ),
        }

    @staticmethod
    def multiclass_classification(target, predictions, sample_weight=None):
        all_labels = [i[11:] for i in predictions.columns.tolist()[:-1]]

        predicted_probas = predictions[predictions.columns[:-1]]
        ll = logloss(
            target, predictions[predictions.columns[:-1]], sample_weight=sample_weight
        )

        if "target" in target.columns.tolist():
            # multiclass coding with integer
            labels = {i: l for i, l in enumerate(all_labels)}
            target = target["target"].map(labels)
        else:
            # multiclass coding with one-hot encoding
            old_columns = target.columns
            t = target[old_columns[0]]
            for l in all_labels:
                t[target[f"target_{l}"] == 1] = l

            target = pd.DataFrame({"target": t})

        # Print the confusion matrix
        predicted_labels = predictions["label"]
        predictions = predictions["label"]
        if not pd.api.types.is_string_dtype(predictions):
            predictions = predictions.astype(str)

        if not pd.api.types.is_string_dtype(target):
            target = target.astype(str)

        conf_matrix = confusion_matrix(
            target, predictions, labels=all_labels, sample_weight=sample_weight
        )

        rows = [f"Predicted as {a}" for a in all_labels]
        cols = [f"Labeled as {a}" for a in all_labels]

        conf_matrix = pd.DataFrame(conf_matrix, columns=rows, index=cols)

        max_metrics = classification_report(
            target,
            predictions,
            digits=6,
            labels=all_labels,
            output_dict=True,
            sample_weight=sample_weight,
        )
        max_metrics["logloss"] = ll

        return {
            "max_metrics": pd.DataFrame(max_metrics).transpose(),
            "confusion_matrix": conf_matrix,
            "additional_plots": AdditionalPlots.plots_multiclass(
                target, predicted_labels, predicted_probas
            ),
        }

    @staticmethod
    def regression(target, predictions, sample_weight=None):
        regression_metrics = {
            "MAE": mean_absolute_error,
            "MSE": mean_squared_error,
            "RMSE": lambda t, p, sample_weight: np.sqrt(
                mean_squared_error(t, p, sample_weight=sample_weight)
            ),
            "R2": r2_score,
            "MAPE": mean_absolute_percentage_error,
        }
        max_metrics = {}
        for k, v in regression_metrics.items():
            max_metrics[k] = v(target, predictions, sample_weight=sample_weight)

        return {
            "max_metrics": pd.DataFrame(
                {
                    "Metric": list(max_metrics.keys()),
                    "Score": list(max_metrics.values()),
                }
            ),
            "additional_plots": AdditionalPlots.plots_regression(target, predictions),
        }

    @staticmethod
    def compute(target, predictions, sample_weight, ml_task):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if ml_task == BINARY_CLASSIFICATION:
                return AdditionalMetrics.binary_classification(
                    target, predictions, sample_weight
                )
            elif ml_task == MULTICLASS_CLASSIFICATION:
                return AdditionalMetrics.multiclass_classification(
                    target, predictions, sample_weight
                )
            elif ml_task == REGRESSION:
                return AdditionalMetrics.regression(target, predictions, sample_weight)

    @staticmethod
    def save(additional_metrics, ml_task, model_desc, model_path):
        try:
            fold_cnt, repeat_cnt = get_fold_repeat_cnt(model_path)
            if ml_task == BINARY_CLASSIFICATION:
                AdditionalMetrics.save_binary_classification(
                    additional_metrics, model_desc, model_path, fold_cnt, repeat_cnt
                )
            elif ml_task == MULTICLASS_CLASSIFICATION:
                AdditionalMetrics.save_multiclass_classification(
                    additional_metrics, model_desc, model_path, fold_cnt, repeat_cnt
                )
            elif ml_task == REGRESSION:
                AdditionalMetrics.save_regression(
                    additional_metrics, model_desc, model_path, fold_cnt, repeat_cnt
                )
        except Exception as e:
            logger.error(
                f"Exception while saving additional metrics. {str(e)}\nContinuing ..."
            )

    @staticmethod
    def add_learning_curves(fout):
        fout.write("\n\n## Learning curves\n")
        fout.write(f"![Learning curves]({LearningCurves.output_file_name})")

    @staticmethod
    def save_binary_classification(
        additional_metrics, model_desc, model_path, fold_cnt, repeat_cnt
    ):
        max_metrics = additional_metrics["max_metrics"].transpose()
        confusion_matrix = additional_metrics["confusion_matrix"]
        threshold = additional_metrics["threshold"]

        with open(os.path.join(model_path, "README.md"), "w", encoding="utf-8") as fout:
            fout.write(model_desc)
            fout.write("\n## Metric details\n{}\n\n".format(max_metrics.to_markdown()))
            fout.write(
                "\n## Confusion matrix (at threshold={})\n{}".format(
                    np.round(threshold, 6), confusion_matrix.to_markdown()
                )
            )
            AdditionalMetrics.add_learning_curves(fout)
            AdditionalMetrics.add_tree_viz(fout, model_path, fold_cnt, repeat_cnt)
            AdditionalMetrics.add_linear_coefs(fout, model_path, fold_cnt, repeat_cnt)
            AdditionalMetrics.add_permutation_importance(
                fout, model_path, fold_cnt, repeat_cnt
            )

            plots = additional_metrics.get("additional_plots")
            if plots is not None:
                AdditionalPlots.append(fout, model_path, plots)

            AdditionalMetrics.add_shap_importance(
                fout, model_path, fold_cnt, repeat_cnt
            )
            AdditionalMetrics.add_shap_binary(fout, model_path, fold_cnt, repeat_cnt)

            fout.write("\n\n[<< Go back](../README.md)\n")

    @staticmethod
    def save_multiclass_classification(
        additional_metrics, model_desc, model_path, fold_cnt, repeat_cnt
    ):
        max_metrics = additional_metrics["max_metrics"].transpose()
        confusion_matrix = additional_metrics["confusion_matrix"]

        with open(os.path.join(model_path, "README.md"), "w", encoding="utf-8") as fout:
            fout.write(model_desc)
            fout.write("\n### Metric details\n{}\n\n".format(max_metrics.to_markdown()))
            fout.write(
                "\n## Confusion matrix\n{}".format(confusion_matrix.to_markdown())
            )
            AdditionalMetrics.add_learning_curves(fout)
            AdditionalMetrics.add_tree_viz(fout, model_path, fold_cnt, repeat_cnt)
            AdditionalMetrics.add_linear_coefs(fout, model_path, fold_cnt, repeat_cnt)
            AdditionalMetrics.add_permutation_importance(
                fout, model_path, fold_cnt, repeat_cnt
            )

            plots = additional_metrics.get("additional_plots")
            if plots is not None:
                AdditionalPlots.append(fout, model_path, plots)

            AdditionalMetrics.add_shap_importance(
                fout, model_path, fold_cnt, repeat_cnt
            )
            AdditionalMetrics.add_shap_multiclass(
                fout, model_path, fold_cnt, repeat_cnt
            )

            fout.write("\n\n[<< Go back](../README.md)\n")

    @staticmethod
    def save_regression(
        additional_metrics, model_desc, model_path, fold_cnt, repeat_cnt
    ):
        max_metrics = additional_metrics["max_metrics"]
        with open(os.path.join(model_path, "README.md"), "w", encoding="utf-8") as fout:
            fout.write(model_desc)
            fout.write(
                "\n### Metric details:\n{}\n\n".format(
                    tabulate(max_metrics.values, max_metrics.columns, tablefmt="pipe")
                )
            )
            AdditionalMetrics.add_learning_curves(fout)
            AdditionalMetrics.add_tree_viz(fout, model_path, fold_cnt, repeat_cnt)
            AdditionalMetrics.add_linear_coefs(fout, model_path, fold_cnt, repeat_cnt)
            AdditionalMetrics.add_permutation_importance(
                fout, model_path, fold_cnt, repeat_cnt
            )

            plots = additional_metrics.get("additional_plots")
            if plots is not None:
                AdditionalPlots.append(fout, model_path, plots)

            AdditionalMetrics.add_shap_importance(
                fout, model_path, fold_cnt, repeat_cnt
            )
            AdditionalMetrics.add_shap_regression(
                fout, model_path, fold_cnt, repeat_cnt
            )

            fout.write("\n\n[<< Go back](../README.md)\n")

    @staticmethod
    def add_linear_coefs(fout, model_path, fold_cnt, repeat_cnt):

        coef_files = [f for f in os.listdir(model_path) if "_coefs.csv" in f]
        if not len(coef_files):
            return

        # check if multiclass
        df = pd.read_csv(os.path.join(model_path, coef_files[0]), index_col=0)
        if df.shape[0] > 100:
            return
        multiclass = df.shape[1] > 1

        if multiclass:
            fout.write("\n\n## Coefficients\n")

            for repeat in range(repeat_cnt):
                repeat_str = f", repeat #{repeat+1}" if repeat_cnt > 1 else ""
                for fold in range(fold_cnt):
                    learner_name = construct_learner_name(fold, repeat, repeat_cnt)
                    fname = learner_name + "_coefs.csv"
                    if fname in coef_files:
                        fout.write(
                            f"\n### Coefficients learner #{fold+1}{repeat_str}\n"
                        )
                        df = pd.read_csv(os.path.join(model_path, fname), index_col=0)
                        fout.write(df.to_markdown() + "\n")

        else:
            df_all = []
            for repeat in range(repeat_cnt):
                repeat_str = f"_Repeat_{repeat+1}" if repeat_cnt > 1 else ""
                for fold in range(fold_cnt):
                    learner_name = construct_learner_name(fold, repeat, repeat_cnt)
                    fname = learner_name + "_coefs.csv"
                    if fname in coef_files:
                        df = pd.read_csv(os.path.join(model_path, fname), index_col=0)
                        df.columns = [f"Learner_{fold+1}{repeat_str}"]
                        df_all += [df]

            df = pd.concat(df_all, axis=1)
            df["m"] = df.mean(axis=1)

            df = df.sort_values("m", axis=0, ascending=False)
            df = df.drop("m", axis=1)

            fout.write("\n\n## Coefficients\n")
            fout.write(df.to_markdown() + "\n")

    @staticmethod
    def add_tree_viz(fout, model_path, fold_cnt, repeat_cnt):

        tree_viz = [f for f in os.listdir(model_path) if "_tree.svg" in f]
        if len(tree_viz):
            fout.write("\n\n## Decision Tree \n")
            for repeat in range(repeat_cnt):
                repeat_str = f", Repeat #{repeat+1}" if repeat_cnt > 1 else ""
                for fold in range(fold_cnt):
                    learner_name = construct_learner_name(fold, repeat, repeat_cnt)
                    fname = learner_name + "_tree.svg"
                    if fname in tree_viz:
                        fout.write(f"\n### Tree #{fold+1}{repeat_str}\n")
                        fout.write(f"![Tree {fold+1}{repeat_str}]({fname})")
                    try:
                        fname = os.path.join(model_path, learner_name + "_rules.txt")
                        if os.path.exists(fname):
                            fout.write("\n\n### Rules\n\n")
                            with open(fname, "r") as fin:
                                fout.write(fin.read() + "\n\n")
                    except Exception as e:
                        logger.info("Problem with adding rules to report. " + str(e))

    @staticmethod
    def add_permutation_importance(fout, model_path, fold_cnt, repeat_cnt):
        # permutation importance
        imp_data = [
            f
            for f in os.listdir(model_path)
            if "_importance.csv" in f and "shap" not in f
        ]
        if not len(imp_data):
            return

        df_all = []
        for repeat in range(repeat_cnt):
            repeat_str = f", Repeat {repeat+1}" if repeat_cnt > 1 else ""
            for fold in range(fold_cnt):
                learner_name = construct_learner_name(fold, repeat, repeat_cnt)
                fname = learner_name + "_importance.csv"
                if fname in imp_data:
                    df = pd.read_csv(os.path.join(model_path, fname), index_col=0)
                    df.columns = [f"Learner {fold+1}{repeat_str}"]
                    df_all += [df]

        df = pd.concat(df_all, axis=1)

        df["m"] = df.mean(axis=1)
        df = df.sort_values(by="m", ascending=False)
        df = df.drop("m", axis=1)

        # limit to max 25 features in the plot
        ax = df.head(25).plot.barh(figsize=(10, 7))
        ax.invert_yaxis()
        ax.set_xlabel("Mean of feature importance")
        fig = ax.get_figure()
        fig.tight_layout(pad=2.0)
        if df.shape[0] > 25:
            ax.set_title("Top-25 important features")
        else:
            ax.set_title("Feature importance")

        fig.savefig(os.path.join(model_path, "permutation_importance.png"))
        fout.write("\n\n## Permutation-based Importance\n")
        fout.write(f"![Permutation-based Importance](permutation_importance.png)")

        if "random_feature" in df.index.tolist():

            df["counter"] = 0
            df = df.fillna(
                0
            )  # there might be not-used features between different learners
            max_counter = 0.0
            for col in df.columns:
                if "Learner" not in col:
                    continue
                score = max(0, df[col]["random_feature"]) + 1e-6
                df["counter"] += (df[col] <= score).astype(int)
                max_counter += 1.0

            """ version 1
            df["min_score"] = df.min(axis=1)
            df["max_score"] = df.max(axis=1)
            random_feature_score = max(
                0.0, float(df["max_score"]["random_feature"])
            )  # it should be at least 0
            drop_features = df.index[
                df["min_score"] < random_feature_score + 1e-6
            ].tolist()
            """

            # version 2 - should be better
            threshold = max_counter / 2.0
            drop_features = df.index[df["counter"] >= threshold].tolist()

            fname = os.path.join(os.path.dirname(model_path), "drop_features.json")
            with open(fname, "w") as fout:
                fout.write(json.dumps(drop_features, indent=4))

            fname = os.path.join(
                os.path.dirname(model_path),
                f"features_scores_threshold_{threshold}.csv",
            )
            df.to_csv(fname, index=False)

    @staticmethod
    def add_shap_importance(fout, model_path, fold_cnt, repeat_cnt):
        try:
            # SHAP Importance
            imp_data = [
                f for f in os.listdir(model_path) if "_shap_importance.csv" in f
            ]
            if not len(imp_data):
                return

            df_all = []
            for repeat in range(repeat_cnt):
                repeat_str = f", Repeat {repeat+1}" if repeat_cnt > 1 else ""
                for fold in range(fold_cnt):
                    learner_name = construct_learner_name(fold, repeat, repeat_cnt)
                    fname = learner_name + "_shap_importance.csv"
                    if fname in imp_data:
                        df = pd.read_csv(os.path.join(model_path, fname), index_col=0)
                        df.columns = [f"Learner {fold+1}{repeat_str}"]
                        df_all += [df]

            df = pd.concat(df_all, axis=1)

            df["m"] = df.mean(axis=1)
            df = df.sort_values(by="m", ascending=False)
            df = df.drop("m", axis=1)

            # limit to max 25 features in the plot
            ax = df.head(25).plot.barh(figsize=(10, 7))
            ax.invert_yaxis()
            ax.set_xlabel("mean(|SHAP value|) average impact on model output magnitude")
            fig = ax.get_figure()
            fig.tight_layout(pad=2.0)
            if df.shape[0] > 25:
                ax.set_title("SHAP Top-25 important features")
            else:
                ax.set_title("SHAP feature importance")
            fig.savefig(os.path.join(model_path, "shap_importance.png"))
            fout.write("\n\n## SHAP Importance\n")
            fout.write(f"![SHAP Importance](shap_importance.png)")
        except Exception as e:
            logger.error(
                f"Exception while saving SHAP importance. {str(e)}\nContinuing ..."
            )

    @staticmethod
    def add_shap_binary(fout, model_path, fold_cnt, repeat_cnt):
        try:
            # Dependence SHAP
            dep_plots = [
                f for f in os.listdir(model_path) if "_shap_dependence.png" in f
            ]
            if not len(dep_plots):
                return

            fout.write("\n\n## SHAP Dependence plots\n")
            for repeat in range(repeat_cnt):
                repeat_str = f", Repeat {repeat+1}" if repeat_cnt > 1 else ""
                for fold in range(fold_cnt):
                    learner_name = construct_learner_name(fold, repeat, repeat_cnt)
                    fname = learner_name + "_shap_dependence.png"
                    if fname in dep_plots:
                        fout.write(f"\n### Dependence (Fold {fold+1}{repeat_str})\n")
                        fout.write(
                            f"![SHAP Dependence from Fold {fold+1}{repeat_str}]({fname})"
                        )

            # SHAP Decisions
            dec_plots = [
                f
                for f in os.listdir(model_path)
                if "_shap_class" in f and "decisions.png" in f
            ]
            if not len(dec_plots):
                return

            fout.write("\n\n## SHAP Decision plots\n")
            for target in [0, 1]:
                for decision_type in ["worst", "best"]:
                    for repeat in range(repeat_cnt):
                        repeat_str = f", Repeat {repeat+1}" if repeat_cnt > 1 else ""
                        for fold in range(fold_cnt):
                            learner_name = construct_learner_name(
                                fold, repeat, repeat_cnt
                            )
                            fname = (
                                learner_name
                                + f"_shap_class_{target}_{decision_type}_decisions.png"
                            )
                            if fname in dec_plots:
                                fout.write(
                                    f"\n### Top-10 {decision_type.capitalize()} decisions for class {target} (Fold {fold+1}{repeat_str})\n"
                                )
                                fout.write(
                                    f"![SHAP {decision_type} decisions class {target} from Fold {fold+1}{repeat_str}]({fname})"
                                )

        except Exception as e:
            logger.error(
                f"Exception while saving SHAP explanations. {str(e)}\nContinuing ..."
            )

    @staticmethod
    def add_shap_regression(fout, model_path, fold_cnt, repeat_cnt):
        try:
            # Dependence SHAP
            dep_plots = [
                f for f in os.listdir(model_path) if "_shap_dependence.png" in f
            ]
            if not len(dep_plots):
                return

            fout.write("\n\n## SHAP Dependence plots\n")
            for repeat in range(repeat_cnt):
                repeat_str = f", Repeat {repeat+1}" if repeat_cnt > 1 else ""
                for fold in range(fold_cnt):
                    learner_name = construct_learner_name(fold, repeat, repeat_cnt)
                    fname = learner_name + "_shap_dependence.png"
                    if fname in dep_plots:
                        fout.write(f"\n### Dependence (Fold {fold+1}{repeat_str})\n")
                        fout.write(
                            f"![SHAP Dependence from Fold {fold+1}{repeat_str}]({fname})"
                        )

            # SHAP Decisions
            dec_plots = [f for f in os.listdir(model_path) if "decisions.png" in f]
            if not len(dec_plots):
                return

            fout.write("\n\n## SHAP Decision plots\n")
            for decision_type in ["worst", "best"]:
                for repeat in range(repeat_cnt):
                    repeat_str = f", Repeat {repeat+1}" if repeat_cnt > 1 else ""
                    for fold in range(fold_cnt):
                        learner_name = construct_learner_name(fold, repeat, repeat_cnt)
                        fname = learner_name + f"_shap_{decision_type}_decisions.png"
                        if fname in dec_plots:
                            fout.write(
                                f"\n### Top-10 {decision_type.capitalize()} decisions (Fold {fold+1}{repeat_str})\n"
                            )
                            fout.write(
                                f"![SHAP {decision_type} decisions from fold {fold+1}{repeat_str}]({fname})"
                            )
        except Exception as e:
            logger.error(
                f"Exception while saving SHAP explanations. {str(e)}\nContinuing ..."
            )

    @staticmethod
    def add_shap_multiclass(fout, model_path, fold_cnt, repeat_cnt):
        try:
            # Dependence SHAP
            dep_plots = [f for f in os.listdir(model_path) if "_shap_dependence" in f]
            if not len(dep_plots):
                return

            # get number of classes
            start_ind = 0
            for i, a in enumerate(dep_plots[0].split("_")):
                if a == "class":
                    start_ind = i + 1
                    break

            classes = []
            for l in dep_plots:
                a = l.split("_")
                classes += ["".join(a[start_ind:])[:-4]]
            classes = np.unique(classes)

            fout.write("\n\n## SHAP Dependence plots\n")

            for repeat in range(repeat_cnt):
                repeat_str = f", Repeat {repeat+1}" if repeat_cnt > 1 else ""
                for fold in range(fold_cnt):
                    learner_name = construct_learner_name(fold, repeat, repeat_cnt)
                    for t in classes:
                        fname = learner_name + f"_shap_dependence_class_{t}.png"
                        if fname in dep_plots:
                            fout.write(
                                f"\n### Dependence {t} (Fold {fold+1}{repeat_str})\n"
                            )
                            fout.write(
                                f"![SHAP Dependence from fold {fold+1}{repeat_str}]({fname})"
                            )

            # SHAP Decisions
            dec_plots = [
                f
                for f in os.listdir(model_path)
                if "_sample_" in f and "decisions.png" in f
            ]
            if not len(dec_plots):
                return

            fout.write("\n\n## SHAP Decision plots\n")
            for decision_type in ["worst", "best"]:
                for sample in [0, 1, 2, 3]:
                    for repeat in range(repeat_cnt):
                        repeat_str = f", Repeat {repeat+1}" if repeat_cnt > 1 else ""
                        for fold in range(fold_cnt):
                            learner_name = construct_learner_name(
                                fold, repeat, repeat_cnt
                            )
                            fname = (
                                learner_name
                                + f"_sample_{sample}_{decision_type}_decisions.png"
                            )
                            if fname in dec_plots:
                                fout.write(
                                    f"\n### {decision_type.capitalize()} decisions for selected sample {sample+1} (Fold {fold+1}{repeat_str})\n"
                                )
                                fout.write(
                                    f"![SHAP {decision_type} decisions from Fold {fold+1}{repeat_str}]({fname})"
                                )
        except Exception as e:
            logger.error(
                f"Exception while saving SHAP explanations. {str(e)}\nContinuing ..."
            )
