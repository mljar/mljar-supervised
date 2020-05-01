import os
import logging
import copy
import numpy as np
import pandas as pd
import time
import uuid

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
)
from supervised.utils.metric import logloss

logger = logging.getLogger(__name__)
from supervised.utils.config import LOG_LEVEL

logger.setLevel(LOG_LEVEL)
from supervised.utils.learning_curves import LearningCurves
from tabulate import tabulate


class AdditionalMetrics:
    @staticmethod
    def binary_classification(target, predictions):

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
            details["f1"] += [f1_score(target, response)]
            details["accuracy"] += [accuracy_score(target, response)]
            details["precision"] += [precision_score(target, response)]
            details["recall"] += [recall_score(target, response)]
            if i == 0:
                details["mcc"] += [0.0]
            else:
                details["mcc"] += [matthews_corrcoef(target, response)]

        # max metrics
        max_metrics = {
            "logloss": {
                "score": logloss(target, predictions),
                "threshold": None,
            },  # there is no threshold for LogLoss
            "auc": {
                "score": roc_auc_score(target, predictions),
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
        # confusion matrix
        conf_matrix = confusion_matrix(
            target, predictions > max_metrics["f1"]["threshold"]
        )
        conf_matrix = pd.DataFrame(
            conf_matrix,
            columns=["Predicted as negative", "Predicted as positive"],
            index=["Labeled as negative", "Labeled as positive"],
        )

        return {
            "metric_details": pd.DataFrame(details),
            "max_metrics": pd.DataFrame(max_metrics),
            "confusion_matrix": conf_matrix,
            "threshold": float(max_metrics["f1"]["threshold"]),
        }

    @staticmethod
    def multiclass_classification(target, predictions):

        all_labels = [i[11:] for i in predictions.columns.tolist()[:-1]]

        ll = logloss(target, predictions[predictions.columns[:-1]])

        labels = {i: l for i, l in enumerate(all_labels)}
        predictions = predictions["label"]
        target = target["target"].map(labels)
        # Print the confusion matrix
        conf_matrix = confusion_matrix(target, predictions, labels=all_labels)

        rows = [f"Predicted as {a}" for a in all_labels]
        cols = [f"Labeled as {a}" for a in all_labels]

        conf_matrix = pd.DataFrame(conf_matrix, columns=rows, index=cols)

        max_metrics = classification_report(
            target, predictions, digits=6, labels=all_labels, output_dict=True
        )
        max_metrics["logloss"] = ll

        return {
            "max_metrics": pd.DataFrame(max_metrics).transpose(),
            "confusion_matrix": conf_matrix,
        }

    @staticmethod
    def regression(target, predictions):
        regression_metrics = {
            "MAE": mean_absolute_error,
            "MSE": mean_squared_error,
            "RMSE": lambda t, p: np.sqrt(mean_squared_error(t, p)),
            "R2": r2_score,
        }
        max_metrics = {}
        for k, v in regression_metrics.items():
            max_metrics[k] = v(target, predictions)

        return {
            "max_metrics": pd.DataFrame(
                {
                    "Metric": list(max_metrics.keys()),
                    "Score": list(max_metrics.values()),
                }
            )
        }

    @staticmethod
    def compute(target, predictions, ml_task):

        if ml_task == BINARY_CLASSIFICATION:
            return AdditionalMetrics.binary_classification(target, predictions)
        elif ml_task == MULTICLASS_CLASSIFICATION:
            return AdditionalMetrics.multiclass_classification(target, predictions)
        elif ml_task == REGRESSION:
            return AdditionalMetrics.regression(target, predictions)

    @staticmethod
    def save(additional_metrics, ml_task, model_desc, model_path):
        if ml_task == BINARY_CLASSIFICATION:
            AdditionalMetrics.save_binary_classification(
                additional_metrics, model_desc, model_path
            )
        elif ml_task == MULTICLASS_CLASSIFICATION:
            AdditionalMetrics.save_multiclass_classification(
                additional_metrics, model_desc, model_path
            )
        elif ml_task == REGRESSION:
            AdditionalMetrics.save_regression(
                additional_metrics, model_desc, model_path
            )

    @staticmethod
    def add_learning_curves(fout):
        fout.write("\n\n## Learning curves\n")
        fout.write(f"![Learning curves]({LearningCurves.output_file_name})")

    @staticmethod
    def save_binary_classification(additional_metrics, model_desc, model_path):
        max_metrics = additional_metrics["max_metrics"].transpose()
        confusion_matrix = additional_metrics["confusion_matrix"]
        threshold = additional_metrics["threshold"]

        with open(os.path.join(model_path, "README.md"), "w") as fout:
            fout.write(model_desc)
            fout.write("\n## Metric details\n{}\n\n".format(max_metrics.to_markdown()))
            fout.write(
                "\n## Confusion matrix (at threshold={})\n{}".format(
                    np.round(threshold, 6), confusion_matrix.to_markdown()
                )
            )
            AdditionalMetrics.add_learning_curves(fout)
            AdditionalMetrics.add_tree_viz(fout, model_path)
            AdditionalMetrics.add_linear_coefs(fout, model_path)
            AdditionalMetrics.add_importance(fout, model_path)
            AdditionalMetrics.add_shap_importance(fout, model_path)
            AdditionalMetrics.add_shap_binary(fout, model_path)
            

    @staticmethod
    def save_multiclass_classification(additional_metrics, model_desc, model_path):
        max_metrics = additional_metrics["max_metrics"].transpose()
        confusion_matrix = additional_metrics["confusion_matrix"]

        with open(os.path.join(model_path, "README.md"), "w") as fout:
            fout.write(model_desc)
            fout.write("\n### Metric details\n{}\n\n".format(max_metrics.to_markdown()))
            fout.write(
                "\n## Confusion matrix\n{}".format(confusion_matrix.to_markdown())
            )
            AdditionalMetrics.add_learning_curves(fout)
            AdditionalMetrics.add_tree_viz(fout, model_path)
            AdditionalMetrics.add_linear_coefs(fout, model_path)
            AdditionalMetrics.add_importance(fout, model_path)
            AdditionalMetrics.add_shap_importance(fout, model_path)

    @staticmethod
    def save_regression(additional_metrics, model_desc, model_path):
        max_metrics = additional_metrics["max_metrics"]
        with open(os.path.join(model_path, "README.md"), "w") as fout:
            fout.write(model_desc)
            fout.write(
                "\n### Metric details:\n{}\n\n".format(
                    tabulate(max_metrics.values, max_metrics.columns, tablefmt="pipe")
                )
            )
            AdditionalMetrics.add_learning_curves(fout)
            AdditionalMetrics.add_tree_viz(fout, model_path)
            AdditionalMetrics.add_linear_coefs(fout, model_path)
            AdditionalMetrics.add_importance(fout, model_path)
            AdditionalMetrics.add_shap_importance(fout, model_path)

    @staticmethod
    def add_linear_coefs(fout, model_path):
        coef_files = [f for f in os.listdir(model_path) if "_coefs.csv" in f]
        if len(coef_files):
            fout.write("\n\n## Coefficients\n")
            for l in range(len(coef_files)):
                fout.write(f"\n### Coefficients learner #{l+1}\n")
                df = pd.read_csv(
                    os.path.join(model_path, f"learner_{l+1}_coefs.csv"), index_col=0
                )
                fout.write(df.to_markdown() + "\n")

    @staticmethod
    def add_tree_viz(fout, model_path):

        tree_viz = [f for f in os.listdir(model_path) if "_tree.svg" in f]
        if len(tree_viz):
            fout.write("\n\n## Tree visualizations\n")
            for l in range(len(tree_viz)):
                fout.write(f"\n### Tree #{l+1}\n")
                f_path = f"learner_{l+1}_tree.svg"
                fout.write(f"![Tree {l+1}]({f_path})")

    @staticmethod
    def add_importance(fout, model_path):

        imp_plots = [f for f in os.listdir(model_path) if "_importance.png" in f]
        if not len(imp_plots):
            return

        fout.write("\n\n## Features importance\n")
        for l in range(len(imp_plots)):
            fout.write(f"\n### Features importance (Fold #{l+1})\n")
            f_path = f"learner_{l+1}_importance.png"
            fout.write(f"![Imprtance from fold {l+1}]({f_path})")



    @staticmethod
    def add_shap_importance(fout, model_path):

        # SHAP Importance
        imp_data = [f for f in os.listdir(model_path) if "_shap_importance.csv" in f]
        if not len(imp_data):
            return
        df_all = []
        for l in range(len(imp_data)):
            f_path = os.path.join(model_path, f"learner_{l+1}_shap_importance.csv")
            df = pd.read_csv(f_path, index_col=0)
            df.columns = [f"Learner {l+1}"]
            df_all += [df]
        
        df = pd.concat(df_all, axis=1)
        ax = df.plot.barh(figsize=(10,7))
        ax.set_xlabel("mean(|SHAP value|) average impact on model output magnitude")
        fig = ax.get_figure()
        fig.tight_layout(pad=2.0)
        fig.savefig(os.path.join(model_path, "shap_importance.png"))
        fout.write("\n\n## SHAP Importance\n")
        fout.write(f"![SHAP Importance](shap_importance.png)")


    @staticmethod
    def add_shap_binary(fout, model_path):

        # Dependence SHAP
        dep_plots = [f for f in os.listdir(model_path) if "_shap_dependence.png" in f]
        if not len(dep_plots):
            return

        fout.write("\n\n## SHAP Dependence plots\n")
        for l in range(len(dep_plots)):
            fout.write(f"\n### Dependence (Fold #{l+1})\n")
            f_path = f"learner_{l+1}_shap_dependence.png"
            fout.write(f"![SHAP Dependence from fold {l+1}]({f_path})")


        # SHAP Decisions
        dec_plots = [f for f in os.listdir(model_path) if "_shap_class" in f and "decisions.png" in f]
        if not len(dec_plots):
            return

        fout.write("\n\n## SHAP Decision plots\n")
        for target in [0, 1]:
            for decision_type in ["worst", "best"]:
                for l in range(len(dep_plots)):
                    fout.write(f"\n### {decision_type.capitalize()} decisions for class {target} (Fold #{l+1})\n")
                    f_path = f"learner_{l+1}_shap_class_{target}_{decision_type}_decisions.png"
                    fout.write(f"![SHAP {decision_type} decisions class {target} from fold {l+1}]({f_path})")
