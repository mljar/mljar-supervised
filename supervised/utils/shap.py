import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
import shap


logger = logging.getLogger(__name__)
from supervised.utils.config import LOG_LEVEL

logger.setLevel(LOG_LEVEL)
import warnings


class PlotSHAP:
    @staticmethod
    def is_available(algorithm, X_train, y_train, ml_task):

        # https://github.com/mljar/mljar-supervised/issues/112 disable for NN
        # https://github.com/mljar/mljar-supervised/issues/114 disable for CatBoost
        if algorithm.algorithm_short_name in ["Baseline", "Neural Network", "CatBoost"]:
            return False
        if (
            algorithm.algorithm_short_name == "Xgboost"
            and algorithm.learner_params["booster"] == "gblinear"
        ):
            # Xgboost gblinear is not supported by SHAP
            return False
        # disable for large number of columns
        if X_train.shape[1] > 500:
            warnings.warn(
                "Disable SHAP explanations because of number of columns > 500."
            )
            return False
        if ml_task == MULTICLASS_CLASSIFICATION and len(np.unique(y_train)) > 100:
            warnings.warn(
                "Disable SHAP explanations because of large number of classes (> 100)."
            )
            return False
        if X_train.shape[0] < 20:
            warnings.warn(
                "Disable SHAP explanations because of small number of samples (< 20)."
            )
            return False
        return True

    @staticmethod
    def get_explainer(algorithm, X_train):

        explainer = None
        if algorithm.algorithm_short_name in [
            "Xgboost",
            "Decision Tree",
            "Random Forest",
            "LightGBM",
            "Extra Trees",
            "CatBoost",
        ]:
            explainer = shap.TreeExplainer(algorithm.model)
        elif algorithm.algorithm_short_name in ["Linear"]:
            explainer = shap.LinearExplainer(algorithm.model, X_train)
        # elif algorithm.algorithm_short_name in ["Neural Network"]:
        #    explainer = shap.KernelExplainer(algorithm.model.predict, X_train)  # slow

        return explainer

    @staticmethod
    def get_sample(X_validation, y_validation):
        # too many samples in the data, down-sample it
        SAMPLES_LIMIT = 1000
        if X_validation.shape[0] > SAMPLES_LIMIT:
            X_validation.reset_index(inplace=True, drop=True)
            y_validation.reset_index(inplace=True, drop=True)
            X_vald = X_validation.sample(SAMPLES_LIMIT)
            y_vald = y_validation[X_vald.index]
        else:
            X_vald = X_validation
            y_vald = y_validation
        return X_vald, y_vald

    def get_predictions(algorithm, X_vald, y_vald, ml_task):
        # compute predictions on down-sampled data
        predictions = algorithm.predict(X_vald)

        if ml_task == MULTICLASS_CLASSIFICATION:
            oh = OneHotEncoder(sparse=False)
            y_encoded = oh.fit_transform(np.array(y_vald).reshape(-1, 1))
            residua = np.sum(np.abs(np.array(y_encoded) - predictions), axis=1)
        else:
            residua = np.abs(np.array(y_vald) - predictions)

        df_preds = pd.DataFrame(
            {"res": residua, "lp": range(residua.shape[0]), "target": np.array(y_vald)},
            index=X_vald.index,
        )
        df_preds = df_preds.sort_values(by="res", ascending=False)

        return df_preds

    @staticmethod
    def summary(shap_values, X_vald, model_file_path, learner_name, class_names):
        fig = plt.gcf()
        classes = None
        if class_names is not None and len(class_names):
            classes = class_names

        shap.summary_plot(
            shap_values, X_vald, plot_type="bar", show=False, class_names=classes
        )
        fig.tight_layout(pad=2.0)
        fig.savefig(os.path.join(model_file_path, f"{learner_name}_shap_summary.png"))
        plt.close("all")

        vals = None
        if isinstance(shap_values, list):
            for sh in shap_values:
                v = np.abs(sh).mean(0)
                vals = v if vals is None else vals + v
        else:
            vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(
            list(zip(X_vald.columns, vals)), columns=["feature", "shap_importance"]
        )
        feature_importance.sort_values(
            by=["shap_importance"], ascending=False, inplace=True
        )
        feature_importance.to_csv(
            os.path.join(model_file_path, f"{learner_name}_shap_importance.csv"),
            index=False,
        )

    @staticmethod
    def dependence(shap_values, X_vald, model_file_path, learner_name, file_postfix=""):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = plt.figure(figsize=(14, 7))
            plots_cnt = np.min([9, X_vald.shape[1]])
            cols_cnt = 3
            rows_cnt = 3
            if plots_cnt < 4:
                rows_cnt = 1
            elif plots_cnt < 7:
                rows_cnt = 2
            for i in range(plots_cnt):
                ax = fig.add_subplot(rows_cnt, cols_cnt, i + 1)
                shap.dependence_plot(
                    f"rank({i})",
                    shap_values,
                    X_vald,
                    show=False,
                    title=f"Importance #{i+1}",
                    ax=ax,
                )

            fig.tight_layout(pad=2.0)
            fig.savefig(
                os.path.join(
                    model_file_path, f"{learner_name}_shap_dependence{file_postfix}.png"
                )
            )
            plt.close("all")

    @staticmethod
    def compute(
        algorithm,
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_file_path,
        learner_name,
        class_names,
        ml_task,
    ):
        if not PlotSHAP.is_available(algorithm, X_train, y_train, ml_task):
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = PlotSHAP.get_explainer(algorithm, X_train)
                X_vald, y_vald = PlotSHAP.get_sample(X_validation, y_validation)
                shap_values = explainer.shap_values(X_vald)

            # fix problem with 1 or 2 dimensions for binary classification
            expected_value = explainer.expected_value
            if ml_task == BINARY_CLASSIFICATION and isinstance(shap_values, list):
                shap_values = shap_values[1]
                expected_value = explainer.expected_value[1]

            # Summary SHAP plot
            PlotSHAP.summary(
                shap_values, X_vald, model_file_path, learner_name, class_names
            )
            # Dependence SHAP plots
            if ml_task == MULTICLASS_CLASSIFICATION:
                for t in np.unique(y_vald):
                    PlotSHAP.dependence(
                        shap_values[t],
                        X_vald,
                        model_file_path,
                        learner_name,
                        f"_class_{class_names[t]}",
                    )
            else:
                PlotSHAP.dependence(shap_values, X_vald, model_file_path, learner_name)

            # Decision SHAP plots
            df_preds = PlotSHAP.get_predictions(algorithm, X_vald, y_vald, ml_task)

            if ml_task == REGRESSION:
                PlotSHAP.decisions_regression(
                    df_preds,
                    shap_values,
                    expected_value,
                    X_vald,
                    y_vald,
                    model_file_path,
                    learner_name,
                )
            elif ml_task == BINARY_CLASSIFICATION:
                PlotSHAP.decisions_binary(
                    df_preds,
                    shap_values,
                    expected_value,
                    X_vald,
                    y_vald,
                    model_file_path,
                    learner_name,
                )
            else:
                PlotSHAP.decisions_multiclass(
                    df_preds,
                    shap_values,
                    expected_value,
                    X_vald,
                    y_vald,
                    model_file_path,
                    learner_name,
                    class_names,
                )
        except Exception as e:
            print(
                f"Exception while producing SHAP explanations. {str(e)}\nContinuing ..."
            )

    @staticmethod
    def decisions_regression(
        df_preds,
        shap_values,
        expected_value,
        X_vald,
        y_vald,
        model_file_path,
        learner_name,
    ):
        fig = plt.gcf()
        shap.decision_plot(
            expected_value,
            shap_values[df_preds.lp[:10], :],
            X_vald.loc[df_preds.index[:10]],
            show=False,
        )
        fig.tight_layout(pad=2.0)
        fig.savefig(
            os.path.join(model_file_path, f"{learner_name}_shap_worst_decisions.png")
        )
        plt.close("all")

        fig = plt.gcf()
        shap.decision_plot(
            expected_value,
            shap_values[df_preds.lp[-10:], :],
            X_vald.loc[df_preds.index[-10:]],
            show=False,
        )
        fig.tight_layout(pad=2.0)
        fig.savefig(
            os.path.join(model_file_path, f"{learner_name}_shap_best_decisions.png")
        )
        plt.close("all")

    @staticmethod
    def decisions_binary(
        df_preds,
        shap_values,
        expected_value,
        X_vald,
        y_vald,
        model_file_path,
        learner_name,
    ):
        # classes are from 0 ...
        for t in np.unique(y_vald):
            fig = plt.gcf()
            shap.decision_plot(
                expected_value,
                shap_values[df_preds[df_preds.target == t].lp[:10], :],
                X_vald.loc[df_preds[df_preds.target == t].index[:10]],
                show=False,
            )
            fig.tight_layout(pad=2.0)
            fig.savefig(
                os.path.join(
                    model_file_path,
                    f"{learner_name}_shap_class_{t}_worst_decisions.png",
                )
            )
            plt.close("all")

            fig = plt.gcf()
            shap.decision_plot(
                expected_value,
                shap_values[df_preds[df_preds.target == t].lp[-10:], :],
                X_vald.loc[df_preds[df_preds.target == t].index[-10:]],
                show=False,
            )
            fig.tight_layout(pad=2.0)
            fig.savefig(
                os.path.join(
                    model_file_path, f"{learner_name}_shap_class_{t}_best_decisions.png"
                )
            )
            plt.close("all")

    @staticmethod
    def decisions_multiclass(
        df_preds,
        shap_values,
        expected_value,
        X_vald,
        y_vald,
        model_file_path,
        learner_name,
        class_names,
    ):

        for decision_type in ["worst", "best"]:
            m = 1 if decision_type == "worst" else -1
            for i in range(4):

                fig = plt.gcf()
                shap.multioutput_decision_plot(
                    list(expected_value),
                    shap_values,
                    row_index=df_preds.lp.iloc[m * i],
                    show=False,
                    legend_labels=class_names,
                    title=f"It should be {class_names[df_preds.target.iloc[m*i]]}",
                )
                fig.tight_layout(pad=2.0)
                fig.savefig(
                    os.path.join(
                        model_file_path,
                        f"{learner_name}_sample_{i}_{decision_type}_decisions.png",
                    )
                )
                plt.close("all")
