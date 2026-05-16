import json
from textwrap import dedent


def app_support_source():
    return dedent(
        """
        import base64
        import io
        import json
        from functools import lru_cache
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        from supervised import AutoML

        ROOT = Path(__file__).resolve().parent
        MANIFEST_PATH = ROOT / "mljar_app.json"
        AUTOML_PATH = ROOT / "automl"


        @lru_cache(maxsize=1)
        def load_bundle():
            with MANIFEST_PATH.open("r", encoding="utf-8") as fin:
                manifest = json.load(fin)
            automl = AutoML(results_path=str(AUTOML_PATH))
            return {"manifest": manifest, "automl": automl}


        def manifest():
            return load_bundle()["manifest"]


        def feature_schema():
            return manifest().get("feature_schema", [])


        def create_widget(mr, feature):
            name = feature["name"]
            key = f"feature-{name}"
            label = name.replace("_", " ")
            widget = feature.get("widget")
            if widget == "select" and feature.get("choices"):
                choices = [str(choice) for choice in feature.get("choices", [])]
                value = str(feature.get("default", choices[0] if choices else ""))
                return mr.Select(label=label, choices=choices, value=value, key=key)
            if widget == "checkbox":
                return mr.CheckBox(label=label, value=bool(feature.get("default", False)), key=key)
            if widget == "number":
                return mr.NumberInput(
                    label=label,
                    value=feature.get("default", 0.0),
                    min=feature.get("min", 0.0),
                    max=feature.get("max", 1.0),
                    step=feature.get("step", 1.0),
                    key=key,
                )
            return mr.TextInput(label=label, value=str(feature.get("default", "")), key=key)


        def widgets_to_dataframe(widgets):
            row = {}
            for feature in feature_schema():
                name = feature["name"]
                value = getattr(widgets[name], "value", None)
                row[name] = _cast_scalar(feature, value)
            return pd.DataFrame([row], columns=[feature["name"] for feature in feature_schema()])


        def predict_single(input_df):
            bundle = load_bundle()
            predictions = bundle["automl"].predict_all(input_df)
            task = bundle["manifest"]["model_task"]
            result = {
                "task": task,
                "input": input_df.iloc[0].to_dict(),
                "table": predictions,
                "prediction": None,
                "probabilities": [],
            }
            if task == "regression":
                result["prediction"] = float(predictions["prediction"].iloc[0])
            else:
                result["prediction"] = predictions["label"].iloc[0]
                for column in predictions.columns:
                    if column == "label":
                        continue
                    result["probabilities"].append(
                        {"label": column.replace("prediction_", ""), "value": float(predictions[column].iloc[0])}
                    )
            return result


        def batch_predict(upload_widget):
            if not getattr(upload_widget, "value", b""):
                return None, "Upload a CSV file to score a batch."
            try:
                batch_df = pd.read_csv(io.BytesIO(upload_widget.value))
            except Exception as exc:
                return None, f"Failed to read CSV: {exc}"

            expected = [feature["name"] for feature in feature_schema()]
            missing = [column for column in expected if column not in batch_df.columns]
            if missing:
                return None, "Missing required columns: " + ", ".join(missing)

            ordered = batch_df[expected].copy()
            for feature in feature_schema():
                ordered[feature["name"]] = ordered[feature["name"]].map(
                    lambda value, spec=feature: _cast_scalar(spec, value)
                )
            predictions = load_bundle()["automl"].predict_all(ordered)
            scored = pd.concat([batch_df.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
            return scored, None


        def csv_download_payload(df):
            csv_data = df.to_csv(index=False)
            return base64.b64encode(csv_data.encode("utf-8")).decode("ascii")


        def render_single_dashboard(mr, result):
            if result["task"] == "regression":
                mr.Markdown(f"## Predicted value\\n\\n`{result['prediction']:.6g}`")
            else:
                headline = f"## Predicted label\\n\\n`{result['prediction']}`"
                if result["probabilities"]:
                    top = max(result["probabilities"], key=lambda item: item["value"])
                    headline += f"\\n\\nTop probability: `{top['value']:.3f}`"
                mr.Markdown(headline)


        def plot_probabilities(result):
            if not result["probabilities"]:
                return
            labels = [item["label"] for item in result["probabilities"]]
            values = [item["value"] for item in result["probabilities"]]
            fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(labels))))
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, values, color="#0b7285")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("Probability")
            ax.set_title("Class probabilities")
            plt.tight_layout()
            plt.show()


        def plot_feature_importance():
            gfi = manifest().get("global_feature_importance", {})
            if not gfi.get("available"):
                return
            top = gfi.get("top", [])[:8]
            if not top:
                return
            labels = [item["feature"] for item in top]
            values = [item["mean_rank"] if "mean_rank" in item else item["importance"] for item in top]
            fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(labels))))
            ax.barh(labels[::-1], values[::-1], color="#faa307")
            ax.set_title("Global feature importance")
            plt.tight_layout()
            plt.show()


        def plot_feature_context(input_df):
            schema = feature_schema()
            gfi = manifest().get("global_feature_importance", {})
            priority = [item["feature"] for item in gfi.get("top", [])]
            selected = []
            for name in priority:
                feature = next((item for item in schema if item["name"] == name), None)
                if feature is not None and "distribution" in feature:
                    selected.append(feature)
                if len(selected) == 3:
                    break
            if not selected:
                selected = [feature for feature in schema if "distribution" in feature][:3]
            for feature in selected:
                if feature["kind"] == "numeric":
                    _plot_numeric_context(feature, input_df.iloc[0][feature["name"]])
                elif feature["kind"] in {"categorical", "boolean"}:
                    _plot_categorical_context(feature, input_df.iloc[0][feature["name"]])


        def plot_batch_summary(scored_df):
            task = manifest()["model_task"]
            fig, ax = plt.subplots(figsize=(8, 4))
            if task == "regression":
                scored_df["prediction"].plot(kind="hist", bins=12, color="#577590", ax=ax)
                ax.set_title("Prediction distribution")
                ax.set_xlabel("Prediction")
            else:
                scored_df["label"].astype(str).value_counts().plot(kind="bar", color="#577590", ax=ax)
                ax.set_title("Predicted label counts")
                ax.set_xlabel("Label")
            plt.tight_layout()
            plt.show()


        def _plot_numeric_context(feature, sample_value):
            distribution = feature.get("distribution", {})
            edges = distribution.get("bin_edges")
            counts = distribution.get("counts")
            if not edges or not counts:
                return
            centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]
            widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(centers, counts, width=widths, color="#8ecae6", align="center", edgecolor="white")
            try:
                sample_numeric = float(sample_value)
                ax.axvline(sample_numeric, color="#d00000", linestyle="--", linewidth=2)
            except Exception:
                pass
            ax.set_title(f"{feature['name']} in training data")
            plt.tight_layout()
            plt.show()


        def _plot_categorical_context(feature, sample_value):
            distribution = feature.get("distribution", {})
            labels = [str(label) for label in distribution.get("labels", [])]
            counts = distribution.get("counts", [])
            if not labels or not counts:
                return
            colors = ["#023047" if str(sample_value) == label else "#8ecae6" for label in labels]
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(labels, counts, color=colors)
            ax.set_title(f"{feature['name']} in training data")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            plt.show()


        def _cast_scalar(feature, value):
            if value is None:
                return None
            if feature["kind"] == "numeric":
                if value == "":
                    return None
                if feature.get("dtype") == "int":
                    return int(float(value))
                return float(value)
            if feature["kind"] == "boolean":
                return _normalize_bool(value)
            if feature["kind"] == "datetime":
                if value == "":
                    return None
                return pd.to_datetime(value)
            return str(value)


        def _normalize_bool(value):
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            return text in {"true", "1", "yes", "y"}
        """
    ).strip() + "\n"


def config_toml(title):
    return dedent(
        f"""
        [main]
        title = {json.dumps(title)}
        footer = "Generated with MLJAR AutoML"
        favicon_emoji = "📈"

        [welcome]
        header = {json.dumps(title)}
        message = "Mercury prediction apps generated from a trained MLJAR AutoML model."
        """
    ).lstrip()


def requirements_txt():
    return (
        "# Requires Python 3.10+ because Mercury v3 uses Python 3.10 syntax.\n"
        "mercury\n"
        "mljar-supervised\n"
        "matplotlib\n"
        "pandas\n"
        "numpy\n"
    )


def runtime_txt():
    return "python-3.10\n"


def readme_md():
    return dedent(
        """
        # Generated MLJAR AutoML Mercury App

        This workspace was generated by `AutoML.app()`.

        ## Python version

        Mercury v3 requires **Python 3.10 or newer**.

        If you try to run `mercury` with Python 3.9, it will fail with syntax errors
        from Mercury itself, for example around `str | None`.

        ## Local run

        1. Create a Python 3.10+ virtual environment.
        2. Install dependencies:

        ```bash
        pip install -r requirements.txt
        ```

        3. Start Mercury:

        ```bash
        mercury
        ```

        ## Files

        - `predict_single.ipynb` - single-sample prediction dashboard
        - `predict_batch.ipynb` - batch CSV scoring
        - `automl/` - copied trained AutoML artifacts
        - `mljar_app.json` - generated app manifest
        """
    ).lstrip()


def single_notebook_source():
    return {
        "cells": [
            markdown_cell("# Single Prediction Dashboard"),
            code_cell(
                dedent(
                    """
                    import mercury as mr
                    from app_support import (
                        create_widget,
                        feature_schema,
                        plot_feature_context,
                        plot_feature_importance,
                        plot_probabilities,
                        predict_single,
                        render_single_dashboard,
                        widgets_to_dataframe,
                    )

                    mr.Markdown(
                        "Use the widgets in the sidebar to set a sample and re-run the notebook to inspect the prediction."
                    )
                    """
                ).strip()
            ),
            code_cell(
                dedent(
                    """
                    widgets = {}
                    for feature in feature_schema():
                        widgets[feature["name"]] = create_widget(mr, feature)
                    input_df = widgets_to_dataframe(widgets)
                    """
                ).strip()
            ),
            code_cell(
                dedent(
                    """
                    result = predict_single(input_df)
                    render_single_dashboard(mr, result)
                    mr.JSON(result["input"])
                    mr.Table(result["table"])
                    """
                ).strip()
            ),
            code_cell(
                dedent(
                    """
                    plot_probabilities(result)
                    plot_feature_context(input_df)
                    plot_feature_importance()
                    """
                ).strip()
            ),
        ],
        "metadata": notebook_metadata("Single Prediction Dashboard"),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def batch_notebook_source():
    return {
        "cells": [
            markdown_cell("# Batch Prediction"),
            code_cell(
                dedent(
                    """
                    import mercury as mr
                    from app_support import batch_predict, csv_download_payload, plot_batch_summary

                    uploader = mr.UploadFile(
                        label="Upload CSV for batch scoring",
                        accept=".csv",
                        max_file_size="100MB",
                    )
                    """
                ).strip()
            ),
            code_cell(
                dedent(
                    """
                    scored_df, error_message = batch_predict(uploader)
                    if error_message:
                        mr.Markdown(error_message)
                    elif scored_df is not None:
                        mr.Markdown(f"## Scored rows\\n\\n`{len(scored_df)}`")
                        mr.Table(scored_df.head(20))
                        mr.Download(
                            csv_download_payload(scored_df),
                            filename="predictions.csv",
                            mime="text/csv",
                            is_base64=True,
                            label="Download predictions",
                            position="inline",
                        )
                        plot_batch_summary(scored_df)
                    else:
                        mr.Markdown("Upload a CSV file to begin batch prediction.")
                    """
                ).strip()
            ),
        ],
        "metadata": notebook_metadata("Batch Prediction"),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def notebook_metadata(title):
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "mercury": {
            "title": title,
            "description": title,
            "show-code": False,
            "full-width": True,
        },
    }
