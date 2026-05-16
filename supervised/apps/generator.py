import json
import os
import shutil

from supervised.apps.metadata import collect_app_metadata
from supervised.apps.templates import (
    app_support_source,
    batch_notebook_source,
    config_toml,
    readme_md,
    requirements_txt,
    runtime_txt,
    single_notebook_source,
)
from supervised.exceptions import AutoMLException


def generate_app(automl, path="app", overwrite=False, title=None):
    results_path = automl._get_results_path()
    _ensure_automl_ready(automl, results_path)
    output_path = os.path.abspath(path)
    _prepare_output_dir(output_path, overwrite=overwrite)

    manifest = collect_app_metadata(automl, title=title)
    try:
        _write_workspace(output_path, results_path, manifest)
    except Exception:
        shutil.rmtree(output_path, ignore_errors=True)
        raise
    return output_path


def _ensure_automl_ready(automl, results_path):
    if results_path is None or not os.path.exists(os.path.join(results_path, "params.json")):
        raise AutoMLException(
            "This model has not been fitted yet. Please call `fit()` first."
        )
    if automl._best_model is None:
        automl.load(results_path)


def _prepare_output_dir(output_path, overwrite=False):
    if os.path.exists(output_path):
        if not overwrite:
            raise AutoMLException(
                f"Cannot generate app. Directory '{output_path}' already exists."
            )
        shutil.rmtree(output_path)
    os.makedirs(output_path)


def _write_workspace(output_path, results_path, manifest):
    shutil.copytree(results_path, os.path.join(output_path, "automl"))
    _write_json(os.path.join(output_path, "mljar_app.json"), manifest)
    _write_json(os.path.join(output_path, "predict_single.ipynb"), single_notebook_source())
    _write_json(os.path.join(output_path, "predict_batch.ipynb"), batch_notebook_source())
    _write_text(os.path.join(output_path, "app_support.py"), app_support_source())
    _write_text(os.path.join(output_path, "config.toml"), config_toml(manifest["title"]))
    _write_text(os.path.join(output_path, "requirements.txt"), requirements_txt())
    _write_text(os.path.join(output_path, "runtime.txt"), runtime_txt())
    _write_text(os.path.join(output_path, "README.md"), readme_md())


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)
        fout.write("\n")


def _write_text(path, payload):
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(payload)
