import importlib
import json
import os
import shutil
import shlex
import tempfile
import zipfile

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


PUBLISHABLE_WORKSPACE_FILES = (
    "predict_single.ipynb",
    "predict_batch.ipynb",
    "app_support.py",
    "mljar_app.json",
    "config.toml",
    "requirements.txt",
    "runtime.txt",
    "automl.zip",
)


def generate_app(automl, path=None, overwrite=False, title=None, verbose=True):
    results_path = automl._get_results_path()
    _ensure_automl_ready(automl, results_path)
    output_path = _resolve_output_path(results_path, path)
    _prepare_output_dir(output_path, overwrite=overwrite)

    selected_models = _selected_model_names(automl)
    manifest = collect_app_metadata(
        automl,
        title=title,
        selected_models=selected_models,
    )
    try:
        _write_workspace(output_path, results_path, manifest, selected_models)
    except Exception:
        shutil.rmtree(output_path, ignore_errors=True)
        raise
    if verbose:
        _print_verbose_summary(output_path)
    return output_path


def publishable_workspace_paths(output_path):
    return [
        os.path.join(output_path, filename) for filename in PUBLISHABLE_WORKSPACE_FILES
    ]


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


def _resolve_output_path(results_path, path):
    if path is None:
        return os.path.abspath(os.path.join(results_path, "app"))
    return os.path.abspath(path)


def _selected_model_names(automl):
    if automl._best_model is None:
        automl.load(automl._get_results_path())
    best_model_name = automl._best_model.get_name()
    return automl.models_needed_on_predict(best_model_name)


def _write_workspace(output_path, results_path, manifest, selected_models):
    _create_automl_zip(output_path, results_path, selected_models)
    _write_json(os.path.join(output_path, "mljar_app.json"), manifest)
    notebook_sources = {
        "predict_single.ipynb": single_notebook_source(),
        "predict_batch.ipynb": batch_notebook_source(),
    }
    for notebook in manifest.get("notebooks", []):
        filename = notebook.get("filename")
        payload = notebook_sources.get(filename)
        if payload is not None:
            _write_json(os.path.join(output_path, filename), payload)
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


def _print_verbose_summary(output_path):
    quoted_path = shlex.quote(output_path)
    mercury_command = f"mercury --working-dir={quoted_path}"
    manifest = _load_manifest(output_path)

    print(f"App directory: {output_path}")
    if manifest and not manifest.get("supports", {}).get("single_sample", True):
        feature_count = len(manifest.get("feature_schema", []))
        print(
            "Single prediction UI was skipped because the model has "
            f"{feature_count} features, which is above the 15-feature limit."
        )
        print("Generated app includes batch CSV prediction only.")
    print(f"Enter app directory: cd {quoted_path}")
    try:
        importlib.import_module("mercury")
    except ImportError:
        print(
            "Mercury is not available in the current Python environment. "
            "Install it with: pip install -r requirements.txt"
        )
    print(f"Start Mercury: {mercury_command}")


def _load_manifest(output_path):
    manifest_path = os.path.join(output_path, "mljar_app.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def _create_automl_zip(output_path, results_path, selected_models):
    zip_path = os.path.join(output_path, "automl.zip")
    with tempfile.TemporaryDirectory() as temp_dir:
        bundle_root = os.path.join(temp_dir, "automl")
        os.makedirs(bundle_root)
        _write_json(
            os.path.join(bundle_root, "params.json"),
            _build_minimal_params(results_path, selected_models),
        )
        _copy_root_runtime_files(results_path, bundle_root, selected_models)
        _copy_selected_model_dirs(results_path, bundle_root, selected_models)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(temp_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zf.write(file_path, arcname)


def _build_minimal_params(results_path, selected_models):
    with open(os.path.join(results_path, "params.json"), "r", encoding="utf-8") as fin:
        params = json.load(fin)

    params["saved"] = [name for name in params.get("saved", []) if name in selected_models]
    params["load_on_predict"] = list(selected_models)
    if "stacked" in params:
        params["stacked"] = [
            name for name in params.get("stacked", []) if name in selected_models
        ]
    params["results_path"] = "automl"
    return params


def _copy_root_runtime_files(results_path, bundle_root, selected_models):
    root_files = ["data_info.json"]
    if _needs_golden_features(results_path, selected_models):
        root_files.append("golden_features.json")

    for filename in root_files:
        source = os.path.join(results_path, filename)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(bundle_root, filename))


def _needs_golden_features(results_path, selected_models):
    for model_name in selected_models:
        framework_path = os.path.join(results_path, model_name, "framework.json")
        if not os.path.exists(framework_path):
            continue
        with open(framework_path, "r", encoding="utf-8") as fin:
            framework = json.load(fin)
        for preprocessing in framework.get("preprocessing", []):
            if "golden_features" in preprocessing:
                return True
    return False


def _copy_selected_model_dirs(results_path, bundle_root, selected_models):
    for model_name in selected_models:
        source = os.path.join(results_path, model_name)
        target = os.path.join(bundle_root, model_name)
        if os.path.isdir(source):
            shutil.copytree(source, target, ignore=_ignore_non_runtime_files)


def _ignore_non_runtime_files(_, names):
    ignored = []
    for name in names:
        lower = name.lower()
        if lower in {"readme.md", "readme.html"}:
            ignored.append(name)
            continue
        if lower.endswith((".png", ".svg", ".html", ".md", ".log")):
            ignored.append(name)
    return ignored
