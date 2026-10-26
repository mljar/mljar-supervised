"""Export without Mercury, then validate the bundle in a separate app environment.

Run `export DIRECTORY` with the base wheel, install that same wheel plus each
generated requirements.txt into a fresh environment, then run `run DIRECTORY`.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def export(root):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification, make_friedman1
    from supervised import AutoML

    assert importlib.util.find_spec("mercury") is None, "Export must work in the base installation"
    assert importlib.util.find_spec("torch") is None
    expected = {}
    for case in ("binary", "multiclass", "regression", "discovery"):
        regression = case in ("regression", "discovery")
        if regression:
            X, y = make_friedman1(n_samples=250, n_features=6, random_state=42)
        else:
            X, y = make_classification(n_samples=100, n_features=6, n_informative=4,
                                       n_classes=2 if case == "binary" else 3, random_state=42)
            y = np.array([f"class_{label}" for label in y])
        X = pd.DataFrame(X, columns=list("abcdef"))
        training = root / (case + "-training")
        model = AutoML(results_path=str(training), mode="Compete", algorithms=["Decision Tree"],
                        total_time_limit=120, n_jobs=1, verbose=0, explain_level=0,
                        golden_features=False, feature_discovery=case == "discovery",
                        features_selection=False, kmeans_features=False, mix_encoding=False,
                        boost_on_errors=False, start_random_models=1, hill_climbing_steps=0,
                        train_ensemble=False, stack_models=False,
                        validation_strategy=dict(validation_type="split", train_ratio=0.8))
        model.fit(X, y)
        if case == "discovery":
            # Require this test to export a discovery model, regardless of ranking.
            model._best_model = next(m for m in model._models if "FeatureDiscovery" in m.get_name())
            params_path = training / "params.json"
            params = json.loads(params_path.read_text())
            params["best_model"] = model._best_model.get_name()
            params_path.write_text(json.dumps(params))
        app = Path(model.app(path=str(root / case), verbose=False))
        manifest = json.loads((app / "mljar_app.json").read_text())
        default = pd.DataFrame([{f["name"]: f["default"] for f in manifest["feature_schema"]}])
        expected[case] = dict(single=model.predict(default).tolist(), batch=model.predict(X.iloc[:5]).tolist(),
                              regression=regression)
        X.iloc[:5].to_csv(app / "validation.csv", index=False)
        training.rename(root / (case + "-source-unavailable"))
    (root / "expected.json").write_text(json.dumps(expected))
    print("Exported four apps without Mercury or Torch", flush=True)


def server_smoke(app):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (app / "server.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "mercury_app", "--ip=127.0.0.1", f"--port={port}",
             "--no-browser", "--allow-root", f"--working-dir={app}"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError((app / "server.log").read_text())
                try:
                    with urlopen(f"http://127.0.0.1:{port}", timeout=1) as response:
                        assert response.status == 200
                        assert response.read()
                        return
                except (URLError, TimeoutError):
                    time.sleep(0.2)
            raise AssertionError("Mercury did not start within the smoke-test deadline")
        finally:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def run(root):
    import nbformat
    from nbclient import NotebookClient

    assert importlib.util.find_spec("torch") is None
    expected = json.loads((root / "expected.json").read_text())
    for case, values in expected.items():
        app = root / case
        for notebook_name in ("predict_single.ipynb", "predict_batch.ipynb"):
            notebook = nbformat.read(app / notebook_name, as_version=4)
            single = notebook_name.startswith("predict_single")
            if not single:
                notebook.cells.insert(2, nbformat.v4.new_code_cell(
                    "from pathlib import Path\nuploader.values = [list(Path('validation.csv').read_bytes())]\n"
                    "uploader.filenames = ['validation.csv']"))
            actual = "[result['prediction']]" if single else "scored_df['prediction' if 'prediction' in scored_df else 'label'].tolist()"
            comparison = "assert_allclose" if values["regression"] else "assert_array_equal"
            wanted = values["single" if single else "batch"]
            notebook.cells.append(nbformat.v4.new_code_cell(
                f"import numpy as np\nassert APP_IMPORT_ERROR is None\n"
                f"np.testing.{comparison}({actual}, {wanted!r})\n"))
            NotebookClient(notebook, timeout=120, kernel_name="python3",
                           resources={"metadata": {"path": str(app)}}).execute()
            nbformat.write(notebook, app / ("validated-" + notebook_name))
            print(f"PASS: {case} {notebook_name}", flush=True)
    server_smoke(root / "binary")
    print("PASS: Mercury HTTP startup and cleanup", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["export", "run"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (export if args.action == "export" else run)(root)
