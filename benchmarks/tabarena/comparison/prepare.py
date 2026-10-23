"""Export the official TabArena split once; both versions consume identical files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from tabarena.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper

from common import BASE, SMALL, TABARENA_REVISION, lock, read_json, sha256, write_json


def prepare(name):
    root = BASE / "comparison-data" / TABARENA_REVISION / name / "r0f0"
    metadata_file = BASE / "source/packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/TabArena-v0.1_tasks_metadata.csv"
    with metadata_file.open() as stream:
        metadata = next(row for row in csv.DictReader(stream)
                        if row["dataset_name"] == name and row["split_index"] == "r0f0")
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / "prepare.lock"):
        if (root / "dataset.json").exists():
            saved = read_json(root / "dataset.json")
            for file, digest in saved["files_sha256"].items():
                if sha256(root / file) != digest:
                    raise RuntimeError(f"Cached split changed: {root / file}")
            print(f"Using cached official split: {name} ({root})", flush=True)
            return
        print(f"Loading official TabArena r0f0: {name} (OpenML task {metadata['task_id_str']})", flush=True)
        task = OpenMLTaskWrapper.from_task_id(int(metadata["task_id_str"]))
        assert task.problem_type == metadata["problem_type"]
        train_indices, test_indices = task.get_split_indices(fold=0, repeat=0, sample=0)
        assert not np.intersect1d(train_indices, test_indices).size
        X, y, X_test, y_test = task.get_train_test_split(fold=0, repeat=0, sample=0)
        for filename, frame in [("X_train", X), ("X_test", X_test),
                                ("y_train", y.to_frame()), ("y_test", y_test.to_frame())]:
            frame.to_parquet(root / f"{filename}.parquet", index=True)
        np.savez(root / "split_indices.npz", train=train_indices, test=test_indices)
        categorical = X.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
        target_summary = ({str(label): int(count) for label, count in y.value_counts().items()}
                          if task.problem_type != "regression"
                          else {key: float(value) for key, value in y.describe().items()})
        description = task.task.get_dataset().description or ""
        (root / "description.md").write_text(description)
        info = dict(dataset=name, task_id=task.task_id, dataset_id=task.dataset_id,
                    source_url=f"https://www.openml.org/t/{task.task_id}",
                    description=" ".join(description.split())[:1200], domain=metadata["domain"],
                    problem_type=task.problem_type, metric=metadata["eval_metric"], target=task.label,
                    repeat=0, fold=0, split="r0f0", tabarena_revision=TABARENA_REVISION,
                    train_rows=len(X), test_rows=len(X_test), features=len(X.columns),
                    categorical_features=len(categorical), numerical_features=len(X.columns)-len(categorical),
                    categorical_columns=categorical, training_missing_cells=int(X.isna().sum().sum()),
                    training_target_summary=target_summary,
                    feature_dtypes={str(k): str(v) for k, v in X.dtypes.items()})
        info["files_sha256"] = {p.name: sha256(p) for p in root.iterdir()
                                if p.suffix in [".parquet", ".npz", ".md"]}
        write_json(root / "dataset.json", info)
        print(f"Ready: {len(X):,} train / {len(X_test):,} test rows, {len(X.columns)} features", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=SMALL)
    for dataset in parser.parse_args().datasets:
        prepare(dataset)
