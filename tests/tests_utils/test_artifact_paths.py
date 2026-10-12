import ntpath
import os
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from supervised.utils.artifact_paths import (
    portable_artifact_path,
    resolve_artifact_path,
)


@pytest.mark.parametrize(
    "subpath", ["model/learner.bin", r"model\learner.bin", r"model\nested/learner.bin"]
)
def test_legacy_separators_resolve_on_host(tmp_path, subpath):
    expected = (
        tmp_path
        / "model"
        / ("nested/learner.bin" if "nested" in subpath else "learner.bin")
    )
    assert resolve_artifact_path(tmp_path, subpath) == str(expected)


def test_serialization_is_portable_in_both_directions():
    saved = portable_artifact_path(PureWindowsPath("model/nested"), "learner.bin")
    assert saved == "model/nested/learner.bin"
    assert PureWindowsPath(ntpath.join(r"C:\deploy", saved)).parts == (
        "C:\\",
        "deploy",
        "model",
        "nested",
        "learner.bin",
    )
    assert portable_artifact_path(PurePosixPath("model/nested"), "learner.bin") == saved


def test_results_root_is_not_normalized():
    root = os.path.join("deployment", "literal\\directory")
    assert resolve_artifact_path(root, r"model\learner.bin") == os.path.join(
        root, "model", "learner.bin"
    )
