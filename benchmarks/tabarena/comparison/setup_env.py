"""Create isolated published-package environments without activating a venv."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from common import (BASE, CANDIDATE_VERSIONS, SDM_REVISION, TABARENA_REVISION,
                    clean_environment, lock, sha256, write_json, read_json)

HERE = Path(__file__).resolve().parent


def command(args, *, cwd=BASE, env=None):
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env or clean_environment(), check=True)


def ensure_environment(version):
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Install uv first: python -m pip install uv")
    BASE.mkdir(parents=True, exist_ok=True)
    # Dependency updates get a new directory, preserving environments of older runs.
    profile = sha256(HERE / "constraints.txt")[:12]
    root = BASE / "comparison-environments" / f"{version}-{profile}"
    python = root / "bin/python"
    marker = root / "benchmark-environment.json"
    spec = {"version": version, "constraints_sha256": sha256(HERE / "constraints.txt"),
            "python": "3.11", "torch": "2.11.0+cu128" if version in CANDIDATE_VERSIONS else None,
            "sdm_revision": SDM_REVISION if version in CANDIDATE_VERSIONS else None}
    if version in ("1.4.0rc3", "1.4.0rc4"):
        spec["features_goldmine"] = "1.1.0"
    with lock(root.parent / f"{version}.setup.lock"):
        if marker.exists():
            if read_json(marker) != spec:
                raise RuntimeError(f"Environment configuration changed; retain {root} and select a fresh environment directory")
            return python
        if not python.exists():
            command([uv, "venv", "--seed", "--python", "3.11", root])
        if version in CANDIDATE_VERSIONS:
            command([uv, "pip", "install", "--python", python, "torch==2.11.0",
                     "--index-url", "https://download.pytorch.org/whl/cu128"])
        package = f"mljar-supervised{'[all]' if version in CANDIDATE_VERSIONS else ''}=={version}"
        requirements = [package, "psutil", "pyarrow", "pytest"]
        if version in ("1.4.0rc3", "1.4.0rc4"):
            requirements.append("features-goldmine==1.1.0")
        if version in CANDIDATE_VERSIONS:
            requirements += [f"torch=={spec['torch']}",
                             f"git+https://github.com/NVIDIA/structured-data-models.git@{SDM_REVISION}"]
        # Pick up a just-published candidate even if uv cached the previous index.
        command([uv, "pip", "install", "--python", python, "--refresh-package", "mljar-supervised",
                 "--constraint", HERE / "constraints.txt", *requirements])
        command([uv, "pip", "check", "--python", python])
        frozen = subprocess.check_output([uv, "pip", "freeze", "--python", str(python)],
                                         cwd=BASE, env=clean_environment(), text=True)
        (root / "requirements-frozen.txt").write_text(frozen)
        write_json(marker, spec)
    return python


def ensure_tabarena():
    """Reuse the existing pinned data environment; bootstrap it on a fresh checkout."""
    uv = shutil.which("uv")
    source = BASE / "source"
    python = BASE / "venv/bin/python"
    with lock(BASE / "comparison-environments/tabarena.setup.lock"):
        if not source.exists():
            command(["git", "clone", "https://github.com/autogluon/tabarena.git", source])
            command(["git", "checkout", TABARENA_REVISION], cwd=source)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if revision != TABARENA_REVISION:
            raise RuntimeError(f"TabArena checkout must be pinned to {TABARENA_REVISION}; found {revision}")
        if not python.exists():
            if not uv:
                raise RuntimeError("Install uv first: python -m pip install uv")
            command([uv, "venv", "--seed", "--python", "3.11", python.parent.parent])
            command([uv, "pip", "install", "--python", python, "--prerelease=allow",
                     "-e", source / "packages/bencheval", "-e", source / "packages/tabarena"])
    return python
