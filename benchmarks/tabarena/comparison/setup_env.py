"""Create isolated published-package or local-wheel benchmark environments."""
from __future__ import annotations

import shutil
import subprocess
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile, BadZipFile

from common import (BASE, CANDIDATE_VERSIONS, DISCOVERY_VERSIONS, SDM_REVISION, TABARENA_REVISION,
                    clean_environment, lock, sha256, write_json, read_json)

HERE = Path(__file__).resolve().parent


def command(args, *, cwd=BASE, env=None):
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env or clean_environment(), check=True)


def local_wheel_info(wheel, version):
    wheel = Path(wheel).resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError(f"Local wheel does not exist: {wheel}")
    try:
        with ZipFile(wheel) as archive:
            metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("Local wheel must contain exactly one package metadata file")
            package = BytesParser().parsebytes(archive.read(metadata[0]))
    except BadZipFile as error:
        raise ValueError(f"Invalid wheel: {wheel}") from error
    if package.get("Name", "").lower().replace("_", "-") != "mljar-supervised" or package.get("Version") != version:
        raise ValueError(f"Local wheel must contain mljar-supervised {version}")
    return dict(filename=wheel.name, sha256=sha256(wheel))


def ensure_environment(version, local_wheel=None):
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Install uv first: python -m pip install uv")
    BASE.mkdir(parents=True, exist_ok=True)
    # Dependency updates get a new directory, preserving environments of older runs.
    profile = sha256(HERE / "constraints.txt")[:12]
    artifact = local_wheel_info(local_wheel, version) if local_wheel else None
    if artifact:
        profile += f"-local-{artifact['sha256'][:12]}"
    root = BASE / "comparison-environments" / f"{version}-{profile}"
    python = root / "bin/python"
    marker = root / "benchmark-environment.json"
    spec = {"version": version, "constraints_sha256": sha256(HERE / "constraints.txt"),
            "python": "3.11", "torch": "2.11.0+cu128" if version in CANDIDATE_VERSIONS else None,
            "sdm_revision": SDM_REVISION if version in CANDIDATE_VERSIONS else None}
    if version in DISCOVERY_VERSIONS:
        spec["features_goldmine"] = "1.1.0"
    if artifact:
        spec["local_wheel"] = artifact
    with lock(root.parent / f"{version}.setup.lock"):
        if marker.exists():
            if read_json(marker) != spec:
                raise RuntimeError(f"Environment configuration changed; retain {root} and select a fresh environment directory")
            if artifact and sha256(root / "artifacts" / artifact["filename"]) != artifact["sha256"]:
                raise ValueError("Retained local wheel changed; use a fresh environment")
            return python
        if not python.exists():
            command([uv, "venv", "--seed", "--python", "3.11", root])
        if artifact:
            retained_wheel = root / "artifacts" / artifact["filename"]
            retained_wheel.parent.mkdir(exist_ok=True)
            shutil.copy2(local_wheel, retained_wheel)
            if sha256(retained_wheel) != artifact["sha256"]:
                raise ValueError("Local wheel changed while preparing the environment")
        if version in CANDIDATE_VERSIONS:
            command([uv, "pip", "install", "--python", python, "torch==2.11.0",
                     "--index-url", "https://download.pytorch.org/whl/cu128"])
        package = f"mljar-supervised{'[all]' if version in CANDIDATE_VERSIONS else ''}=={version}"
        if artifact:
            package = f"mljar-supervised{'[all]' if version in CANDIDATE_VERSIONS else ''} @ {retained_wheel.as_uri()}"
        requirements = [package, "psutil", "pyarrow", "pytest"]
        if version in DISCOVERY_VERSIONS:
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
