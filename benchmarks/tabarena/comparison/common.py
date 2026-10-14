"""Shared benchmark configuration; deliberately uses only the standard library."""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# A frozen copy passes its original workspace via the environment.
ROOT = Path(os.environ.get("MLJAR_BENCHMARK_ROOT", ROOT))
BASE = ROOT / "performance-results/tabarena"
STABLE_VERSION = "1.3.2"
CANDIDATE_VERSIONS = ("1.4.0rc1", "1.4.0rc2", "1.4.0rc3")
DEFAULT_CANDIDATE = "1.4.0rc3"
VERSIONS = (STABLE_VERSION, *CANDIDATE_VERSIONS)
TABARENA_REVISION = "10b1ecb9fa628b8f69db6763f8aeaafdd5d5ef77"
SDM_REVISION = "b97cb914e2bdcf0d165fc2c83b5fdf795aad6a69"
NEW_ALGORITHMS = ["RealMLP", "TabM", "Kumo", "TabICLv2"]
SMALL = [
    "credit-g", "Bank_Customer_Churn", "Amazon_employee_access",
    "polish_companies_bankruptcy", "maternal_health_risk", "anneal", "splice",
    "students_dropout_and_academic_success", "airfoil_self_noise",
    "healthcare_insurance_expenses", "houses", "superconductivity",
]
SMOKE = ["credit-g", "maternal_health_risk", "airfoil_self_noise"]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def lock(path, *, wait=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError(f"Another benchmark process holds {path}") from None
        yield


def clean_environment(n_jobs=1, device="cpu"):
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(name, None)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = str(n_jobs)
    env.update(PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", MPLBACKEND="Agg",
               MLJAR_BENCHMARK_ROOT=str(ROOT), TOKENIZERS_PARALLELISM="false")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # Preserve an explicitly chosen GPU, otherwise use the first one.
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def cpu_affinity(version, n_jobs):
    """Partition physical cores, keeping SMT siblings in the same worker."""
    if version not in VERSIONS:
        raise ValueError(f"Unsupported MLJAR version: {version}")
    allowed = sorted(os.sched_getaffinity(0))
    cores = {}
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((topology / "physical_package_id").read_text().strip(),
               (topology / "core_id").read_text().strip())
        cores.setdefault(key, []).append(cpu)
    groups = sorted(cores.values(), key=min)
    # Reserve two physical cores on larger workstations.
    count = (len(groups) - (2 if len(groups) >= 10 else 0)) // 2
    if count < 1:
        raise ValueError("Two parallel versions require at least two physical cores")
    offset = (0 if version == STABLE_VERSION else 1) * count
    selected = [cpu for group in groups[offset:offset + count] for cpu in group]
    if n_jobs > len(selected):
        raise ValueError(f"This worker has {len(selected)} logical CPUs; lower --n-jobs")
    return sorted(selected)[:n_jobs]
