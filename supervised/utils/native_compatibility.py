"""Bounded macOS native-runtime preflight. Never import Torch in the host."""

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time

from supervised.exceptions import NativeCompatibilityError

TORCH_ALGORITHMS = frozenset({"RealMLP", "TabM", "Kumo", "TabICLv2"})
TIMEOUT_SECONDS = 3.0
MIN_AVAILABLE_BYTES = 1024**3
MAX_CHILD_RSS = 768 * 1024**2
MAX_OUTPUT_BYTES = 64 * 1024
PROBE_REVISION = 1
_cache = {}
_lock = threading.Lock()
_PACKAGES = ("torch", "lightgbm", "xgboost", "scikit-learn", "numpy", "pytabkit", "structured-data-models")


def _result(status, reason, **details):
    return dict(status=status, reason=reason, **details)


def allowed(result):
    return result["status"] in ("not_needed", "passed")


def is_macos():
    return sys.platform == "darwin"


def torch_loaded():
    return "torch" in sys.modules


def validate_dependencies(algorithms):
    # Existing metadata-only validation takes precedence over a native probe.
    if not is_macos():
        return
    modules = {"RealMLP": "realmlp", "TabM": "tabm", "Kumo": "sdm", "TabICLv2": "sdm"}
    for module in dict.fromkeys(modules[name] for name in algorithms if name in modules):
        importlib.import_module(f"supervised.algorithms.{module}").check_dependencies()


def require_algorithm(algorithm, context):
    validate_dependencies([algorithm])
    return require_compatible(algorithm in TORCH_ALGORITHMS, context)


def _file_identity(path):
    path = Path(path).resolve()
    try:
        stat = path.stat()
        return (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino)
    except OSError:
        return (str(path), None)


def _inspect():
    """Inspect files/module state, not OpenMP APIs or CUDA availability."""
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    binaries = []
    for name, relative in (
        ("torch", "lib/libomp.dylib"), ("torch", "lib/libtorch_cpu.dylib"),
        ("sklearn", ".dylibs/libomp.dylib"),
        ("lightgbm", "lib/lib_lightgbm.dylib"),
        ("xgboost", "lib/libxgboost.dylib"),
    ):
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.submodule_search_locations:
            for directory in spec.submodule_search_locations:
                binaries.append(_file_identity(Path(directory) / relative))
    for path in ("/opt/homebrew/opt/libomp/lib/libomp.dylib",
                 "/usr/local/opt/libomp/lib/libomp.dylib", "/opt/local/lib/libomp/libomp.dylib"):
        binaries.append(_file_identity(path))
    # Preserve insertion order: later imports can change native initialization.
    loaded = [(name, getattr(module, "__file__", None))
              for name, module in tuple(sys.modules.items())
              if name in ("torch", "torch._C", "lightgbm", "xgboost", "sklearn")]
    state = dict(revision=PROBE_REVISION, executable=sys.executable,
                 python=sys.version, platform=sys.platform, machine=platform.machine(),
                 versions=versions, binaries=binaries, loaded=loaded,
                 environment={name: os.environ.get(name) for name in (
                     "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OMP_THREAD_LIMIT",
                     "KMP_DUPLICATE_LIB_OK", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
                 )})
    key = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    return key, versions


def _stop(process):
    """Kill the isolated process group and reap it, including on cancellation."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=0.1)
    except subprocess.TimeoutExpired:
        # Do not extend the caller's deadline for an uninterruptible OS task.
        threading.Thread(target=process.wait, daemon=True).start()


def _run_child(order, deadline, *, script=None):
    """One child; caller supplies the deadline shared by every probe case."""
    import psutil

    if deadline - time.monotonic() <= 0.15:
        return _result("inconclusive", "timeout")
    script = script or Path(__file__).with_name("_native_probe.py")
    try:
        process = subprocess.Popen(
            [sys.executable, "-X", "faulthandler", str(script), order],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
    except OSError as error:
        return _result("inconclusive", "launch_failed", diagnostic=str(error))
    captured = bytearray()

    def drain():
        try:
            while chunk := process.stdout.read(4096):
                remaining = MAX_OUTPUT_BYTES - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])
        finally:
            process.stdout.close()

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    outcome = None
    try:
        while process.poll() is None:
            # Reserve a short cleanup window inside the shared three seconds.
            remaining = deadline - time.monotonic() - 0.15
            if remaining <= 0:
                outcome = _result("inconclusive", "timeout")
                break
            try:
                if psutil.Process(process.pid).memory_info().rss > MAX_CHILD_RSS:
                    outcome = _result("inconclusive", "probe_memory_limit")
                    break
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                outcome = _result("inconclusive", "memory_monitor_unavailable")
                break
            time.sleep(min(0.01, remaining))
    finally:
        _stop(process)
        reader.join(timeout=max(0, min(0.05, deadline - time.monotonic())))
    output = bytes(captured).decode("utf-8", errors="replace")
    if outcome is not None:
        return dict(outcome, diagnostic=output)
    if process.returncode is None:
        return _result("inconclusive", "cleanup_pending", diagnostic=output)
    if process.returncode < 0:
        return _result("blocked", "native_crash", signal=-process.returncode, diagnostic=output)
    if process.returncode != 0:
        if '"reason": "import_error"' in output:
            return _result("inconclusive", "import_error", diagnostic=output)
        return _result("inconclusive", "probe_error", diagnostic=output)
    try:
        payload = json.loads(output.strip().splitlines()[-1])
        if payload != {"status": "passed", "order": order}:
            raise ValueError("Unexpected probe response")
    except (ValueError, IndexError):
        return _result("inconclusive", "invalid_response", diagnostic=output)
    return _result("passed", "probe_passed")


def check_native_compatibility(needs_torch=False):
    """Return evidence for the tested native workload, never a universal guarantee."""
    if not is_macos():
        return _result("not_needed", "platform")
    if not needs_torch and not torch_loaded():
        return _result("not_needed", "torch_unused")
    started = time.monotonic()
    deadline = started + TIMEOUT_SECONDS
    try:
        if importlib.util.find_spec("torch") is None:
            # The learner's existing missing-dependency error remains authoritative.
            return _result("not_needed", "torch_absent")
        key, versions = _inspect()
    except (OSError, ValueError, ImportError) as error:
        return _result("inconclusive", "inspection_failed", diagnostic=str(error))
    if not _lock.acquire(timeout=max(0, deadline - time.monotonic())):
        return _result("inconclusive", "timeout")
    try:
        if key in _cache:
            return copy.deepcopy(_cache[key])
        try:
            import psutil
            free = psutil.virtual_memory().available
        except (ImportError, OSError):
            return _result("inconclusive", "memory_monitor_unavailable")
        if free < MIN_AVAILABLE_BYTES:
            return _result("inconclusive", "insufficient_probe_memory", available_bytes=free)
        result = None
        for order in ("boosting-first", "torch-first"):
            result = _run_child(order, deadline)
            result["probe_order"] = order
            if not allowed(result):
                break
        result.update(fingerprint=key, versions=versions,
                      elapsed_seconds=round(time.monotonic() - started, 4))
        # An inconclusive run can be retried by a later API request, never here.
        if result["status"] in ("passed", "blocked"):
            if len(_cache) >= 16:
                _cache.clear()
            _cache[key] = copy.deepcopy(result)
        return result
    finally:
        _lock.release()


def raise_incompatible(result, context):
    raise NativeCompatibilityError(
        f"Cannot run {context}: native runtime compatibility on macOS "
        f"is not established ({result['reason']}). The check is limited to 3 seconds. "
        "See https://supervised.mljar.com/install/#native-runtime-compatibility. "
        "No model was substituted.", result,
    )


def require_compatible(needs_torch=False, context="this model"):
    result = check_native_compatibility(needs_torch)
    if not allowed(result):
        raise_incompatible(result, context)
    return result
