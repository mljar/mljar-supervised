import os
import shutil
import socket
import subprocess
import time
import webbrowser
from urllib import error as urllib_error
from urllib import request as urllib_request

from supervised.apps.generator import generate_app
from supervised.exceptions import AutoMLException


LOCAL_APP_HOST = "127.0.0.1"
LOCAL_APP_START_TIMEOUT_SECONDS = 30
LOCAL_APP_POLL_INTERVAL_SECONDS = 0.5


def run_local_app_from_automl(automl):
    app_dir = generate_app(automl, overwrite=True, verbose=False)
    mercury_path = shutil.which("mercury")
    if mercury_path is None:
        raise AutoMLException(
            "Mercury is not installed or not available in PATH. "
            "Install it with: pip install -r requirements.txt"
        )

    port = _get_free_port()
    url = f"http://{LOCAL_APP_HOST}:{port}"
    os.makedirs(app_dir, exist_ok=True)
    log_path = os.path.join(app_dir, ".local_app.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                mercury_path,
                f"--ip={LOCAL_APP_HOST}",
                f"--port={port}",
                "--no-browser",
            ],
            cwd=app_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    _wait_until_ready(url, process, log_path)
    webbrowser.open(url)
    print(f"Local app URL: {url}")
    print("Press Ctrl+C to stop the local app.")
    automl._local_app_process = process
    automl._local_app_url = url
    return _wait_for_local_app_shutdown(process, log_path, url)


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOCAL_APP_HOST, 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def _wait_until_ready(url, process, log_path):
    deadline = time.time() + LOCAL_APP_START_TIMEOUT_SECONDS
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise AutoMLException(_process_failure_message(process.returncode, log_path))
        try:
            with urllib_request.urlopen(url, timeout=2) as response:
                if 200 <= getattr(response, "status", 200) < 500:
                    return
        except urllib_error.URLError as exc:
            last_error = exc
            time.sleep(LOCAL_APP_POLL_INTERVAL_SECONDS)

    raise AutoMLException(
        "Mercury server did not start in time. "
        f"Tried URL {url}. Last error: {last_error}"
    )


def _process_failure_message(returncode, log_path):
    tail = _read_log_tail(log_path)
    message = f"Mercury server exited with code {returncode}."
    if tail:
        message += f" Last log output:\n{tail}"
    return message


def _wait_for_local_app_shutdown(process, log_path, url):
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        _stop_process(process)
        return url

    if returncode not in (None, 0):
        raise AutoMLException(_process_failure_message(returncode, log_path))
    return url


def _stop_process(process, timeout=5):
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _read_log_tail(log_path, max_chars=2000):
    if not os.path.exists(log_path):
        return ""
    with open(log_path, "r", encoding="utf-8", errors="replace") as fin:
        content = fin.read()
    if len(content) <= max_chars:
        return content
    return content[-max_chars:]
