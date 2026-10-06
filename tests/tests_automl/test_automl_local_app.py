import io
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from supervised import AutoML
from supervised.apps.local_runner import (
    _process_failure_message,
    _stop_process,
    run_local_app_from_automl,
)
from supervised.exceptions import AutoMLException


class AutoMLLocalAppTests(unittest.TestCase):
    @patch("supervised.apps.local_runner.webbrowser.open")
    @patch("supervised.apps.local_runner._wait_until_ready")
    @patch("supervised.apps.local_runner._get_free_port", return_value=9988)
    @patch("supervised.apps.local_runner.subprocess.Popen")
    @patch("supervised.apps.local_runner.shutil.which", return_value="/tmp/mercury")
    @patch("supervised.apps.local_runner.generate_app", return_value="/tmp/appdir")
    def test_local_app_runs_mercury_and_opens_browser(
        self,
        mock_generate,
        _mock_which,
        mock_popen,
        _mock_port,
        mock_wait,
        mock_open,
    ):
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [KeyboardInterrupt(), None]
        mock_popen.return_value = process
        model = AutoML(results_path="LocalAppModel")

        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            url = run_local_app_from_automl(model)

        self.assertEqual(url, "http://127.0.0.1:9988")
        mock_generate.assert_called_once_with(model, overwrite=True, verbose=False)
        mock_wait.assert_called_once()
        mock_open.assert_called_once_with("http://127.0.0.1:9988")
        self.assertEqual(model._local_app_url, "http://127.0.0.1:9988")
        self.assertIs(model._local_app_process, process)
        self.assertIn("Local app URL: http://127.0.0.1:9988", stdout.getvalue())
        self.assertIn("Press Ctrl+C to stop the local app.", stdout.getvalue())
        process.terminate.assert_called_once()

    @patch("supervised.apps.local_runner.generate_app", return_value="/tmp/appdir")
    @patch("supervised.apps.local_runner.shutil.which", return_value=None)
    def test_local_app_raises_when_mercury_missing(self, _mock_which, _mock_generate):
        model = AutoML(results_path="LocalAppModel")
        with self.assertRaises(AutoMLException) as ctx:
            run_local_app_from_automl(model)
        self.assertIn("Install it with: pip install -r requirements.txt", str(ctx.exception))

    def test_process_failure_message_includes_log_tail(self):
        with patch("supervised.apps.local_runner._read_log_tail", return_value="boom"):
            message = _process_failure_message(2, "/tmp/fake.log")
        self.assertIn("Mercury server exited with code 2.", message)
        self.assertIn("boom", message)

    def test_stop_process_kills_when_terminate_times_out(self):
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired(cmd="mercury", timeout=5), None]

        _stop_process(process)

        process.terminate.assert_called_once()
        process.kill.assert_called_once()
