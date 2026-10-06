import io
import json
import os
import shutil
import unittest
from urllib import error as urllib_error
from unittest.mock import patch

from sklearn import datasets

from supervised import AutoML
from supervised.apps.generator import PUBLISHABLE_WORKSPACE_FILES
from supervised.apps.publisher import (
    BrowserTokenSession,
    PlatformApiError,
    PlatformClient,
    generate_machine_learning_slug,
)
from supervised.exceptions import AutoMLException


iris = datasets.load_iris()


class FakePublishingClient:
    created_site = None
    sites = []
    upload_calls = []
    registered_calls = []

    def __init__(self, base_url=None, token=None):
        self.base_url = base_url
        self.token = token

    def list_sites(self):
        return list(self.sites)

    def create_site(self, title, subdomain, domain):
        payload = {
            "id": 101,
            "title": title,
            "subdomain": subdomain,
            "domain": domain,
            "full_domain": f"{subdomain}.{domain}",
        }
        self.created_site = payload
        type(self).created_site = payload
        return payload

    def presign_upload(self, site_id, filename, filesize):
        return f"https://upload.example/{site_id}/{filename}"

    def upload_bytes(self, upload_url, body):
        type(self).upload_calls.append((upload_url, len(body)))

    def register_uploaded_file(self, site_id, filename, filesize):
        type(self).registered_calls.append((site_id, filename, filesize))

    def find_site_by_url(self, url):
        hostname = url.replace("https://", "").replace("http://", "").strip().rstrip("/")
        for site in self.sites:
            if site["full_domain"] == hostname:
                return site
        return None


class FailingUploadClient(FakePublishingClient):
    def upload_bytes(self, upload_url, body):
        raise PlatformApiError("SignatureDoesNotMatch", status=403, payload={})


class FailingCreateSiteClient(FakePublishingClient):
    def create_site(self, title, subdomain, domain):
        raise PlatformApiError("http_400", status=400, payload={})


class BrowserTokenSessionTests(unittest.TestCase):
    @patch("supervised.apps.publisher.webbrowser.open", return_value=False)
    @patch("supervised.apps.publisher.PlatformClient")
    def test_authenticate_prints_login_url_when_browser_cant_open(
        self, mock_client_class, _mock_open
    ):
        mock_client = mock_client_class.return_value
        mock_client._request.side_effect = [
            {"session_id": "session-1", "poll_token": "poll-1"},
            {"status": "completed"},
            {"token": "app.jwt.token"},
        ]
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            token = BrowserTokenSession(
                base_url="https://platform.mljar.com",
                open_browser=True,
                timeout=1,
            ).authenticate()

        self.assertEqual(token, "app.jwt.token")
        self.assertIn("/app/desktop-login?session_id=session-1&auth_mode=signin", stdout.getvalue())


class PlatformClientTests(unittest.TestCase):
    @patch("supervised.apps.publisher.http.client.HTTPSConnection")
    def test_upload_bytes_does_not_force_content_type_header(self, mock_urlopen):
        connection = mock_urlopen.return_value
        response = connection.getresponse.return_value
        response.status = 200
        response.read.return_value = b""

        client = PlatformClient(base_url="https://platform.mljar.com", token="token")
        client.upload_bytes("https://upload.example/file", b"body")

        connection.request.assert_called_once_with("PUT", "/file", body=b"body", headers={})
        connection.close.assert_called_once()

    @patch("supervised.apps.publisher.http.client.HTTPSConnection")
    def test_upload_bytes_preserves_non_json_http_error_body(self, mock_connection_class):
        connection = mock_connection_class.return_value
        response = connection.getresponse.return_value
        response.status = 403
        response.read.return_value = b"<Error><Code>SignatureDoesNotMatch</Code></Error>"

        client = PlatformClient(base_url="https://platform.mljar.com", token="token")

        with self.assertRaises(PlatformApiError) as ctx:
            client.upload_bytes("https://upload.example/file", b"body")

        self.assertIn("SignatureDoesNotMatch", str(ctx.exception))
        self.assertEqual(
            ctx.exception.payload.get("raw_body"),
            "<Error><Code>SignatureDoesNotMatch</Code></Error>",
        )
        connection.close.assert_called_once()

    @patch("supervised.apps.publisher.urllib_request.urlopen")
    def test_request_preserves_non_json_http_error_body(self, mock_urlopen):
        mock_urlopen.side_effect = urllib_error.HTTPError(
            url="https://platform.mljar.com/api/sites/",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=io.BytesIO(b"<Error><Code>SignatureDoesNotMatch</Code></Error>"),
        )

        client = PlatformClient(base_url="https://platform.mljar.com", token="token")

        with self.assertRaises(PlatformApiError) as ctx:
            client._request("GET", "/api/sites/")

        self.assertIn("SignatureDoesNotMatch", str(ctx.exception))
        self.assertEqual(
            ctx.exception.payload.get("raw_body"),
            "<Error><Code>SignatureDoesNotMatch</Code></Error>",
        )


class AutoMLPublishAppTests(unittest.TestCase):
    automl_dir = "AutoMLPublishAppTest"
    app_dir = os.path.join(automl_dir, "app")

    def setUp(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)
        shutil.rmtree(self.app_dir, ignore_errors=True)
        FakePublishingClient.created_site = None
        FakePublishingClient.sites = []
        FakePublishingClient.upload_calls = []
        FakePublishingClient.registered_calls = []

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)
        shutil.rmtree(self.app_dir, ignore_errors=True)

    def _trained_model(self):
        model = AutoML(
            algorithms=["Baseline"],
            explain_level=0,
            verbose=0,
            random_state=1,
            results_path=self.automl_dir,
        )
        model.fit(iris.data, iris.target)
        return model

    @patch("supervised.apps.publisher.generate_machine_learning_slug", return_value="feature-lab")
    @patch("supervised.apps.publisher.PlatformClient", FakePublishingClient)
    @patch("supervised.apps.publisher.BrowserTokenSession.authenticate", return_value="app.jwt.token")
    def test_publish_app_creates_new_site_and_uploads_runtime_files(
        self, _mock_authenticate, _mock_slug
    ):
        model = self._trained_model()

        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            url = model.publish_app(open_browser=False)

        self.assertEqual(url, "https://feature-lab.ismvp.org")
        self.assertIsNotNone(FakePublishingClient.created_site)
        self.assertEqual(FakePublishingClient.created_site["domain"], "ismvp.org")
        self.assertEqual(
            sorted(filename for _, filename, _ in FakePublishingClient.registered_calls),
            sorted(PUBLISHABLE_WORKSPACE_FILES),
        )
        self.assertIn(
            "mljar_app.json",
            [filename for _, filename, _ in FakePublishingClient.registered_calls],
        )
        self.assertEqual(len(FakePublishingClient.upload_calls), len(PUBLISHABLE_WORKSPACE_FILES))
        output = stdout.getvalue()
        self.assertIn("Start app publish", output)
        self.assertIn("Creating app workspace", output)
        self.assertIn("Signing in to MLJAR platform", output)
        self.assertIn("Creating app URL", output)
        self.assertIn("Created app URL: https://feature-lab.ismvp.org", output)
        self.assertIn("Uploading file: predict_single.ipynb", output)
        self.assertIn("Finished. You can access your app at: https://feature-lab.ismvp.org", output)

    @patch("supervised.apps.publisher.generate_machine_learning_slug", return_value="feature-lab")
    @patch("supervised.apps.publisher.PlatformClient", FakePublishingClient)
    @patch("supervised.apps.publisher.BrowserTokenSession.authenticate", return_value="app.jwt.token")
    def test_publish_app_overwrites_default_workspace_if_it_exists(
        self, _mock_authenticate, _mock_slug
    ):
        model = self._trained_model()
        os.makedirs(self.app_dir)
        stale_file = os.path.join(self.app_dir, "stale.txt")
        with open(stale_file, "w", encoding="utf-8") as fout:
            fout.write("stale")

        with patch("sys.stdout", new_callable=io.StringIO):
            url = model.publish_app(open_browser=False)

        self.assertEqual(url, "https://feature-lab.ismvp.org")
        self.assertFalse(os.path.exists(stale_file))
        self.assertTrue(FakePublishingClient.registered_calls)

    @patch("supervised.apps.publisher.generate_machine_learning_slug", return_value="feature-lab")
    @patch("supervised.apps.publisher.PlatformClient", FakePublishingClient)
    @patch("supervised.apps.publisher.BrowserTokenSession.authenticate", return_value="app.jwt.token")
    def test_publish_app_saves_and_prints_last_successful_url(
        self, _mock_authenticate, _mock_slug
    ):
        model = self._trained_model()

        first_stdout = io.StringIO()
        with patch("sys.stdout", first_stdout):
            first_url = model.publish_app(open_browser=False)

        self.assertEqual(first_url, "https://feature-lab.ismvp.org")
        state_path = os.path.join(self.automl_dir, "publish_app_state.json")
        self.assertTrue(os.path.exists(state_path))
        with open(state_path, "r", encoding="utf-8") as fin:
            state = json.load(fin)
        self.assertEqual(state.get("last_published_url"), first_url)

        FakePublishingClient.sites = [FakePublishingClient.created_site]
        FakePublishingClient.created_site = None

        second_stdout = io.StringIO()
        with patch("sys.stdout", second_stdout):
            second_url = model.publish_app(open_browser=False)

        self.assertEqual(second_url, first_url)
        self.assertIsNone(FakePublishingClient.created_site)
        self.assertIn(f"Last published app URL: {first_url}", second_stdout.getvalue())
        self.assertIn(f"Checking existing app URL: {first_url}", second_stdout.getvalue())
        self.assertIn(f"Using app URL: {first_url}", second_stdout.getvalue())

    @patch("supervised.apps.publisher.generate_machine_learning_slug", return_value="feature-lab")
    @patch("supervised.apps.publisher.PlatformClient", FakePublishingClient)
    @patch("supervised.apps.publisher.BrowserTokenSession.authenticate", return_value="app.jwt.token")
    def test_publish_app_creates_new_site_when_last_published_url_is_missing(
        self, _mock_authenticate, _mock_slug
    ):
        model = self._trained_model()

        first_stdout = io.StringIO()
        with patch("sys.stdout", first_stdout):
            first_url = model.publish_app(open_browser=False)

        self.assertEqual(first_url, "https://feature-lab.ismvp.org")
        FakePublishingClient.created_site = None
        FakePublishingClient.sites = []

        second_stdout = io.StringIO()
        with patch("sys.stdout", second_stdout):
            second_url = model.publish_app(open_browser=False)

        self.assertEqual(second_url, first_url)
        self.assertIsNotNone(FakePublishingClient.created_site)
        output = second_stdout.getvalue()
        self.assertIn(f"Checking existing app URL: {first_url}", output)
        self.assertIn(
            f"Last published app URL not found on platform: {first_url}",
            output,
        )
        self.assertIn("Creating app URL", output)

    @patch("supervised.apps.publisher.PlatformClient", FakePublishingClient)
    @patch("supervised.apps.publisher.BrowserTokenSession.authenticate", return_value="app.jwt.token")
    def test_publish_app_updates_existing_site_from_url(self, _mock_authenticate):
        FakePublishingClient.sites = [
            {
                "id": 303,
                "title": "Existing App",
                "subdomain": "model-lab",
                "domain": "ismvp.org",
                "full_domain": "model-lab.ismvp.org",
            }
        ]
        model = self._trained_model()

        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            url = model.publish_app(url="https://model-lab.ismvp.org", open_browser=False)

        self.assertEqual(url, "https://model-lab.ismvp.org")
        self.assertIsNone(FakePublishingClient.created_site)
        self.assertTrue(FakePublishingClient.registered_calls)
        self.assertEqual(
            {site_id for site_id, _, _ in FakePublishingClient.registered_calls},
            {303},
        )
        output = stdout.getvalue()
        self.assertIn("Checking existing app URL: https://model-lab.ismvp.org", output)
        self.assertIn("Using app URL: https://model-lab.ismvp.org", output)

    @patch("supervised.apps.publisher.BrowserTokenSession.authenticate", return_value="app.jwt.token")
    def test_publish_app_raises_when_target_url_not_found(self, _mock_authenticate):
        model = self._trained_model()
        with patch("supervised.apps.publisher.PlatformClient", FakePublishingClient):
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                url = model.publish_app(url="https://missing.ismvp.org", open_browser=False)
        self.assertIsNone(url)
        self.assertIn("Publish app failed:", stdout.getvalue())
        self.assertIn("Could not find Mercury app", stdout.getvalue())

    @patch("supervised.apps.publisher.generate_machine_learning_slug", return_value="feature-lab")
    @patch("supervised.apps.publisher.PlatformClient", FailingUploadClient)
    @patch("supervised.apps.publisher.BrowserTokenSession.authenticate", return_value="app.jwt.token")
    def test_publish_app_reports_filename_when_upload_fails(
        self, _mock_authenticate, _mock_slug
    ):
        model = self._trained_model()

        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            url = model.publish_app(open_browser=False)

        self.assertIsNone(url)
        message = stdout.getvalue()
        self.assertIn("Publish app failed:", message)
        self.assertIn("Failed to upload", message)
        self.assertIn("predict_single.ipynb", message)
        self.assertIn("SignatureDoesNotMatch", message)

    @patch("supervised.apps.publisher.generate_machine_learning_slug", return_value="feature-lab")
    @patch("supervised.apps.publisher.PlatformClient", FailingCreateSiteClient)
    @patch("supervised.apps.publisher.BrowserTokenSession.authenticate", return_value="app.jwt.token")
    def test_publish_app_reports_create_site_context_for_opaque_bad_request(
        self, _mock_authenticate, _mock_slug
    ):
        model = self._trained_model()

        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            url = model.publish_app(open_browser=False)

        self.assertIsNone(url)
        message = stdout.getvalue()
        self.assertIn("Publish app failed:", message)
        self.assertIn("https://feature-lab.ismvp.org", message)
        self.assertIn("The platform rejected the app creation request.", message)
        self.assertIn("account limits, permissions, or invalid app settings", message)

    def test_publish_app_raises_when_model_not_trained(self):
        model = AutoML(
            algorithms=["Baseline"],
            explain_level=0,
            verbose=0,
            random_state=1,
            results_path=self.automl_dir,
        )
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            url = model.publish_app(open_browser=False)
        self.assertIsNone(url)
        self.assertIn("Publish app failed:", stdout.getvalue())
        self.assertIn("Please call `fit()` first.", stdout.getvalue())

    def test_generate_machine_learning_slug_is_readable(self):
        slug = generate_machine_learning_slug(existing_slugs={"boost-lab"})
        self.assertRegex(slug, r"^[a-z0-9]+-[a-z0-9]+-[a-f0-9]{4}$")
        self.assertNotEqual(slug, "boost-lab")
