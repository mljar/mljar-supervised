import io
import os
import shutil
import unittest
from unittest.mock import patch

from sklearn import datasets

from supervised import AutoML
from supervised.apps.generator import PUBLISHABLE_WORKSPACE_FILES
from supervised.apps.publisher import BrowserTokenSession, generate_machine_learning_slug
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

        url = model.publish_app(open_browser=False)

        self.assertEqual(url, "https://feature-lab.ismvp.org")
        self.assertIsNotNone(FakePublishingClient.created_site)
        self.assertEqual(FakePublishingClient.created_site["domain"], "ismvp.org")
        self.assertEqual(
            sorted(filename for _, filename, _ in FakePublishingClient.registered_calls),
            sorted(PUBLISHABLE_WORKSPACE_FILES),
        )
        self.assertEqual(len(FakePublishingClient.upload_calls), len(PUBLISHABLE_WORKSPACE_FILES))

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

        url = model.publish_app(url="https://model-lab.ismvp.org", open_browser=False)

        self.assertEqual(url, "https://model-lab.ismvp.org")
        self.assertIsNone(FakePublishingClient.created_site)
        self.assertTrue(FakePublishingClient.registered_calls)
        self.assertEqual(
            {site_id for site_id, _, _ in FakePublishingClient.registered_calls},
            {303},
        )

    @patch("supervised.apps.publisher.BrowserTokenSession.authenticate", return_value="app.jwt.token")
    def test_publish_app_raises_when_target_url_not_found(self, _mock_authenticate):
        model = self._trained_model()
        with patch("supervised.apps.publisher.PlatformClient", FakePublishingClient):
            with self.assertRaises(AutoMLException):
                model.publish_app(url="https://missing.ismvp.org", open_browser=False)

    def test_publish_app_raises_when_model_not_trained(self):
        model = AutoML(
            algorithms=["Baseline"],
            explain_level=0,
            verbose=0,
            random_state=1,
            results_path=self.automl_dir,
        )
        with self.assertRaises(AutoMLException):
            model.publish_app(open_browser=False)

    def test_generate_machine_learning_slug_is_readable(self):
        slug = generate_machine_learning_slug(existing_slugs={"boost-lab"})
        self.assertRegex(slug, r"^[a-z0-9]+-[a-z0-9]+(?:-[a-f0-9]{4})?$")
        self.assertNotEqual(slug, "boost-lab")
