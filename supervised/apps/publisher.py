import hashlib
import http.client
import json
import os
import platform
import secrets
import socket
import time
import uuid
import webbrowser
from importlib.metadata import PackageNotFoundError, version
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from supervised.apps.generator import generate_app, publishable_workspace_paths
from supervised.exceptions import AutoMLException


DEFAULT_PLATFORM_BASE_URL = os.environ.get(
    "MLJAR_PLATFORM_BASE_URL", "https://platform.mljar.com"
).rstrip("/")
DEFAULT_PLATFORM_DOMAIN = os.environ.get("MLJAR_PLATFORM_DEFAULT_DOMAIN", "ismvp.org")
LOGIN_TIMEOUT_SECONDS = 300
LOGIN_POLL_INTERVAL_SECONDS = 2
CREATE_SITE_MAX_ATTEMPTS = 20
PUBLISH_STATE_FILENAME = "publish_app_state.json"

ML_SLUG_PREFIXES = (
    "boost",
    "cluster",
    "feature",
    "forest",
    "gradient",
    "kernel",
    "metric",
    "model",
    "neuron",
    "signal",
    "target",
    "tensor",
    "vector",
)

ML_SLUG_SUFFIXES = (
    "atlas",
    "engine",
    "flow",
    "forge",
    "insight",
    "lab",
    "pilot",
    "radar",
    "scope",
    "studio",
)


class PlatformApiError(Exception):
    def __init__(self, message, status=None, payload=None):
        super().__init__(message)
        self.status = status
        self.payload = payload or {}


def _package_version():
    try:
        return version("mljar-supervised")
    except PackageNotFoundError:
        return "unknown"


def _device_payload():
    device_install_id = f"mljar-supervised-{uuid.uuid4().hex}"
    host = socket.gethostname()
    platform_name = platform.system().lower() or "python"
    fingerprint_hash = hashlib.sha256(
        f"{device_install_id}|{platform_name}|{host}".encode("utf-8")
    ).hexdigest()
    return {
        "device_install_id": device_install_id,
        "platform": platform_name,
        "app_version": _package_version(),
        "device_label": host,
        "client_fingerprint_hash": fingerprint_hash,
    }


def generate_machine_learning_slug(existing_slugs=None):
    existing = existing_slugs or set()
    while True:
        slug = (
            f"{secrets.choice(ML_SLUG_PREFIXES)}-"
            f"{secrets.choice(ML_SLUG_SUFFIXES)}-"
            f"{secrets.token_hex(2)}"
        )
        if slug not in existing:
            return slug


def _normalize_url(url):
    candidate = str(url or "").strip()
    if not candidate:
        return ""
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    return candidate


def _site_url(site_payload):
    full_domain = str(site_payload.get("full_domain") or "").strip()
    if not full_domain:
        domain = str(site_payload.get("domain") or "").strip()
        subdomain = str(site_payload.get("subdomain") or "").strip()
        if domain and subdomain:
            full_domain = f"{subdomain}.{domain}"
    if not full_domain:
        raise AutoMLException("Platform site response is missing domain details.")
    return f"https://{full_domain}"


def _workspace_title(output_path, fallback_title=None):
    manifest_path = os.path.join(output_path, "mljar_app.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fin:
            manifest = json.load(fin)
        title = str(manifest.get("title") or "").strip()
        if title:
            return title
    if fallback_title:
        return str(fallback_title).strip()
    return os.path.basename(output_path.rstrip(os.sep)) or "MLJAR AutoML App"


def _publish_state_path(automl):
    results_path = automl._get_results_path()
    if not results_path:
        return None
    return os.path.join(results_path, PUBLISH_STATE_FILENAME)


def _load_publish_state(automl):
    state_path = _publish_state_path(automl)
    if not state_path or not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as fin:
            payload = json.load(fin)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _save_publish_state(automl, app_url):
    state_path = _publish_state_path(automl)
    if not state_path:
        return
    payload = {"last_published_url": app_url}
    with open(state_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)
        fout.write("\n")


class PlatformClient:
    def __init__(self, base_url=DEFAULT_PLATFORM_BASE_URL, token=None):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _request(
        self, method, path_or_url, data=None, headers=None, absolute=False, with_auth=True
    ):
        target = (
            path_or_url
            if absolute
            else urllib_parse.urljoin(f"{self.base_url}/", path_or_url.lstrip("/"))
        )
        body = None
        request_headers = {"Accept": "application/json"}
        if headers:
            request_headers.update(headers)
        if self.token and with_auth:
            request_headers["Authorization"] = f"Bearer {self.token}"
        if data is not None:
            if isinstance(data, (bytes, bytearray)):
                body = data
            else:
                body = json.dumps(data).encode("utf-8")
                request_headers.setdefault("Content-Type", "application/json")
        req = urllib_request.Request(
            target, data=body, headers=request_headers, method=method
        )
        try:
            with urllib_request.urlopen(req, timeout=30) as response:
                response_body = response.read()
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type and response_body:
                    return json.loads(response_body.decode("utf-8"))
                return response_body
        except urllib_error.HTTPError as exc:
            raw = exc.read()
            payload = {}
            message = f"http_{exc.code}"
            if raw:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    payload = {"raw_body": raw.decode("utf-8", errors="replace")}
            if isinstance(payload, dict):
                message = (
                    payload.get("detail")
                    or payload.get("message")
                    or payload.get("code")
                    or payload.get("raw_body")
                    or message
                )
            raise PlatformApiError(message, status=exc.code, payload=payload)
        except urllib_error.URLError as exc:
            raise AutoMLException(
                f"Could not connect to MLJAR platform: {getattr(exc, 'reason', exc)}"
            )

    def list_sites(self):
        return self._request("GET", "/api/sites/")

    def create_site(self, title, subdomain, domain):
        return self._request(
            "POST",
            "/api/sites/",
            data={
                "title": title,
                "subdomain": subdomain,
                "domain": domain,
                "is_public": True,
            },
        )

    def presign_upload(self, site_id, filename, filesize):
        path = "/api/presigned-url-put/{site_id}/{filename}/{filesize}".format(
            site_id=urllib_parse.quote(str(site_id), safe=""),
            filename=urllib_parse.quote(filename, safe=""),
            filesize=urllib_parse.quote(str(filesize), safe=""),
        )
        payload = self._request("GET", path)
        if not isinstance(payload, dict) or not payload.get("url"):
            raise AutoMLException("Platform did not return an upload URL.")
        return payload["url"]

    def upload_bytes(self, upload_url, body):
        parsed = urllib_parse.urlparse(upload_url)
        connection_class = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        connection = connection_class(parsed.netloc, timeout=30)
        try:
            connection.request("PUT", path, body=body, headers={})
            response = connection.getresponse()
            response_body = response.read()
            if 200 <= response.status < 300:
                return response_body

            payload = {}
            message = f"http_{response.status}"
            if response_body:
                try:
                    payload = json.loads(response_body.decode("utf-8"))
                except Exception:
                    payload = {"raw_body": response_body.decode("utf-8", errors="replace")}
            if isinstance(payload, dict):
                message = (
                    payload.get("detail")
                    or payload.get("message")
                    or payload.get("code")
                    or payload.get("raw_body")
                    or message
                )
            raise PlatformApiError(message, status=response.status, payload=payload)
        except OSError as exc:
            raise AutoMLException(f"Could not upload file to storage: {exc}")
        finally:
            connection.close()

    def register_uploaded_file(self, site_id, filename, filesize):
        self._request(
            "POST",
            "/api/file-uploaded",
            data={"site_id": site_id, "filename": filename, "filesize": filesize},
        )

    def find_site_by_url(self, url):
        normalized = _normalize_url(url)
        if not normalized:
            return None
        parsed = urllib_parse.urlparse(normalized)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise AutoMLException("Provided app URL is invalid.")
        for site in self.list_sites():
            full_domain = str(site.get("full_domain") or "").strip().lower()
            if full_domain == hostname:
                return site
        return None


class BrowserTokenSession:
    def __init__(self, base_url=DEFAULT_PLATFORM_BASE_URL, open_browser=True, timeout=LOGIN_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.open_browser = open_browser
        self.timeout = timeout
        self.device = _device_payload()
        self.client = PlatformClient(base_url=self.base_url)

    def authenticate(self):
        session = self.client._request(
            "POST",
            "/api/app/auth/session/start",
            data=self.device,
        )
        session_id = str(session.get("session_id") or "")
        poll_token = str(session.get("poll_token") or "")
        if not session_id or not poll_token:
            raise AutoMLException("Platform did not return a valid login session.")
        login_url = (
            f"{self.base_url}/app/desktop-login?"
            f"session_id={urllib_parse.quote(session_id)}&auth_mode=signin"
        )
        opened = False
        if self.open_browser:
            try:
                opened = bool(webbrowser.open(login_url))
            except Exception:
                opened = False
        if not opened:
            print("Open this URL in your browser to sign in to MLJAR platform:")
            print(login_url)
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            status_payload = self.client._request(
                "GET",
                "/api/app/auth/session/status?"
                f"session_id={urllib_parse.quote(session_id)}&poll_token={urllib_parse.quote(poll_token)}",
            )
            status = str(status_payload.get("status") or "")
            if status == "completed":
                exchanged = self.client._request(
                    "POST",
                    "/api/app/auth/session/exchange",
                    data={
                        **self.device,
                        "session_id": session_id,
                        "poll_token": poll_token,
                    },
                )
                token = str(exchanged.get("token") or "")
                if not token:
                    raise AutoMLException("Platform login succeeded but no token was returned.")
                return token
            if status in {"expired", "consumed"}:
                raise AutoMLException("Platform login session expired. Please try again.")
            time.sleep(LOGIN_POLL_INTERVAL_SECONDS)
        raise AutoMLException("Timed out waiting for MLJAR platform login to complete.")


def _read_publishable_files(output_path):
    files = []
    for file_path in publishable_workspace_paths(output_path):
        if not os.path.exists(file_path):
            raise AutoMLException(
                f"Generated app is missing required publish file '{os.path.basename(file_path)}'."
            )
        with open(file_path, "rb") as fin:
            body = fin.read()
        files.append((os.path.basename(file_path), body))
    return files


def _create_site_with_generated_slug(client, title):
    sites = client.list_sites()
    existing_slugs = {
        str(site.get("subdomain") or "").strip().lower()
        for site in sites
        if str(site.get("domain") or "").strip().lower() == DEFAULT_PLATFORM_DOMAIN
    }
    for _ in range(CREATE_SITE_MAX_ATTEMPTS):
        slug = generate_machine_learning_slug(existing_slugs)
        target_url = f"https://{slug}.{DEFAULT_PLATFORM_DOMAIN}"
        try:
            return client.create_site(title, slug, DEFAULT_PLATFORM_DOMAIN)
        except PlatformApiError as exc:
            payload_text = json.dumps(exc.payload, sort_keys=True).lower()
            if exc.status == 400 and ("subdomain" in payload_text or "unique" in payload_text):
                existing_slugs.add(slug)
                continue
            message = (
                f"Failed to create Mercury app at '{target_url}' "
                f"with title '{title}': {exc}"
            )
            if exc.status == 400 and str(exc) == "http_400":
                message += (
                    " The platform rejected the app creation request. "
                    "This can happen because of account limits, permissions, "
                    "or invalid app settings."
                )
            raise AutoMLException(message)
    raise AutoMLException("Failed to generate a unique Mercury slug for the new app.")


def _upload_workspace_files(client, site, output_path, verbose=False):
    for filename, body in _read_publishable_files(output_path):
        try:
            if verbose:
                print(f"Uploading file: {filename}")
            upload_url = client.presign_upload(site["id"], filename, len(body))
            client.upload_bytes(upload_url, body)
            client.register_uploaded_file(site["id"], filename, len(body))
        except PlatformApiError as exc:
            raise AutoMLException(
                f"Failed to upload '{filename}' for site '{site.get('id')}': {exc}"
            )


def publish_app_from_automl(
    automl,
    url=None,
    path=None,
    overwrite=False,
    title=None,
    open_browser=True,
    timeout=LOGIN_TIMEOUT_SECONDS,
    verbose=True,
):
    try:
        previous_url = _load_publish_state(automl).get("last_published_url")
        if verbose:
            print("Start app publish")
            if previous_url:
                print(f"Last published app URL: {previous_url}")
        overwrite_output = overwrite or path is None
        if verbose:
            print("Creating app workspace")
        output_path = generate_app(
            automl, path=path, overwrite=overwrite_output, title=title, verbose=False
        )
        if verbose:
            print("Signing in to MLJAR platform")
        session = BrowserTokenSession(
            base_url=DEFAULT_PLATFORM_BASE_URL,
            open_browser=open_browser,
            timeout=timeout,
        )
        token = session.authenticate()
        client = PlatformClient(base_url=DEFAULT_PLATFORM_BASE_URL, token=token)

        target_url = url or previous_url
        if target_url:
            if verbose:
                print(f"Checking existing app URL: {target_url}")
            site = client.find_site_by_url(target_url)
            if site is None:
                if url:
                    raise AutoMLException(f"Could not find Mercury app for URL '{url}'.")
                site_title = _workspace_title(output_path, fallback_title=title)
                if verbose:
                    print(
                        f"Last published app URL not found on platform: {target_url}"
                    )
                    print("Creating app URL")
                site = _create_site_with_generated_slug(client, site_title)
                if verbose:
                    print(f"Created app URL: {_site_url(site)}")
        else:
            site_title = _workspace_title(output_path, fallback_title=title)
            if verbose:
                print("Creating app URL")
            site = _create_site_with_generated_slug(client, site_title)
            if verbose:
                print(f"Created app URL: {_site_url(site)}")

        if verbose and target_url and site is not None:
            print(f"Using app URL: {_site_url(site)}")

        _upload_workspace_files(client, site, output_path, verbose=verbose)
        app_url = _site_url(site)
        _save_publish_state(automl, app_url)
        if verbose:
            print(f"Finished. You can access your app at: {app_url}")
        return app_url
    except AutoMLException as exc:
        print(f"Publish app failed: {exc}")
        return None
