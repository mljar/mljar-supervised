from html.parser import HTMLParser

import pytest
from IPython.display import HTML, IFrame

from supervised import AutoML


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.classes = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if "class" in attrs:
            self.classes.append(attrs["class"].split())
        if tag == "a":
            self.links.append(attrs)


@pytest.fixture
def saved_report(tmp_path, monkeypatch):
    (tmp_path / "params.json").write_text("{}")
    (tmp_path / "README.md").write_text(
        "# AutoML Leaderboard\n\n[Model](1_Baseline/README.md)\n\n"
        "[Documentation](https://mljar.com/)\n"
    )
    for name in ["1_Baseline", "optuna"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "README.md").write_text(
            f"# {name}\n\n[Back](../README.md)\n"
        )
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    return AutoML(results_path=str(tmp_path)), tmp_path / "README.html"


def test_report_navigation_and_wrapper(saved_report):
    model, path = saved_report
    report = model.report()
    assert isinstance(report, HTML)
    assert report.data == path.read_text()
    parser = ReportParser()
    parser.feed(report.data)
    assert any("mljar-automl-report" in classes for classes in parser.classes)
    for link in parser.links:
        if "onclick" in link:
            # Each toggle must refer to a section in this same report.
            targets = link["onclick"].split("toggleShow('")[1:]
            assert targets
            assert all(target.split("'")[0] in parser.ids for target in targets)
    assert any(link.get("href") == "https://mljar.com/" for link in parser.links)


@pytest.mark.parametrize("cached_version", [None, 1])
def test_older_cached_report_rebuilt_without_training(
    saved_report, monkeypatch, cached_version
):
    model, path = saved_report
    version = (
        ""
        if cached_version is None
        else f'<meta name="mljar-report-version" content="{cached_version}">'
    )
    path.write_text(f"<html><head>{version}</head><body>Old cache</body></html>")
    markdown_paths = list(path.parent.rglob("*.md"))
    original_markdown = {p: p.read_bytes() for p in markdown_paths}

    def no_training(*args, **kwargs):
        pytest.fail("Report migration must not train models")

    monkeypatch.setattr(model, "fit", no_training)
    model.report()
    assert "Old cache" not in path.read_text()
    assert {p: p.read_bytes() for p in markdown_paths} == original_markdown
    # A current cache is reused without reconverting any Markdown.
    monkeypatch.setattr(model, "_md_to_html", no_training)
    assert model.report().data == path.read_text()


def test_kaggle_uses_upgraded_cached_report(saved_report, monkeypatch):
    model, path = saved_report
    path.write_text("<html><body>Old cache</body></html>")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    report = model.report(width=700, height=800)
    assert isinstance(report, IFrame)
    assert report.src == str(path)
    assert report.width == 700
    assert report.height == 800
    assert "Old cache" not in path.read_text()
