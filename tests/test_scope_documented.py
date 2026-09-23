"""Guards that the AMD-only, nightly-only scope stays explicitly documented.

The previous incarnation of this dashboard narrowed to AMD nightlies through
four stacked filters and said so nowhere, so a reader had to reverse-engineer
the scope from the code. These tests fail if that documentation is removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import perf_eval

ROOT = Path(perf_eval.__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "perf_eval"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestReadme:
    @pytest.fixture
    def readme(self):
        return read(ROOT / "README.md").lower()

    def test_states_what_is_covered(self, readme):
        assert "covers" in readme

    def test_names_amd_as_the_hardware_scope(self, readme):
        assert "amd" in readme and "mi" in readme

    def test_names_nightly_as_the_run_scope(self, readme):
        assert "nightly" in readme

    def test_names_the_excluded_nvidia_hardware(self, readme):
        assert "h200" in readme or "nvidia" in readme

    def test_says_adhoc_builds_are_excluded(self, readme):
        assert "ad-hoc" in readme or "adhoc" in readme

    def test_explains_why_nightly_only(self, readme):
        # The reason matters: an ad-hoc run may cover one workload at one
        # concurrency, which would distort a trend line.
        assert "concurrency" in readme


class TestModuleDocstrings:
    @pytest.mark.parametrize("module", ["collect_artifacts.py", "aggregate.py", "normalize.py"])
    def test_docstring_declares_the_scope(self, module):
        source = read(SCRIPTS / module)
        docstring = source.split('"""')[1].lower()
        assert "scope" in docstring, module
        assert "amd" in docstring, module

    @pytest.mark.parametrize("module", ["collect_artifacts.py", "aggregate.py"])
    def test_docstring_declares_nightly_only(self, module):
        docstring = read(SCRIPTS / module).split('"""')[1].lower()
        assert "nightly" in docstring, module

    def test_package_docstring_declares_the_scope(self):
        docstring = read(SCRIPTS / "__init__.py").split('"""')[1].lower()
        assert "amd" in docstring and "nightly" in docstring


class TestNamedPredicates:
    def test_the_amd_filter_is_a_named_predicate(self):
        assert callable(__import__("perf_eval.normalize", fromlist=["x"]).is_amd_workload)

    def test_the_nightly_filter_is_a_named_predicate(self):
        module = __import__("perf_eval.collect_artifacts", fromlist=["x"])
        assert callable(module.is_nightly_build)

    def test_the_aggregator_reapplies_the_scope_through_one_predicate(self):
        module = __import__("perf_eval.aggregate", fromlist=["x"])
        assert callable(module._is_in_scope)


class TestPublishedPayload:
    def test_the_scope_travels_with_the_data(self):
        from conftest import perf_result
        from perf_eval import aggregate as agg

        scope = agg.aggregate([perf_result()])["scope"]
        assert scope["hardware"] == "amd"
        assert scope["runs"] == "nightly"
        assert scope["description"]


class TestDashboardUi:
    @pytest.fixture
    def html(self):
        return read(ROOT / "site" / "index.html")

    def test_the_page_states_the_scope_in_the_header(self, html):
        assert "AMD nightly runs only" in html

    def test_the_page_names_the_excluded_hardware(self, html):
        assert "H200" in html or "NVIDIA" in html

    def test_the_page_says_adhoc_builds_are_excluded(self, html):
        assert "ad-hoc" in html

    def test_the_title_carries_the_scope(self, html):
        assert "AMD nightly" in html.split("<title>")[1].split("</title>")[0]
