"""Guards on the vendored third-party assets.

Vendoring trades a CDN's guarantees for a committed file, so the digest has to
be asserted here instead: a silently swapped, truncated or partially
downloaded bundle should fail CI rather than break the charts in production.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

import perf_eval

ROOT = Path(perf_eval.__file__).resolve().parents[2]
VENDOR = ROOT / "site" / "vendor"
INDEX = ROOT / "site" / "index.html"

# Digest published by cdnjs for
# https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js
# Keep in sync with site/vendor/README.md.
CHART_JS = "chart.umd.min.js"
CHART_JS_VERSION = "4.4.1"
CHART_JS_SRI = "sha384-bs/nf9FbdNouRbMiFcrcZfLXYPKiPaGVGplVbv7dLGECccEXDW+S3zjqSKR5ZEaD"


def sri_digest(path: Path) -> str:
    digest = hashlib.sha384(path.read_bytes()).digest()
    return "sha384-" + base64.b64encode(digest).decode("ascii")


class TestChartJs:
    def test_is_present(self):
        assert (VENDOR / CHART_JS).is_file()

    def test_matches_the_upstream_integrity_digest(self):
        # Equal to the cdnjs SRI hash means vendoring changed where the file
        # comes from, not what it is.
        assert sri_digest(VENDOR / CHART_JS) == CHART_JS_SRI

    def test_is_the_expected_version(self):
        # The digest above already pins the exact bytes. This catches the
        # narrower mistake of updating the file and its digest together while
        # leaving the documented version number stale. The minified bundle has
        # no banner comment, so the version lives inline.
        source = (VENDOR / CHART_JS).read_text(encoding="utf-8", errors="replace")
        assert CHART_JS_VERSION in source

    def test_is_a_umd_bundle_exporting_chart(self):
        head = (VENDOR / CHART_JS).read_text(encoding="utf-8", errors="replace")[:600]
        assert "module.exports" in head
        assert ".Chart=" in head

    def test_is_plausibly_complete(self):
        # A truncated download would still satisfy a naive existence check.
        assert (VENDOR / CHART_JS).stat().st_size > 150_000


class TestMitAttribution:
    """MIT requires the notice to travel with every copy we distribute.

    Loading from a CDN distributed nothing; committing the file and publishing
    it to gh-pages makes this project a redistributor, which is what attaches
    the obligation. The minified bundle carries no banner of its own, so the
    notice sits beside it and these tests keep it there.
    """

    LICENSE_FILE = CHART_JS + ".LICENSE.txt"
    COPYRIGHT = "Copyright (c) 2014-2022 Chart.js Contributors"

    @pytest.fixture
    def notice(self):
        return (VENDOR / self.LICENSE_FILE).read_text(encoding="utf-8")

    def test_the_bundle_itself_carries_no_notice(self):
        # If a future version does embed a banner this test fails, which is a
        # prompt to reconsider the separate file rather than a real problem.
        source = (VENDOR / CHART_JS).read_text(encoding="utf-8", errors="replace")
        assert "@license" not in source
        assert "Copyright" not in source

    def test_notice_file_exists(self):
        assert (VENDOR / self.LICENSE_FILE).is_file()

    def test_includes_the_copyright_line(self, notice):
        assert self.COPYRIGHT in notice

    def test_includes_the_permission_notice_verbatim(self, notice):
        # This sentence is the actual obligation; paraphrasing it would not
        # satisfy the licence.
        assert (
            "The above copyright notice and this permission notice shall be "
            "included in all copies or substantial portions of the Software."
        ) in notice

    def test_includes_the_grant_and_the_warranty_disclaimer(self, notice):
        assert "Permission is hereby granted, free of charge" in notice
        assert 'THE SOFTWARE IS PROVIDED "AS IS"' in notice
        assert "The MIT License (MIT)" in notice

    def test_identifies_what_it_covers(self, notice):
        assert CHART_JS in notice
        assert CHART_JS_VERSION in notice

    def test_third_party_notices_records_the_dependency(self):
        text = (ROOT / "THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8")
        assert "Chart.js" in text
        assert self.COPYRIGHT in text
        assert "MIT" in text
        assert self.LICENSE_FILE in text

    def test_the_notice_is_published_with_the_site(self, tmp_path):
        # The obligation attaches to the copy we serve, so the built site has
        # to carry the notice, not just the repository.
        import build_site

        payload = tmp_path / "perf_eval.json"
        payload.write_text('{"models": [], "summary": {}}', encoding="utf-8")
        out = build_site.build(build_site.DEFAULT_SITE, payload, tmp_path / "_site")
        published = out / "vendor" / self.LICENSE_FILE
        assert published.is_file()
        assert self.COPYRIGHT in published.read_text(encoding="utf-8")
        # And the bundle it covers is published too.
        assert (out / "vendor" / CHART_JS).is_file()


class TestProjectLicence:
    """This project's own licence, distinct from the bundled dependency's."""

    COPYRIGHT = "Copyright © Advanced Micro Devices, Inc. All rights reserved."

    @pytest.fixture
    def licence(self):
        return (ROOT / "LICENSE").read_text(encoding="utf-8")

    def test_licence_file_exists(self, licence):
        assert licence.strip()

    def test_is_mit_with_the_amd_copyright(self, licence):
        assert self.COPYRIGHT in licence
        assert "MIT License" in licence

    def test_carries_the_full_mit_terms(self, licence):
        assert "Permission is hereby granted, free of charge" in licence
        assert (
            "The above copyright notice and this permission notice shall be included in all"
        ) in licence
        assert 'THE SOFTWARE IS PROVIDED "AS IS"' in licence

    def test_is_declared_in_package_metadata(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'license = "MIT"' in text
        assert 'license-files = ["LICENSE"]' in text

    def test_readme_points_at_it(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "MIT" in readme
        assert "THIRD-PARTY-NOTICES.md" in readme

    def test_no_bundled_asset_is_more_restrictive_than_mit(self):
        # Every notice under site/vendor/ must be MIT. A copyleft dependency
        # would conflict with this project's licence and needs a decision, not
        # a silent commit.
        notices = list(VENDOR.glob("*.LICENSE.txt"))
        assert notices, "expected at least one bundled licence notice"
        for path in notices:
            assert "The MIT License (MIT)" in path.read_text(encoding="utf-8"), path.name


class TestProvenanceIsRecorded:
    @pytest.fixture
    def vendor_readme(self):
        return (VENDOR / "README.md").read_text(encoding="utf-8")

    def test_vendor_readme_exists(self, vendor_readme):
        assert vendor_readme.strip()

    def test_records_the_digest(self, vendor_readme):
        assert CHART_JS_SRI in vendor_readme

    def test_records_the_upstream_url_and_licence(self, vendor_readme):
        assert "cdnjs.cloudflare.com" in vendor_readme
        assert "MIT" in vendor_readme

    def test_project_readme_explains_the_choice(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "vendored" in readme.lower()
        assert CHART_JS_SRI in readme


class TestAmdLogo:
    """The favicon and header logo: Buildkite's ``:amd:`` emoji image."""

    LOGO = "amd.png"
    # Keep in sync with site/vendor/README.md and THIRD-PARTY-NOTICES.md.
    SHA256 = "5da4382060eb39e1f5e9bf9d0f3c3c6b5ac7ebbef3e578f7a9f5bf3cbb1fe7e6"

    @pytest.fixture
    def html(self):
        return INDEX.read_text(encoding="utf-8")

    def test_matches_the_upstream_bytes(self):
        assert hashlib.sha256((VENDOR / self.LOGO).read_bytes()).hexdigest() == self.SHA256

    def test_is_a_png(self):
        assert (VENDOR / self.LOGO).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_is_the_favicon(self, html):
        assert f'<link rel="icon" type="image/png" href="vendor/{self.LOGO}"' in html

    def test_is_in_the_header(self, html):
        header = html.split("<h1>")[1].split("</h1>")[0]
        assert f'src="vendor/{self.LOGO}"' in header

    def test_is_not_hotlinked(self, html):
        assert "buildkiteassets.com" not in html
        assert "raw.githubusercontent.com/buildkite" not in html

    def test_provenance_is_recorded(self):
        for doc in (VENDOR / "README.md", ROOT / "THIRD-PARTY-NOTICES.md"):
            text = doc.read_text(encoding="utf-8")
            assert self.SHA256 in text, doc.name
            assert "buildkite/emojis" in text, doc.name

    def test_carries_no_mit_notice(self):
        # A trademark, not MIT software: a .LICENSE.txt here would misstate its
        # terms and trip the "every notice is MIT" check above.
        assert not (VENDOR / (self.LOGO + ".LICENSE.txt")).exists()


class TestPageLoadsLocally:
    @pytest.fixture
    def html(self):
        return INDEX.read_text(encoding="utf-8")

    def test_chart_js_is_loaded_from_vendor(self, html):
        assert f'src="vendor/{CHART_JS}"' in html

    def test_no_cdn_script_or_style_remains(self, html):
        for host in ("cdnjs.cloudflare.com", "cdn.jsdelivr.net", "unpkg.com"):
            assert f'src="https://{host}' not in html, host
            assert f'href="https://{host}' not in html, host

    def test_nothing_is_fetched_from_a_remote_origin(self, html):
        # Only same-origin fetches; anything else would break an offline load.
        assert "fetch('http" not in html
        assert 'fetch("http' not in html
