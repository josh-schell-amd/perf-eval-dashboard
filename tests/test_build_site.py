"""Tests for assembling the deployable site."""

from __future__ import annotations

import json

import pytest

import build_site


@pytest.fixture
def site(tmp_path):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "index.html").write_text(
        "<html><script>fetch('perf_eval.json').then(r=>r.json());</script></html>",
        encoding="utf-8",
    )
    (site_dir / "extra.css").write_text("body{}", encoding="utf-8")
    return site_dir


@pytest.fixture
def payload(tmp_path):
    path = tmp_path / "data" / "perf_eval.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"models": [], "summary": {}}), encoding="utf-8")
    return path


class TestBuild:
    def test_copies_the_site_and_the_payload(self, site, payload, tmp_path):
        out = build_site.build(site, payload, tmp_path / "_site")
        assert (out / "index.html").is_file()
        assert (out / "extra.css").is_file()
        assert json.loads((out / "perf_eval.json").read_text(encoding="utf-8")) == {
            "models": [],
            "summary": {},
        }

    def test_cache_busts_the_fetch(self, site, payload, tmp_path):
        out = build_site.build(site, payload, tmp_path / "_site")
        html = (out / "index.html").read_text(encoding="utf-8")
        assert "fetch('perf_eval.json?v=" in html

    def test_the_cache_tag_tracks_the_payload_contents(self, site, payload, tmp_path):
        first = build_site.build(site, payload, tmp_path / "a")
        tag_one = (first / "index.html").read_text(encoding="utf-8").split("?v=")[1][:12]
        payload.write_text(json.dumps({"models": [{"model": "x"}]}), encoding="utf-8")
        second = build_site.build(site, payload, tmp_path / "b")
        tag_two = (second / "index.html").read_text(encoding="utf-8").split("?v=")[1][:12]
        assert tag_one != tag_two

    def test_rebuilding_is_idempotent(self, site, payload, tmp_path):
        out = tmp_path / "_site"
        build_site.build(site, payload, out)
        stale = out / "stale.txt"
        stale.write_text("old", encoding="utf-8")
        build_site.build(site, payload, out)
        assert not stale.exists()

    def test_the_event_store_is_never_published(self, site, payload, tmp_path):
        (payload.parent / "events.jsonl").write_text('{"event":"build"}\n', encoding="utf-8")
        out = build_site.build(site, payload, tmp_path / "_site")
        assert not (out / "events.jsonl").exists()
        assert [p.name for p in out.iterdir() if p.name.endswith(".jsonl")] == []


class TestFailureModes:
    def test_missing_payload_raises(self, site, tmp_path):
        with pytest.raises(FileNotFoundError, match="published payload not found"):
            build_site.build(site, tmp_path / "absent.json", tmp_path / "_site")

    def test_missing_site_raises(self, payload, tmp_path):
        with pytest.raises(FileNotFoundError, match="site source directory not found"):
            build_site.build(tmp_path / "absent", payload, tmp_path / "_site")

    def test_missing_index_raises(self, payload, tmp_path):
        empty = tmp_path / "empty-site"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="site entrypoint not found"):
            build_site.build(empty, payload, tmp_path / "_site")

    def test_invalid_payload_json_raises_before_publishing(self, site, tmp_path):
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        out = tmp_path / "_site"
        with pytest.raises(ValueError, match="not valid JSON"):
            build_site.build(site, broken, out)
        assert not out.exists()

    def test_a_failed_build_leaves_the_previous_output(self, site, payload, tmp_path):
        out = build_site.build(site, payload, tmp_path / "_site")
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            build_site.build(site, broken, out)
        assert (out / "perf_eval.json").is_file()

    @pytest.mark.parametrize(
        "html",
        [
            "<html>no data load</html>",
            "<script>fetch('perf_eval.json');fetch('perf_eval.json');</script>",
        ],
    )
    def test_the_page_must_fetch_the_payload_exactly_once(self, payload, tmp_path, html):
        site_dir = tmp_path / "site"
        site_dir.mkdir()
        (site_dir / "index.html").write_text(html, encoding="utf-8")
        with pytest.raises(RuntimeError, match="exactly one"):
            build_site.build(site_dir, payload, tmp_path / "_site")


class TestRealSite:
    def test_the_checked_in_dashboard_builds(self, tmp_path):
        payload = tmp_path / "perf_eval.json"
        payload.write_text(
            json.dumps({"generated_at": "2026-01-01T00:00:00Z", "models": [], "summary": {}}),
            encoding="utf-8",
        )
        out = build_site.build(build_site.DEFAULT_SOURCE, payload, tmp_path / "_site")
        html = (out / "index.html").read_text(encoding="utf-8")
        assert "fetch('perf_eval.json?v=" in html
