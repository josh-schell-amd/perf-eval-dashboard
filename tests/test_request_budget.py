"""Tests pinning how many Buildkite requests a collection run costs.

The point is that the cost is predictable and bounded before anyone hands the
collector a real token. These tests use a fake Buildkite so the arithmetic can
be asserted exactly rather than estimated.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from conftest import STARTED
from perf_eval import collect_artifacts as ca
from perf_eval import store as store_mod

# One filtered artifact listing per result path filter, per build.
LISTINGS_PER_BUILD = len(ca._RESULT_ARTIFACT_PATHS)

# The recipe's runs, one per artifact FakeBuildkite serves (bench-cfg<i>.json).
CONFIGS = {f"cfg{i}": {"isl": 1024, "osl": 1024, "conc": 2**i} for i in range(8)}


def nightly_build(number: int) -> dict:
    # A distinct vLLM commit per build. Nightly identity keys on the commit, so
    # reusing one would make the store correctly fold every build into a single
    # nightly and the request accounting would measure the wrong thing.
    commit = f"{number:040x}"
    finished = STARTED - timedelta(hours=6 * (40 - (number - 1000)))
    day = finished.strftime("%Y-%m-%d")
    return {
        "number": number,
        "branch": "main",
        "message": f"Nightly run {day}: commit {commit}",
        "state": "finished",
        "source": "schedule",
        "web_url": f"https://buildkite.com/vllm/perf-eval/builds/{number}",
        "commit": "f" * 40,
        "created_at": (finished - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": {},
    }


class FakeBuildkite:
    """Minimal stand-in that records every call the collector makes."""

    def __init__(self, builds, artifacts_per_build=2):
        self.builds = builds
        self.artifacts_per_build = artifacts_per_build
        self.build_list_calls = 0
        self.artifact_list_calls = []
        self.downloads = []

    def paginate(self, path, token, params=None, max_pages=10, budget=None):
        params = params or {}
        if path.endswith("/builds"):
            self.build_list_calls += 1
            if budget:
                budget.charge("listing")
            return self.builds
        # Artifact listing, one call per path filter.
        self.artifact_list_calls.append((path, params.get("path")))
        if budget:
            budget.charge("listing")
        if "bench-" not in str(params.get("path")):
            return []
        number = path.split("/builds/")[1].split("/")[0]
        return [
            {
                "id": f"artifact-{number}-{index}",
                "job_id": "job",
                "path": f"results/test_8b-mi355x/bench-cfg{index}.json",
                "sha1sum": f"{index:040x}",
                "download_url": f"https://example.invalid/{number}/{index}",
            }
            for index in range(self.artifacts_per_build)
        ]

    def download(self, url, token, budget=None, label=""):
        self.downloads.append(url)
        if budget:
            budget.charge("download")
        return {
            "model_id": "org/Model",
            "total_token_throughput": 800.0,
            "output_throughput": 200.0,
            "max_concurrency": 128,
        }


@pytest.fixture
def fake(monkeypatch):
    def install(builds, artifacts_per_build=2):
        bk = FakeBuildkite(builds, artifacts_per_build)
        monkeypatch.setattr(ca, "_bk_paginate", bk.paginate)
        monkeypatch.setattr(ca, "_bk_download_json", bk.download)
        monkeypatch.setattr(
            ca,
            "fetch_workload_map",
            lambda _token, ref: {
                "test_8b-mi355x": (
                    {
                        "name": "test_8b-mi355x",
                        "device": "mi355x",
                        "parallelism": {"tensor_parallel_size": 4},
                        "precision": "fp8",
                        "model": "org/Model",
                    },
                    CONFIGS,
                )
            },
        )
        return bk

    return install


class TestRequestAccounting:
    def test_cold_start_cost_is_one_listing_plus_one_per_filter_per_build(self, fake, tmp_path):
        builds = [nightly_build(1000 + i) for i in range(5)]
        bk = fake(builds, artifacts_per_build=2)
        budget = ca.RequestBudget()
        ca.collect(tmp_path / "events.jsonl", days=14, bk_token="t", gh_token="", budget=budget)
        # 1 builds listing + one artifact listing per path filter per nightly.
        assert budget.listings == 1 + LISTINGS_PER_BUILD * len(builds)
        # One download per discovered artifact, first time through.
        assert budget.downloads == len(builds) * 2
        assert bk.build_list_calls == 1

    def test_second_run_skips_already_ingested_builds(self, fake, tmp_path):
        builds = [nightly_build(1000 + i) for i in range(8)]
        store = tmp_path / "events.jsonl"
        fake(builds, artifacts_per_build=2)

        first = ca.RequestBudget()
        ca.collect(store, days=14, bk_token="t", gh_token="", budget=first)

        second = ca.RequestBudget()
        ca.collect(store, days=14, bk_token="t", gh_token="", budget=second)

        # Steady state re-lists only the re-check window, not all eight.
        assert second.skipped_builds == 8 - ca.DEFAULT_RECHECK_BUILDS
        assert second.listings == 1 + LISTINGS_PER_BUILD * ca.DEFAULT_RECHECK_BUILDS
        # And downloads nothing, because every artifact is already known.
        assert second.downloads == 0
        assert second.total < first.total

    def test_the_recheck_window_still_catches_a_retried_job(self, fake, tmp_path):
        builds = [nightly_build(1000 + i) for i in range(5)]
        store = tmp_path / "events.jsonl"
        bk = fake(builds, artifacts_per_build=1)
        ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())

        # A retried job adds a second artifact to the newest build.
        bk.artifacts_per_build = 2
        budget = ca.RequestBudget()
        appended = ca.collect(store, days=14, bk_token="t", gh_token="", budget=budget)
        # The newest builds are re-listed, so the new artifact is picked up.
        assert appended > 0
        assert budget.downloads > 0

    def test_recheck_zero_skips_every_ingested_build(self, fake, tmp_path):
        builds = [nightly_build(1000 + i) for i in range(4)]
        store = tmp_path / "events.jsonl"
        fake(builds)
        ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())

        budget = ca.RequestBudget()
        ca.collect(store, days=14, bk_token="t", gh_token="", recheck_builds=0, budget=budget)
        assert budget.skipped_builds == 4
        assert budget.listings == 1  # only the builds listing


class TestDryRun:
    def test_downloads_nothing_and_writes_nothing(self, fake, tmp_path):
        builds = [nightly_build(1000 + i) for i in range(4)]
        store = tmp_path / "events.jsonl"
        bk = fake(builds, artifacts_per_build=3)
        budget = ca.RequestBudget()
        ca.collect(store, days=14, bk_token="t", gh_token="", dry_run=True, budget=budget)
        assert bk.downloads == []
        assert budget.downloads == 0
        assert not store.exists()

    def test_still_lists_so_the_cost_is_real(self, fake, tmp_path):
        builds = [nightly_build(1000 + i) for i in range(4)]
        fake(builds)
        budget = ca.RequestBudget()
        ca.collect(
            tmp_path / "events.jsonl",
            days=14,
            bk_token="t",
            gh_token="",
            dry_run=True,
            budget=budget,
        )
        assert budget.listings == 1 + LISTINGS_PER_BUILD * 4


class TestCeiling:
    def test_exceeding_the_ceiling_aborts(self, fake, tmp_path):
        builds = [nightly_build(1000 + i) for i in range(20)]
        fake(builds, artifacts_per_build=5)
        with pytest.raises(RuntimeError, match="request ceiling exceeded"):
            ca.collect(
                tmp_path / "events.jsonl",
                days=14,
                bk_token="t",
                gh_token="",
                budget=ca.RequestBudget(max_requests=10),
            )

    def test_the_ceiling_message_says_how_to_proceed(self, fake, tmp_path):
        fake([nightly_build(1000)], artifacts_per_build=5)
        with pytest.raises(RuntimeError) as excinfo:
            ca.collect(
                tmp_path / "events.jsonl",
                days=14,
                bk_token="t",
                gh_token="",
                budget=ca.RequestBudget(max_requests=1),
            )
        assert "--max-requests" in str(excinfo.value)

    def test_a_partial_run_never_writes_a_truncated_store(self, fake, tmp_path):
        store = tmp_path / "events.jsonl"
        fake([nightly_build(1000 + i) for i in range(10)], artifacts_per_build=5)
        with pytest.raises(RuntimeError):
            ca.collect(
                store,
                days=14,
                bk_token="t",
                gh_token="",
                budget=ca.RequestBudget(max_requests=5),
            )
        # The store is only written after the whole scan completes.
        assert not store.exists()

    def test_ceiling_can_be_disabled(self):
        budget = ca.RequestBudget(max_requests=None)
        for _ in range(5000):
            budget.charge("listing")
        assert budget.total == 5000

    def test_default_ceiling_is_documented_and_generous(self):
        assert ca.DEFAULT_MAX_REQUESTS >= 1000
        assert ca.RequestBudget().max_requests == ca.DEFAULT_MAX_REQUESTS


class TestRetriesAreCharged:
    def test_every_http_attempt_counts(self, monkeypatch):
        # A retry is a real request Buildkite sees, so it must be charged or
        # the reported total would understate what we did to them.
        attempts = {"n": 0}

        class Response:
            # 503 rather than 500: only gateway-ish codes are retried, since a
            # 500 is usually persistent and retrying it just adds load.
            status_code = 503
            headers: dict = {}

            def raise_for_status(self):
                raise RuntimeError("boom")

            def json(self):
                return {}

        def fake_get(*_args, **_kwargs):
            attempts["n"] += 1
            return Response()

        monkeypatch.setattr(ca.requests, "get", fake_get)
        monkeypatch.setattr(ca.time, "sleep", lambda _s: None)
        budget = ca.RequestBudget()
        with pytest.raises(RuntimeError):
            ca._bk_get("/x", "token", budget=budget)
        assert budget.listings == attempts["n"] == ca.BK_GET_MAX_ATTEMPTS


class _DownloadResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body
        self.headers: dict = {}

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class TestDownloads:
    SIGNED = "https://bucket.s3.amazonaws.com/a.json?X-Amz-Signature=secret"

    def _serve(self, monkeypatch, responses):
        calls = {"n": 0}

        def fake_get(*_args, **_kwargs):
            response = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            if isinstance(response, Exception):
                raise response
            return response

        monkeypatch.setattr(ca.requests, "get", fake_get)
        monkeypatch.setattr(ca.time, "sleep", lambda _s: None)
        return calls

    def test_a_transient_failure_is_retried(self, monkeypatch):
        calls = self._serve(monkeypatch, [_DownloadResponse(503), _DownloadResponse(200, {"a": 1})])
        budget = ca.RequestBudget()
        assert ca._bk_download_json(self.SIGNED, "t", budget=budget) == {"a": 1}
        assert calls["n"] == budget.downloads == 2

    def test_exhausted_retries_fail_the_run(self, monkeypatch):
        # Returning None would lose the artifact for good: a build that has
        # other results is never listed again.
        self._serve(monkeypatch, [_DownloadResponse(429)])
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            ca._bk_download_json(self.SIGNED, "t", label="#1 results/wl/bench-a.json")

    def test_a_connection_error_is_retried_then_fails(self, monkeypatch):
        self._serve(monkeypatch, [ca.requests.exceptions.ConnectionError(self.SIGNED)])
        with pytest.raises(RuntimeError) as excinfo:
            ca._bk_download_json(self.SIGNED, "t")
        assert "Signature" not in str(excinfo.value)

    @pytest.mark.parametrize("response", [_DownloadResponse(404), _DownloadResponse(200)])
    def test_a_permanent_failure_is_skipped(self, monkeypatch, response):
        self._serve(monkeypatch, [response])
        assert ca._bk_download_json(self.SIGNED, "t") is None

    def test_the_signed_url_is_never_logged(self, monkeypatch, caplog):
        self._serve(monkeypatch, [_DownloadResponse(503), _DownloadResponse(404)])
        with caplog.at_level("WARNING"):
            ca._bk_download_json(self.SIGNED, "t", label="#1 results/wl/bench-a.json")
        assert "results/wl/bench-a.json" in caplog.text
        assert "Signature" not in caplog.text and "amazonaws" not in caplog.text


class TestBudgetSummary:
    def test_reports_listings_and_downloads_separately(self):
        budget = ca.RequestBudget()
        budget.charge("listing")
        budget.charge("download")
        budget.charge("download")
        assert budget.total == 3
        summary = budget.summary()
        assert "3 Buildkite requests" in summary
        assert "1 listings" in summary
        assert "2 downloads" in summary


class TestBoundedByConstruction:
    def test_artifact_listing_pages_are_capped(self):
        assert ca._RESULT_ARTIFACT_MAX_PAGES == 10

    def test_lookback_is_capped(self, tmp_path):
        with pytest.raises(ValueError, match="between 1 and"):
            ca.collect(tmp_path / "e.jsonl", days=31, bk_token="t", gh_token="")

    def test_retries_are_bounded(self):
        assert ca.BK_GET_MAX_ATTEMPTS == 3

    def test_pagination_refuses_to_loop_forever(self):
        with pytest.raises(ValueError, match="max_pages must be positive"):
            ca._bk_paginate("/x", "token", max_pages=0)

    def test_only_result_artifacts_are_listed(self):
        # Narrow path filters keep the pipeline's large sample/log tree out of
        # the listing entirely.
        assert ca._RESULT_ARTIFACT_PATHS == (
            "*results/*/bench-*.json",
            "*results/*/*/results_*.json",
            "*results/*/*/*/results_*.json",
        )


def test_each_build_is_labelled_from_the_recipes_it_ran(fake, monkeypatch, tmp_path):
    # The recipe moved from TP=2 to TP=4 after build 1000 ran.
    old, new = nightly_build(1000), nightly_build(1001)
    old["commit"], new["commit"] = "1" * 40, "2" * 40
    fake([new, old], artifacts_per_build=1)

    def recipes(_token, ref):
        entry = {
            "name": "test_8b-mi355x",
            "device": "mi355x",
            "parallelism": {"tensor_parallel_size": 2 if ref == old["commit"] else 4},
            "precision": "fp8",
            "model": "org/Model",
        }
        return {"test_8b-mi355x": (entry, CONFIGS)}

    monkeypatch.setattr(ca, "fetch_workload_map", recipes)
    store = tmp_path / "events.jsonl"
    ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())

    results = {
        e["build_number"]: e
        for e in store_mod.read_events_strict(store)
        if e["event"] == "perf_result"
    }
    assert [results[n]["parallelism"] for n in (1000, 1001)] == [
        {"tensor_parallel_size": 2},
        {"tensor_parallel_size": 4},
    ]
    # 800 tok/s in total, over the GPUs each build actually used.
    assert results[1000]["metrics"]["tput_per_gpu"] == 400.0
    assert results[1001]["metrics"]["tput_per_gpu"] == 200.0


def test_coverage_expects_what_the_latest_nightly_was_asked_to_run(fake, monkeypatch, tmp_path):
    # Build 1000 ran TP=2 and build 1001 ran TP=4.
    old, new = nightly_build(1000), nightly_build(1001)
    old["commit"], new["commit"] = "1" * 40, "2" * 40
    fake([new, old], artifacts_per_build=1)
    tp_at = {old["commit"]: 2, new["commit"]: 4}

    def recipes(_token, ref):
        entry = {
            "name": "test_8b-mi355x",
            "device": "mi355x",
            "parallelism": {"tensor_parallel_size": tp_at[ref]},
            "precision": "fp8",
            "model": "org/Model",
            "nightly": True,
        }
        return {"test_8b-mi355x": (entry, {"cfg0": CONFIGS["cfg0"]})}

    monkeypatch.setattr(ca, "fetch_workload_map", recipes)
    store = tmp_path / "events.jsonl"
    ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())

    (snapshot,) = [
        e for e in store_mod.read_events_strict(store) if e["event"] == ca.EXPECTED_CONFIGS_EVENT
    ]
    assert [config["parallelism"] for config in snapshot["configs"]] == [
        {"tensor_parallel_size": 4}
    ]
    assert snapshot["recipe_commit"] == new["commit"]


def test_recipes_are_not_refetched_when_nothing_new_ran(fake, monkeypatch, tmp_path):
    # Recipes at a commit never change, so an ingested build and a snapshot
    # already taken for the latest nightly's commit need no GitHub requests.
    fake([nightly_build(1000 + i) for i in range(3)])
    fetched = []

    def recipes(_token, ref):
        fetched.append(ref)
        entry = {
            "name": "test_8b-mi355x",
            "device": "mi355x",
            "parallelism": {"tensor_parallel_size": 4},
            "nightly": True,
        }
        return {"test_8b-mi355x": (entry, CONFIGS)}

    monkeypatch.setattr(ca, "fetch_workload_map", recipes)
    store = tmp_path / "events.jsonl"
    ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())
    assert fetched == ["f" * 40]

    fetched.clear()
    ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())
    assert fetched == []


def test_a_snapshot_predating_accuracy_is_refetched_once(fake, monkeypatch, tmp_path):
    # Otherwise accuracy coverage would stay blind until an unrelated recipe
    # change moved the commit, which can be weeks away.
    fake([nightly_build(1000)])
    fetched = []

    def recipes(_token, ref):
        fetched.append(ref)
        entry = {
            "name": "test_8b-mi355x",
            "device": "mi355x",
            "parallelism": {"tensor_parallel_size": 4},
            "nightly": True,
            "accuracy_tasks": ["gsm8k"],
        }
        return {"test_8b-mi355x": (entry, CONFIGS)}

    monkeypatch.setattr(ca, "fetch_workload_map", recipes)
    store = tmp_path / "events.jsonl"
    ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())

    # Rewrite the snapshot as an older collector would have left it.
    events = store_mod.read_events_strict(store)
    for event in events:
        if event["event"] == ca.EXPECTED_CONFIGS_EVENT:
            event.pop("accuracy")
    store.write_text(
        "".join(json.dumps(e) + "\n" for e in events),
        encoding="utf-8",
    )

    fetched.clear()
    ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())
    assert fetched == ["f" * 40]
    (snapshot,) = [
        e for e in store_mod.read_events_strict(store) if e["event"] == ca.EXPECTED_CONFIGS_EVENT
    ]
    assert snapshot["accuracy"] == [
        {"workload": "test_8b-mi355x", "model": "", "device": "mi355x", "task": "gsm8k"}
    ]

    # Settled: the key is present now, so the next run refetches nothing.
    fetched.clear()
    ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())
    assert fetched == []


def test_an_artifact_for_a_run_the_recipe_lacks_is_skipped(fake, monkeypatch, tmp_path, caplog):
    # Without the run there is no ISL/OSL, and a result with neither is not a config.
    fake([nightly_build(1000)], artifacts_per_build=2)
    entry = {
        "name": "test_8b-mi355x",
        "device": "mi355x",
        "parallelism": {"tensor_parallel_size": 4},
        "model": "org/Model",
    }
    monkeypatch.setattr(
        ca,
        "fetch_workload_map",
        lambda _token, ref: {"test_8b-mi355x": (entry, {"cfg0": CONFIGS["cfg0"]})},
    )
    store = tmp_path / "events.jsonl"
    with caplog.at_level("WARNING"):
        ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())
    results = [e for e in store_mod.read_events_strict(store) if e["event"] == "perf_result"]
    assert [(e["isl"], e["osl"]) for e in results] == [(1024, 1024)]
    assert "No run cfg1" in caplog.text


def test_dry_run_reports_a_plan(fake, tmp_path, caplog):
    builds = [nightly_build(1000 + i) for i in range(3)]
    fake(builds, artifacts_per_build=2)
    with caplog.at_level("INFO"):
        ca.collect(tmp_path / "events.jsonl", days=14, bk_token="t", gh_token="", dry_run=True)
    assert "DRY RUN" in caplog.text
    assert "would download" in caplog.text


def test_store_is_unchanged_by_a_dry_run(fake, tmp_path):
    store = tmp_path / "events.jsonl"
    builds = [nightly_build(1000 + i) for i in range(3)]
    fake(builds)
    ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())
    before = store.read_bytes()
    ca.collect(store, days=14, bk_token="t", gh_token="", dry_run=True)
    assert store.read_bytes() == before
    # And the store really does hold parsed results.
    assert any(
        json.loads(line)["event"] == "perf_result"
        for line in store.read_text(encoding="utf-8").splitlines()
        if line
    )


def test_a_collection_with_nothing_new_leaves_the_payload_unchanged(fake, monkeypatch, tmp_path):
    # The deploy check compares payloads, so anything a no-op collection
    # restamps (the recipe snapshot's time, say) would make every scheduled
    # run deploy.
    from perf_eval import aggregate as agg
    from perf_eval.payload_changed import payload_changed
    from perf_eval.store import read_events_strict

    store = tmp_path / "events.jsonl"
    fake([nightly_build(1000 + i) for i in range(3)])
    recipe = (
        {
            "name": "test_8b-mi355x",
            "device": "mi355x",
            "parallelism": {"tensor_parallel_size": 4},
            "precision": "fp8",
            "model": "org/Model",
            "nightly": True,
        },
        {"cfg0": {"isl": 1024, "osl": 1024, "conc": 128}},
    )
    monkeypatch.setattr(ca, "fetch_workload_map", lambda _token, ref: {"test_8b-mi355x": recipe})
    # A clock that moves on every call, so each collection stamps new times.
    ticks = iter(range(10_000))
    monkeypatch.setattr(ca, "utcnow_iso", lambda: f"2026-06-10T00:{next(ticks) % 60:02d}:00Z")

    ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())
    first = agg.aggregate(read_events_strict(store))
    ca.collect(store, days=14, bk_token="t", gh_token="", budget=ca.RequestBudget())
    second = agg.aggregate(read_events_strict(store))

    assert first["expected"]["configs"], "the recipe snapshot should be recorded"
    assert not payload_changed(first, second)
