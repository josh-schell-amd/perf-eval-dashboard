"""Tests for the bounded, atomic event store."""

from __future__ import annotations

import datetime
import json

import pytest

from conftest import accuracy_result, perf_result
from perf_eval import store

NOW = datetime.datetime(2026, 1, 10, tzinfo=datetime.UTC)


def _read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class TestRoundTrip:
    def test_append_then_read(self, tmp_path):
        path = tmp_path / "events.jsonl"
        store.append_events(path, [perf_result()], now=NOW)
        events = store.read_events_strict(path)
        assert len(events) == 1
        assert events[0]["event"] == "perf_result"

    def test_missing_store_reads_empty(self, tmp_path):
        assert store.read_events_strict(tmp_path / "absent.jsonl") == []

    def test_written_file_ends_with_newline_per_record(self, tmp_path):
        path = tmp_path / "events.jsonl"
        store.append_events(path, [perf_result(), accuracy_result()], now=NOW)
        assert path.read_text(encoding="utf-8").endswith("\n")
        assert len(_read_lines(path)) == 2


class TestStrictRead:
    def test_malformed_line_raises(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text('{"event": "perf_result"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="invalid perf-eval JSONL"):
            store.read_events_strict(path)

    def test_non_object_line_raises(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            store.read_events_strict(path)

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text('\n{"event": "build"}\n\n', encoding="utf-8")
        assert len(store.read_events_strict(path)) == 1

    def test_lenient_read_skips_instead_of_raising(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text('{"event": "build"}\nnot json\n', encoding="utf-8")
        assert len(store.read_events(path)) == 1

    def test_non_canonical_artifact_index_raises(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text(
            json.dumps(
                {
                    "event": store.ARTIFACT_INDEX_EVENT,
                    "schema_version": 99,
                    "identities": [["id", "x", "2026-01-01T00:00:00Z"]],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="artifact identity index is not canonical"):
            store.read_events_strict(path)


class TestCompaction:
    def test_duplicate_results_merge_metrics(self):
        first = perf_result(metrics={"tput_per_gpu": 100.0})
        second = perf_result(metrics={"mean_ttft": 0.25})
        compacted = store.compact_events([first, second], now=NOW)
        results = [e for e in compacted if e["event"] == "perf_result"]
        assert len(results) == 1
        assert results[0]["metrics"] == {"tput_per_gpu": 100.0, "mean_ttft": 0.25}

    def test_duplicate_accuracy_rows_union_by_task_and_metric(self):
        first = accuracy_result(task="gsm8k", metric="exact_match,strict-match", value=0.8)
        second = accuracy_result(task="gsm8k", metric="acc,none", value=0.9)
        compacted = store.compact_events([first, second], now=NOW)
        results = [e for e in compacted if e["event"] == "accuracy_result"]
        assert len(results) == 1
        assert {row["metric"] for row in results[0]["results"]} == {
            "exact_match,strict-match",
            "acc,none",
        }

    def test_the_newer_build_of_a_commit_wins_even_when_stored_first(self):
        # The collector appends newest builds first, so a backfill puts an
        # older rebuild of the same commit later in the file.
        newer = perf_result(build_number=601, date="2026-01-09 12:00:00", value=200.0)
        older = perf_result(build_number=600, date="2026-01-08 12:00:00", value=100.0)
        compacted = store.compact_events([newer, older], now=NOW)
        results = [e for e in compacted if e["event"] == "perf_result"]
        assert len(results) == 1
        assert results[0]["build_number"] == 601
        assert results[0]["metrics"]["tput_per_gpu"] == 200.0

    def test_nightlies_without_a_vllm_commit_stay_separate(self):
        # The perf-eval repo's commit is shared by many nightlies, so it must
        # not stand in for the vLLM commit.
        events = [perf_result(commit="", build_number=n) for n in (700, 701)]
        for event in events:
            event["build_commit"] = "f" * 40
        compacted = store.compact_events(events, now=NOW)
        assert len([e for e in compacted if e["event"] == "perf_result"]) == 2

    def test_distinct_commits_are_distinct_results(self):
        events = [perf_result(commit="a" * 40), perf_result(commit="b" * 40)]
        compacted = store.compact_events(events, now=NOW)
        assert len([e for e in compacted if e["event"] == "perf_result"]) == 2

    def test_distinct_configs_are_distinct_results(self):
        events = [perf_result(conc=128), perf_result(conc=256)]
        compacted = store.compact_events(events, now=NOW)
        assert len([e for e in compacted if e["event"] == "perf_result"]) == 2

    def test_artifact_markers_fold_into_an_index(self):
        marker = {
            "event": store.ARTIFACT_MARKER_EVENT,
            "received_at": "2026-01-09T00:00:00Z",
            "build_number": 5,
            "buildkite_artifact_id": "artifact-1",
        }
        compacted = store.compact_events([marker], now=NOW)
        assert not [e for e in compacted if e["event"] == store.ARTIFACT_MARKER_EVENT]
        index = [e for e in compacted if e["event"] == store.ARTIFACT_INDEX_EVENT]
        assert len(index) == 1
        assert index[0]["schema_version"] == store.ARTIFACT_INDEX_SCHEMA_VERSION
        assert ("id", "artifact-1") in store.artifact_keys_from_event(index[0])

    def test_identity_carried_on_a_result_is_not_duplicated_into_the_index(self):
        event = perf_result()
        event["buildkite_artifact_id"] = "artifact-1"
        compacted = store.compact_events([event], now=NOW)
        assert not [e for e in compacted if e["event"] == store.ARTIFACT_INDEX_EVENT]

    def test_history_beyond_the_window_is_dropped(self):
        old = perf_result(
            commit="c" * 40,
            date="2020-01-01 00:00:00",
            received_at="2020-01-01T00:00:00Z",
            build_number=2,
        )
        recent = perf_result(commit="d" * 40, date="2026-01-09 00:00:00", build_number=3)
        # min_nightlies protection is what keeps the old one alive by default,
        # so exercise the tightest policy directly.
        compacted = store._compact_events_once(
            [old, recent], NOW, history_days=14, min_nightlies=1, auxiliary_days=14
        )
        commits = {e["vllm_commit"] for e in compacted if e["event"] == "perf_result"}
        assert commits == {"d" * 40}

    def test_recent_nightlies_are_protected_from_pruning(self):
        old = perf_result(
            commit="c" * 40,
            date="2020-01-01 00:00:00",
            received_at="2020-01-01T00:00:00Z",
        )
        compacted = store._compact_events_once(
            [old], NOW, history_days=14, min_nightlies=30, auxiliary_days=14
        )
        assert len([e for e in compacted if e["event"] == "perf_result"]) == 1


class TestExpectedConfigsSnapshot:
    """The recipe expectation is a singleton that retention must not discard.

    It is not a result, so it must not be deduped as one — and it must not be
    aged out either: an old expectation is still the current recipe set, and
    dropping it would blind the coverage card entirely.
    """

    def _snapshot(self, received_at: str, workload: str) -> dict:
        return {
            "event": store.EXPECTED_CONFIGS_EVENT,
            "received_at": received_at,
            "configs": [{"workload": workload}],
        }

    def test_only_the_newest_snapshot_survives(self):
        events = [
            self._snapshot("2026-01-01T00:00:00Z", "old"),
            self._snapshot("2026-01-09T00:00:00Z", "new"),
        ]
        compacted = store.compact_events(events, now=NOW)
        snapshots = [e for e in compacted if e["event"] == store.EXPECTED_CONFIGS_EVENT]
        assert len(snapshots) == 1
        assert snapshots[0]["configs"] == [{"workload": "new"}]

    def test_store_order_does_not_decide_the_winner(self):
        events = [
            self._snapshot("2026-01-09T00:00:00Z", "new"),
            self._snapshot("2026-01-01T00:00:00Z", "old"),
        ]
        compacted = store.compact_events(events, now=NOW)
        snapshots = [e for e in compacted if e["event"] == store.EXPECTED_CONFIGS_EVENT]
        assert snapshots[0]["configs"] == [{"workload": "new"}]

    def test_an_old_snapshot_is_not_aged_out(self):
        # Far outside every retention cutoff, but still the current recipes.
        events = [self._snapshot("2020-01-01T00:00:00Z", "ancient")]
        compacted = store._compact_events_once(
            events, NOW, history_days=14, min_nightlies=1, auxiliary_days=14
        )
        assert [e for e in compacted if e["event"] == store.EXPECTED_CONFIGS_EVENT]

    def test_it_survives_a_round_trip_alongside_results(self, tmp_path):
        path = tmp_path / "events.jsonl"
        store.append_events(
            path, [perf_result(), self._snapshot("2026-01-09T00:00:00Z", "wl")], now=NOW
        )
        events = store.read_events_strict(path)
        assert any(e["event"] == store.EXPECTED_CONFIGS_EVENT for e in events)
        assert any(e["event"] == "perf_result" for e in events)


class TestByteBudget:
    def test_oversized_payload_leaves_the_previous_file_intact(self, tmp_path):
        path = tmp_path / "events.jsonl"
        store.append_events(path, [perf_result()], now=NOW)
        before = path.read_bytes()
        with pytest.raises(RuntimeError):
            # 1 byte cannot hold a single record under any retention policy.
            store.write_events_atomic(path, [perf_result()], now=NOW, max_bytes=1)
        assert path.read_bytes() == before

    def test_no_temp_files_are_left_behind(self, tmp_path):
        path = tmp_path / "events.jsonl"
        store.append_events(path, [perf_result()], now=NOW)
        assert [p.name for p in tmp_path.iterdir()] == ["events.jsonl"]

    def test_budget_cannot_exceed_the_hard_cap(self):
        assert store.enforced_byte_budget(10**12, cap=store.EVENTS_MAX_BYTES) == (
            store.EVENTS_MAX_BYTES
        )

    @pytest.mark.parametrize("requested", [0, -1])
    def test_non_positive_budget_rejected(self, requested):
        with pytest.raises(ValueError, match="must be positive"):
            store.enforced_byte_budget(requested, cap=store.EVENTS_MAX_BYTES)

    def test_published_summary_budget_is_tighter_than_the_event_log(self):
        # The summary is downloaded by every browser; the log never is.
        assert store.SUMMARY_MAX_BYTES < store.EVENTS_MAX_BYTES


class TestIdentities:
    def test_nightly_identity_prefers_commit(self):
        assert store.nightly_identity({"vllm_commit": "abc"}) == "commit:abc"

    def test_nightly_identity_falls_back_to_build_number(self):
        assert store.nightly_identity({"build_number": 42}) == "build:42"

    def test_nightly_identity_ignores_the_perf_eval_commit(self):
        event = {"vllm_commit": "", "build_commit": "f" * 40, "build_number": 42}
        assert store.nightly_identity(event) == "build:42"

    def test_result_identity_separates_perf_configs(self):
        left = store.result_identity(perf_result(conc=128))
        right = store.result_identity(perf_result(conc=256))
        assert left != right

    def test_result_identity_folds_a_retried_nightly(self):
        # Same commit, different build number: one nightly, re-run.
        left = store.result_identity(perf_result(build_number=1))
        right = store.result_identity(perf_result(build_number=2))
        assert left == right

    def test_artifact_key_prefers_id(self):
        assert store.artifact_key({"buildkite_artifact_id": "x"}) == ("id", "x")

    def test_artifact_key_metadata_fallback(self):
        key = store.artifact_key(
            {
                "build_number": 3,
                "buildkite_artifact_job_id": "job",
                "buildkite_artifact_path": "./results/a/bench-b.json",
                "buildkite_artifact_sha1": "ABC",
            }
        )
        assert key == ("metadata", 3, "job", "results/a/bench-b.json", "abc")

    def test_artifact_key_requires_a_complete_fallback(self):
        assert store.artifact_key({"build_number": 3}) is None


class TestEventDatetime:
    def test_prefers_run_date_over_ingestion_time(self):
        parsed = store.event_datetime(
            {"date": "2026-01-01 00:00:00", "received_at": "2026-06-01T00:00:00Z"}
        )
        assert parsed is not None
        assert parsed.year == 2026
        assert parsed.month == 1

    def test_unparseable_returns_none(self):
        assert store.event_datetime({"date": "not-a-date"}) is None

    def test_naive_timestamps_are_treated_as_utc(self):
        parsed = store.event_datetime({"date": "2026-01-01 00:00:00"})
        assert parsed is not None
        assert parsed.tzinfo is datetime.UTC
