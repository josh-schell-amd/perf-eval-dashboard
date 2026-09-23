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
        assert store.artifact_keys_from_event(index[0]) == ("artifact-1",)

    def test_identity_carried_on_a_result_is_not_duplicated_into_the_index(self):
        event = perf_result()
        event["buildkite_artifact_id"] = "artifact-1"
        compacted = store.compact_events([event], now=NOW)
        assert not [e for e in compacted if e["event"] == store.ARTIFACT_INDEX_EVENT]

    def test_rewriting_an_unchanged_store_changes_nothing(self):
        # A timestamp stamped on every write would commit the state branch on
        # every run even with no new data.
        marker = {
            "event": store.ARTIFACT_MARKER_EVENT,
            "received_at": "2026-01-09T00:00:00Z",
            "buildkite_artifact_id": "artifact-1",
        }
        once = store.compact_events([perf_result(date="2026-01-09 00:00:00"), marker], now=NOW)
        later = NOW + datetime.timedelta(hours=8)
        assert store.compact_events(once, now=later) == once

    def test_unknown_and_non_nightly_events_are_dropped(self):
        events = [{"event": "build", "received_at": "2026-01-09T00:00:00Z"}]
        events.append(perf_result(nightly=False, date="2026-01-09 00:00:00"))
        assert store.compact_events(events, now=NOW) == []


class TestRetention:
    def _night(self, days_ago: float, n: int) -> dict:
        day = NOW - datetime.timedelta(days=days_ago)
        return perf_result(
            commit=f"{n:040x}",
            date=day.strftime("%Y-%m-%d %H:%M:%S"),
            received_at=day.strftime("%Y-%m-%dT%H:%M:%SZ"),
            build_number=n,
        )

    def test_results_inside_the_window_are_kept_and_older_ones_dropped(self):
        events = [self._night(1, 1), self._night(13.5, 2), self._night(14.5, 3), self._night(40, 4)]
        compacted = store.compact_events(events, now=NOW)
        assert {e["build_number"] for e in compacted if e["event"] == "perf_result"} == {1, 2}

    def test_the_window_matches_the_dashboard(self):
        # One number: the dashboard shows it, the store keeps it, and the
        # collector looks back no further.
        assert store.WINDOW_DAYS == 14

    def test_artifact_ids_expire_with_the_results(self):
        old = {
            "event": store.ARTIFACT_MARKER_EVENT,
            "received_at": "2025-12-01T00:00:00Z",
            "buildkite_artifact_id": "old",
        }
        assert store.compact_events([old], now=NOW) == []


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
        compacted = store.compact_events(events, now=NOW)
        assert [e for e in compacted if e["event"] == store.EXPECTED_CONFIGS_EVENT]

    def test_it_survives_a_round_trip_alongside_results(self, tmp_path):
        path = tmp_path / "events.jsonl"
        store.append_events(
            path, [perf_result(), self._snapshot("2026-01-09T00:00:00Z", "wl")], now=NOW
        )
        events = store.read_events_strict(path)
        assert any(e["event"] == store.EXPECTED_CONFIGS_EVENT for e in events)
        assert any(e["event"] == "perf_result" for e in events)


class TestAtomicWrite:
    def test_no_temp_files_are_left_behind(self, tmp_path):
        path = tmp_path / "events.jsonl"
        store.append_events(path, [perf_result()], now=NOW)
        assert [p.name for p in tmp_path.iterdir()] == ["events.jsonl"]

    def test_a_malformed_store_is_not_overwritten(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError):
            store.append_events(path, [perf_result()], now=NOW)
        assert path.read_text(encoding="utf-8") == "not json\n"


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

    def test_artifact_key_is_the_buildkite_artifact_id(self):
        assert store.artifact_key({"buildkite_artifact_id": " x "}) == "x"
        assert store.artifact_key({"build_number": 3}) is None


class TestEventTime:
    def test_a_result_is_timed_by_when_its_nightly_finished(self):
        event = perf_result(date="2026-01-01 00:00:00", received_at="2026-06-01T00:00:00Z")
        assert store.event_time(event) == datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)

    def test_other_events_are_timed_by_when_they_were_recorded(self):
        marker = {"event": store.ARTIFACT_MARKER_EVENT, "received_at": "2026-01-02T03:04:05Z"}
        assert store.event_time(marker) == datetime.datetime(
            2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC
        )

    def test_unparseable_returns_none(self):
        assert store.parse_time("not-a-date") is None

    def test_naive_timestamps_are_treated_as_utc(self):
        parsed = store.parse_time("2026-01-01 00:00:00")
        assert parsed is not None
        assert parsed.tzinfo is datetime.UTC
