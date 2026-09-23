"""Tests for identity-based merging of two event stores."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from conftest import accuracy_result, perf_result
from perf_eval import merge_events as me
from perf_eval import store


def write_store(path, events):
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


class TestReconcile:
    def test_union_of_distinct_nightlies(self):
        local = [perf_result(commit="a" * 40)]
        remote = [perf_result(commit="b" * 40)]
        merged = me.reconcile_events(local, remote)
        assert len(merged) == 2

    def test_same_result_from_both_sides_orders_by_ingestion_generation(self):
        older = perf_result(value=100.0, received_at="2026-01-01T00:00:00Z")
        newer = perf_result(value=110.0, received_at="2026-01-02T00:00:00Z")
        # Remote carries the newer generation even though local is listed first.
        merged = me.reconcile_events([older], [newer])
        values = [event["metrics"]["tput_per_gpu"] for event in merged]
        assert values == [100.0, 110.0]

    def test_equal_generation_with_disjoint_metrics_is_allowed(self):
        left = perf_result(metrics={"tput_per_gpu": 100.0})
        right = perf_result(metrics={"mean_ttft": 0.25})
        merged = me.reconcile_events([left], [right])
        assert len(merged) == 2

    def test_equal_generation_conflict_fails_closed(self):
        left = perf_result(metrics={"tput_per_gpu": 100.0})
        right = perf_result(metrics={"tput_per_gpu": 999.0})
        with pytest.raises(ValueError, match="equal-timestamp perf-eval conflict"):
            me.reconcile_events([left], [right])

    def test_equal_generation_accuracy_conflict_fails_closed(self):
        left = accuracy_result(value=0.80)
        right = accuracy_result(value=0.95)
        with pytest.raises(ValueError, match="equal-timestamp perf-eval conflict"):
            me.reconcile_events([left], [right])

    def test_non_result_events_pass_through(self):
        build = {"event": "build", "received_at": "2026-01-01T00:00:00Z"}
        merged = me.reconcile_events([build], [])
        assert merged == [build]

    def test_missing_timestamp_is_rejected_rather_than_guessed(self):
        event = perf_result()
        event.pop("received_at")
        event.pop("date")
        with pytest.raises(ValueError, match="no valid timestamp"):
            me.reconcile_events([event], [])

    def test_invalid_received_at_is_rejected(self):
        event = perf_result(received_at="not-a-timestamp")
        with pytest.raises(ValueError, match="invalid received_at"):
            me.reconcile_events([event], [])


class TestMergeEventFiles:
    def test_merges_remote_history_into_local(self, tmp_path):
        local = write_store(tmp_path / "local.jsonl", [perf_result(commit="a" * 40)])
        remote = write_store(tmp_path / "remote.jsonl", [perf_result(commit="b" * 40)])
        count = me.merge_event_files(local, remote)
        assert count == 2
        assert len(store.read_events_strict(local)) == 2

    def test_local_only_merge_still_compacts(self, tmp_path):
        local = write_store(
            tmp_path / "local.jsonl",
            [
                perf_result(metrics={"tput_per_gpu": 100.0}),
                perf_result(metrics={"mean_ttft": 0.25}),
            ],
        )
        assert me.merge_event_files(local) == 1

    def test_empty_remote_leaves_local_untouched(self, tmp_path):
        local = write_store(tmp_path / "local.jsonl", [perf_result()])
        remote = write_store(tmp_path / "remote.jsonl", [])
        before = local.read_bytes()
        with pytest.raises(ValueError, match="no events"):
            me.merge_event_files(local, remote)
        assert local.read_bytes() == before

    def test_malformed_remote_leaves_local_untouched(self, tmp_path):
        local = write_store(tmp_path / "local.jsonl", [perf_result()])
        remote = tmp_path / "remote.jsonl"
        remote.write_text("not json\n", encoding="utf-8")
        before = local.read_bytes()
        with pytest.raises(ValueError, match="invalid perf-eval JSONL"):
            me.merge_event_files(local, remote)
        assert local.read_bytes() == before

    def test_conflicting_remote_leaves_local_untouched(self, tmp_path):
        local = write_store(tmp_path / "local.jsonl", [perf_result(value=100.0)])
        remote = write_store(tmp_path / "remote.jsonl", [perf_result(value=999.0)])
        before = local.read_bytes()
        with pytest.raises(ValueError, match="equal-timestamp perf-eval conflict"):
            me.merge_event_files(local, remote)
        assert local.read_bytes() == before

    def test_fewer_remote_lines_do_not_discard_history(self, tmp_path):
        # A compacted remote store can legitimately hold fewer lines while
        # covering the same nightlies, which is why we merge identities rather
        # than compare line counts.
        # Recent dates: the merge compacts against the real clock.
        days = [datetime.now(UTC) - timedelta(days=5 - index) for index in range(5)]
        local = write_store(
            tmp_path / "local.jsonl",
            [
                perf_result(commit=f"{index:040x}", date=day.strftime("%Y-%m-%d %H:%M:%S"))
                for index, day in enumerate(days)
            ],
        )
        remote = write_store(tmp_path / "remote.jsonl", [perf_result(commit=f"{0:040x}")])
        me.merge_event_files(local, remote)
        commits = {event["vllm_commit"] for event in store.read_events_strict(local)}
        assert len(commits) == 5
