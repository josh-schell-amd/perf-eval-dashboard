"""Tests for the deploy gate that suppresses redundant Pages builds."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from conftest import perf_result
from perf_eval import aggregate as agg
from perf_eval import payload_changed as pc


def payload(**overrides):
    base = {
        "generated_at": "2026-01-01T00:00:00Z",
        "models": [{"model": "m", "nightly_count": 3}],
        "summary": {"models": 1, "nightlies": 3},
        "retention": {"display_window_days": 14},
    }
    base.update(overrides)
    return base


class TestVolatileFields:
    def test_a_new_timestamp_alone_is_not_a_change(self):
        # This is the whole point: aggregate.py restamps generated_at every
        # run, so without this the site would republish identical numbers
        # three times a day.
        old = payload(generated_at="2026-01-01T00:00:00Z")
        new = payload(generated_at="2026-06-01T12:34:56Z")
        assert pc.payload_changed(old, new) is False

    def test_the_volatile_list_stays_minimal(self):
        # Anything else in the payload is real data whose change must deploy.
        assert pc.VOLATILE_FIELDS == ("generated_at",)


class TestMaterialChanges:
    def test_new_nightly_count_is_a_change(self):
        assert pc.payload_changed(payload(), payload(summary={"models": 1, "nightlies": 4})) is True

    def test_new_model_is_a_change(self):
        new = payload(models=[{"model": "m", "nightly_count": 3}, {"model": "n"}])
        assert pc.payload_changed(payload(), new) is True

    def test_a_changed_metric_value_deep_in_the_payload_is_a_change(self):
        old = payload(models=[{"model": "m", "metrics": {"tput": {"latest": 100.0}}}])
        new = payload(models=[{"model": "m", "metrics": {"tput": {"latest": 101.0}}}])
        assert pc.payload_changed(old, new) is True

    def test_a_retention_change_is_a_change(self):
        new = payload(retention={"display_window_days": 7})
        assert pc.payload_changed(payload(), new) is True

    def test_key_order_is_not_a_change(self):
        old = {"models": [], "summary": {}, "generated_at": "a"}
        new = {"summary": {}, "generated_at": "b", "models": []}
        assert pc.payload_changed(old, new) is False

    def test_identical_payloads_are_unchanged(self):
        assert pc.payload_changed(payload(), payload()) is False


class TestFailsTowardsPublishing:
    def test_no_previous_payload_counts_as_changed(self):
        assert pc.payload_changed(None, payload()) is True

    def test_missing_previous_file_counts_as_changed(self, tmp_path):
        assert pc._load(tmp_path / "absent.json") is None

    def test_unreadable_previous_file_counts_as_changed(self, tmp_path):
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        # One redundant deploy is a better failure than a broken collection.
        assert pc._load(broken) is None

    def test_non_object_previous_file_counts_as_changed(self, tmp_path):
        listy = tmp_path / "list.json"
        listy.write_text("[1, 2]", encoding="utf-8")
        assert pc._load(listy) is None


class TestCli:
    def _run(self, tmp_path, monkeypatch, capsys, previous, current):
        cur = tmp_path / "current.json"
        cur.write_text(json.dumps(current), encoding="utf-8")
        args = ["payload_changed.py", "--current", str(cur)]
        if previous is not None:
            prev = tmp_path / "previous.json"
            prev.write_text(json.dumps(previous), encoding="utf-8")
            args += ["--previous", str(prev)]
        monkeypatch.setattr("sys.argv", args)
        code = pc.main()
        return code, capsys.readouterr()

    def test_prints_true_when_changed(self, tmp_path, monkeypatch, capsys):
        code, out = self._run(
            tmp_path, monkeypatch, capsys, payload(), payload(summary={"nightlies": 9})
        )
        assert code == 0
        assert out.out.strip().splitlines()[0] == "true"

    def test_prints_false_when_only_the_timestamp_moved(self, tmp_path, monkeypatch, capsys):
        code, out = self._run(
            tmp_path, monkeypatch, capsys, payload(), payload(generated_at="2027-01-01T00:00:00Z")
        )
        assert code == 0
        assert out.out.strip().splitlines()[0] == "false"

    def test_writes_the_github_output_variable(self, tmp_path, monkeypatch, capsys):
        output = tmp_path / "gh_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        self._run(tmp_path, monkeypatch, capsys, payload(), payload())
        assert "changed=false" in output.read_text(encoding="utf-8")

    def test_missing_current_payload_is_an_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["payload_changed.py", "--current", str(tmp_path / "absent.json")],
        )
        assert pc.main() == 1


class TestAgainstRealAggregateOutput:
    def _nights(self, days_ago: list[float], when: datetime):
        return [
            perf_result(
                commit=f"{index:040x}",
                date=(when - timedelta(days=ago)).strftime("%Y-%m-%d %H:%M:%S"),
                build_number=1000 + index,
            )
            for index, ago in enumerate(days_ago)
        ]

    def test_a_day_passing_is_not_a_change(self):
        # No nightly crosses the window edge, so nothing new to publish.
        when = datetime(2026, 2, 1, tzinfo=UTC)
        events = self._nights([1, 3, 8], when)
        today = agg.build_payload(events, generated_at=when)
        tomorrow = agg.build_payload(events, generated_at=when + timedelta(days=1))
        assert pc.payload_changed(today, tomorrow) is False

    def test_a_nightly_leaving_the_window_is_a_change(self):
        # The payload publishes only the window, so this deploys, at most once
        # per nightly that ages out.
        when = datetime(2026, 2, 1, tzinfo=UTC)
        events = self._nights([1, 3, 13.5], when)
        today = agg.build_payload(events, generated_at=when)
        tomorrow = agg.build_payload(events, generated_at=when + timedelta(days=1))
        assert pc.payload_changed(today, tomorrow) is True

    def test_two_runs_over_the_same_events_differ_only_by_timestamp(self):
        events = [perf_result()]
        first = agg.aggregate(events, generated_at=datetime(2026, 1, 1, tzinfo=UTC))
        second = agg.aggregate(events, generated_at=datetime(2026, 9, 9, tzinfo=UTC))
        assert first["generated_at"] != second["generated_at"]
        assert pc.payload_changed(first, second) is False

    def test_a_new_nightly_registers_as_a_change(self):
        when = datetime(2026, 1, 1, tzinfo=UTC)
        first = agg.aggregate([perf_result(commit="a" * 40)], generated_at=when)
        second = agg.aggregate(
            [
                perf_result(commit="a" * 40, date="2026-01-01 00:00:00"),
                perf_result(commit="b" * 40, date="2026-01-02 00:00:00", value=120.0),
            ],
            generated_at=when,
        )
        assert pc.payload_changed(first, second) is True
