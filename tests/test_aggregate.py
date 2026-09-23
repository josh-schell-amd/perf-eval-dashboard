"""Tests for aggregation, the scope filter, and series-ordering correctness."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from conftest import accuracy_result, days_ago, perf_result
from perf_eval import aggregate as agg


def _only_model(payload):
    assert len(payload["models"]) == 1
    return payload["models"][0]


def _metric(payload, key="tput_per_gpu"):
    model = _only_model(payload)
    assert len(model["perf_configs"]) == 1
    return model["perf_configs"][0]["metrics"][key]


class TestScopeFilter:
    def test_nvidia_results_are_excluded(self):
        events = [
            perf_result(device="h200", model="nvidia-model"),
            perf_result(device="mi355x", model="amd-model"),
        ]
        payload = agg.aggregate(events)
        assert [m["model"] for m in payload["models"]] == ["amd-model"]
        assert payload["summary"]["amd_devices"] == ["mi355x"]

    def test_non_nightly_results_are_excluded(self):
        events = [perf_result(nightly=False)]
        assert agg.aggregate(events)["models"] == []

    def test_nightly_flag_must_be_exactly_true(self):
        events = [perf_result(nightly="yes")]
        assert agg.aggregate(events)["models"] == []

    def test_non_result_events_are_ignored(self):
        events = [{"event": "build", "nightly": True}, perf_result()]
        assert len(agg.aggregate(events)["models"]) == 1

    def test_scope_is_declared_in_the_payload(self):
        payload = agg.aggregate([perf_result()])
        assert payload["scope"]["hardware"] == "amd"
        assert payload["scope"]["runs"] == "nightly"
        assert "NVIDIA" in payload["scope"]["description"]


class TestSeriesOrdering:
    def test_newest_observation_wins_regardless_of_store_order(self):
        # Same nightly (same commit) observed twice. The newer observation
        # appears FIRST in the store, so resolving by list position would pick
        # the stale value.
        events = [
            perf_result(commit="a" * 40, value=100.0, date="2026-01-02 00:00:00"),
            perf_result(commit="a" * 40, value=50.0, date="2026-01-01 00:00:00"),
        ]
        block = _metric(agg.aggregate(events))
        assert block["latest"] == 100.0
        assert len(block["series"]) == 1

    def test_newest_observation_wins_when_store_order_agrees(self):
        events = [
            perf_result(commit="a" * 40, value=50.0, date="2026-01-01 00:00:00"),
            perf_result(commit="a" * 40, value=100.0, date="2026-01-02 00:00:00"),
        ]
        block = _metric(agg.aggregate(events))
        assert block["latest"] == 100.0

    def test_retried_nightly_on_the_same_commit_is_one_point(self):
        # A nightly re-run produces a second build number for one commit. That
        # is one nightly, so it must not appear twice in the trend line.
        events = [
            perf_result(commit="a" * 40, build_number=1, value=100.0),
            perf_result(commit="a" * 40, build_number=2, value=110.0),
        ]
        block = _metric(agg.aggregate(events))
        assert len(block["series"]) == 1
        assert _only_model(agg.aggregate(events))["nightly_count"] == 1

    def test_distinct_nightlies_are_distinct_points_sorted_oldest_first(self):
        events = [
            perf_result(commit="b" * 40, value=200.0, date="2026-01-03 00:00:00"),
            perf_result(commit="a" * 40, value=100.0, date="2026-01-01 00:00:00"),
        ]
        block = _metric(agg.aggregate(events))
        assert [point["value"] for point in block["series"]] == [100.0, 200.0]
        assert block["latest"] == 200.0
        assert block["previous"] == 100.0

    def test_unparseable_date_is_logged_not_crashed(self, caplog):
        event = perf_result(date="not-a-date")
        with caplog.at_level("WARNING"):
            payload = agg.aggregate([event])
        assert len(payload["models"]) == 1
        assert "no parseable date" in caplog.text


class TestStatusThresholds:
    """Perf metrics must move by 0.5% to count; any accuracy change counts."""

    @pytest.mark.parametrize(
        "previous,latest,expected",
        [
            (100.0, 103.0, "good"),  # up on higher-is-better
            (100.0, 97.0, "bad"),  # down on higher-is-better
            (100.0, 100.5, "good"),  # exactly at the threshold counts
            (100.0, 99.5, "bad"),
            (100.0, 100.49, "neutral"),  # just under it does not
            (100.0, 99.9, "neutral"),
            (100.0, 100.0, "neutral"),
        ],
    )
    def test_perf_movement_under_the_threshold_is_neutral(self, previous, latest, expected):
        events = [
            perf_result(commit="a" * 40, value=previous, date="2026-01-01 00:00:00"),
            perf_result(commit="b" * 40, value=latest, date="2026-01-02 00:00:00"),
        ]
        assert _metric(agg.aggregate(events))["status"] == expected

    def test_lower_is_better_inverts_the_verdict(self):
        events = [
            perf_result(commit="a" * 40, metrics={"mean_ttft": 0.10}, date="2026-01-01 00:00:00"),
            perf_result(commit="b" * 40, metrics={"mean_ttft": 0.05}, date="2026-01-02 00:00:00"),
        ]
        block = _metric(agg.aggregate(events), "mean_ttft")
        assert block["direction"] == "lower"
        assert block["status"] == "good"

    def test_first_nightly_is_neutral_with_no_previous(self):
        block = _metric(agg.aggregate([perf_result()]))
        assert block["previous"] is None
        assert block["status"] == "neutral"
        assert block["delta"] is None

    @pytest.mark.parametrize(
        "previous,latest,expected",
        [
            (0.80, 0.81, "good"),  # exactly one point counts
            (0.80, 0.79, "bad"),
            (0.9234, 0.9334, "good"),  # one point despite float error
            (0.80, 0.8099, "neutral"),  # just under a point does not
            (0.9242, 0.9234, "neutral"),  # one gsm8k question
            (0.80, 0.80, "neutral"),
        ],
    )
    def test_accuracy_must_move_by_a_point(self, previous, latest, expected):
        events = [
            accuracy_result(commit="a" * 40, value=previous, date="2026-01-01 00:00:00"),
            accuracy_result(commit="b" * 40, value=latest, date="2026-01-02 00:00:00"),
        ]
        task = _only_model(agg.aggregate(events))["accuracy_tasks"][0]
        assert task["status"] == expected

    def test_zero_previous_never_divides(self):
        events = [
            perf_result(commit="a" * 40, value=0.0, date="2026-01-01 00:00:00"),
            perf_result(commit="b" * 40, value=5.0, date="2026-01-02 00:00:00"),
        ]
        block = _metric(agg.aggregate(events))
        assert block["delta_pct"] is None
        assert block["status"] == "neutral"

    def test_thresholds_are_published_for_the_frontend(self):
        payload = agg.aggregate([perf_result()])
        assert payload["thresholds"] == {"perf_rel": 0.005, "accuracy_abs": 0.01}

    def test_the_thresholds_are_pinned(self):
        # Pinned so changing a threshold is a deliberate, visible change rather
        # than a quiet constant edit.
        assert agg.PERF_REL_THRESHOLD == 0.005
        assert agg.ACCURACY_ABS_THRESHOLD == 0.01

    def test_a_flat_metric_is_not_a_regression(self):
        events = [
            perf_result(commit="a" * 40, value=100.0, date="2026-01-01 00:00:00"),
            perf_result(commit="b" * 40, value=100.0, date="2026-01-02 00:00:00"),
        ]
        block = _metric(agg.aggregate(events))
        assert block["delta"] == 0
        assert block["status"] == "neutral"

    def test_a_flat_accuracy_score_is_not_a_regression(self):
        events = [
            accuracy_result(commit="a" * 40, value=0.8, date="2026-01-01 00:00:00"),
            accuracy_result(commit="b" * 40, value=0.8, date="2026-01-02 00:00:00"),
        ]
        task = _only_model(agg.aggregate(events))["accuracy_tasks"][0]
        assert task["status"] == "neutral"


class TestGrouping:
    def test_configs_are_keyed_by_device_shape_and_concurrency(self):
        events = [perf_result(conc=128), perf_result(conc=256)]
        model = _only_model(agg.aggregate(events))
        assert [config["conc"] for config in model["perf_configs"]] == [128, 256]

    def test_tp_variants_of_one_shape_are_separate_configs(self):
        events = [perf_result(tp=4, value=40.0), perf_result(tp=8, value=60.0)]
        model = _only_model(agg.aggregate(events))
        assert sorted(
            (c["tp"], c["metrics"]["tput_per_gpu"]["latest"]) for c in model["perf_configs"]
        ) == [(4, 40.0), (8, 60.0)]

    def test_config_label_abbreviates_power_of_two_lengths(self):
        events = [perf_result(isl=8192, osl=1024, conc=128, device="mi355x")]
        model = _only_model(agg.aggregate(events))
        assert model["perf_configs"][0]["label"] == "8K in / 1K out @ conc 128 (MI355X)"

    def test_workload_set_is_discovered_not_hard_coded(self):
        events = [perf_result(model="brand-new/Model-1T")]
        assert _only_model(agg.aggregate(events))["model"] == ("brand-new/Model-1T")

    def test_missing_model_gets_a_placeholder_rather_than_being_dropped(self):
        events = [perf_result(model="")]
        assert _only_model(agg.aggregate(events))["model"] == ("(unknown model)")

    def test_provenance_is_kept_on_every_series_point(self):
        block = _metric(agg.aggregate([perf_result()]))
        point = block["series"][0]
        for field in ("vllm_commit", "image", "build_url", "build_number", "date"):
            assert field in point
        assert "nightly_key" not in point
        assert not any(key.startswith("_") for key in point)

    def test_failed_requests_are_kept_on_every_series_point(self):
        event = perf_result()
        event.update(completed_requests=500.0, failed_requests=12.0)
        point = _metric(agg.aggregate([event]))["series"][0]
        assert (point["completed_requests"], point["failed_requests"]) == (500.0, 12.0)

    def test_summary_counts_points_and_nightlies(self):
        events = [
            perf_result(commit="a" * 40, date="2026-01-01 00:00:00"),
            perf_result(commit="b" * 40, date="2026-01-02 00:00:00"),
            accuracy_result(commit="a" * 40, date="2026-01-01 00:00:00"),
        ]
        summary = agg.aggregate(events)["summary"]
        assert summary["models"] == 1
        assert summary["nightlies"] == 2
        assert summary["perf_points"] == 2
        assert summary["accuracy_points"] == 1

    def test_primary_accuracy_tasks_sort_first(self):
        events = [
            accuracy_result(task="b_task", metric="acc,none"),
            accuracy_result(task="a_task", metric="acc,none"),
        ]
        events[0]["results"][0]["primary"] = False
        tasks = _only_model(agg.aggregate(events))["accuracy_tasks"]
        assert tasks[0]["primary"] is True


class TestAccuracyGrouping:
    def test_one_model_on_two_devices_is_two_series(self):
        events = [
            accuracy_result(device="mi300x", value=0.92),
            accuracy_result(device="mi355x", value=0.94),
        ]
        tasks = _only_model(agg.aggregate(events))["accuracy_tasks"]
        assert sorted((t["device"], t["series"][0]["value"]) for t in tasks) == [
            ("mi300x", 0.92),
            ("mi355x", 0.94),
        ]

    def test_sample_len_is_not_a_score(self):
        event = accuracy_result(metric="sample_len", value=1319.0)
        event["results"].append(
            {"task": "gsm8k", "metric": "exact_match,strict-match", "value": 0.9, "primary": False}
        )
        tasks = _only_model(agg.aggregate([event]))["accuracy_tasks"]
        assert [(t["metric"], t["primary"]) for t in tasks] == [("exact_match,strict-match", True)]

    def test_a_backend_name_is_resolved_from_the_recipe_expectation(self):
        # Events stored before the collector read the model from the recipe
        # carry lm-eval's backend name; left alone, every workload would fold
        # into one "model" and each nightly would keep a single one of them.
        expected = {
            "event": "expected_configs",
            "received_at": "2026-01-02T00:00:00Z",
            "configs": [{"workload": "test_8b_mi355x", "model": "meta-llama/Test-8B"}],
        }
        event = accuracy_result(model="local-completions", workload="test_8b_mi355x")
        models = agg.aggregate([expected, event])["models"]
        assert [m["model"] for m in models] == ["meta-llama/Test-8B"]

    def test_a_backend_name_falls_back_to_lm_evals_output_directory(self):
        event = accuracy_result(model="local-completions", workload="other_mi355x")
        event["buildkite_artifact_path"] = (
            "results/other_mi355x/gsm8k/openai__gpt-oss-120b/results_2026-01-01.json"
        )
        models = agg.aggregate([event])["models"]
        assert [m["model"] for m in models] == ["openai/gpt-oss-120b"]

    def test_an_unresolvable_backend_name_falls_back_to_the_workload(self):
        event = accuracy_result(model="local-completions", workload="gone_mi355x")
        models = agg.aggregate([event])["models"]
        assert [m["model"] for m in models] == ["gone_mi355x"]

    def test_two_workloads_for_one_model_and_device_are_separate_series(self):
        events = [
            accuracy_result(workload="x_tp4-mi355x", value=0.9),
            accuracy_result(workload="x_tp8-mi355x", value=0.5),
        ]
        tasks = _only_model(agg.aggregate(events))["accuracy_tasks"]
        assert sorted((t["workload"], t["series"][0]["value"]) for t in tasks) == [
            ("x_tp4-mi355x", 0.9),
            ("x_tp8-mi355x", 0.5),
        ]

    def test_workloads_are_not_folded_into_each_other(self):
        events = [
            accuracy_result(model="local-completions", workload="a_mi355x", value=0.9),
            accuracy_result(model="local-completions", workload="b_mi355x", value=0.5),
        ]
        events[0]["buildkite_artifact_path"] = "results/a_mi355x/gsm8k/org__A/results_1.json"
        events[1]["buildkite_artifact_path"] = "results/b_mi355x/gsm8k/org__B/results_1.json"
        models = agg.aggregate(events)["models"]
        assert {m["model"]: m["accuracy_tasks"][0]["series"][0]["value"] for m in models} == {
            "org/A": 0.9,
            "org/B": 0.5,
        }


class TestWhatCountsAsANightly:
    """The retention limit counts vLLM commits, not calendar days.

    Worth pinning explicitly: "180 nightlies" reads like 180 days, but the
    unit is whatever `nightly_identity` considers one run — the vLLM commit,
    falling back to build number and then timestamp.
    """

    def _events(self, days: int, runs_per_day: int, *, distinct_commits: bool):
        events = []
        for day in range(days):
            for run in range(runs_per_day):
                index = day * runs_per_day + run if distinct_commits else day
                events.append(
                    perf_result(
                        commit=f"{index:040x}",
                        date=f"2026-01-{day + 1:02d} {run:02d}:00:00",
                        build_number=1000 + day * runs_per_day + run,
                    )
                )
        return events

    def test_several_runs_a_day_on_distinct_commits_count_separately(self):
        # 5 runs a day for 14 days on different commits is 70 nightlies.
        payload = agg.aggregate(self._events(14, 5, distinct_commits=True))
        assert payload["summary"]["nightlies"] == 70

    def test_several_runs_a_day_on_one_commit_collapse(self):
        # The same commit re-run 5 times a night is still one nightly: that is
        # the retry dedupe, not data loss.
        payload = agg.aggregate(self._events(14, 5, distinct_commits=False))
        assert payload["summary"]["nightlies"] == 14

    def test_one_run_a_day_is_one_nightly_a_day(self):
        # The real pipeline shape: one scheduled nightly per new vLLM commit.
        payload = agg.aggregate(self._events(14, 1, distinct_commits=True))
        assert payload["summary"]["nightlies"] == 14

    def test_a_run_without_a_commit_falls_back_to_build_number(self):
        events = [perf_result(commit="", build_number=n) for n in (1, 2, 3)]
        for event in events:
            event["vllm_commit"] = ""
            event["build_commit"] = ""
        assert agg.aggregate(events)["summary"]["nightlies"] == 3


class TestBuildPayload:
    """The payload publishes the nightlies from the last WINDOW_DAYS."""

    def _nights(self, ages: list[float]):
        return [
            perf_result(commit=f"{index:040x}", date=days_ago(age), build_number=1000 + index)
            for index, age in enumerate(ages)
        ]

    def test_only_nightlies_inside_the_window_are_published(self):
        payload = agg.build_payload(self._nights([1, 5, 13.5, 14.5, 29]))
        assert len(_metric(payload)["series"]) == 3

    def test_nothing_in_the_window_publishes_no_models(self):
        payload = agg.build_payload(self._nights([40, 25]))
        assert payload["models"] == []

    def test_the_window_is_published_so_the_page_cannot_drift(self):
        payload = agg.build_payload([perf_result()])
        assert payload["retention"] == {"display_window_days": agg.WINDOW_DAYS}


class TestExpectedIsPublished:
    """The coverage card needs the recipe-derived expectation in the payload.

    The page cannot reach GitHub, so the collector snapshots it into the store
    and the aggregator republishes the newest snapshot.
    """

    # `configs` is typed loosely on purpose: the store holds whatever JSON
    # arrived, and one test feeds it malformed entries to prove they are
    # dropped rather than published.
    def _snapshot(self, received_at: str, configs: list) -> dict:
        return {
            "event": "expected_configs",
            "received_at": received_at,
            "configs": configs,
        }

    def test_absent_snapshot_publishes_an_empty_expectation(self):
        payload = agg.aggregate([perf_result()])
        assert payload["expected"] == {"recorded_at": "", "configs": []}

    def test_the_snapshot_is_published(self):
        configs = [{"workload": "wl-mi355x", "device": "mi355x", "conc": 64}]
        payload = agg.aggregate([perf_result(), self._snapshot("2026-01-05T00:00:00Z", configs)])
        assert payload["expected"]["configs"] == configs
        assert payload["expected"]["recorded_at"] == "2026-01-05T00:00:00Z"

    def test_the_newest_snapshot_wins(self):
        old = self._snapshot("2026-01-01T00:00:00Z", [{"workload": "old"}])
        new = self._snapshot("2026-02-01T00:00:00Z", [{"workload": "new"}])
        # Listed oldest-last to prove order in the store does not decide it.
        payload = agg.aggregate([new, old])
        assert payload["expected"]["configs"] == [{"workload": "new"}]

    def test_malformed_entries_are_dropped_rather_than_published(self):
        snapshot = self._snapshot("2026-01-05T00:00:00Z", [{"workload": "ok"}, "nonsense", 7])
        payload = agg.aggregate([snapshot])
        assert payload["expected"]["configs"] == [{"workload": "ok"}]

    def test_the_snapshot_is_not_mistaken_for_a_result(self):
        snapshot = self._snapshot("2026-01-05T00:00:00Z", [{"workload": "wl"}])
        payload = agg.aggregate([snapshot])
        assert payload["models"] == []
        assert payload["summary"]["nightlies"] == 0


class TestBaselinesArePublished:
    """The page labels and thresholds come from data, not hard-coded JS."""

    @pytest.fixture
    def payload(self):
        return agg.aggregate([perf_result()])

    def test_one_model_is_published(self, payload):
        assert [b["id"] for b in payload["baselines"]] == ["previous"]

    def test_default_baseline_names_a_published_model(self, payload):
        assert payload["default_baseline"] in {b["id"] for b in payload["baselines"]}

    def test_the_model_declares_what_the_ui_needs(self, payload):
        for baseline in payload["baselines"]:
            for field in ("id", "label", "kind", "baseline_window", "warn", "description"):
                assert field in baseline, baseline["id"]
            # Accuracy's zero is intended, so assert presence and type rather
            # than a positive floor.
            assert isinstance(baseline["warn"], float)
            assert isinstance(baseline["accuracy_abs"], float)

    def test_the_comparison_is_against_the_previous_run(self, payload):
        previous = next(b for b in payload["baselines"] if b["id"] == "previous")
        assert previous["kind"] == "previous"
        assert previous["baseline_window"] == 1
        assert previous["warn"] == agg.PERF_REL_THRESHOLD
        # No second level: a single threshold is the whole model.
        assert previous["alert"] is None

    def test_no_model_smooths_across_nightlies(self, payload):
        # Reducing measurement noise is the benchmark's job — perf-eval already
        # median-aggregates repeated runs before ingestion. Smoothing again here
        # would blur the night-to-night change this dashboard exists to show.
        for baseline in payload["baselines"]:
            assert baseline["kind"] == "previous", baseline["id"]
            assert baseline["baseline_window"] == 1, baseline["id"]

    def test_the_rationale_travels_with_the_data(self, payload):
        description = payload["baselines"][0]["description"].lower()
        assert "run before it" in description
        assert "neutral" in description
        assert "0.5%" in description


class TestMetricDisplayMetadata:
    @pytest.fixture
    def metric_meta(self):
        return agg.aggregate([perf_result()])["metric_meta"]

    def test_display_order_is_explicit(self, metric_meta):
        # The published JSON is key-sorted, so insertion order cannot survive
        # the round trip and the order has to be carried as a field.
        orders = [meta["order"] for meta in metric_meta.values()]
        assert len(orders) == len(set(orders))
        assert min(orders) == 0

    def test_headline_throughput_metrics_come_first(self, metric_meta):
        assert metric_meta["tput_per_gpu"]["order"] == 0
        assert metric_meta["output_tput_per_gpu"]["order"] == 1
        # Secondary percentiles sort after the headline set.
        assert metric_meta["median_tpot"]["order"] > metric_meta["mean_ttft"]["order"]

    def test_latency_metrics_declare_a_display_unit_and_scale(self, metric_meta):
        # A chart axis has to pick one unit for every point, so the conversion
        # is per metric rather than per value.
        ttft = metric_meta["mean_ttft"]
        assert ttft["unit"] == "s"
        assert ttft["display_unit"] == "ms"
        assert ttft["display_scale"] == 1000

    def test_throughput_metrics_need_no_conversion(self, metric_meta):
        assert "display_scale" not in metric_meta["tput_per_gpu"]

    def test_every_metric_declares_digits(self, metric_meta):
        for key, meta in metric_meta.items():
            assert isinstance(meta.get("digits"), int), key

    def test_accuracy_is_included_with_an_order(self, metric_meta):
        assert metric_meta["accuracy"]["direction"] == "higher"
        assert "order" in metric_meta["accuracy"]


class TestGeneratedAt:
    def test_is_utc_iso_with_a_trailing_z(self):
        stamp = agg.aggregate([perf_result()])["generated_at"]
        parsed = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        assert abs(datetime.now(UTC) - parsed) < timedelta(minutes=5)

    def test_pipeline_provenance_is_published(self):
        payload = agg.aggregate([perf_result()])
        assert payload["pipeline"]["org"] == "vllm"
        assert payload["pipeline"]["slug"] == "perf-eval"
