"""Tests for the Buildkite artifact collector's pure logic.

No network is touched: every helper exercised here is I/O-free by design.
"""

from __future__ import annotations

import pytest

from perf_eval import collect_artifacts as ca

COMMIT = "93d8f834dd8acf33eb0e2a75b2711b628cb6e226"
NIGHTLY_MESSAGE = f"Nightly run 2026-06-30: commit {COMMIT}"


def build(**overrides):
    payload = {
        "number": 42,
        "branch": "main",
        "message": NIGHTLY_MESSAGE,
        "state": "finished",
        "source": "schedule",
        "web_url": "https://buildkite.com/vllm/perf-eval/builds/42",
        "commit": "f" * 40,
        "env": {},
    }
    payload.update(overrides)
    return payload


class TestNightlyScopeFilter:
    def test_structured_message_is_a_nightly(self):
        assert ca.is_nightly_build(build()) is True

    def test_nightly_env_on_main_is_a_nightly(self):
        assert ca.is_nightly_build(build(message="manual", env={"NIGHTLY": "1"})) is True

    def test_scheduled_build_mentioning_nightly_is_a_nightly(self):
        assert ca.is_nightly_build(build(message="nightly sweep", source="schedule")) is True

    def test_adhoc_build_is_not_a_nightly(self):
        assert ca.is_nightly_build(build(message="debug run", source="ui")) is False

    def test_nightly_env_off_main_is_not_a_nightly(self):
        assert (
            ca.is_nightly_build(build(branch="feature/x", message="manual", env={"NIGHTLY": "1"}))
            is False
        )

    def test_scheduled_mention_off_main_is_not_a_nightly(self):
        assert (
            ca.is_nightly_build(build(branch="pr-1", message="nightly", source="schedule")) is False
        )

    def test_structured_message_wins_even_off_main(self):
        # The structured message carries the nightly date and commit, so it is
        # authoritative regardless of branch.
        assert ca.is_nightly_build(build(branch="release")) is True

    def test_non_nightly_returns_no_info(self):
        assert ca.nightly_info(build(message="debug", source="ui")) is None


class TestNightlyInfo:
    def test_extracts_the_commit_from_the_message(self):
        info = ca.nightly_info(build())
        assert info == {"vllm_commit": COMMIT, "branch": "main"}

    def test_falls_back_to_vllm_commit_env(self):
        info = ca.nightly_info(build(message="manual", env={"NIGHTLY": "1", "VLLM_COMMIT": COMMIT}))
        assert info is not None
        assert info["vllm_commit"] == COMMIT

    def test_falls_back_to_the_image_tag(self):
        info = ca.nightly_info(
            build(
                message="manual",
                env={"NIGHTLY": "1", "VLLM_IMAGE": f"vllm/vllm-openai-rocm:nightly-{COMMIT}"},
            )
        )
        assert info is not None
        assert info["vllm_commit"] == COMMIT


class TestAmdImage:
    def test_prefers_an_explicit_rocm_image(self):
        image = "myrepo/vllm-rocm:custom"
        assert ca.amd_image({"VLLM_IMAGE": image}, COMMIT) == image

    def test_ignores_a_cuda_image_and_synthesizes_a_rocm_one(self):
        assert ca.amd_image({"VLLM_IMAGE": "vllm/vllm-openai:nightly"}, COMMIT) == (
            f"{ca.AMD_IMAGE_REPO}:nightly-{COMMIT}"
        )

    def test_without_a_commit_falls_back_to_the_bare_tag(self):
        assert ca.amd_image({}, "") == f"{ca.AMD_IMAGE_REPO}:nightly"


class TestClassifyArtifact:
    @pytest.mark.parametrize("prefix", ["", "./"])
    def test_perf_bench_artifact(self, prefix):
        result = ca.classify_artifact(
            f"{prefix}results/minimax_m2_5-mi355x/bench-8k-in-1k-out.json"
        )
        assert result == ("perf", "minimax_m2_5-mi355x", "8k-in-1k-out")

    def test_accuracy_artifact(self):
        result = ca.classify_artifact("results/minimax_m2_5-mi355x/gsm8k/results_2026-01-01.json")
        assert result == ("accuracy", "minimax_m2_5-mi355x", "gsm8k")

    @pytest.mark.parametrize("prefix", ["", "./"])
    def test_accuracy_artifact_in_lm_evals_model_directory(self, prefix):
        # The layout a real nightly produces: lm-eval nests its output under a
        # directory named after the sanitized model id.
        result = ca.classify_artifact(
            f"{prefix}results/gpt_oss_120b-mi355x/gsm8k/openai__gpt-oss-120b/"
            "results_2026-09-22T13-04-18.930123.json"
        )
        assert result == ("accuracy", "gpt_oss_120b-mi355x", "gsm8k")

    @pytest.mark.parametrize(
        "path",
        [
            "results/wl/gsm8k/samples_gsm8k.jsonl",  # large, never downloaded
            "results/wl/gsm8k/model/samples_gsm8k_2026-01-01.jsonl",
            "results/wl/gsm8k/a/b/results_2026-01-01.json",  # deeper than lm-eval writes
            "results/wl/aiperf-profile/profile_export.json",
            "logs/build.txt",
            "",
        ],
    )
    def test_unrecognized_paths_are_skipped(self, path):
        assert ca.classify_artifact(path) is None


class TestParseTp:
    @pytest.mark.parametrize(
        "serve_args,expected",
        [
            ("--tensor-parallel-size 8", 8),
            ("--tensor-parallel-size=8", 8),
            ("-tp 4", 4),
            ("--tp 2", 2),
            ("--tensor-parallel-size 4 --data-parallel-size 2", 8),
            ("", 1),
            ("--tensor-parallel-size abc", 1),
            ("--tensor-parallel-size", 1),
        ],
    )
    def test_effective_parallel_degree(self, serve_args, expected):
        assert ca.parse_tp(serve_args) == expected


class TestPrecisionFromModel:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("org/Model-FP8", "fp8"),
            ("org/model-fp4-instruct", "fp4"),
            ("org/model-int4", "int4"),
            ("org/Model-Instruct", "bf16"),
        ],
    )
    def test_inferred_from_the_model_id(self, model, expected):
        assert ca.precision_from_model(model) == expected


class TestWorkloadEntry:
    def test_projects_recipe_fields_and_configs(self):
        entry, configs = ca.workload_entry(
            {
                "name": "minimax_m2_5-mi355x",
                "gpu": "MI355X",
                "vllm": {
                    "model": "org/Model-FP8",
                    "serve_args": "--tensor-parallel-size 8",
                },
                "vllm_bench": {
                    "configs": [
                        {
                            "name": "8k-in-1k-out",
                            "input_len": 8192,
                            "output_len": 1024,
                            "max_concurrency": 128,
                        }
                    ]
                },
            }
        )
        assert entry["device"] == "mi355x"
        assert entry["tp"] == 8
        assert entry["precision"] == "fp8"
        # Keyed on the expanded run name. perf-eval suffixes every run with
        # -conc-<value>, even a single value, and the artifact is named after
        # the run — so keying on the bare name would miss on every lookup and
        # silently drop ISL/OSL from every result.
        assert configs["8k-in-1k-out-conc-128"] == {"isl": 8192, "osl": 1024, "conc": 128}
        assert "8k-in-1k-out" not in configs

    def test_metadata_overrides_inference(self):
        entry, _ = ca.workload_entry(
            {
                "name": "wl",
                "gpu": "MI355X",
                "vllm": {"model": "org/Model", "serve_args": "--tensor-parallel-size 8"},
                "vllm_bench": {"metadata": {"device": "mi300x", "tp": 4, "precision": "mxfp4"}},
            }
        )
        assert (entry["device"], entry["tp"], entry["precision"]) == ("mi300x", 4, "mxfp4")

    def test_unnamed_configs_are_skipped(self):
        _, configs = ca.workload_entry(
            {"name": "wl", "gpu": "MI355X", "vllm_bench": {"configs": [{"input_len": 1}]}}
        )
        assert configs == {}

    def test_a_concurrency_sweep_expands_to_one_config_per_value(self):
        # Every AMD recipe sweeps concurrency, so this is the normal case, not
        # an edge case. Each swept run is a separate artifact.
        _, configs = ca.workload_entry(
            {
                "name": "wl-mi355x",
                "gpu": "MI355X",
                "vllm_bench": {
                    "configs": [
                        {
                            "name": "1k-in-1k-out",
                            "input_len": 1024,
                            "output_len": 1024,
                            "num_prompts": [10, 256, 512],
                            "max_concurrency": [1, 64, 128],
                        }
                    ]
                },
            }
        )
        assert set(configs) == {
            "1k-in-1k-out-conc-1",
            "1k-in-1k-out-conc-64",
            "1k-in-1k-out-conc-128",
        }
        # Shape is carried onto every swept run, not just the first.
        for run, config in configs.items():
            assert config["isl"] == 1024, run
            assert config["osl"] == 1024, run
        assert configs["1k-in-1k-out-conc-64"]["conc"] == 64

    def test_nightly_is_captured_for_the_expectation(self):
        entry, _ = ca.workload_entry({"name": "wl", "gpu": "MI355X", "nightly": True})
        assert entry["nightly"] is True
        entry, _ = ca.workload_entry({"name": "wl", "gpu": "MI355X"})
        assert entry["nightly"] is False

    def test_the_declared_accuracy_tasks_are_captured(self):
        entry, _ = ca.workload_entry(
            {
                "name": "wl",
                "gpu": "MI355X",
                "lm_eval": {"timeout": 6000, "tasks": [{"name": "gsm8k", "num_fewshot": 5}]},
            }
        )
        assert entry["accuracy_tasks"] == ["gsm8k"]


class TestLmEvalTasks:
    """The recipe's `lm_eval.tasks`, which name the accuracy results expected."""

    def test_task_names_are_read_from_the_mappings(self):
        recipe = {"lm_eval": {"tasks": [{"name": "gsm8k"}, {"name": "mmlu"}]}}
        assert ca.lm_eval_tasks(recipe) == ["gsm8k", "mmlu"]

    def test_a_recipe_without_lm_eval_expects_no_accuracy(self):
        assert ca.lm_eval_tasks({"name": "wl"}) == []
        assert ca.lm_eval_tasks({"lm_eval": None}) == []
        assert ca.lm_eval_tasks({"lm_eval": {"timeout": 60}}) == []

    def test_a_bare_string_task_is_accepted(self):
        assert ca.lm_eval_tasks({"lm_eval": {"tasks": ["gsm8k"]}}) == ["gsm8k"]

    def test_unnamed_and_malformed_tasks_are_skipped(self):
        recipe = {"lm_eval": {"tasks": [{"num_fewshot": 5}, {"name": "  "}, 7, None, "gsm8k"]}}
        assert ca.lm_eval_tasks(recipe) == ["gsm8k"]

    def test_duplicates_collapse_and_order_is_stable(self):
        recipe = {"lm_eval": {"tasks": [{"name": "mmlu"}, {"name": "gsm8k"}, {"name": "mmlu"}]}}
        assert ca.lm_eval_tasks(recipe) == ["gsm8k", "mmlu"]


class TestExpectedAccuracy:
    """Accuracy coverage is measured against the recipes, like perf coverage.

    Inferring it from the last WINDOW_DAYS of results meant a workload whose
    lm-eval step had been failing for longer than the window dropped out of its
    own denominator, so the page stopped reporting it missing at exactly the
    point the outage became serious.
    """

    def _recipe(self, name, device, *, nightly=True, tasks=("gsm8k",)):
        return ca.workload_entry(
            {
                "name": name,
                "gpu": device.upper(),
                "nightly": nightly,
                "vllm": {"model": "org/Model-FP8", "serve_args": "--tensor-parallel-size 8"},
                "lm_eval": {"tasks": [{"name": t} for t in tasks]},
            }
        )

    def test_one_entry_per_declared_task(self):
        expected = ca.expected_accuracy(
            {"wl-mi355x": self._recipe("wl-mi355x", "mi355x", tasks=("gsm8k", "mmlu"))}
        )
        assert [e["task"] for e in expected] == ["gsm8k", "mmlu"]

    def test_carries_the_fields_coverage_matches_on(self):
        (expected,) = ca.expected_accuracy({"wl-mi355x": self._recipe("wl-mi355x", "mi355x")})
        assert expected == {
            "workload": "wl-mi355x",
            "model": "org/Model-FP8",
            "device": "mi355x",
            "task": "gsm8k",
        }

    def test_a_workload_without_lm_eval_expects_no_accuracy(self):
        recipes = {"wl-mi355x": self._recipe("wl-mi355x", "mi355x", tasks=())}
        assert ca.expected_accuracy(recipes) == []

    def test_it_shares_the_scope_filters_with_perf(self):
        non_nightly = {"wl-mi355x": self._recipe("wl-mi355x", "mi355x", nightly=False)}
        assert ca.expected_accuracy(non_nightly) == []
        nvidia = {"wl-h200": self._recipe("wl-h200", "h200")}
        assert ca.expected_accuracy(nvidia) == []

    def test_output_is_deterministic(self):
        recipes = {
            "b-mi355x": self._recipe("b-mi355x", "mi355x"),
            "a-mi300x": self._recipe("a-mi300x", "mi300x"),
        }
        assert ca.expected_accuracy(recipes) == ca.expected_accuracy(recipes)
        assert [e["workload"] for e in ca.expected_accuracy(recipes)][0] == "a-mi300x"

    def test_a_workload_broken_all_window_is_still_expected(self):
        # The regression this guards: expectation must not depend on results.
        recipes = {"wl-mi355x": self._recipe("wl-mi355x", "mi355x")}
        assert [e["task"] for e in ca.expected_accuracy(recipes)] == ["gsm8k"]


class TestExpectedConfigs:
    """Coverage is measured against the recipes, not against recent reporting.

    An expectation derived from recent data forgets whatever has been absent
    long enough, so the longer a workload stays broken the healthier the
    dashboard would claim to be. The recipes never decay.
    """

    def _recipe(self, name, device, *, nightly=True, concurrencies=(64, 128)):
        return ca.workload_entry(
            {
                "name": name,
                "gpu": device.upper(),
                "nightly": nightly,
                "vllm": {"model": "org/Model-FP8", "serve_args": "--tensor-parallel-size 8"},
                "vllm_bench": {
                    "configs": [
                        {
                            "name": "1k-in-1k-out",
                            "input_len": 1024,
                            "output_len": 1024,
                            "max_concurrency": list(concurrencies),
                        }
                    ]
                },
            }
        )

    def test_one_entry_per_expanded_run(self):
        expected = ca.expected_configs({"wl-mi355x": self._recipe("wl-mi355x", "mi355x")})
        assert len(expected) == 2
        assert {e["conc"] for e in expected} == {64, 128}

    def test_carries_the_fields_coverage_matches_on(self):
        expected = ca.expected_configs({"wl-mi355x": self._recipe("wl-mi355x", "mi355x")})
        for field in (
            "workload",
            "run",
            "model",
            "device",
            "precision",
            "tp",
            "isl",
            "osl",
            "conc",
        ):
            assert field in expected[0], field

    def test_non_nightly_recipes_are_not_expected(self):
        recipes = {"wl-mi355x": self._recipe("wl-mi355x", "mi355x", nightly=False)}
        assert ca.expected_configs(recipes) == []

    def test_nvidia_recipes_are_not_expected(self):
        recipes = {"wl-h200": self._recipe("wl-h200", "h200")}
        assert ca.expected_configs(recipes) == []

    def test_output_is_deterministic(self):
        recipes = {
            "b-mi355x": self._recipe("b-mi355x", "mi355x"),
            "a-mi300x": self._recipe("a-mi300x", "mi300x"),
        }
        assert ca.expected_configs(recipes) == ca.expected_configs(recipes)
        # Sorted by workload, so a diff of the published payload stays readable.
        assert [e["workload"] for e in ca.expected_configs(recipes)][0] == "a-mi300x"


class TestPerfEvent:
    def _identity(self):
        return {
            "build_number": 42,
            "build_url": "https://buildkite.com/vllm/perf-eval/builds/42",
            "build_commit": "f" * 40,
            "branch": "main",
            "vllm_commit": COMMIT,
            "date": "2026-06-30T04:00:00Z",
            "image": f"vllm/vllm-openai-rocm:nightly-{COMMIT}",
        }

    def _entry(self, **overrides):
        entry = {
            "name": "wl-mi355x",
            "device": "mi355x",
            "tp": 4,
            "precision": "fp8",
            "model": "org/Model",
        }
        entry.update(overrides)
        return entry

    def test_canonical_perf_event(self):
        raw = {
            "model_id": "org/Model",
            "total_token_throughput": 800.0,
            "output_throughput": 200.0,
            "max_concurrency": 64,
        }
        event = ca.perf_event(
            raw,
            entry=self._entry(),
            config={"isl": 8192, "osl": 1024, "conc": 128},
            identity=self._identity(),
        )
        assert event is not None
        assert event["event"] == "perf_result"
        assert event["nightly"] is True
        assert event["metrics"]["tput_per_gpu"] == 200.0
        assert event["conc"] == 128
        assert event["vllm_commit"] == COMMIT

    def test_request_counts_are_recorded(self):
        raw = {"total_token_throughput": 800.0, "completed": 500, "failed": 12}
        event = ca.perf_event(raw, entry=self._entry(), config={}, identity=self._identity())
        assert event is not None
        assert (event["completed_requests"], event["failed_requests"]) == (500.0, 12.0)

    def test_concurrency_falls_back_to_the_raw_result(self):
        event = ca.perf_event(
            {"total_token_throughput": 10.0, "max_concurrency": 64},
            entry=self._entry(),
            config={},
            identity=self._identity(),
        )
        assert event is not None
        assert event["conc"] == 64

    def test_nvidia_entry_is_dropped(self):
        identity = self._identity()
        identity["image"] = "vllm/vllm-openai:nightly"
        event = ca.perf_event(
            {"total_token_throughput": 10.0},
            entry=self._entry(name="wl-h200", device="h200"),
            config={},
            identity=identity,
        )
        assert event is None

    @pytest.mark.parametrize(
        "raw",
        [
            {"model_id": "org/Model"},  # nothing numeric at all
            {"error": "server crashed", "completed": 0},
            {"total_token_throughput": 0.0, "mean_ttft_ms": 120.0},
            {"total_token_throughput": float("nan")},
        ],
    )
    def test_a_failed_benchmark_is_skipped_not_published_as_zero(self, raw):
        # Zero throughput would read as a 100% regression tonight and a false
        # recovery tomorrow.
        event = ca.perf_event(raw, entry=self._entry(), config={}, identity=self._identity())
        assert event is None


class TestAccuracyEvent:
    def test_model_comes_from_the_recipe_not_lm_evals_backend(self):
        results = {
            "config": {
                "model": "local-completions",
                "model_args": "model=openai/gpt-oss-120b,base_url=http://x/v1/completions",
            },
            "results": {"gsm8k": {"sample_len": 1319, "exact_match,flexible-extract": 0.76}},
        }
        event = ca.accuracy_event(
            results,
            workload="gpt_oss_120b-mi355x",
            task="gsm8k",
            entry={"device": "mi355x", "model": "openai/gpt-oss-120b"},
            identity={"image": f"{ca.AMD_IMAGE_REPO}:nightly-{COMMIT}", "build_number": 1},
        )
        assert event is not None
        assert event["model"] == "openai/gpt-oss-120b"
        assert [row["metric"] for row in event["results"]] == ["exact_match,flexible-extract"]


class TestEventKey:
    def test_perf_key_separates_configs(self):
        base = {
            "event": "perf_result",
            "build_number": 1,
            "model": "m",
            "device": "mi355x",
            "isl": 1,
            "osl": 1,
        }
        assert ca.event_key({**base, "conc": 1}) != ca.event_key({**base, "conc": 2})

    def test_perf_key_separates_tp_variants_of_one_shape(self):
        base = {
            "event": "perf_result",
            "build_number": 1,
            "model": "m",
            "device": "mi355x",
            "isl": 1,
            "osl": 1,
            "conc": 1,
        }
        assert ca.event_key({**base, "tp": 4}) != ca.event_key({**base, "tp": 8})

    def test_accuracy_key_folds_task_rows(self):
        event = {
            "event": "accuracy_result",
            "build_number": 1,
            "workload": "wl",
            "results": [
                {"task": "gsm8k", "metric": "acc,none"},
                {"task": "gsm8k", "metric": "exact_match,strict-match"},
            ],
        }
        # Row order must not change the identity.
        reversed_event = {**event, "results": list(reversed(event["results"]))}
        assert ca.event_key(event) == ca.event_key(reversed_event)


class TestArtifactProvenance:
    def test_captures_stable_fields_and_normalizes_them(self):
        provenance = ca.artifact_provenance(
            {
                "id": "artifact-1",
                "job_id": "job-1",
                "path": "./results/wl/bench-a.json",
                "sha1sum": "ABCDEF",
                "download_url": "https://example.invalid/presigned?sig=secret",
            },
            42,
        )
        assert provenance["buildkite_artifact_path"] == "results/wl/bench-a.json"
        assert provenance["buildkite_artifact_sha1"] == "abcdef"

    def test_never_persists_a_presigned_download_url(self):
        provenance = ca.artifact_provenance(
            {"id": "a", "download_url": "https://example.invalid/presigned?sig=secret"}, 1
        )
        assert not any("sig=secret" in str(value) for value in provenance.values())
        assert "download_url" not in provenance


class TestLookbackGuard:
    @pytest.mark.parametrize("days", [0, -1, 31, 999])
    def test_out_of_range_lookback_is_rejected(self, days, tmp_path):
        with pytest.raises(ValueError, match="lookback must be between"):
            ca.collect(tmp_path / "events.jsonl", days=days, bk_token="t", gh_token="")
