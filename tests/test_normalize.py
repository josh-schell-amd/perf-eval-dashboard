"""Tests for the pure normalizers, including the AMD-only scope filter."""

from __future__ import annotations

import pytest

from perf_eval import normalize as nz


class TestAmdScopeFilter:
    @pytest.mark.parametrize("device", ["mi300x", "mi355x", "MI355X", " mi250 "])
    def test_amd_devices_match(self, device):
        assert nz.is_amd_device(device) is True
        assert nz.is_amd_workload(device=device) is True

    @pytest.mark.parametrize("device", ["h200", "b200", "a100", "", None])
    def test_nvidia_devices_excluded(self, device):
        assert nz.is_amd_device(device) is False
        assert nz.is_amd_workload(device=device) is False

    @pytest.mark.parametrize(
        "workload",
        ["minimax_m2_5_mi355x", "kimi_k3_mi355x", "deepseek_v4_pro-mi300x"],
    )
    def test_amd_workload_stems_match(self, workload):
        assert nz.is_amd_workload(workload=workload) is True

    @pytest.mark.parametrize("workload", ["gpt_oss_120b_h200", "glm_5_3_flash_b200"])
    def test_nvidia_workload_stems_excluded(self, workload):
        assert nz.is_amd_workload(workload=workload) is False

    def test_rocm_image_marks_amd(self):
        assert nz.is_amd_workload(image="vllm/vllm-openai-rocm:nightly-abc") is True
        assert nz.is_amd_workload(image="vllm/vllm-openai:nightly-abc") is False

    def test_any_single_amd_signal_is_enough(self):
        assert nz.is_amd_workload(workload="unknown", image="", device="mi355x") is True


class TestCommitFromImage:
    def test_nightly_tag(self):
        commit = "93d8f834dd8acf33eb0e2a75b2711b628cb6e226"
        assert nz.commit_from_image(f"vllm/vllm-openai-rocm:nightly-{commit}") == commit

    def test_embedded_long_hex(self):
        commit = "93d8f834dd8a"
        assert nz.commit_from_image(f"repo/img:v1-{commit}-rocm") == commit

    @pytest.mark.parametrize("image", ["", "repo/img", "repo/img:latest", "repo/img:v1.2.3"])
    def test_no_commit(self, image):
        assert nz.commit_from_image(image) == ""


class TestNumericGuards:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "x", None])
    def test_rejected(self, value):
        assert nz.to_float(value) is None

    def test_accepted(self):
        assert nz.to_float("1.5") == 1.5
        assert nz.to_int("7") == 7
        assert nz.to_int("x") is None


class TestTransformPerf:
    def test_per_gpu_division_and_unit_conversion(self):
        raw = {
            "total_token_throughput": 800.0,
            "output_throughput": 200.0,
            "mean_ttft_ms": 250.0,
            "mean_tpot_ms": 20.0,
        }
        metrics = nz.transform_perf(raw, tp=4)
        assert metrics["tput_per_gpu"] == 200.0
        assert metrics["output_tput_per_gpu"] == 50.0
        assert metrics["input_tput_per_gpu"] == 150.0
        assert metrics["mean_ttft"] == 0.25
        assert metrics["mean_tpot"] == 0.02
        # Interactivity is derived from TPOT as 1000 / tpot_ms.
        assert metrics["mean_intvty"] == 50.0

    def test_latency_keeps_sub_millisecond_precision(self):
        # One 0.1 ms step on a 5 ms TPOT is 2%, four times the regression threshold.
        metrics = nz.transform_perf({"mean_tpot_ms": 5.234}, tp=1)
        assert metrics["mean_tpot"] == pytest.approx(0.005234)

    def test_tp_zero_or_none_is_treated_as_one(self):
        raw = {"total_token_throughput": 10.0, "output_throughput": 4.0}
        assert nz.transform_perf(raw, tp=0)["tput_per_gpu"] == 10.0
        assert nz.transform_perf(raw, tp=None)["tput_per_gpu"] == 10.0

    def test_zero_tpot_yields_zero_interactivity_not_division_error(self):
        metrics = nz.transform_perf({"mean_tpot_ms": 0.0}, tp=1)
        assert metrics["mean_intvty"] == 0.0

    def test_unknown_metrics_are_dropped(self):
        metrics = nz.transform_perf({"some_other_ms": 5.0}, tp=1)
        assert "some_other" not in metrics

    def test_nan_latency_is_skipped(self):
        metrics = nz.transform_perf({"mean_ttft_ms": float("nan")}, tp=1)
        assert "mean_ttft" not in metrics


class TestAccuracyRows:
    def test_flattens_and_marks_primary(self):
        payload = {
            "data": {
                "results": {
                    "gsm8k": {
                        "exact_match,strict-match": 0.81,
                        "exact_match_stderr,strict-match": 0.01,
                        "alias": "gsm8k",
                    }
                }
            }
        }
        rows = nz.accuracy_rows(payload)
        assert rows == [
            {
                "task": "gsm8k",
                "metric": "exact_match,strict-match",
                "value": 0.81,
                "primary": True,
            }
        ]

    def test_without_a_preferred_metric_the_first_is_primary(self):
        payload = {"data": {"results": {"t": {"a": 0.1, "b": 0.2}}}}
        rows = nz.accuracy_rows(payload)
        assert [row["primary"] for row in rows] == [True, False]

    def test_sample_len_is_dropped_and_flexible_extract_leads(self):
        # The key order a real gsm8k artifact has: the question count first.
        payload = {
            "data": {
                "results": {
                    "gsm8k": {
                        "alias": "gsm8k",
                        "sample_len": 1319,
                        "exact_match,strict-match": 0.52,
                        "exact_match_stderr,strict-match": 0.01,
                        "exact_match,flexible-extract": 0.76,
                    }
                }
            }
        }
        rows = nz.accuracy_rows(payload)
        assert [(row["metric"], row["primary"]) for row in rows] == [
            ("exact_match,strict-match", False),
            ("exact_match,flexible-extract", True),
        ]


class TestModelFromEval:
    def test_reads_the_model_from_model_args_not_the_backend(self):
        payload = {
            "data": {
                "config": {
                    "model": "local-completions",
                    "model_args": "model=openai/gpt-oss-120b,base_url=http://x/v1/completions",
                }
            }
        }
        assert nz.model_from_eval(payload) == "openai/gpt-oss-120b"

    def test_pretrained_is_still_understood(self):
        payload = {"data": {"config": {"model": "hf", "model_args": "pretrained=org/M,dtype=auto"}}}
        assert nz.model_from_eval(payload) == "org/M"

    def test_a_backend_name_alone_is_not_a_model(self):
        payload = {"data": {"config": {"model": "local-completions"}}}
        assert nz.model_from_eval(payload) == ""


class TestNormalizeEvalPayload:
    def _payload(self, **overrides):
        payload = {
            "kind": "results",
            "workload": "test_8b_mi355x",
            "device": "mi355x",
            "image": "vllm/vllm-openai-rocm:nightly-abc123def456",
            "nightly": True,
            "data": {
                "config": {"model_name": "meta-llama/Test-8B"},
                "results": {"gsm8k": {"exact_match,strict-match": 0.8}},
            },
        }
        payload.update(overrides)
        return payload

    def test_canonical_shape(self):
        event = nz.normalize_eval_payload(self._payload())
        assert event is not None
        assert event["event"] == "accuracy_result"
        assert event["model"] == "meta-llama/Test-8B"
        assert event["nightly"] is True
        assert event["vllm_commit"] == "abc123def456"
        assert event["results"][0]["value"] == 0.8

    def test_nvidia_payload_is_dropped(self):
        payload = self._payload(
            workload="test_8b_h200",
            device="h200",
            image="vllm/vllm-openai:nightly-abc123def456",
        )
        assert nz.normalize_eval_payload(payload) is None

    def test_empty_results_dropped(self):
        assert nz.normalize_eval_payload(self._payload(data={"results": {}})) is None

    def test_wrong_kind_dropped(self):
        assert nz.normalize_eval_payload(self._payload(kind="samples")) is None

    def test_workload_is_model_fallback(self):
        payload = self._payload(data={"results": {"gsm8k": {"acc,none": 0.5}}})
        event = nz.normalize_eval_payload(payload)
        assert event is not None
        assert event["model"] == "test_8b_mi355x"

    def test_model_args_pretrained_fallback(self):
        payload = self._payload(
            data={
                "config": {"model_args": "pretrained=org/Model-9B,dtype=auto"},
                "results": {"gsm8k": {"acc,none": 0.5}},
            }
        )
        event = nz.normalize_eval_payload(payload)
        assert event is not None
        assert event["model"] == "org/Model-9B"


class TestMetricRegistry:
    def test_every_metric_declares_direction_label_and_unit(self):
        for key, meta in nz.METRIC_META.items():
            assert meta["direction"] in {"higher", "lower"}, key
            assert meta["label"], key
            assert "unit" in meta, key

    def test_throughput_is_higher_better_and_latency_lower_better(self):
        assert nz.METRIC_META["tput_per_gpu"]["direction"] == "higher"
        assert nz.METRIC_META["mean_ttft"]["direction"] == "lower"
        assert nz.METRIC_META["mean_tpot"]["direction"] == "lower"
        assert nz.METRIC_META["mean_intvty"]["direction"] == "higher"

    def test_accuracy_is_higher_better(self):
        assert nz.ACCURACY_DIRECTION == "higher"
