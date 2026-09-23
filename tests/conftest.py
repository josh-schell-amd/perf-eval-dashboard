"""Shared event builders for the perf-eval collector tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

ROCM_IMAGE = "vllm/vllm-openai-rocm"
CUDA_IMAGE = "vllm/vllm-openai"

# The code keeps the last WINDOW_DAYS by the real clock, so fixtures are dated
# relative to one instant: events built "at the same time" stay identical.
STARTED = datetime.now(UTC).replace(microsecond=0)


def days_ago(days: float) -> str:
    """A result ``date``, in Buildkite's format."""
    return (STARTED - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def received_days_ago(days: float) -> str:
    """A ``received_at`` stamp, in the collector's format."""
    return (STARTED - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def image_for(device: str, commit: str) -> str:
    """Realistic image URI for a device.

    The AMD filter is an OR across device, workload stem and image, so a
    fixture claiming an NVIDIA device must also carry a non-ROCm image or it
    is not actually exercising the exclusion.
    """
    repo = ROCM_IMAGE if device.lower().startswith("mi") else CUDA_IMAGE
    return f"{repo}:nightly-{commit}"


def perf_result(
    *,
    commit: str = "a" * 40,
    value: float = 100.0,
    metrics: dict | None = None,
    date: str | None = None,
    received_at: str | None = None,
    device: str = "mi355x",
    model: str = "meta-llama/Test-8B",
    build_number: int = 1,
    # Typed as object, not bool: events carry whatever JSON arrived, and tests
    # deliberately pass non-boolean values to prove only a literal True counts.
    nightly: object = True,
    isl: int = 1024,
    osl: int = 1024,
    conc: int = 128,
    tp: int = 8,
) -> dict:
    return {
        "event": "perf_result",
        "received_at": received_at or received_days_ago(1),
        "nightly": nightly,
        "model": model,
        "device": device,
        "precision": "fp8",
        "tp": tp,
        "isl": isl,
        "osl": osl,
        "conc": conc,
        "date": date or days_ago(1),
        "build_number": build_number,
        "build_url": f"https://buildkite.com/vllm/perf-eval/builds/{build_number}",
        "build_commit": "",
        "branch": "main",
        "image": image_for(device, commit),
        "vllm_commit": commit,
        "metrics": metrics if metrics is not None else {"tput_per_gpu": value},
    }


def accuracy_result(
    *,
    commit: str = "a" * 40,
    value: float = 0.80,
    task: str = "gsm8k",
    metric: str = "exact_match,strict-match",
    date: str | None = None,
    received_at: str | None = None,
    device: str = "mi355x",
    model: str = "meta-llama/Test-8B",
    workload: str | None = None,
    build_number: int = 1,
    nightly: object = True,
) -> dict:
    # The workload stem encodes the hardware, so it tracks the device unless a
    # test overrides it deliberately.
    workload = workload if workload is not None else f"test_8b_{device}"
    return {
        "event": "accuracy_result",
        "received_at": received_at or received_days_ago(1),
        "nightly": nightly,
        "model": model,
        "workload": workload,
        "task": task,
        "device": device,
        "date": date or days_ago(1),
        "build_number": build_number,
        "build_url": f"https://buildkite.com/vllm/perf-eval/builds/{build_number}",
        "build_commit": "",
        "branch": "main",
        "image": image_for(device, commit),
        "vllm_commit": commit,
        "results": [{"task": task, "metric": metric, "value": value, "primary": True}],
    }
