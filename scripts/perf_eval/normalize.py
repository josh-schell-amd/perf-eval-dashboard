"""Pure functions turning raw perf-eval results into canonical events.

Scope: AMD only. ``is_amd_workload`` is the one predicate for it, and every
normalizer returns None for an NVIDIA run, so none reaches the store.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

# Device tags as perf-eval emits them (``mi355x``, ``mi300x``).
_AMD_DEVICE_RE = re.compile(r"^mi\d+", re.IGNORECASE)
# Workload stems encode the hardware suffix (``minimax_m2_5_mi355x``).
_AMD_WORKLOAD_RE = re.compile(r"(?:^|[_-])mi\d+[a-z]?(?:$|[_-])", re.IGNORECASE)

# Published perf metrics, in display order. ``unit`` is how the value is stored
# (latencies in seconds, as perf-eval does); ``display_*`` is how it is shown.
# Throughputs are divided by TP at ingest, so the unit carries the /GPU rather
# than the label: ATOM publishes the same metrics unnormalized, and a bare
# "tok/s" here would read as comparable to its numbers when it is TP times
# smaller.
METRIC_META: dict[str, dict] = {
    "tput_per_gpu": {
        "label": "Total Throughput",
        "unit": "tok/s/GPU",
        "better": "higher",
        "digits": 1,
    },
    "output_tput_per_gpu": {
        "label": "Output Throughput",
        "unit": "tok/s/GPU",
        "better": "higher",
        "digits": 1,
    },
    "mean_intvty": {
        "label": "Interactivity",
        "unit": "tok/s/user",
        "better": "higher",
        "digits": 1,
    },
    "mean_ttft": {
        "label": "Mean TTFT",
        "unit": "s",
        "better": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 0,
    },
    "p99_ttft": {
        "label": "P99 TTFT",
        "unit": "s",
        "better": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 0,
    },
    "mean_tpot": {
        "label": "Mean TPOT",
        "unit": "s",
        "better": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 2,
    },
    "mean_itl": {
        "label": "Mean ITL",
        "unit": "s",
        "better": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 2,
    },
    "median_ttft": {
        "label": "Median TTFT",
        "unit": "s",
        "better": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 0,
    },
    "median_tpot": {
        "label": "Median TPOT",
        "unit": "s",
        "better": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 2,
    },
    "p99_tpot": {
        "label": "P99 TPOT",
        "unit": "s",
        "better": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 2,
    },
    "input_tput_per_gpu": {
        "label": "Input Throughput",
        "unit": "tok/s/GPU",
        "better": "higher",
        "digits": 1,
    },
}

# Accuracy is always "higher is better" and lives on a 0..1 scale.
ACCURACY_BETTER = "higher"


def utcnow_iso() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    """Parse a float, rejecting NaN and infinities.

    A NaN would serialize as invalid JSON and an infinity would poison every
    delta computed against it, so both are treated as missing.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def is_amd_device(device: str | None) -> bool:
    return bool(device) and bool(_AMD_DEVICE_RE.match(str(device).strip()))


def is_amd_workload(
    workload: str | None = None,
    image: str | None = None,
    device: str | None = None,
) -> bool:
    """True if the device, workload stem or image marks the run as AMD.

    Any one is enough: an artifact may carry only some of the three.
    """
    return (
        is_amd_device(device)
        or bool(workload and _AMD_WORKLOAD_RE.search(str(workload)))
        or bool(image and "rocm" in str(image).lower())
    )


def build_identity(payload: dict) -> dict:
    """Pull a compact build-identity block out of a result payload."""
    return {
        "build_number": to_int(payload.get("buildkite_build_number")),
        "build_url": payload.get("buildkite_build_url") or "",
        "build_commit": payload.get("buildkite_commit") or "",
        "branch": payload.get("buildkite_branch") or "",
        "image": (payload.get("image") or "").strip(),
        "vllm_commit": (payload.get("vllm_commit") or "").strip(),
    }


# vLLM's world size (vllm/config/parallel.py): each of these multiplies the GPUs
# a server uses. Decode context and expert parallelism reuse those GPUs.
_GPU_FACTORS = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "prefill_context_parallel_size",
    "data_parallel_size",
)
_SIZE_LABELS = {
    "tensor_parallel_size": "TP",
    "pipeline_parallel_size": "PP",
    "prefill_context_parallel_size": "PCP",
    "decode_context_parallel_size": "DCP",
    "data_parallel_size": "DP",
}
_FLAG_LABELS = {"enable_expert_parallel": "EP"}


def parallelism_of(record: dict) -> dict:
    """A result's or expected config's parallelism, by vLLM flag name, defaults left out.

    Records written before the map existed carry only ``tp``.
    """
    if isinstance(record.get("parallelism"), dict):
        return record["parallelism"]
    tp = to_int(record.get("tp")) or 1
    return {"tensor_parallel_size": tp} if tp > 1 else {}


def parallel_key(parallelism: dict) -> tuple:
    return tuple(sorted(parallelism.items()))


def gpu_count(parallelism: dict) -> int:
    count = 1
    for factor in _GPU_FACTORS:
        count *= max(to_int(parallelism.get(factor)) or 1, 1)
    return count


def parallel_label(parallelism: dict) -> str:
    """``TP8``, or ``TP4×DP2 · EP``; a flag with no short name shows as itself."""
    sizes = [f"TP{parallelism.get('tensor_parallel_size', 1)}"] + [
        f"{label}{parallelism[name]}"
        for name, label in _SIZE_LABELS.items()
        if name != "tensor_parallel_size" and name in parallelism
    ]
    flags = [label for name, label in _FLAG_LABELS.items() if parallelism.get(name) is True]
    others = [
        name if value is True else f"{name}={value}"
        for name, value in sorted(parallelism.items())
        if name not in _SIZE_LABELS and name not in _FLAG_LABELS
    ]
    return " · ".join(["×".join(sizes), *flags, *others])


def perf_metrics(payload: dict) -> dict[str, float]:
    """Extract the registry metrics present in an already-canonical payload."""
    out: dict[str, float] = {}
    for key in METRIC_META:
        value = to_float(payload.get(key))
        if value is not None:
            out[key] = value
    return out


def transform_perf(raw: dict, *, gpus: int | None) -> dict[str, float]:
    """Turn a raw ``vllm bench serve`` result into per-GPU metrics.

    Mirrors perf-eval's ``ingest_perf.transform``, except that it divides by TP x DP
    and this by every GPU the server uses (``gpu_count``); they differ only once a
    recipe uses PP or PCP. Only ``METRIC_META`` metrics are kept, so a new upstream
    field cannot widen the published schema.
    """
    gpus = max(int(gpus or 1), 1)
    total = to_float(raw.get("total_token_throughput")) or 0.0
    output = to_float(raw.get("output_throughput")) or 0.0
    metrics: dict[str, float] = {
        "tput_per_gpu": total / gpus,
        "output_tput_per_gpu": output / gpus,
        "input_tput_per_gpu": (total - output) / gpus,
    }
    for key, value in raw.items():
        if not isinstance(key, str) or not key.endswith("_ms"):
            continue
        millis = to_float(value)
        if millis is None:
            continue
        base = key[: -len("_ms")]
        metrics[base] = millis / 1000.0
        if "tpot" in base:
            metrics[base.replace("tpot", "intvty")] = 1000.0 / millis if millis else 0.0
    return {k: v for k, v in metrics.items() if k in METRIC_META}


# The number of questions in an lm-eval task block (1319 for gsm8k), not a score.
_SAMPLE_COUNT = "sample_len"

# The metric headlined per task, in preference order, else the first score.
# Flexible extract first: strict match also grades the answer format.
PRIMARY_METRIC_PREFERENCE = (
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "acc_norm,none",
    "acc,none",
)


def is_score_metric(metric: str) -> bool:
    """False for the numbers lm-eval reports beside each score: its standard
    error (``exact_match_stderr,strict-match``) and the question count."""
    key = str(metric)
    return key != _SAMPLE_COUNT and "stderr" not in key


def score_rows(rows: list[dict]) -> list[dict]:
    """Keep only score rows and flag one headline metric per task.

    Returns new dicts, so it is safe to apply to rows read from the store.
    Applied again at aggregation, so events stored before a rule change are
    judged by the current rule rather than the one they were written with.
    """
    kept = [dict(row) for row in rows if is_score_metric(row.get("metric", ""))]
    by_task: dict[str, list[dict]] = {}
    for row in kept:
        by_task.setdefault(row["task"], []).append(row)
    for task_rows in by_task.values():
        metrics = [row["metric"] for row in task_rows]
        chosen = next((m for m in PRIMARY_METRIC_PREFERENCE if m in metrics), metrics[0])
        for row in task_rows:
            row["primary"] = row["metric"] == chosen
    return kept


def accuracy_rows(payload: dict) -> list[dict]:
    """Flatten lm-eval ``results`` into ``{task, metric, value, primary}`` rows.

    lm-eval reports each task as a dict of ``metric,filter`` keys. We keep the
    numeric score metrics and flag one per task as ``primary`` so the frontend
    can headline a single score without hard-coding metric names.
    """
    data = payload.get("data") or {}
    results = data.get("results") or {}
    rows: list[dict] = []
    for task_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for raw_key, raw_value in metrics.items():
            value = to_float(raw_value)
            if value is None:
                continue
            rows.append({"task": str(task_name), "metric": str(raw_key), "value": value})
    return score_rows(rows)


def normalize_eval_payload(payload: dict) -> dict | None:
    """Canonicalize an lm-eval ``results`` push.

    Returns ``None`` for a non-AMD workload or an empty result set.
    """
    if not isinstance(payload, dict) or payload.get("kind") != "results":
        return None
    workload = (payload.get("workload") or "").strip()
    image = (payload.get("image") or "").strip()
    device = (payload.get("device") or "").strip()
    if not is_amd_workload(workload=workload, image=image, device=device):
        return None
    rows = accuracy_rows(payload)
    if not rows:
        return None
    identity = build_identity(payload)
    return {
        "event": "accuracy_result",
        "received_at": utcnow_iso(),
        "nightly": bool(payload.get("nightly")),
        "model": (payload.get("model") or "").strip(),
        "workload": workload,
        "task": (payload.get("task") or "").strip(),
        "device": device,
        **identity,
        "results": rows,
    }
