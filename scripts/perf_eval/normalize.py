"""Pure normalizers turning raw perf-eval results into canonical events.

SCOPE: **AMD only.** ``is_amd_workload`` is the single named predicate that
enforces it, and every normalizer in this module returns ``None`` for a
non-AMD run. NVIDIA workloads (H200/B200/A100) execute in the same upstream
pipeline; they are dropped here rather than downstream so a NVIDIA result can
never reach the event store. Nightly filtering is a *build*-level concern and
lives in ``collect_artifacts.is_nightly_build``.

Everything here is deliberately free of I/O and module-level state so it can
be unit-tested without a live Buildkite or GitHub.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

# AMD GPU device tags. perf-eval's ``parse_workload.py`` emits ``gpu.lower()``
# as the device, e.g. ``mi355x`` / ``mi300x``. NVIDIA tags (``h200`` / ``b200``
# / ``a100``) deliberately do not match.
_AMD_DEVICE_RE = re.compile(r"^mi\d+", re.IGNORECASE)
# Workload stems encode the hardware suffix (``minimax_m2_5_mi355x``).
_AMD_WORKLOAD_RE = re.compile(r"(?:^|[_-])mi\d+[a-z]?(?:$|[_-])", re.IGNORECASE)

# Perf metrics we surface, with display metadata.
#
# ``direction`` drives the red/green arrow logic in the frontend: ``higher``
# means a larger value is better (throughput), ``lower`` means smaller is
# better (latency). Embedding it in the published payload keeps the frontend
# data-driven, so a new metric renders with the right higher/lower-is-better
# hint automatically.
#
# ``unit`` is the *canonical* unit the value is stored in, matching perf-eval's
# ``ingest_perf.py`` (latencies in seconds). ``display_unit`` and
# ``display_scale`` say how to present it, because a chart axis has to pick one
# unit for every point rather than switching per value. Ratios are
# scale-invariant, so regression detection is unaffected either way.
#
# Insertion order is the display order: the headline metrics that answer "is
# tonight's build healthy" come first, the secondary percentiles after.
# ``aggregate.py`` turns this into an explicit ``order`` field, because the
# published JSON is key-sorted and cannot carry insertion order.
METRIC_META: dict[str, dict] = {
    "tput_per_gpu": {
        "label": "Total tok/s/GPU",
        "unit": "tok/s",
        "direction": "higher",
        "digits": 1,
    },
    "output_tput_per_gpu": {
        "label": "Output tok/s/GPU",
        "unit": "tok/s",
        "direction": "higher",
        "digits": 1,
    },
    "mean_intvty": {
        "label": "Interactivity",
        "unit": "tok/s/user",
        "direction": "higher",
        "digits": 1,
    },
    "mean_ttft": {
        "label": "Mean TTFT",
        "unit": "s",
        "direction": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 0,
    },
    "p99_ttft": {
        "label": "P99 TTFT",
        "unit": "s",
        "direction": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 0,
    },
    "mean_tpot": {
        "label": "Mean TPOT",
        "unit": "s",
        "direction": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 2,
    },
    "mean_e2el": {
        "label": "Mean E2E latency",
        "unit": "s",
        "direction": "lower",
        "digits": 2,
    },
    "mean_itl": {
        "label": "Mean ITL",
        "unit": "s",
        "direction": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 2,
    },
    "median_ttft": {
        "label": "Median TTFT",
        "unit": "s",
        "direction": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 0,
    },
    "median_tpot": {
        "label": "Median TPOT",
        "unit": "s",
        "direction": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 2,
    },
    "p99_tpot": {
        "label": "P99 TPOT",
        "unit": "s",
        "direction": "lower",
        "display_unit": "ms",
        "display_scale": 1000,
        "digits": 2,
    },
    "input_tput_per_gpu": {
        "label": "Input tok/s/GPU",
        "unit": "tok/s",
        "direction": "higher",
        "digits": 1,
    },
}

# Accuracy is always "higher is better" and lives on a 0..1 scale.
ACCURACY_DIRECTION = "higher"


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


def commit_from_image(image: str) -> str:
    """Extract a vLLM commit SHA embedded in an image tag, if present.

    Mirrors the regex in perf-eval's ``parse_workload.py`` so the dashboard
    derives the same commit the pipeline tagged the run with.
    """
    if not image:
        return ""
    _, sep, tag = image.rpartition(":")
    if not sep:
        return ""
    tag = tag.split("@", 1)[0]
    match = re.match(r"nightly-([0-9a-f]{7,40})(?:[-_.].*)?$", tag, re.IGNORECASE) or re.search(
        r"(?:^|[-_.])([0-9a-f]{12,40})(?:$|[-_.])", tag, re.IGNORECASE
    )
    return match.group(1) if match else ""


def is_amd_device(device: str | None) -> bool:
    return bool(device) and bool(_AMD_DEVICE_RE.match(str(device).strip()))


def is_amd_workload(
    workload: str | None = None,
    image: str | None = None,
    device: str | None = None,
) -> bool:
    """True if any signal marks this run as AMD/ROCm.

    AMD if the device is an MIxxx tag, the workload stem carries an MIxxx
    suffix, or the image is a ROCm image. Everything else (H200/B200/A100) is
    treated as NVIDIA and excluded from this dashboard.

    Any one signal is sufficient, because a given artifact may carry only some
    of the three.
    """
    return (
        is_amd_device(device)
        or bool(workload and _AMD_WORKLOAD_RE.search(str(workload)))
        or bool(image and "rocm" in str(image).lower())
    )


def build_identity(payload: dict) -> dict:
    """Pull a compact build-identity block out of a result payload."""
    image = (payload.get("image") or "").strip()
    commit = (payload.get("vllm_commit") or "").strip() or commit_from_image(image)
    return {
        "build_number": to_int(payload.get("buildkite_build_number")),
        "build_url": payload.get("buildkite_build_url") or "",
        "build_commit": payload.get("buildkite_commit") or "",
        "branch": payload.get("buildkite_branch") or "",
        "image": image,
        "vllm_commit": commit,
    }


def perf_metrics(payload: dict) -> dict[str, float]:
    """Extract the registry metrics present in an already-canonical payload."""
    out: dict[str, float] = {}
    for key in METRIC_META:
        value = to_float(payload.get(key))
        if value is not None:
            out[key] = value
    return out


def transform_perf(raw: dict, *, tp: int | None) -> dict[str, float]:
    """Turn a raw ``vllm bench serve`` result into canonical per-GPU metrics.

    Mirrors perf-eval's ``ingest_perf.transform``: aggregate throughput is
    divided by ``tp`` for the per-GPU columns, ``*_ms`` latencies are converted
    to seconds, and interactivity (tok/s) is derived from TPOT. Only metrics in
    ``METRIC_META`` survive, so an upstream field addition cannot silently
    widen the published schema.

    ``tp`` is optional because it comes from a workload recipe that may omit
    it; a missing or zero value is treated as 1 rather than dividing by zero.
    """
    tp = max(int(tp or 1), 1)
    total = to_float(raw.get("total_token_throughput")) or 0.0
    output = to_float(raw.get("output_throughput")) or 0.0
    metrics: dict[str, float] = {
        "tput_per_gpu": total / tp,
        "output_tput_per_gpu": output / tp,
        "input_tput_per_gpu": (total - output) / tp,
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
    return {k: round(v, 4) for k, v in metrics.items() if k in METRIC_META}


# lm-eval client backends. lm-eval's ``config.model`` holds one of these, not
# the model under test, so it must never be read as a model id.
LM_EVAL_BACKENDS = frozenset(
    {
        "local-completions",
        "local-chat-completions",
        "openai-completions",
        "openai-chat-completions",
        "vllm",
        "sglang",
        "hf",
    }
)

# Keys in an lm-eval task block that are bookkeeping rather than scores.
# ``sample_len`` is the number of questions (1319 for gsm8k).
_NON_SCORE_METRICS = frozenset({"alias", "sample_len"})

# The metric headlined per task, in preference order; otherwise the first
# score. Flexible extract leads because strict match also grades the answer
# format: on gsm8k, gpt-oss-120b scores about 0.52 strict against 0.76
# flexible, while other models agree within a point.
PRIMARY_METRIC_PREFERENCE = (
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "acc_norm,none",
    "acc,none",
)


def model_from_eval(payload: dict) -> str:
    """Best-effort model id from an lm-eval ``results`` payload."""
    data = payload.get("data") or {}
    config = data.get("config") or {}
    args = config.get("model_args")
    if isinstance(args, str):
        match = re.search(r"(?:^|,)\s*(?:model|pretrained)=([^,]+)", args)
        if match:
            return match.group(1).strip()
    elif isinstance(args, dict):
        for key in ("model", "pretrained"):
            if args.get(key):
                return str(args[key]).strip()
    for candidate in (config.get("model_name"), data.get("model_name")):
        if candidate and str(candidate) not in LM_EVAL_BACKENDS:
            return str(candidate)
    return ""


def is_score_metric(metric: str) -> bool:
    key = str(metric)
    return key not in _NON_SCORE_METRICS and "stderr" not in key


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
        "model": model_from_eval(payload) or workload,
        "workload": workload,
        "task": (payload.get("task") or "").strip(),
        "device": device,
        **identity,
        "results": rows,
    }
