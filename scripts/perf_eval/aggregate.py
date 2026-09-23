#!/usr/bin/env python3
"""Fold the event store into the published ``perf_eval.json``.

Scope: AMD only, nightly only, re-applied here instead of trusted from ingest
so a stray event cannot widen what the page shows. Reads only the local store.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf_eval import (  # noqa: E402
    BUILDKITE_ORG,
    BUILDKITE_PIPELINE_SLUG,
    PIPELINE_URL,
    WINDOW_DAYS,
)
from perf_eval.normalize import (  # noqa: E402
    ACCURACY_DIRECTION,
    LM_EVAL_BACKENDS,
    METRIC_META,
    is_amd_workload,
    score_rows,
)
from perf_eval.store import (  # noqa: E402
    EXPECTED_CONFIGS_EVENT,
    RESULT_EVENTS,
    finished_at,
    nightly_identity,
    read_events_strict,
    received_at,
    write_json_atomic,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STORE = ROOT / "data" / "events.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "perf_eval.json"

# Smallest move that counts as a regression or improvement: 0.5% relative for
# perf, one point absolute for accuracy (0..1 scale). README: Regression detection.
PERF_REL_THRESHOLD = 0.005
ACCURACY_ABS_THRESHOLD = 0.01

# The regression rule, published so the page labels it from data.
BASELINES = [
    {
        "id": "previous",
        "label": "vs previous run",
        "short": "prev",
        "kind": "previous",
        "baseline_window": 1,
        "warn": PERF_REL_THRESHOLD,
        "alert": None,
        "accuracy_abs": ACCURACY_ABS_THRESHOLD,
        "description": (
            "Compares the newest nightly against the run before it. A perf "
            "metric counts as regressed or improved only when it moves by "
            f"{PERF_REL_THRESHOLD:.1%} or more, an accuracy score when it moves "
            f"by {ACCURACY_ABS_THRESHOLD * 100:g} point or more; smaller moves are "
            "neutral."
        ),
    },
]
DEFAULT_BASELINE = "previous"

_EPOCH = datetime.min.replace(tzinfo=UTC)


def _finished_at(result: dict) -> datetime:
    """When a result's nightly finished, sortable.

    Unparseable timestamps sort first rather than crashing the build, but they
    are logged: a silently mis-ordered series is far harder to notice than a
    warning in the collector output.
    """
    parsed = finished_at(result)
    if parsed is None:
        log.warning(
            "perf-eval result has no parseable date; sorting it oldest "
            "(event=%s model=%s build=%s)",
            result.get("event"),
            result.get("model"),
            result.get("build_number"),
        )
        return _EPOCH
    return parsed


def _provenance(event: dict) -> dict:
    return {
        "vllm_commit": (event.get("vllm_commit") or "").strip(),
        "build_commit": (event.get("build_commit") or "").strip(),
        "image": (event.get("image") or "").strip(),
        "build_url": event.get("build_url") or "",
        "build_number": event.get("build_number"),
    }


def _config_label(device: str, isl, osl, conc) -> str:
    def fmt_len(value):
        if value is None:
            return "?"
        return f"{value // 1024}K" if value and value % 1024 == 0 and value >= 1024 else str(value)

    return f"{fmt_len(isl)} in / {fmt_len(osl)} out @ conc {conc} ({(device or '').upper()})"


def _status(direction: str, latest: float, previous: float | None, *, rel: bool) -> dict:
    """Compute delta and red/green status for the latest-vs-previous nightly."""
    out = {
        "latest": latest,
        "previous": previous,
        "direction": direction,
        "delta": None,
        "delta_pct": None,
        "status": "neutral",
    }
    if previous is None:
        return out
    delta = latest - previous
    out["delta"] = delta
    if previous != 0:
        out["delta_pct"] = (delta / abs(previous)) * 100.0

    # Guarding on delta rather than on the threshold keeps a flat value neutral
    # even if a threshold is ever set to zero, where `abs(delta) >= 0` would
    # always be true.
    if delta == 0:
        return out
    if rel:
        moved = previous != 0 and abs(delta / previous) >= PERF_REL_THRESHOLD
    else:
        # Tolerance so a move of exactly one point is not lost to float error.
        moved = abs(delta) >= ACCURACY_ABS_THRESHOLD - 1e-9
    if not moved:
        return out

    improved = delta > 0 if direction == "higher" else delta < 0
    out["status"] = "good" if improved else "bad"
    return out


def _series_from_points(points: list[dict]) -> list[dict]:
    """One point per nightly, the newest winning by time (the store is not
    time-ordered), sorted oldest first."""
    by_night: dict[str, dict] = {}
    for point in points:
        current = by_night.get(point["nightly_key"])
        if current is None or point["_ts"] >= current["_ts"]:
            by_night[point["nightly_key"]] = point
    return sorted(by_night.values(), key=lambda point: point["_ts"])


def _strip_internal(series: list[dict]) -> list[dict]:
    return [
        {k: v for k, v in point.items() if not k.startswith("_") and k != "nightly_key"}
        for point in series
    ]


def build_perf_configs(perf_events: list[dict]) -> list[dict]:
    """Per-config metric series, a config being device, TP, precision and shape."""
    configs: dict[tuple, dict] = {}
    for event in perf_events:
        device = (event.get("device") or "").strip()
        isl, osl, conc = event.get("isl"), event.get("osl"), event.get("conc")
        tp, precision = event.get("tp"), event.get("precision") or ""
        key = (device, tp, precision, isl, osl, conc)
        config = configs.setdefault(
            key,
            {
                "device": device,
                "isl": isl,
                "osl": osl,
                "conc": conc,
                "tp": tp,
                "precision": precision,
                "label": _config_label(device, isl, osl, conc),
                "_metric_points": {},
            },
        )
        timestamp = _finished_at(event)
        night = nightly_identity(event)
        provenance = _provenance(event)
        for metric, value in (event.get("metrics") or {}).items():
            config["_metric_points"].setdefault(metric, []).append(
                {
                    "nightly_key": night,
                    "_ts": timestamp,
                    "date": event.get("date") or "",
                    "value": value,
                    "completed_requests": event.get("completed_requests"),
                    "failed_requests": event.get("failed_requests"),
                    **provenance,
                }
            )

    out = []
    for config in configs.values():
        metric_points = config.pop("_metric_points")
        metrics_out = {}
        for metric, points in metric_points.items():
            meta = METRIC_META.get(metric, {"direction": "higher"})
            series = _series_from_points(points)
            latest = series[-1]["value"]
            previous = series[-2]["value"] if len(series) >= 2 else None
            block = _status(meta["direction"], latest, previous, rel=True)
            block.update(
                {
                    "label": meta.get("label", metric),
                    "unit": meta.get("unit", ""),
                    "series": _strip_internal(series),
                }
            )
            metrics_out[metric] = block
        config["metrics"] = metrics_out
        out.append(config)
    # Stable, human-friendly ordering: device, then concurrency, then ISL/OSL.
    out.sort(
        key=lambda c: (
            c["device"],
            c.get("conc") or 0,
            c.get("isl") or 0,
            c.get("osl") or 0,
            c.get("tp") or 0,
            c["precision"],
        )
    )
    return out


def build_accuracy_tasks(eval_events: list[dict]) -> list[dict]:
    """Per-(workload, device, task, metric) accuracy series."""
    tasks: dict[tuple, dict] = {}
    for event in eval_events:
        timestamp = _finished_at(event)
        night = nightly_identity(event)
        provenance = _provenance(event)
        device = (event.get("device") or "").strip()
        workload = (event.get("workload") or "").strip()
        for row in score_rows(event.get("results") or []):
            key = (workload, device, row["task"], row["metric"])
            entry = tasks.setdefault(
                key,
                {
                    "workload": workload,
                    "device": device,
                    "task": row["task"],
                    "metric": row["metric"],
                    "primary": bool(row.get("primary")),
                    "_points": [],
                },
            )
            entry["primary"] = entry["primary"] or bool(row.get("primary"))
            entry["_points"].append(
                {
                    "nightly_key": night,
                    "_ts": timestamp,
                    "date": event.get("date") or "",
                    "value": row["value"],
                    **provenance,
                }
            )

    out = []
    for entry in tasks.values():
        series = _series_from_points(entry.pop("_points"))
        latest = series[-1]["value"]
        previous = series[-2]["value"] if len(series) >= 2 else None
        entry.update(_status(ACCURACY_DIRECTION, latest, previous, rel=False))
        entry["series"] = _strip_internal(series)
        out.append(entry)
    out.sort(key=lambda t: (not t["primary"], t["device"], t["workload"], t["task"], t["metric"]))
    return out


def _accuracy_model(event: dict, workload_models: dict[str, str]) -> str:
    """The model an accuracy event measured.

    Older events carry lm-eval's backend name (``local-completions``) instead;
    those resolve from the recipe, then from lm-eval's output directory.
    """
    model = (event.get("model") or "").strip()
    if model and model not in LM_EVAL_BACKENDS:
        return model
    workload = (event.get("workload") or "").strip()
    if workload_models.get(workload):
        return workload_models[workload]
    parts = (event.get("buildkite_artifact_path") or "").strip().lstrip("./").split("/")
    if len(parts) == 5 and "__" in parts[3]:
        return parts[3].replace("__", "/", 1)
    # The workload before the backend name: a backend name would fold every
    # unresolved workload into one "model".
    return workload or model


def _latest_identity(events: list[dict]) -> dict:
    if not events:
        return {}
    latest = max(events, key=_finished_at)
    return {
        "date": latest.get("date") or "",
        **_provenance(latest),
    }


def _expected_from_events(events: list[dict]) -> dict:
    """The newest expected-configs snapshot, for the coverage card."""
    newest: dict | None = None
    newest_at: datetime | None = None
    for event in events:
        if event.get("event") != EXPECTED_CONFIGS_EVENT:
            continue
        observed_at = received_at(event) or _EPOCH
        if newest_at is None or observed_at >= newest_at:
            newest, newest_at = event, observed_at
    if newest is None:
        return {"recorded_at": "", "configs": []}
    configs = [config for config in newest.get("configs") or [] if isinstance(config, dict)]
    return {"recorded_at": newest.get("received_at") or "", "configs": configs}


def _is_in_scope(event: dict) -> bool:
    """The AMD-only, nightly-only scope filter, re-applied at aggregation."""
    if event.get("event") not in RESULT_EVENTS:
        return False
    if event.get("nightly") is not True:
        return False
    return is_amd_workload(
        workload=event.get("workload"),
        image=event.get("image"),
        device=event.get("device"),
    )


def aggregate(events: list[dict]) -> dict:
    """Fold the event log into the frontend payload (AMD + nightly only)."""
    perf_by_model: dict[str, list[dict]] = {}
    eval_by_model: dict[str, list[dict]] = {}
    devices: set[str] = set()
    nightlies: set[str] = set()
    expected = _expected_from_events(events)
    workload_models = {
        str(config.get("workload")): str(config.get("model"))
        for config in expected["configs"]
        if config.get("workload") and config.get("model")
    }

    for event in events:
        if not _is_in_scope(event):
            continue
        if event["event"] == "perf_result":
            model = (event.get("model") or "").strip() or "(unknown model)"
            perf_by_model.setdefault(model, []).append(event)
        elif event["event"] == "accuracy_result":
            model = _accuracy_model(event, workload_models) or "(unknown model)"
            eval_by_model.setdefault(model, []).append(event)
        if event.get("device"):
            devices.add(event["device"])
        nightlies.add(nightly_identity(event))

    models = []
    perf_points = accuracy_points = 0
    for model in sorted(set(perf_by_model) | set(eval_by_model)):
        perf_events = perf_by_model.get(model, [])
        eval_events = eval_by_model.get(model, [])
        perf_configs = build_perf_configs(perf_events)
        accuracy_tasks = build_accuracy_tasks(eval_events)
        perf_points += sum(len(m["series"]) for c in perf_configs for m in c["metrics"].values())
        accuracy_points += sum(len(t["series"]) for t in accuracy_tasks)
        # One lookup per event, bound and then tested: a separate guard and
        # value expression could drift apart and let a None reach sorted().
        eval_devices = {
            device for event in eval_events if (device := (event.get("device") or "").strip())
        }
        model_devices = sorted({c["device"] for c in perf_configs if c["device"]} | eval_devices)
        models.append(
            {
                "model": model,
                "devices": model_devices,
                "latest": _latest_identity(perf_events + eval_events),
                "nightly_count": len({nightly_identity(e) for e in perf_events + eval_events}),
                "perf_configs": perf_configs,
                "accuracy_tasks": accuracy_tasks,
            }
        )

    # The published JSON is key-sorted, so insertion order cannot survive the
    # round trip. Publish the display order explicitly instead.
    metric_meta = {
        key: {**meta, "order": order} for order, (key, meta) in enumerate(METRIC_META.items())
    }
    metric_meta["accuracy"] = {
        "label": "Accuracy",
        "unit": "",
        "direction": ACCURACY_DIRECTION,
        "digits": 4,
        "order": len(METRIC_META),
    }

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": {
            "hardware": "amd",
            "runs": "nightly",
            "description": (
                "AMD (MI-series) workloads from scheduled nightly builds of the "
                "vllm/perf-eval pipeline. NVIDIA workloads and ad-hoc builds are "
                "deliberately excluded."
            ),
        },
        "pipeline": {
            "org": BUILDKITE_ORG,
            "slug": BUILDKITE_PIPELINE_SLUG,
            "url": PIPELINE_URL,
        },
        "metric_meta": metric_meta,
        "thresholds": {
            "perf_rel": PERF_REL_THRESHOLD,
            "accuracy_abs": ACCURACY_ABS_THRESHOLD,
        },
        "baselines": BASELINES,
        "default_baseline": DEFAULT_BASELINE,
        "expected": expected,
        "models": models,
        "summary": {
            "models": len(models),
            "amd_devices": sorted(devices),
            "nightlies": len(nightlies),
            "perf_points": perf_points,
            "accuracy_points": accuracy_points,
        },
    }


def build_payload(events: list[dict]) -> dict:
    """The published payload: the nightlies from the last WINDOW_DAYS.

    A nightly is in if its latest result is inside the window, so a nightly
    is never split across the edge.
    """
    window_start = datetime.now(UTC) - timedelta(days=WINDOW_DAYS)
    latest: dict[str, datetime] = {}
    for event in events:
        if _is_in_scope(event):
            identity = nightly_identity(event)
            latest[identity] = max(latest.get(identity, _EPOCH), _finished_at(event))
    published = {identity for identity, last in latest.items() if last >= window_start}

    payload = aggregate(
        [e for e in events if not _is_in_scope(e) or nightly_identity(e) in published]
    )
    payload["retention"] = {"display_window_days": WINDOW_DAYS}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="Path to events.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to perf_eval.json")
    args = parser.parse_args()

    events = read_events_strict(Path(args.store))
    payload = build_payload(events)
    out = Path(args.output)
    write_json_atomic(out, payload)
    log.info(
        "Wrote %s: %d models, %d nightlies, %d perf points, %d accuracy points",
        out,
        payload["summary"]["models"],
        payload["summary"]["nightlies"],
        payload["summary"]["perf_points"],
        payload["summary"]["accuracy_points"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
