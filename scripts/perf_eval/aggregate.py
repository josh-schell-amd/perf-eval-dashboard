#!/usr/bin/env python3
"""Fold the perf-eval event store into the published ``perf_eval.json``.

SCOPE: **AMD only, nightly only.** Both filters are re-applied here rather
than trusted from ingest, so a hand-seeded or legacy event can never widen
what the dashboard presents. See ``README.md`` for why nightly-only is the
right window: an ad-hoc run may cover a single workload at one concurrency,
so mixing it into a trend line would make latest-vs-previous meaningless.

Design goals:

* **Self-updating workload set** — models, perf configs and accuracy tasks are
  discovered from the data, never hard-coded, so adding, removing or renaming
  a workload upstream is reflected automatically while older runs stay in each
  metric's time series.
* **Traceable provenance** — every series point keeps the vLLM commit, image
  and Buildkite build URL that produced it.
* **Data-driven framing** — each metric carries its ``direction`` (higher or
  lower is better) and a red/green ``status`` derived from the latest-versus-
  previous nightly delta.

This collector performs no network requests; it only reads the local store.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf_eval import BUILDKITE_ORG, BUILDKITE_PIPELINE_SLUG, PIPELINE_URL  # noqa: E402
from perf_eval.normalize import (  # noqa: E402
    ACCURACY_DIRECTION,
    LM_EVAL_BACKENDS,
    METRIC_META,
    is_amd_workload,
    score_rows,
)
from perf_eval.store import (  # noqa: E402
    ARTIFACT_IDENTITY_DAYS,
    EXPECTED_CONFIGS_EVENT,
    PERF_EVAL_HISTORY_DAYS,
    RESULT_EVENTS,
    SUMMARY_MAX_BYTES,
    encoded_json,
    enforced_byte_budget,
    event_datetime,
    nightly_identity,
    read_events_strict,
    write_json_atomic,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STORE = ROOT / "data" / "events.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "perf_eval.json"

# A perf metric has to move by at least 0.5% relative to count as a regression
# or improvement. Below that, the movement is still shown as a number but is
# neutral.
#
# Chosen, not measured: the AMD workloads run at `repetitions: 1`, so there is
# no run-to-run spread to derive a floor from. With no floor at all, about half
# of the regressions flagged on a typical night were under 0.5%, down to 0.015%,
# and latencies stored at 0.1 ms resolution make one rounding step on a fast
# metric look like a regression. The page states this threshold wherever it
# reports a count, so what it hides is never invisible.
#
# Once `repetitions: 3` lands on the AMD recipes upstream, the spread across
# those repetitions is a measurable noise floor and should replace this value.
#
# Accuracy scores are on a 0..1 scale and must move by 0.01 (one point)
# absolute. They are not reproducible night to night: every AMD workload's
# gsm8k score moves every night, and one gsm8k question out of 1319 is worth
# 0.00076. The expected run-to-run spread on gsm8k is about 0.007, so a point
# sits just above it.
PERF_REL_THRESHOLD = 0.005
ACCURACY_ABS_THRESHOLD = 0.01

# How many days the dashboard renders. Published in the payload so the page
# and this module cannot drift apart, and so retention can guarantee the page
# is never handed less history than it claims to show.
DISPLAY_WINDOW_DAYS = 14

# The regression model, published so the page labels it from data rather than
# hard-coding the rule. A list of one, because the structure makes adding a
# second model later a data change rather than a frontend change.
#
# Why only "newest versus the run before it", and no smoothing across
# nightlies: reducing measurement noise is the benchmark's job, not the
# dashboard's. perf-eval already does it properly — ``lib/aggregate_perf.py``
# repeats a benchmark on the same warm server ``repetitions`` times and
# median-aggregates every numeric field before ingestion. Smoothing again here
# would blur the night-to-night change this dashboard exists to show, while
# only pretending to fix noise that belongs upstream. If a metric is too noisy
# to compare night to night, the fix is a higher ``repetitions`` in the
# workload recipe.
#
# The page recomputes this from the series inside the time window; the
# per-metric ``status`` on each metric block is the same rule applied over
# whole history, kept for machine consumers of this JSON.
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


def _parse_ts(event: dict) -> datetime:
    """Sortable timestamp for an event.

    Unparseable timestamps sort first rather than crashing the build, but they
    are logged: a silently mis-ordered series is far harder to notice than a
    warning in the collector output.
    """
    parsed = event_datetime(event)
    if parsed is None:
        log.warning(
            "perf-eval event has no parseable timestamp; sorting it oldest "
            "(event=%s model=%s build=%s)",
            event.get("event"),
            event.get("model"),
            event.get("build_number"),
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
    """One entry per nightly, newest observation winning, sorted oldest-first.

    Resolution is by timestamp, not by position in the event store. The store
    is append-ordered rather than time-ordered, so picking the last matching
    row would let a re-ingested older observation silently override a newer
    one for the same nightly.
    """
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
    """Group perf events into per-config metric time series.

    Keyed on TP and precision as well as shape, the same identity the page
    uses: a TP 4 and a TP 8 recipe for one model and device are two series.
    """
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
        timestamp = _parse_ts(event)
        night = nightly_identity(event)
        provenance = _provenance(event)
        for metric, value in (event.get("metrics") or {}).items():
            config["_metric_points"].setdefault(metric, []).append(
                {
                    "nightly_key": night,
                    "_ts": timestamp,
                    "date": event.get("date") or "",
                    "value": value,
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
    """Group accuracy events into per-(device, task, metric) time series.

    Keyed on device as well as task: one model can run on several AMD devices
    (MiniMax-M2.5 on mi300x and mi355x), and those are separate series, not
    two observations of one.
    """
    tasks: dict[tuple, dict] = {}
    for event in eval_events:
        timestamp = _parse_ts(event)
        night = nightly_identity(event)
        provenance = _provenance(event)
        device = (event.get("device") or "").strip()
        workload = (event.get("workload") or "").strip()
        for row in score_rows(event.get("results") or []):
            # Workload too: two recipes for one model and device would
            # otherwise share a series and each nightly would keep one of them.
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
                    "date": event.get("date") or event.get("received_at") or "",
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

    Events collected before the collector read the model id from the recipe
    carry lm-eval's backend name (``local-completions``) instead, which would
    fold every workload into one "model". Those are resolved from the recipe
    expectation for their workload, then from lm-eval's own output directory
    (``results/<workload>/<task>/<org>__<name>/results_*.json``).
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
    latest = max(events, key=_parse_ts)
    return {
        "date": latest.get("date") or latest.get("received_at") or "",
        **_provenance(latest),
    }


def _expected_from_events(events: list[dict]) -> dict:
    """The newest recipe-derived expectation, for the coverage card.

    Published separately from the results because it answers a different
    question: not "what did we measure" but "what should have been measured".
    Coverage compares the two, which is the only way a workload that has never
    reported can show up as missing.
    """
    newest: dict | None = None
    newest_at: datetime | None = None
    for event in events:
        if event.get("event") != EXPECTED_CONFIGS_EVENT:
            continue
        observed_at = _parse_ts(event)
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


def aggregate(events: list[dict], *, generated_at: datetime | None = None) -> dict:
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
        if event["event"] == "accuracy_result":
            model = _accuracy_model(event, workload_models) or "(unknown model)"
        else:
            model = (event.get("model") or "").strip() or "(unknown model)"
        if event.get("device"):
            devices.add(event["device"])
        nightlies.add(nightly_identity(event))
        if event["event"] == "perf_result":
            perf_by_model.setdefault(model, []).append(event)
        else:
            eval_by_model.setdefault(model, []).append(event)

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
        "generated_at": (generated_at or datetime.now(UTC))
        .astimezone(UTC)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
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


def _retention_candidates(available: int, floor: int) -> list[int]:
    """Descending nightly counts to try, always ending at ``floor``.

    Starts by publishing everything and halves down, so a payload that fits
    needs one pass and a large one converges in a handful. There is no fixed
    ceiling: the only limits are what the store holds and the byte budget.
    """
    candidates: list[int] = []
    limit = max(available, floor)
    while limit > floor:
        candidates.append(limit)
        limit = max(floor, limit // 2)
    candidates.append(floor)
    return candidates


def bounded_aggregate(
    events: list[dict],
    *,
    generated_at: datetime | None = None,
    max_bytes: int = SUMMARY_MAX_BYTES,
    display_window_days: int = DISPLAY_WINDOW_DAYS,
) -> dict:
    """Build a payload, shedding only history the dashboard does not render.

    Publishes as much history as fits the byte budget, starting from
    everything the store holds. History is shed a whole nightly at a time, so
    a partial nightly is never presented as a complete comparison.

    Nightlies inside the display window are never shed. The window is a
    promise the page makes to the reader; quietly publishing less than it
    would make the page show a shorter history than it claims, with nothing
    on screen saying so. If even the window cannot fit, this raises rather
    than publishing a payload that silently under-delivers.
    """
    max_bytes = enforced_byte_budget(max_bytes, cap=SUMMARY_MAX_BYTES)
    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC)
    nightly_latest: dict[str, datetime] = {}
    for event in events:
        if not _is_in_scope(event):
            continue
        identity = nightly_identity(event)
        observed_at = _parse_ts(event)
        nightly_latest[identity] = max(nightly_latest.get(identity, observed_at), observed_at)
    ordered_nightlies = [
        identity
        for identity, _ in sorted(nightly_latest.items(), key=lambda item: (item[1], item[0]))
    ]

    window_start = timestamp - timedelta(days=display_window_days)
    protected = {
        identity for identity, last_seen in nightly_latest.items() if last_seen >= window_start
    }

    payload: dict | None = None
    for limit in _retention_candidates(len(ordered_nightlies), len(protected)):
        allowed = set(ordered_nightlies[-limit:]) | protected
        selected = [
            event
            for event in events
            if not _is_in_scope(event) or nightly_identity(event) in allowed
        ]
        payload = aggregate(selected, generated_at=timestamp)
        payload["retention"] = {
            "display_window_days": display_window_days,
            "event_history_days": PERF_EVAL_HISTORY_DAYS,
            "artifact_identity_days": ARTIFACT_IDENTITY_DAYS,
            "max_bytes": max_bytes,
            "nightlies_available": len(ordered_nightlies),
            "nightlies_published": len(allowed),
            "trimmed": len(allowed) < len(ordered_nightlies),
            # Deliberately not publishing the in-window nightly count. It is
            # derived from "now" rather than from the data, so it drifts as the
            # window slides and would trigger a deploy every day with no new
            # results — the same problem `generated_at` causes, which the
            # deploy gate exists to suppress. The page counts in-window
            # nightlies itself for the footer.
        }
        if len(encoded_json(payload)) <= max_bytes:
            return payload

    required = len(encoded_json(payload)) if payload is not None else 0
    raise RuntimeError(
        f"perf_eval.json needs {required} bytes for just the {len(protected)} nightlies "
        f"inside the {display_window_days}-day display window, over the {max_bytes} byte "
        "budget. Raise SUMMARY_MAX_BYTES or narrow DISPLAY_WINDOW_DAYS — do not let the "
        "published payload fall short of what the page renders."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="Path to events.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to perf_eval.json")
    args = parser.parse_args()

    events = read_events_strict(Path(args.store))
    payload = bounded_aggregate(events, generated_at=datetime.now(UTC))
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
