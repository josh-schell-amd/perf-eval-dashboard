#!/usr/bin/env python3
"""Atomically merge a local and a published perf-eval event store.

Stores are merged by result identity, not line count: compaction folds
duplicates, so a newer store can hold fewer lines. Duplicate results are
ordered by when they were recorded, and two conflicting results recorded at
the same moment fail the merge rather than one silently winning. The store's
own writer then compacts the result. With only --local, this just validates
and compacts that store.

No network requests.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf_eval.store import (  # noqa: E402
    RESULT_EVENTS,
    event_time,
    parse_time,
    read_events_strict,
    result_identity,
    write_events_atomic,
)

_TIMESTAMP_FIELDS = frozenset({"date", "received_at"})
_ARTIFACT_FIELDS = frozenset(
    {
        "buildkite_artifact_id",
        "buildkite_artifact_job_id",
        "buildkite_artifact_path",
        "buildkite_artifact_sha1",
    }
)


def _canonical_event(event: dict) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":"))


def _revision_timestamp(event: dict) -> datetime:
    """When a result was recorded (`received_at`), else when its nightly ran."""
    if event.get("received_at"):
        parsed = parse_time(event["received_at"])
        if parsed is None:
            raise ValueError(f"perf-eval result has invalid received_at: {event['received_at']!r}")
        return parsed
    parsed = event_time(event)
    if parsed is None:
        raise ValueError("perf-eval result has no valid received_at or date")
    return parsed


def _meaningful(value) -> bool:
    return value not in (None, "", [], {})


def _assert_compatible_value(left, right, *, identity: tuple, field: str) -> None:
    if _meaningful(left) and _meaningful(right) and left != right:
        raise ValueError(
            "equal-timestamp perf-eval conflict for "
            f"{identity!r} field {field!r}: {left!r} != {right!r}"
        )


def _accuracy_rows(event: dict) -> dict[tuple[str, str], dict]:
    return {
        (str(row.get("task") or ""), str(row.get("metric") or "")): row
        for row in event.get("results") or []
        if isinstance(row, dict)
    }


def _assert_equal_revision_compatible(left: dict, right: dict, identity: tuple) -> None:
    ignored = _TIMESTAMP_FIELDS | _ARTIFACT_FIELDS | {"metrics", "results"}
    if left.get("event") == "accuracy_result":
        # One workload emits a separate event per task, while the stable store
        # identity deliberately folds those task rows into one nightly result.
        ignored = ignored | {"task"}
    for field in sorted((set(left) & set(right)) - ignored):
        _assert_compatible_value(left[field], right[field], identity=identity, field=field)

    if left.get("event") == "perf_result":
        left_metrics = left.get("metrics") or {}
        right_metrics = right.get("metrics") or {}
        for metric in sorted(set(left_metrics) & set(right_metrics)):
            _assert_compatible_value(
                left_metrics[metric],
                right_metrics[metric],
                identity=identity,
                field=f"metrics.{metric}",
            )
        return

    left_rows = _accuracy_rows(left)
    right_rows = _accuracy_rows(right)
    for row_key in sorted(set(left_rows) & set(right_rows)):
        left_row = left_rows[row_key]
        right_row = right_rows[row_key]
        for field in sorted(set(left_row) & set(right_row)):
            _assert_compatible_value(
                left_row[field],
                right_row[field],
                identity=identity,
                field=f"results.{row_key[0]}.{row_key[1]}.{field}",
            )


def reconcile_events(local_events: list[dict], remote_events: list[dict]) -> list[dict]:
    """Union stores with source-neutral, timestamp-monotonic result ordering."""
    combined = [*local_events, *remote_events]
    groups: dict[tuple, list[dict]] = {}
    for event in combined:
        if event.get("event") in RESULT_EVENTS:
            groups.setdefault(result_identity(event), []).append(event)

    ordered_groups: dict[tuple, list[dict]] = {}
    for identity, events in groups.items():
        stamped = [(_revision_timestamp(event), _canonical_event(event), event) for event in events]
        stamped.sort(key=lambda row: (row[0], row[1]))
        for _, equal_group in itertools.groupby(stamped, key=lambda row: row[0]):
            equal_events = [row[2] for row in equal_group]
            for left, right in itertools.combinations(equal_events, 2):
                _assert_equal_revision_compatible(left, right, identity)
        ordered_groups[identity] = [row[2] for row in stamped]

    reconciled: list[dict] = []
    emitted: set[tuple] = set()
    for event in combined:
        if event.get("event") not in RESULT_EVENTS:
            reconciled.append(event)
            continue
        identity = result_identity(event)
        if identity in emitted:
            continue
        reconciled.extend(ordered_groups[identity])
        emitted.add(identity)
    return reconciled


def merge_event_files(
    local_path: Path,
    remote_path: Path | None = None,
    *,
    now: datetime | None = None,
) -> int:
    """Merge remote history into local, replacing local only after validation."""
    local_events = read_events_strict(local_path)
    remote_events = read_events_strict(remote_path) if remote_path else []
    if remote_path is not None and not remote_events:
        raise ValueError(f"invalid perf-eval remote store {remote_path}: no events")

    return write_events_atomic(local_path, reconcile_events(local_events, remote_events), now=now)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True, help="Local events.jsonl")
    parser.add_argument("--remote", type=Path, help="Published events.jsonl")
    args = parser.parse_args()

    count = merge_event_files(args.local, args.remote)
    print(f"Merged perf-eval event store: {count} records -> {args.local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
