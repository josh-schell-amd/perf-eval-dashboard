#!/usr/bin/env python3
"""Merge a local and a published event store by result identity, atomically.

Duplicates are ordered by ``received_at``; two conflicting results recorded at
the same moment fail the merge instead of one silently winning.
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
    read_events_strict,
    received_at,
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
    """When the collector recorded a result (`received_at`)."""
    parsed = received_at(event)
    if parsed is None:
        raise ValueError(f"perf-eval result has no valid received_at: {event.get('received_at')!r}")
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

    kind = left.get("event")
    if kind == "perf_result":
        # Perf measurements are metrics, by name.
        left_metrics = left.get("metrics") or {}
        right_metrics = right.get("metrics") or {}
        for metric in sorted(set(left_metrics) & set(right_metrics)):
            _assert_compatible_value(
                left_metrics[metric],
                right_metrics[metric],
                identity=identity,
                field=f"metrics.{metric}",
            )
    elif kind == "accuracy_result":
        # Accuracy measurements are rows, by (task, metric).
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
    else:
        raise ValueError(f"not a result event: {kind!r}")


def _revision_order(event: dict) -> tuple[datetime, str]:
    # The canonical JSON only breaks ties, so equal-timestamp copies come out
    # in the same order whichever store listed them first.
    return _revision_timestamp(event), _canonical_event(event)


def _revisions_by_identity(events: list[dict]) -> dict[tuple, list[dict]]:
    """Every copy of each result, oldest-recorded first."""
    revisions_by_identity: dict[tuple, list[dict]] = {}
    for event in events:
        if event.get("event") in RESULT_EVENTS:
            revisions_by_identity.setdefault(result_identity(event), []).append(event)

    for identity, revisions in revisions_by_identity.items():
        revisions.sort(key=_revision_order)
        for _, tied in itertools.groupby(revisions, key=_revision_timestamp):
            for left, right in itertools.combinations(tied, 2):
                _assert_equal_revision_compatible(left, right, identity)
    return revisions_by_identity


def reconcile_events(local_events: list[dict], remote_events: list[dict]) -> list[dict]:
    """Both stores' events, with every copy of one result gathered where it first appears.

    Copies are ordered oldest-recorded first, so compaction's "later wins"
    means "recorded later", whichever store a copy came from.
    """
    combined = [*local_events, *remote_events]
    revisions_by_identity = _revisions_by_identity(combined)

    reconciled: list[dict] = []
    emitted: set[tuple] = set()
    for event in combined:
        if event.get("event") not in RESULT_EVENTS:
            reconciled.append(event)
            continue
        identity = result_identity(event)
        if identity not in emitted:
            reconciled.extend(revisions_by_identity[identity])
            emitted.add(identity)
    return reconciled


def merge_event_files(local_path: Path, remote_path: Path | None = None) -> int:
    """Merge remote history into local, replacing local only after validation."""
    local_events = read_events_strict(local_path)
    remote_events = read_events_strict(remote_path) if remote_path else []
    if remote_path is not None and not remote_events:
        raise ValueError(f"invalid perf-eval remote store {remote_path}: no events")

    return write_events_atomic(local_path, reconcile_events(local_events, remote_events))


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
