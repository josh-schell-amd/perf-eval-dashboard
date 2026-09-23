"""Atomic event store for perf-eval results.

``events.jsonl`` is a JSONL log of canonical events. Every write goes through
:func:`write_events_atomic`, which applies retention (a fixed number of days)
and replaces the file atomically, so a crash mid-write never leaves a
truncated log.

Both budgets are ceilings that fail the write, never targets that trim to fit:

* :data:`EVENTS_MAX_BYTES` for the unpublished event log.
* :data:`SUMMARY_MAX_BYTES` for the published ``perf_eval.json``.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Ceilings, far above normal use (30 days of the event log is about 1 MB). A
# write that would exceed one fails instead of dropping history to fit.
EVENTS_MAX_BYTES = 24 * 1024 * 1024
SUMMARY_MAX_BYTES = 8 * 1024 * 1024

# The collector can re-scan up to 30 days of Buildkite builds, so results are
# kept that long: dropping them sooner would let a backfill download them
# again. The page itself shows 14 days.
MAX_ARTIFACT_LOOKBACK_DAYS = 30
PERF_EVAL_HISTORY_DAYS = MAX_ARTIFACT_LOOKBACK_DAYS
# The newest nightlies are kept whatever their age, so if the nightly stops,
# the page can still say how old the last run is.
PERF_EVAL_MIN_NIGHTLIES = 2

# Downloaded-artifact identities outlive the results a little, so pruning a
# result never makes its artifact look new to the collector.
ARTIFACT_IDENTITY_DAYS = 45

ARTIFACT_MARKER_EVENT = "buildkite_artifact_ingested"
ARTIFACT_INDEX_EVENT = "buildkite_artifact_identity_index"
ARTIFACT_INDEX_SCHEMA_VERSION = 1

# A snapshot of the configs the upstream recipes say should run, recorded on
# each collection so coverage can be measured against what is *expected*
# rather than against what happened to report recently. Only the newest
# snapshot is retained: it describes the recipes as they are now, and an older
# one would resurrect workloads that have since been removed upstream.
EXPECTED_CONFIGS_EVENT = "expected_configs"

RESULT_EVENTS = frozenset({"perf_result", "accuracy_result"})

# Days to keep non-nightly and bookkeeping events.
AUXILIARY_EVENT_DAYS = 30

_EPOCH = datetime.datetime.min.replace(tzinfo=datetime.UTC)


def enforced_byte_budget(requested: int, *, cap: int) -> int:
    """Clamp a requested budget so a caller can never raise the hard cap.

    Tests pass small budgets to exercise retention; nothing may pass a budget
    larger than the cap the store was designed around.
    """
    if requested <= 0:
        raise ValueError("perf-eval byte budget must be positive")
    return min(requested, cap)


def event_datetime(event: dict) -> datetime.datetime | None:
    """Best-effort UTC timestamp for an event, or None if nothing parses."""
    for raw in (
        event.get("date"),
        event.get("finished_at"),
        event.get("created_at"),
        event.get("received_at"),
        event.get("generated_at"),
    ):
        if not raw:
            continue
        text = str(raw).strip().replace("Z", "+00:00")
        for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = (
                    datetime.datetime.fromisoformat(text)
                    if fmt is None
                    else datetime.datetime.strptime(text, fmt)
                )
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.UTC)
            return parsed.astimezone(datetime.UTC)
    return None


def iso(value: datetime.datetime) -> str:
    return value.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def nightly_identity(event: dict) -> str:
    """Stable identity for the nightly an event belongs to, for store dedupe.

    Commit is the most meaningful axis ("which vLLM produced this"). Because
    this dashboard ingests scheduled nightlies only, one commit means one
    nightly, so a retried nightly on the same commit correctly folds into a
    single observation rather than appearing twice.

    Only the vLLM commit counts. ``build_commit`` is the perf-eval repo's own
    commit, which stays the same across many nightlies, so falling back to it
    would fold separate nights into one.
    """
    commit = str(event.get("vllm_commit") or "").strip()
    if commit:
        return f"commit:{commit}"
    if event.get("build_number") is not None:
        return f"build:{event['build_number']}"
    observed_at = event_datetime(event)
    return f"date:{iso(observed_at) if observed_at else event.get('received_at', '')}"


def artifact_key(record: dict) -> tuple | None:
    """Return the exact stable identity for an artifact-bearing record.

    Buildkite artifact IDs are globally stable and preferred. The remaining
    fields provide a conservative fallback for older responses without an ID.
    Download URLs are intentionally excluded: their signatures expire and must
    never be persisted.
    """
    artifact_id = str(record.get("buildkite_artifact_id") or "").strip()
    if artifact_id:
        return "id", artifact_id

    build_number = record.get("build_number")
    job_id = str(record.get("buildkite_artifact_job_id") or "").strip()
    path = str(record.get("buildkite_artifact_path") or "").strip().lstrip("./")
    sha1 = str(record.get("buildkite_artifact_sha1") or "").strip().lower()
    if build_number is not None and job_id and path and sha1:
        return "metadata", build_number, job_id, path, sha1
    return None


def _indexed_artifact_rows(
    event: dict,
) -> list[tuple[tuple, datetime.datetime | None]]:
    if event.get("event") != ARTIFACT_INDEX_EVENT:
        key = artifact_key(event)
        return [(key, event_datetime(event))] if key is not None else []

    rows: list[tuple[tuple, datetime.datetime | None]] = []
    if event.get("schema_version") != ARTIFACT_INDEX_SCHEMA_VERSION:
        return rows
    for encoded in event.get("identities") or []:
        if not isinstance(encoded, list) or not encoded:
            continue
        if encoded[0] == "id" and len(encoded) == 3 and str(encoded[1]).strip():
            key = ("id", str(encoded[1]).strip())
            stamp = event_datetime({"received_at": encoded[2]})
        elif encoded[0] == "metadata" and len(encoded) == 6:
            key = (
                "metadata",
                encoded[1],
                str(encoded[2]).strip(),
                str(encoded[3]).strip().lstrip("./"),
                str(encoded[4]).strip().lower(),
            )
            stamp = event_datetime({"received_at": encoded[5]})
            if not all(key[2:]):
                continue
        else:
            continue
        rows.append((key, stamp))
    return rows


def artifact_keys_from_event(event: dict) -> tuple[tuple, ...]:
    """Expose direct and compact-index identities to pre-download dedupe."""
    return tuple(key for key, _ in _indexed_artifact_rows(event))


def _encode_artifact_identity(key: tuple, observed_at: datetime.datetime) -> list:
    if key[0] == "id":
        return ["id", key[1], iso(observed_at)]
    return ["metadata", *key[1:], iso(observed_at)]


def result_identity(event: dict) -> tuple:
    """Stable identity for one measured result, used for dedupe and merging."""
    base = (
        event.get("event"),
        nightly_identity(event),
        str(event.get("model") or "").strip(),
        str(event.get("device") or "").strip(),
    )
    if event.get("event") == "perf_result":
        return base + (
            event.get("tp"),
            event.get("precision"),
            event.get("isl"),
            event.get("osl"),
            event.get("conc"),
        )
    return base + (str(event.get("workload") or "").strip(),)


def merge_result_events(previous: dict, current: dict) -> dict:
    """Fold repeated pushes for one result into a single complete observation.

    A nightly may report the same config across several artifacts (one per
    metric family, or one per lm-eval task), so metrics and task rows union
    rather than replace.
    """
    merged = dict(previous)
    merged.update(current)
    if current.get("event") == "perf_result":
        merged["metrics"] = {
            **(previous.get("metrics") or {}),
            **(current.get("metrics") or {}),
        }
    else:
        rows = {}
        for event in (previous, current):
            for row in event.get("results") or []:
                if not isinstance(row, dict):
                    continue
                rows[(str(row.get("task") or ""), str(row.get("metric") or ""))] = row
        merged["results"] = [rows[key] for key in sorted(rows)]
    return merged


def encoded_events(events: list[dict]) -> bytes:
    return "".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events
    ).encode("utf-8")


def encoded_json(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _compact_events_once(
    events: list[dict],
    now: datetime.datetime,
    *,
    history_days: int,
    min_nightlies: int,
    auxiliary_days: int,
) -> list[dict]:
    normalized: list[tuple[int, dict, datetime.datetime]] = []
    artifact_identities: dict[tuple, datetime.datetime] = {}
    for position, raw in enumerate(events):
        if not isinstance(raw, dict):
            continue
        event = dict(raw)
        observed_at = event_datetime(event)
        if observed_at is None:
            observed_at = now
            event.setdefault("received_at", iso(now))
        normalized.append((position, event, observed_at))
        for key, identity_at in _indexed_artifact_rows(event):
            stamp = identity_at or observed_at
            if stamp > artifact_identities.get(key, _EPOCH):
                artifact_identities[key] = stamp

    nightly_latest: dict[str, datetime.datetime] = {}
    for _, event, observed_at in normalized:
        if event.get("event") in RESULT_EVENTS and event.get("nightly") is True:
            identity = nightly_identity(event)
            nightly_latest[identity] = max(nightly_latest.get(identity, observed_at), observed_at)
    protected_nightlies = {
        identity
        for identity, _ in sorted(
            nightly_latest.items(),
            key=lambda item: (item[1], item[0]),
        )[-min_nightlies:]
    }

    history_cutoff = now - datetime.timedelta(days=history_days)
    auxiliary_cutoff = now - datetime.timedelta(days=auxiliary_days)
    deduped_results: dict[tuple, tuple[int, dict, datetime.datetime]] = {}
    retained_other: list[tuple[int, dict, datetime.datetime]] = []
    newest_expected: tuple[datetime.datetime, dict] | None = None
    for position, event, observed_at in normalized:
        kind = event.get("event")
        if kind in {ARTIFACT_MARKER_EVENT, ARTIFACT_INDEX_EVENT}:
            continue
        if kind == EXPECTED_CONFIGS_EVENT:
            # Singleton: keep only the newest, ignoring the retention cutoffs.
            # An expectation is never "too old to matter" — it is the current
            # recipe set, and dropping it would blind the coverage card.
            if newest_expected is None or observed_at >= newest_expected[0]:
                newest_expected = (observed_at, event)
            continue
        if kind in RESULT_EVENTS:
            keep = (
                event.get("nightly") is True
                and (
                    observed_at >= history_cutoff or nightly_identity(event) in protected_nightlies
                )
            ) or (event.get("nightly") is not True and observed_at >= auxiliary_cutoff)
            if not keep:
                continue
            key = result_identity(event)
            previous = deduped_results.get(key)
            if previous is None:
                deduped_results[key] = (position, event, observed_at)
            else:
                # Newest observation wins, by timestamp rather than position:
                # the collector appends newest builds first, so on a backfill
                # an older rebuild of the same commit lands later in the file.
                if observed_at >= previous[2]:
                    merged = merge_result_events(previous[1], event)
                else:
                    merged = merge_result_events(event, previous[1])
                deduped_results[key] = (
                    previous[0],
                    merged,
                    max(previous[2], observed_at),
                )
            continue
        if observed_at >= auxiliary_cutoff:
            retained_other.append((position, event, observed_at))

    retained = [*retained_other, *deduped_results.values()]
    # Preserve first-observed order. The aggregator sorts every metric series
    # by timestamp itself, so chronological reordering here would change
    # otherwise identical dashboard output merely because the store was
    # compacted.
    retained.sort(key=lambda row: (row[0], json.dumps(row[1], sort_keys=True)))
    compacted = [event for _, event, _ in retained]

    if newest_expected is not None:
        compacted.append(newest_expected[1])

    direct_keys = {key for event in compacted for key in artifact_keys_from_event(event)}
    identity_cutoff = now - datetime.timedelta(days=ARTIFACT_IDENTITY_DAYS)
    index_rows = [
        _encode_artifact_identity(key, observed_at)
        for key, observed_at in artifact_identities.items()
        if key not in direct_keys and observed_at >= identity_cutoff
    ]
    index_rows.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
    if index_rows:
        compacted.append(
            {
                "event": ARTIFACT_INDEX_EVENT,
                "schema_version": ARTIFACT_INDEX_SCHEMA_VERSION,
                "generated_at": iso(now),
                "retention_days": ARTIFACT_IDENTITY_DAYS,
                "identities": index_rows,
            }
        )
    return compacted


def compact_events(
    events: list[dict],
    *,
    now: datetime.datetime | None = None,
    max_bytes: int = EVENTS_MAX_BYTES,
) -> list[dict]:
    """Apply the fixed retention; fail rather than drop history to fit."""
    max_bytes = enforced_byte_budget(max_bytes, cap=EVENTS_MAX_BYTES)
    current_time = (now or datetime.datetime.now(datetime.UTC)).astimezone(datetime.UTC)
    compacted = _compact_events_once(
        events,
        current_time,
        history_days=PERF_EVAL_HISTORY_DAYS,
        min_nightlies=PERF_EVAL_MIN_NIGHTLIES,
        auxiliary_days=AUXILIARY_EVENT_DAYS,
    )
    size = len(encoded_events(compacted))
    if size > max_bytes:
        raise RuntimeError(
            f"perf-eval events need {size} bytes for {PERF_EVAL_HISTORY_DAYS} days of "
            f"history, over the {max_bytes} byte ceiling. Nothing was written. Raise "
            "EVENTS_MAX_BYTES rather than letting history be dropped to fit."
        )
    return compacted


def _atomic_write_bytes(path: Path, payload: bytes, *, max_bytes: int) -> None:
    """Check size in memory, then atomically replace the destination.

    The size check happens before the destination is touched, so an oversized
    payload leaves the previous good file in place.
    """
    if len(payload) > max_bytes:
        raise RuntimeError(
            f"perf-eval payload exceeds its byte budget: {len(payload)} > {max_bytes} bytes"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_events_atomic(
    store_path: Path,
    events: list[dict],
    *,
    now: datetime.datetime | None = None,
    max_bytes: int = EVENTS_MAX_BYTES,
) -> int:
    budget = enforced_byte_budget(max_bytes, cap=EVENTS_MAX_BYTES)
    compacted = compact_events(events, now=now, max_bytes=budget)
    _atomic_write_bytes(store_path, encoded_events(compacted), max_bytes=budget)
    return len(compacted)


def append_events(
    store_path: Path,
    events: list[dict],
    *,
    now: datetime.datetime | None = None,
    max_bytes: int = EVENTS_MAX_BYTES,
) -> int:
    """Append events, then compact and rewrite the whole store atomically."""
    return write_events_atomic(
        store_path,
        [*read_events_strict(store_path), *events],
        now=now,
        max_bytes=max_bytes,
    )


def read_events(store_path: Path) -> list[dict]:
    """Read every well-formed event, skipping malformed lines with a warning.

    Prefer :func:`read_events_strict` anywhere a silent drop could lose data.
    """
    if not store_path.exists():
        return []
    out: list[dict] = []
    for line in store_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            log.warning("skipping malformed event line")
            continue
        if not isinstance(event, dict):
            log.warning("skipping non-object event line")
            continue
        out.append(event)
    return out


def read_events_strict(store_path: Path) -> list[dict]:
    """Read a JSONL store, failing closed on any non-empty malformed line.

    Every writer replaces the store wholesale, so tolerating a corrupt line
    here would quietly delete the data it represents on the next write.
    """
    if not store_path.exists():
        return []
    out: list[dict] = []
    for line_number, line in enumerate(
        store_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid perf-eval JSONL at {store_path}:{line_number}: {exc.msg}"
            ) from exc
        if not isinstance(event, dict):
            raise ValueError(
                f"invalid perf-eval JSONL at {store_path}:{line_number}: "
                "event must be a JSON object"
            )
        if event.get("event") == ARTIFACT_INDEX_EVENT:
            identities = event.get("identities")
            decoded = _indexed_artifact_rows(event)
            invalid_index = (
                event.get("schema_version") != ARTIFACT_INDEX_SCHEMA_VERSION
                or not isinstance(identities, list)
                or len(decoded) != len(identities)
                or any(
                    stamp is None or (key[0] == "metadata" and key[1] is None)
                    for key, stamp in decoded
                )
            )
            if invalid_index:
                raise ValueError(
                    f"invalid perf-eval JSONL at {store_path}:{line_number}: "
                    "artifact identity index is not canonical"
                )
        out.append(event)
    return out


def write_json_atomic(
    path: Path,
    payload: dict,
    *,
    max_bytes: int = SUMMARY_MAX_BYTES,
) -> None:
    _atomic_write_bytes(
        path,
        encoded_json(payload),
        max_bytes=enforced_byte_budget(max_bytes, cap=SUMMARY_MAX_BYTES),
    )
