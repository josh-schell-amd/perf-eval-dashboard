"""The event store, data/events.jsonl: one event per line.

Every write compacts it and replaces the file atomically. Event kinds:

    perf_result, accuracy_result   one nightly result, timed by `date`
    expected_configs               the configs the recipes define; newest kept
    buildkite_artifact_ingested    an artifact already downloaded; folded into
    buildkite_artifact_identity_index   one index event at compaction
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from perf_eval import WINDOW_DAYS

log = logging.getLogger(__name__)

RESULT_EVENTS = frozenset({"perf_result", "accuracy_result"})
EXPECTED_CONFIGS_EVENT = "expected_configs"
ARTIFACT_MARKER_EVENT = "buildkite_artifact_ingested"
ARTIFACT_INDEX_EVENT = "buildkite_artifact_identity_index"


def parse_time(value: object) -> datetime | None:
    """A UTC datetime from an ISO timestamp or date string, or None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def finished_at(result: dict) -> datetime | None:
    """When the nightly behind a result finished: its `date`."""
    return parse_time(result.get("date"))


def received_at(event: dict) -> datetime | None:
    """When the collector recorded an event: its `received_at`."""
    return parse_time(event.get("received_at"))


def iso(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def nightly_identity(event: dict) -> str:
    """The nightly a result belongs to: its vLLM commit, else its build number.

    Not build_commit, which is the perf-eval repo's commit and is shared by
    many nightlies.
    """
    commit = str(event.get("vllm_commit") or "").strip()
    return f"commit:{commit}" if commit else f"build:{event.get('build_number')}"


def result_identity(event: dict) -> tuple:
    """What makes two result events the same result, for dedupe.

    A perf result is one config (TP, precision, ISL/OSL, concurrency); an
    accuracy result is one workload, whose tasks are rows inside it.
    """
    kind = event.get("event")
    base = (
        kind,
        nightly_identity(event),
        str(event.get("model") or "").strip(),
        str(event.get("device") or "").strip(),
    )
    if kind == "perf_result":
        return base + (
            event.get("tp"),
            event.get("precision"),
            event.get("isl"),
            event.get("osl"),
            event.get("conc"),
        )
    if kind == "accuracy_result":
        return base + (str(event.get("workload") or "").strip(),)
    raise ValueError(f"not a result event: {kind!r}")


def merge_result_events(older: dict, newer: dict) -> dict:
    """Combine two events for one result; newer fields win.

    One nightly can report a result across several artifacts, so the
    measurements are unioned: perf metrics by name, accuracy rows by
    (task, metric).
    """
    kind = newer.get("event")
    merged = {**older, **newer}
    if kind == "perf_result":
        merged["metrics"] = {**(older.get("metrics") or {}), **(newer.get("metrics") or {})}
    elif kind == "accuracy_result":
        rows = {}
        for event in (older, newer):
            for row in event.get("results") or []:
                if isinstance(row, dict):
                    rows[(str(row.get("task") or ""), str(row.get("metric") or ""))] = row
        merged["results"] = [rows[key] for key in sorted(rows)]
    else:
        raise ValueError(f"not a result event: {kind!r}")
    return merged


def artifact_key(record: dict) -> str | None:
    """The Buildkite artifact ID a record came from, if it has one."""
    return str(record.get("buildkite_artifact_id") or "").strip() or None


def _index_row(row: object) -> tuple[str, datetime | None] | None:
    """One ``["id", <artifact ID>, <downloaded at>]`` entry, or None if malformed."""
    if not (isinstance(row, list) and len(row) == 3 and row[0] == "id"):
        return None
    _, artifact_id, downloaded_at = row
    artifact_id = str(artifact_id).strip()
    return (artifact_id, parse_time(downloaded_at)) if artifact_id else None


def _artifact_rows(event: dict) -> list[tuple[str, datetime | None]]:
    """(artifact ID, downloaded at) for every artifact an event names."""
    if event.get("event") == ARTIFACT_INDEX_EVENT:
        rows = (_index_row(row) for row in event.get("identities") or [])
        return [row for row in rows if row is not None]
    artifact_id = artifact_key(event)
    return [(artifact_id, received_at(event))] if artifact_id else []


def artifact_keys_from_event(event: dict) -> tuple[str, ...]:
    """Artifact IDs an event shows as already downloaded."""
    return tuple(key for key, _ in _artifact_rows(event))


def _recorded(event: dict) -> datetime:
    """``received_at`` for ordering; an unparseable one sorts oldest."""
    return received_at(event) or datetime.min.replace(tzinfo=UTC)


def _add_result(
    results: dict[tuple, tuple[int, dict, datetime]], position: int, event: dict, when: datetime
) -> None:
    """Fold a result into ``results`` by identity, keeping its first-seen position.
    Older and newer are by nightly time: a backfill can append an older rebuild later."""
    key = result_identity(event)
    if key not in results:
        results[key] = (position, event, when)
        return
    first_seen, kept, kept_when = results[key]
    if when >= kept_when:
        results[key] = (first_seen, merge_result_events(older=kept, newer=event), when)
    else:
        results[key] = (first_seen, merge_result_events(older=event, newer=kept), kept_when)


def compact_events(events: list[dict]) -> list[dict]:
    """The events worth keeping, in first-seen order: nightly results from the
    last WINDOW_DAYS (duplicates merged), the newest expected_configs snapshot,
    and one index of the artifacts downloaded in that time."""
    cutoff = datetime.now(UTC) - timedelta(days=WINDOW_DAYS)

    results: dict[tuple, tuple[int, dict, datetime]] = {}
    newest_expected: dict | None = None
    artifacts: dict[str, datetime] = {}
    for position, event in enumerate(events):
        for artifact_id, seen in _artifact_rows(event):
            if seen and seen >= cutoff and seen > artifacts.get(artifact_id, cutoff):
                artifacts[artifact_id] = seen

        kind = event.get("event")
        if kind == EXPECTED_CONFIGS_EVENT:
            # The current recipe set, so it is kept however old it is.
            if newest_expected is None or _recorded(event) >= _recorded(newest_expected):
                newest_expected = event
        elif kind in RESULT_EVENTS and event.get("nightly") is True:
            when = finished_at(event)
            if when is not None and when >= cutoff:
                _add_result(results, position, event, when)
        # Every other kind is dropped; artifact IDs were collected above.

    compacted = [event for _, event, _ in sorted(results.values(), key=lambda row: row[0])]
    if newest_expected is not None:
        compacted.append(newest_expected)

    # Artifacts whose result event is kept already carry their ID.
    carried = {artifact_key(event) for event in compacted}
    index = sorted(
        ["id", artifact_id, iso(seen)]
        for artifact_id, seen in artifacts.items()
        if artifact_id not in carried
    )
    if index:
        compacted.append({"event": ARTIFACT_INDEX_EVENT, "identities": index})
    return compacted


def encoded_events(events: list[dict]) -> bytes:
    return "".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events
    ).encode("utf-8")


def encoded_json(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_atomic(path: Path, data: bytes) -> None:
    """Write to a temp file beside path, then rename it over path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # mkstemp creates the file owner-only (0600); give the result normal
        # read permissions (0644), since it replaces the real file.
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_events_atomic(store_path: Path, events: list[dict]) -> int:
    """Compact events and replace the store with them; return how many were kept."""
    compacted = compact_events(events)
    _write_atomic(store_path, encoded_events(compacted))
    return len(compacted)


def append_events(store_path: Path, events: list[dict]) -> int:
    """Add events to the store, then compact and rewrite it."""
    return write_events_atomic(store_path, [*read_events_strict(store_path), *events])


def write_json_atomic(path: Path, payload: dict) -> None:
    _write_atomic(path, encoded_json(payload))


def read_events_strict(store_path: Path) -> list[dict]:
    """Read the store, failing on any malformed line.

    Every write replaces the whole file, so skipping a bad line here would
    delete the data it held on the next write.
    """
    if not store_path.exists():
        return []
    out: list[dict] = []
    for number, line in enumerate(store_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        where = f"invalid perf-eval JSONL at {store_path}:{number}"
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}: {exc.msg}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"{where}: event must be a JSON object")
        if event.get("event") == ARTIFACT_INDEX_EVENT:
            identities = event.get("identities")
            rows = _artifact_rows(event)
            if (
                not isinstance(identities, list)
                or len(rows) != len(identities)
                or any(seen is None for _, seen in rows)
            ):
                raise ValueError(f"{where}: artifact identity index is not canonical")
        out.append(event)
    return out
