#!/usr/bin/env python3
"""Report when perf-eval nightlies actually finish, to set the collection cron.

The upstream pipeline's schedule lives in the Buildkite UI, not in the
``perf-eval`` repo, so there is nothing in code to read it from — and a nightly
sweeping a dozen models across several GPU types takes hours, so its finish
time drifts. Rather than guess a cron, measure one:

    BUILDKITE_TOKEN=bkua_... python scripts/dev_build_times.py --days 30

Read-only: it lists builds and nothing else. No artifacts are downloaded, so
it is cheap to run repeatedly.

This is a development tool for a one-off decision. It is not used by CI.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perf_eval import BUILDKITE_ORG, BUILDKITE_PIPELINE_SLUG  # noqa: E402
from perf_eval.collect_artifacts import (  # noqa: E402
    _bk_paginate,
    is_nightly_build,
    nightly_info,
    use_system_certificates,
)
from perf_eval.store import event_datetime  # noqa: E402


def _parse(stamp: str | None):
    return event_datetime({"date": stamp}) if stamp else None


def collect(days: int, token: str) -> list[dict]:
    builds = _bk_paginate(
        f"/organizations/{BUILDKITE_ORG}/pipelines/{BUILDKITE_PIPELINE_SLUG}/builds",
        token,
        {"branch": "main", "state": "finished"},
        max_pages=5,
    )
    rows = []
    cutoff = datetime.now().timestamp() - days * 86400
    for build in builds:
        started = _parse(build.get("started_at") or build.get("created_at"))
        finished = _parse(build.get("finished_at"))
        if not finished or finished.timestamp() < cutoff:
            continue
        info = nightly_info(build) if is_nightly_build(build) else None
        rows.append(
            {
                "number": build.get("number"),
                "nightly": info is not None,
                "started": started,
                "finished": finished,
                "hours": ((finished - started).total_seconds() / 3600) if started else None,
                "commit": (info or {}).get("vllm_commit", ""),
            }
        )
    return sorted(rows, key=lambda r: r["finished"])


def report(rows: list[dict], days: int) -> None:
    nightlies = [r for r in rows if r["nightly"]]
    print(f"Finished perf-eval builds on main in the last {days} days: {len(rows)}")
    print(f"  of which nightlies: {len(nightlies)}\n")

    if not nightlies:
        print("No nightlies found. Either the window is too short, or the nightly")
        print("is not tagged the way collect_artifacts.is_nightly_build expects.")
        return

    print(f"{'build':>8}  {'finished (UTC)':<20} {'hour':>5} {'duration':>9}  commit")
    for row in nightlies:
        duration = f"{row['hours']:.1f}h" if row["hours"] is not None else "?"
        print(
            f"{row['number']:>8}  {row['finished'].strftime('%Y-%m-%d %H:%M:%S'):<20} "
            f"{row['finished'].hour:>5} {duration:>9}  {row['commit'][:12]}"
        )

    hours = [r["finished"].hour + r["finished"].minute / 60 for r in nightlies]
    durations = [r["hours"] for r in nightlies if r["hours"] is not None]
    median_hour = statistics.median(hours)
    print("\nFinish time, UTC hour of day:")
    print(f"  earliest {min(hours):.1f}   median {median_hour:.1f}   latest {max(hours):.1f}")
    if durations:
        print(
            f"Run duration: min {min(durations):.1f}h  "
            f"median {statistics.median(durations):.1f}h  max {max(durations):.1f}h"
        )

    # One pass an hour after the latest observed finish catches every nightly
    # seen in this window; a second a few hours later absorbs drift and lets a
    # failed run retry without waiting a full day.
    primary = int((max(hours) + 1) % 24)
    retry = int((max(hours) + 5) % 24)
    print("\nSuggested cron for .github/workflows/collect.yml:")
    print(f"    - cron: '17 {primary},{retry} * * *'")
    print(
        f"\n  {primary:02d}:17 UTC is an hour after the latest finish observed here; "
        f"{retry:02d}:17 UTC\n  is a retry pass for drift and failed runs. Widen the "
        "window with --days if\n  this covered only a few nightlies."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30, help="Lookback window (default: 30)")
    args = parser.parse_args()

    # Before any connection is made.
    use_system_certificates()

    token = os.getenv("BUILDKITE_TOKEN") or ""
    if not token:
        print("BUILDKITE_TOKEN not set; cannot list perf-eval builds.", file=sys.stderr)
        return 1

    report(collect(args.days, token), args.days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
