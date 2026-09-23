#!/usr/bin/env python3
"""Whether a freshly built payload differs from the published one, ignoring
``generated_at``.

Prints ``true`` or ``false`` and writes ``changed=<value>`` to ``GITHUB_OUTPUT``
for the deploy step. A missing or unreadable previous payload counts as changed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Fields that change on every run; any other change deploys.
VOLATILE_FIELDS = ("generated_at",)


def _material(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in VOLATILE_FIELDS}


def _canonical(payload: dict) -> str:
    return json.dumps(_material(payload), sort_keys=True, separators=(",", ":"))


def payload_changed(previous: dict | None, current: dict) -> bool:
    """True if anything beyond the volatile timestamps differs."""
    if previous is None:
        return True
    return _canonical(previous) != _canonical(current)


def _load(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Treat an unreadable baseline as absent rather than failing the run:
        # the consequence is one redundant deploy, not a broken collection.
        return None
    return loaded if isinstance(loaded, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True, help="Freshly built payload")
    parser.add_argument("--previous", type=Path, help="Previously published payload")
    args = parser.parse_args()

    current = _load(args.current)
    if current is None:
        print("Current payload is missing or unreadable", file=sys.stderr)
        return 1

    changed = payload_changed(_load(args.previous), current)
    print("true" if changed else "false")

    output = os.getenv("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"changed={'true' if changed else 'false'}\n")

    if not changed:
        print(
            "Only the generated_at timestamp moved; nothing new to publish.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
