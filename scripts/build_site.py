#!/usr/bin/env python3
"""Build the deployable site: copy site/ into a fresh _site/ with perf_eval.json.

The page's data fetch gets a content-hash query so browsers pick up new data
after a deploy. The events.jsonl store is never published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "site"
DEFAULT_PAYLOAD = ROOT / "data" / "perf_eval.json"
DEFAULT_OUTPUT = ROOT / "_site"

PAYLOAD_NAME = "perf_eval.json"
FETCH_CALL = f"fetch('{PAYLOAD_NAME}')"


def build(source_dir: Path, payload_path: Path, output_dir: Path) -> Path:
    """Build output_dir from source_dir and the payload; return output_dir.

    Everything is checked before output_dir is touched, so a failed build
    leaves the previous output in place.
    """
    index = source_dir / "index.html"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"site source directory not found: {source_dir}")
    if not index.is_file():
        raise FileNotFoundError(f"site entrypoint not found: {index}")
    if not payload_path.is_file():
        raise FileNotFoundError(
            f"published payload not found: {payload_path} (run aggregate.py first)"
        )

    payload_bytes = payload_path.read_bytes()
    try:
        json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{payload_path} is not valid JSON ({exc}); the page could not load it, "
            "so nothing was built"
        ) from exc

    html = index.read_text(encoding="utf-8")
    found = html.count(FETCH_CALL)
    if found != 1:
        raise RuntimeError(
            f"{index} must load its data with exactly one {FETCH_CALL} call, so the "
            f"build can add the cache tag; found {found}"
        )
    tag = hashlib.sha256(payload_bytes).hexdigest()[:12]
    html = html.replace(FETCH_CALL, f"fetch('{PAYLOAD_NAME}?v={tag}')")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    (output_dir / PAYLOAD_NAME).write_bytes(payload_bytes)

    print(f"Built {output_dir} ({len(payload_bytes)} bytes of data, cache tag {tag})")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Site source (site/)")
    parser.add_argument(
        "--payload", type=Path, default=DEFAULT_PAYLOAD, help="Path to perf_eval.json"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Where to build the site (_site/)"
    )
    args = parser.parse_args()

    build(args.source, args.payload, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
