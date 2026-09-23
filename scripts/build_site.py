#!/usr/bin/env python3
"""Assemble the deployable site from ``site/`` plus the published payload.

Copies ``site/`` into ``_site/``, drops ``data/perf_eval.json`` alongside
``index.html`` (the page fetches it as a sibling), and cache-busts the fetch so
a browser holding a stale copy of the JSON picks up a fresh deploy.

Only ``perf_eval.json`` is published. The private ``events.jsonl`` event store
is never copied into the site.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SITE = ROOT / "site"
DEFAULT_PAYLOAD = ROOT / "data" / "perf_eval.json"
DEFAULT_OUTPUT = ROOT / "_site"

PAYLOAD_NAME = "perf_eval.json"
# Matches the fetch in site/index.html, with or without an existing ?v= tag.
_FETCH_RE = re.compile(r"(fetch\(\s*['\"])perf_eval\.json(?:\?v=[^'\"]*)?(['\"])")


def build(site_dir: Path, payload_path: Path, output_dir: Path) -> Path:
    if not site_dir.is_dir():
        raise FileNotFoundError(f"site directory not found: {site_dir}")
    index = site_dir / "index.html"
    if not index.is_file():
        raise FileNotFoundError(f"site entrypoint not found: {index}")
    if not payload_path.is_file():
        raise FileNotFoundError(
            f"published payload not found: {payload_path} (run aggregate.py first)"
        )

    payload_bytes = payload_path.read_bytes()
    # Fail loudly here rather than shipping a page that renders nothing.
    json.loads(payload_bytes)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(site_dir, output_dir)

    (output_dir / PAYLOAD_NAME).write_bytes(payload_bytes)

    digest = hashlib.sha256(payload_bytes).hexdigest()[:12]
    html = (output_dir / "index.html").read_text(encoding="utf-8")
    patched, count = _FETCH_RE.subn(rf"\1{PAYLOAD_NAME}?v={digest}\2", html)
    if count == 0:
        raise RuntimeError(
            f"no fetch('{PAYLOAD_NAME}') call found in index.html; "
            "cache-busting would silently do nothing"
        )
    (output_dir / "index.html").write_text(patched, encoding="utf-8")

    print(f"Built {output_dir} ({len(payload_bytes)} bytes of data, cache tag {digest})")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE, help="Source site directory")
    parser.add_argument(
        "--payload", type=Path, default=DEFAULT_PAYLOAD, help="Path to perf_eval.json"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    args = parser.parse_args()

    build(args.site, args.payload, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
