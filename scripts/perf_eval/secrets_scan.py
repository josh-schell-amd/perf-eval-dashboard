#!/usr/bin/env python3
"""Fast local check for credentials accidentally committed to the tree.

Run locally::

    python scripts/perf_eval/secrets_scan.py

Run in CI (``.github/workflows/secrets-scan.yml``) on every push and pull
request. Exits non-zero if anything looks like a real
credential, so a PR is blocked before the secret reaches a default branch.

Scope, deliberately narrow
--------------------------
This looks for **known token shapes only**: a fixed prefix followed by a run
of at least N characters from a known alphabet. That is all a GitHub or
Buildkite token is, so it is expressed as a table of those three facts rather
than as regular expressions — adding a provider is one line of data, and there
is no pattern to misread in review.

It deliberately does *not* try to detect "anything that looks secret". An
earlier version flagged any run of 40+ hex characters, which in practice
detected git commit SHAs rather than credentials, and needed three separate
suppression mechanisms to stay usable. Other providers and git history are
left to GitHub secret scanning and push protection; this file attempts
neither.

Allowlist
---------
Paths skipped for cost rather than correctness: vendored third-party bundles,
build output, virtualenvs and caches. Keep it short — if you find yourself
adding a path so a real-looking match goes away, check it is genuinely safe
first.
"""

from __future__ import annotations

import string
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]

BASE62 = frozenset(string.ascii_letters + string.digits)
BASE62_UNDERSCORE = BASE62 | {"_"}
LOWER_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class TokenShape:
    """A credential format: a prefix, then a run of allowed characters.

    ``min_length`` counts characters *after* the prefix, and is tuned so that
    placeholders such as ``ghp_...`` or ``bkua_<token>`` do not match.
    """

    label: str
    prefix: str
    min_length: int
    alphabet: frozenset[str]


TOKEN_SHAPES: tuple[TokenShape, ...] = (
    TokenShape("GitHub PAT (classic)", "ghp_", 36, BASE62),
    TokenShape("GitHub OAuth token", "gho_", 36, BASE62),
    TokenShape("GitHub user-to-server token", "ghu_", 36, BASE62),
    TokenShape("GitHub server-to-server token", "ghs_", 36, BASE62),
    TokenShape("GitHub refresh token", "ghr_", 36, BASE62),
    TokenShape("GitHub fine-grained PAT", "github_pat_", 50, BASE62_UNDERSCORE),
    TokenShape("Buildkite API token", "bkua_", 40, LOWER_HEX),
    TokenShape("HuggingFace token", "hf_", 34, BASE62),
)

# Skipped for cost, not because they are trusted: vendored bundles, build
# output, dependency trees and caches.
PATH_ALLOWLIST = (
    ".git/",
    "node_modules/",
    "site/vendor/",  # vendored third-party bundles
    "_site/",  # build output, contains a copy of the above
    ".venv/",
    ".tox/",
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
    "tests/test_secrets_scan.py",  # constructs sample tokens to test detection
)

# Source and config only, so we do not walk large binary fixtures.
SCAN_SUFFIXES = (
    ".py",
    ".js",
    ".ts",
    ".mjs",
    ".cjs",
    ".html",
    ".css",
    ".yml",
    ".yaml",
    ".json",
    ".jsonl",
    ".sh",
    ".toml",
    ".md",
    ".txt",
    ".cfg",
    ".ini",
)


def _run_length(text: str, start: int, alphabet: frozenset[str]) -> int:
    """How many characters from ``alphabet`` run consecutively from ``start``."""
    index = start
    while index < len(text) and text[index] in alphabet:
        index += 1
    return index - start


def find_tokens(line: str) -> list[tuple[TokenShape, str]]:
    """Every known token shape present in one line, with the matched text."""
    found: list[tuple[TokenShape, str]] = []
    for shape in TOKEN_SHAPES:
        search_from = 0
        while True:
            at = line.find(shape.prefix, search_from)
            if at == -1:
                break
            tail_start = at + len(shape.prefix)
            tail = _run_length(line, tail_start, shape.alphabet)
            if tail >= shape.min_length:
                found.append((shape, line[at : tail_start + tail]))
            # Advance past the prefix, not past the tail: overlapping prefixes
            # are rare but skipping the tail could hide a second match.
            search_from = tail_start
    return found


def _is_allowlisted(rel: str) -> bool:
    # Dependency installs can live below several project-local roots, so treat
    # node_modules as a path component rather than only a repository root.
    if "node_modules" in PurePosixPath(rel).parts:
        return True
    return any(rel == entry or rel.startswith(entry) for entry in PATH_ALLOWLIST)


def _iter_candidate_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _is_allowlisted(rel):
            continue
        if path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        yield path, rel


def scan_text(text: str, rel: str) -> list[str]:
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for shape, matched in find_tokens(line):
            # Report only the prefix and a little context: the finding should
            # not itself become a copy of the credential in CI logs.
            findings.append(
                f"{rel}:{lineno}: {shape.label}: {matched[: len(shape.prefix) + 4]}… "
                f"({len(matched)} chars) — use a placeholder or an env var"
            )
    return findings


def main() -> int:
    all_findings: list[str] = []
    for path, rel in _iter_candidate_files(ROOT):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"WARN: could not read {rel}: {exc}", file=sys.stderr)
            continue
        all_findings.extend(scan_text(text, rel))

    if all_findings:
        print("Secrets scan found suspicious matches:")
        for finding in all_findings:
            print(f"  - {finding}")
        print()
        print(
            "If this is a placeholder, shorten it so it no longer looks like a "
            "real token. Never add a path to PATH_ALLOWLIST just to silence a "
            "genuine match."
        )
        return 1

    print(f"No secrets detected ({len(TOKEN_SHAPES)} token shapes checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
