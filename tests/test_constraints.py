"""constraints.txt pins every direct dependency CI installs.

The workflows install with ``pip install -c constraints.txt``, so an unpinned
direct dependency would still float to whatever is newest on each run. That
is how a ruff or pyright release could break CI with no code change, and how
an unreviewed release could run in the collect job next to both tokens.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _name(requirement: str) -> str:
    return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].strip().lower().replace("_", "-")


def _pins() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, version = line.partition("==")
        assert sep, f"constraints.txt entry is not an exact pin: {line!r}"
        pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def test_every_direct_dependency_is_pinned():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    direct = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        direct.extend(extra)
    pins = _pins()
    missing = sorted({_name(req) for req in direct} - set(pins))
    assert not missing, f"add exact pins to constraints.txt for: {', '.join(missing)}"
