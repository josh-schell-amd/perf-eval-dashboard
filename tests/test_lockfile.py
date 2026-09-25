"""CI installs exactly what uv.lock pins.

An install that bypasses the lock would float to whatever is newest on each
run. That is how a ruff or pyright release could break CI with no code change,
and how an unreviewed release could run in the collect job next to both tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"


def test_the_lock_is_committed():
    assert (ROOT / "uv.lock").is_file()


@pytest.mark.skipif(not WORKFLOWS.is_dir(), reason="workflows not present")
class TestWorkflowsInstallFromTheLock:
    @pytest.mark.parametrize("name", ["ci.yml", "collect.yml"])
    def test_installs_with_a_locked_sync(self, name):
        assert "uv sync --locked" in (WORKFLOWS / name).read_text(encoding="utf-8")

    def test_no_workflow_installs_around_the_lock(self):
        for path in WORKFLOWS.glob("*.yml"):
            assert "pip install" not in path.read_text(encoding="utf-8"), path.name

    def test_the_token_holding_job_gets_runtime_packages_only(self):
        assert "--no-dev" in (WORKFLOWS / "collect.yml").read_text(encoding="utf-8")
