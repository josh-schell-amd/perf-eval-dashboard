"""Guards keeping the Buildkite token scoped and out of the repository.

The token is a repo secret injected as step-scoped env on the single step that
needs it. These tests fail if a change would widen where it can be used or let
one land in the tree.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import perf_eval

ROOT = Path(perf_eval.__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
WORKFLOWS = ROOT / ".github" / "workflows"

TOKEN_NAMES = ("BUILDKITE_TOKEN", "BUILDKITE_API_TOKEN")
# The only modules permitted to read the Buildkite token. Adding an entry here
# should be a deliberate decision: every one widens where a credential can
# reach. Both of these are read-only and pinned to the vllm org.
TOKEN_ENTRYPOINTS = {
    "perf_eval/collect_artifacts.py",  # the collector, run by CI
    "dev_build_times.py",  # local tool that lists build finish times
}
# Read-only by construction: these must never issue a write.
READ_ONLY_MODULES = ("perf_eval/collect_artifacts.py", "dev_build_times.py")


class TestOrgIsPinned:
    def test_buildkite_org_is_vllm(self):
        assert perf_eval.BUILDKITE_ORG == "vllm"

    def test_pipeline_slug_is_perf_eval(self):
        assert perf_eval.BUILDKITE_PIPELINE_SLUG == "perf-eval"

    def test_every_buildkite_url_interpolates_the_pinned_org(self):
        # A pinned org is what keeps the token from being pointed at an
        # unrelated Buildkite organization.
        pattern = re.compile(r"organizations/\{(\w+)\}")
        for path in SCRIPTS.rglob("*.py"):
            for name in pattern.findall(path.read_text(encoding="utf-8")):
                assert name == "BUILDKITE_ORG", f"{path}: org comes from {name}"


class TestTokenReach:
    def test_only_the_artifact_collector_reads_the_token(self):
        offenders = set()
        for path in SCRIPTS.rglob("*.py"):
            rel = path.relative_to(SCRIPTS).as_posix()
            source = path.read_text(encoding="utf-8")
            if any(name in source for name in TOKEN_NAMES) and rel not in TOKEN_ENTRYPOINTS:
                offenders.add(rel)
        assert offenders == set(), f"unexpected Buildkite token references: {offenders}"

    @pytest.mark.parametrize("module", READ_ONLY_MODULES)
    def test_token_holders_only_issue_reads(self, module):
        source = (SCRIPTS / module).read_text(encoding="utf-8")
        for verb in ("requests.post", "requests.put", "requests.patch", "requests.delete"):
            assert verb not in source, f"{module}: {verb} would need a write-scoped token"

    @pytest.mark.parametrize("module", READ_ONLY_MODULES)
    def test_token_holders_fail_closed_without_a_token(self, module):
        source = (SCRIPTS / module).read_text(encoding="utf-8")
        assert 'os.getenv("BUILDKITE_TOKEN")' in source, module
        assert "BUILDKITE_TOKEN not set" in source, module

    @pytest.mark.parametrize("module", READ_ONLY_MODULES)
    def test_token_holders_never_disable_tls_verification(self, module):
        """A TLS-inspecting proxy is fixed by trusting the OS store.

        Never by turning verification off in a process holding a credential.
        Checked against the parsed syntax tree rather than the text, so a
        comment or docstring *naming* the anti-pattern does not trip it.
        """
        tree = ast.parse((SCRIPTS / module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                disabled = (
                    keyword.arg == "verify"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is False
                )
                assert not disabled, f"{module} disables TLS verification"

    @pytest.mark.parametrize("module", READ_ONLY_MODULES)
    def test_token_holders_trust_the_os_certificate_store(self, module):
        source = (SCRIPTS / module).read_text(encoding="utf-8")
        assert "use_system_certificates()" in source, module

    @pytest.mark.parametrize("module", ["aggregate.py", "merge_events.py", "normalize.py"])
    def test_offline_modules_cannot_reach_the_network(self, module):
        source = (SCRIPTS / "perf_eval" / module).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(?:import requests|from requests)", source, re.M), module
        assert not re.search(r"^\s*(?:import urllib|from urllib)", source, re.M), module
        assert "api.buildkite.com" not in source, module


@pytest.mark.skipif(not WORKFLOWS.is_dir(), reason="workflows not present")
class TestWorkflowTokenHandling:
    def _workflow_text(self, name):
        return (WORKFLOWS / name).read_text(encoding="utf-8")

    def test_no_workflow_contains_a_literal_token(self):
        for path in WORKFLOWS.glob("*.yml"):
            text = path.read_text(encoding="utf-8")
            for name in TOKEN_NAMES:
                for line in text.splitlines():
                    if f"{name}:" not in line:
                        continue
                    # Every assignment must reference the secrets context.
                    assert "secrets." in line, f"{path.name}: {line.strip()}"

    def test_the_token_is_scoped_to_the_ingest_step_only(self):
        text = self._workflow_text("collect.yml")
        # Exactly one step may receive it, and nothing above `steps:` may,
        # since a workflow- or job-level env block reaches every step.
        assert text.count("BUILDKITE_TOKEN: ${{ secrets.BUILDKITE_TOKEN }}") == 1
        preamble = text.split("steps:")[0]
        assert "BUILDKITE_TOKEN" not in preamble

    def test_ci_workflow_never_receives_the_token(self):
        for name in ("ci.yml", "secrets-scan.yml"):
            text = self._workflow_text(name)
            for token in TOKEN_NAMES:
                assert token not in text, f"{name} must not receive {token}"

    def test_pushing_checkouts_do_not_persist_credentials(self):
        text = self._workflow_text("collect.yml")
        assert "persist-credentials: false" in text
