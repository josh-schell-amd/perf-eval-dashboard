"""Tests for the local secret scanner that gates every push and pull request.

The scanner deliberately detects known token shapes only. Other providers and
git history are left to GitHub push protection; see the module docstring in
secrets_scan.py.
"""

from __future__ import annotations

import pytest

from perf_eval import secrets_scan as ss

# Built by concatenation so this file never contains a literal token-shaped
# string — otherwise the scanner would flag its own test suite.
GITHUB_PAT = "ghp_" + "A" * 36
BUILDKITE = "bkua_" + "0" * 40
HUGGINGFACE = "hf_" + "E" * 34
FINE_GRAINED = "github_pat_" + "D" * 50


class TestDetection:
    @pytest.mark.parametrize(
        "token,label",
        [
            (GITHUB_PAT, "GitHub PAT (classic)"),
            ("gho_" + "B" * 40, "GitHub OAuth token"),
            ("ghu_" + "C" * 36, "GitHub user-to-server token"),
            ("ghs_" + "C" * 40, "GitHub server-to-server token"),
            ("ghr_" + "C" * 36, "GitHub refresh token"),
            (FINE_GRAINED, "GitHub fine-grained PAT"),
            (BUILDKITE, "Buildkite API token"),
            (HUGGINGFACE, "HuggingFace token"),
        ],
    )
    def test_flags_each_known_shape(self, token, label):
        findings = ss.scan_text(f"token = '{token}'", "demo.py")
        assert findings, f"scanner missed {label}"
        assert label in findings[0]

    def test_finds_a_token_anywhere_on_the_line(self):
        assert ss.scan_text(f"x=1; y='{GITHUB_PAT}'; z=2", "demo.py")

    def test_finds_two_tokens_on_one_line(self):
        assert len(ss.scan_text(f"{GITHUB_PAT} {BUILDKITE}", "demo.py")) == 2

    def test_reports_file_and_line(self):
        findings = ss.scan_text(f"a\nb\n{GITHUB_PAT}", "pkg/demo.py")
        assert findings[0].startswith("pkg/demo.py:3:")


class TestPlaceholdersAreNotFlagged:
    @pytest.mark.parametrize(
        "line",
        [
            "placeholder = 'ghp_...'",
            "placeholder = 'bkua_...'",
            "placeholder = 'hf_...'",
            'export GITHUB_TOKEN="ghp_..."  # docstring example',
            'export BUILDKITE_TOKEN="bkua_..."',
            "BUILDKITE_TOKEN: ${{ secrets.BUILDKITE_TOKEN }}",
            "gh secret set BUILDKITE_TOKEN",
        ],
    )
    def test_ignored(self, line):
        assert ss.scan_text(line, "demo.py") == []

    def test_a_short_tail_is_not_a_token(self):
        assert ss.scan_text("ghp_" + "A" * 35, "demo.py") == []

    def test_the_exact_minimum_length_is_a_token(self):
        assert ss.scan_text("ghp_" + "A" * 36, "demo.py")

    def test_wrong_alphabet_is_not_a_token(self):
        # Buildkite tokens are lower-case hex; uppercase is a different thing.
        assert ss.scan_text("bkua_" + "G" * 40, "demo.py") == []


class TestNoHashHeuristic:
    """Long hex used to be flagged. It detected commit SHAs, not secrets."""

    def test_a_bare_commit_sha_is_not_flagged(self):
        assert ss.scan_text("x = '" + "a" * 40 + "'", "demo.py") == []

    def test_a_pinned_action_reference_is_not_flagged(self):
        # This is what needed an unreadable suppression regex before.
        assert ss.scan_text(f"      - uses: actions/checkout@{'a' * 40} # v4", "ci.yml") == []

    def test_a_sha256_digest_is_not_flagged(self):
        assert ss.scan_text("integrity: sha384-" + "b" * 64, "demo.yml") == []

    def test_the_suppression_machinery_is_gone(self):
        # Deleting the rule is what let these three go away.
        assert not hasattr(ss, "HASH_PATTERN")
        assert not hasattr(ss, "PINNED_ACTION_PATTERN")
        assert not hasattr(ss, "HASH_CONTEXT_HINTS")


class TestShapeTableIsData:
    def test_every_shape_declares_the_three_facts(self):
        for shape in ss.TOKEN_SHAPES:
            assert shape.label
            assert shape.prefix
            assert shape.min_length > 0
            assert shape.alphabet

    def test_prefixes_are_unique(self):
        prefixes = [shape.prefix for shape in ss.TOKEN_SHAPES]
        assert len(prefixes) == len(set(prefixes))

    def test_the_scanner_owns_no_regular_expressions(self):
        # The whole point of the rewrite: nothing here to misread in review.
        source = (ss.ROOT / "scripts" / "perf_eval" / "secrets_scan.py").read_text(encoding="utf-8")
        assert "import re" not in source
        assert "re.compile" not in source


class TestRunLength:
    def test_counts_only_the_allowed_alphabet(self):
        assert ss._run_length("abc!def", 0, ss.BASE62) == 3

    def test_zero_at_a_boundary(self):
        assert ss._run_length("!abc", 0, ss.BASE62) == 0

    def test_runs_to_end_of_string(self):
        assert ss._run_length("abc", 0, ss.BASE62) == 3


class TestAllowlist:
    @pytest.mark.parametrize(
        "rel",
        [
            "site/vendor/chart.umd.min.js",
            "_site/index.html",
            "node_modules/pkg/index.js",
            "site/assets/node_modules/x.js",
            ".venv/lib/x.py",
        ],
    )
    def test_skipped(self, rel):
        assert ss._is_allowlisted(rel) is True

    @pytest.mark.parametrize(
        "rel",
        [
            "site/index.html",
            "scripts/perf_eval/collect_artifacts.py",
            "scripts/perf_eval/secrets_scan.py",
            ".github/workflows/collect.yml",
            "README.md",
            "data/perf_eval.json",
        ],
    )
    def test_scanned(self, rel):
        assert ss._is_allowlisted(rel) is False

    def test_generated_data_is_scanned_now(self):
        # data/ was only excluded because the hash heuristic flagged the commit
        # SHAs in it. With that gone, generated output gets real coverage.
        assert ss._is_allowlisted("data/events.jsonl") is False

    def test_the_scanner_scans_itself(self):
        # It no longer needs to be exempt, because the shape table contains
        # bare prefixes rather than anything token-shaped.
        assert ss._is_allowlisted("scripts/perf_eval/secrets_scan.py") is False


class TestFindingsDoNotLeak:
    def test_the_report_truncates_the_match(self):
        finding = ss.scan_text(f"token = '{GITHUB_PAT}'", "demo.py")[0]
        assert GITHUB_PAT not in finding
        assert "ghp_" in finding


class TestRepositoryIsClean:
    def test_the_working_tree_has_no_detectable_secrets(self):
        findings: list[str] = []
        for path, rel in ss._iter_candidate_files(ss.ROOT):
            findings.extend(ss.scan_text(path.read_text(encoding="utf-8", errors="replace"), rel))
        assert findings == []
