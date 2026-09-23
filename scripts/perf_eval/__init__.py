"""Collectors for the AMD nightly perf-eval dashboard.

Scope, stated once here and repeated in every module that enforces it: this
dashboard covers **AMD (MI-series) workloads from scheduled nightly builds
only**. NVIDIA workloads (H200/B200/A100) run in the same upstream
``vllm/perf-eval`` pipeline and are deliberately excluded, as are ad-hoc and
pull-request builds. See ``README.md`` for the rationale. NOTE: come back to this *josh*
"""

BUILDKITE_ORG = "vllm"
BUILDKITE_PIPELINE_SLUG = "perf-eval"
BUILDKITE_API_BASE = "https://api.buildkite.com/v2"
PIPELINE_URL = f"https://buildkite.com/{BUILDKITE_ORG}/{BUILDKITE_PIPELINE_SLUG}"

# Public repo holding the workload recipes (device / tp / precision / bench
# sizes). Read anonymously or with the workflow's built-in GITHUB_TOKEN.
WORKLOAD_REPO = "vllm-project/perf-eval"
