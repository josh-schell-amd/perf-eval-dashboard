"""The AMD nightly perf-eval dashboard's pipeline.

Scope: AMD (MI-series) workloads from scheduled nightly builds only; NVIDIA
workloads and ad-hoc builds from the same pipeline are excluded.
"""

BUILDKITE_ORG = "vllm"
BUILDKITE_PIPELINE_SLUG = "perf-eval"
BUILDKITE_API_BASE = "https://api.buildkite.com/v2"
PIPELINE_URL = f"https://buildkite.com/{BUILDKITE_ORG}/{BUILDKITE_PIPELINE_SLUG}"

# Public repo holding the workload recipes.
WORKLOAD_REPO = "vllm-project/perf-eval"

# Days of nightlies shown, collected, stored and published.
WINDOW_DAYS = 14
