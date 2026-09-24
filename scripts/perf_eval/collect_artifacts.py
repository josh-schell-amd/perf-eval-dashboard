#!/usr/bin/env python3
"""Collect AMD nightly perf-eval results from Buildkite artifacts into the store.

Scope: AMD workloads (``is_amd_workload``) in nightly builds (``is_nightly_build``).
Lists finished nightly builds on main, downloads each AMD workload's
``bench-*.json`` (perf) and ``results_*.json`` (accuracy), labels them from the
recipes at the build's perf-eval commit, and appends the new ones. Every
Buildkite and GitHub request is a GET.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf_eval import (  # noqa: E402
    BUILDKITE_API_BASE,
    BUILDKITE_ORG,
    BUILDKITE_PIPELINE_SLUG,
    WINDOW_DAYS,
    WORKLOAD_REPO,
)
from perf_eval.normalize import (  # noqa: E402
    commit_from_image,
    is_amd_workload,
    normalize_eval_payload,
    to_float,
    transform_perf,
    utcnow_iso,
)
from perf_eval.store import (  # noqa: E402
    ARTIFACT_MARKER_EVENT,
    EXPECTED_CONFIGS_EVENT,
    RESULT_EVENTS,
    append_events,
    artifact_key,
    artifact_keys_from_event,
    finished_at,
    read_events_strict,
    received_at,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STORE = ROOT / "data" / "events.jsonl"

AMD_IMAGE_REPO = "vllm/vllm-openai-rocm"

# Running out of retries is an error: an empty page would pass for a complete listing.
BK_GET_MAX_ATTEMPTS = 3
BK_GET_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504, 520, 522, 524})
BK_GET_RETRY_BACKOFF_SECONDS = 2

# "Nightly run 2026-06-30: commit 93d8f834dd8acf33eb0e2a75b2711b628cb6e226"
_NIGHTLY_MSG_RE = re.compile(
    r"nightly\s+run\s+(\d{4}-\d{2}-\d{2}).*?commit\s+([0-9a-f]{7,40})",
    re.IGNORECASE | re.DOTALL,
)
_NIGHTLY_WORD_RE = re.compile(r"\bnightly\b", re.IGNORECASE)

# ``results/<workload>/bench-<run>.json`` (perf) and
# ``results/<workload>/<task>/[<model>/]results_*.json`` (accuracy; lm-eval adds <model>).
_PERF_ARTIFACT_RE = re.compile(r"^results/(?P<wl>[^/]+)/bench-(?P<cfg>.+)\.json$")
_ACC_ARTIFACT_RE = re.compile(
    r"^results/(?P<wl>[^/]+)/(?P<task>[^/]+)/(?:[^/]+/)?results_[^/]*\.json$"
)
# Buildkite path filters: just the result files, not the much larger sample/log tree.
_RESULT_ARTIFACT_PATHS = (
    "*results/*/bench-*.json",
    "*results/*/*/results_*.json",
    "*results/*/*/*/results_*.json",
)
_RESULT_ARTIFACT_MAX_PAGES = 10

# Result events retain the exact Buildkite artifact that produced them.
_ARTIFACT_PROVENANCE_FIELDS = (
    "buildkite_artifact_id",
    "buildkite_artifact_job_id",
    "buildkite_artifact_path",
    "buildkite_artifact_sha1",
)

# Ingested builds are not re-listed, except the newest few: a retried job can
# add artifacts to a finished build.
DEFAULT_RECHECK_BUILDS = 3

# Enough for a cold start over the full window; a logic error stops here.
DEFAULT_MAX_REQUESTS = 1500


class RequestBudget:
    """Counts Buildkite requests, retries included, and raises past a ceiling."""

    def __init__(self, max_requests: int | None = DEFAULT_MAX_REQUESTS):
        self.listings = 0
        self.downloads = 0
        self.skipped_builds = 0
        self.max_requests = max_requests

    @property
    def total(self) -> int:
        return self.listings + self.downloads

    def charge(self, kind: str) -> None:
        if kind == "download":
            self.downloads += 1
        else:
            self.listings += 1
        if self.max_requests is not None and self.total > self.max_requests:
            raise RuntimeError(
                f"perf-eval Buildkite request ceiling exceeded: {self.total} > "
                f"{self.max_requests}. Re-run with a smaller --days, or raise "
                "--max-requests deliberately if a backfill genuinely needs it."
            )

    def summary(self) -> str:
        return (
            f"{self.total} Buildkite requests "
            f"({self.listings} listings, {self.downloads} downloads)"
        )


# ---------------------------------------------------------------------------
# Pure helpers (no I/O) — unit tested without a live Buildkite/GitHub
# ---------------------------------------------------------------------------


def use_system_certificates() -> bool:
    """Verify TLS against the OS trust store instead of certifi's bundle.

    Needed behind proxies that re-sign TLS with a CA only the OS trusts.
    Returns False if truststore is not installed.
    """
    try:
        import truststore
    except ModuleNotFoundError:
        log.debug("truststore not installed; using certifi's CA bundle")
        return False
    truststore.inject_into_ssl()
    return True


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def parse_tp(serve_args: str) -> int:
    """Effective parallel degree (TP * DP) from serve_args; defaults to 1.

    Mirrors perf-eval's ``parse_workload.parse_tp`` so per-GPU throughput is
    computed identically to what the pipeline posts to its own dashboard.
    """
    tokens = (serve_args or "").split()

    def find(*names: str) -> int | None:
        for index, token in enumerate(tokens):
            if "=" in token:
                key, _, value = token.partition("=")
                if key in names:
                    try:
                        return int(value)
                    except ValueError:
                        return None
            elif token in names and index + 1 < len(tokens):
                try:
                    return int(tokens[index + 1])
                except ValueError:
                    return None
        return None

    tp = find("--tensor-parallel-size", "-tp", "--tp") or 1
    dp = find("--data-parallel-size", "-dp", "--dp") or 1
    return tp * dp


def precision_from_model(model: str) -> str:
    """Infer a precision tag from the model id (mirrors parse_workload)."""
    name = (model or "").lower()
    for marker in ("fp4", "fp8", "int4", "int8", "bf16", "fp16"):
        if marker in name:
            return marker
    return "bf16"


def lm_eval_tasks(data: dict) -> list[str]:
    """The lm-eval task names a recipe declares, deduplicated and sorted.

    ``lm_eval.tasks`` entries are mappings with a ``name``; a bare string is
    accepted too, since the task name is all coverage needs. The name is the
    ``<task>`` directory in the artifact path, so it matches what the accuracy
    events carry.
    """
    block = data.get("lm_eval")
    if not isinstance(block, dict):
        return []
    names = set()
    for task in block.get("tasks") or []:
        if isinstance(task, dict):
            name = (task.get("name") or "").strip()
        elif isinstance(task, str):
            name = task.strip()
        else:
            continue
        if name:
            names.add(name)
    return sorted(names)


def workload_entry(data: dict) -> tuple[dict, dict]:
    """A recipe as (entry, configs by run name).

    Runs are named ``<config>-conc-<concurrency>``, as perf-eval expands them
    and names their artifacts (``bench-<run>.json``).
    """
    gpu = (data.get("gpu") or "").strip()
    vllm = data.get("vllm") or {}
    bench = data.get("vllm_bench") or {}
    meta = bench.get("metadata") or {}
    model = (vllm.get("model") or "").strip()
    serve_args = vllm.get("serve_args") or ""
    configs = {}
    for config in bench.get("configs") or []:
        name = config.get("name")
        if not name:
            continue
        raw_concurrency = config.get("max_concurrency")
        # A sweep is a list; a single value still expands to one suffixed run.
        concurrencies = raw_concurrency if isinstance(raw_concurrency, list) else [raw_concurrency]
        for concurrency in concurrencies:
            configs[f"{name}-conc-{concurrency}"] = {
                "isl": config.get("input_len"),
                "osl": config.get("output_len"),
                "conc": concurrency,
            }
    entry = {
        "name": (data.get("name") or "").strip(),
        "gpu": gpu,
        "device": (meta.get("device") or gpu.lower()).strip(),
        "tp": meta.get("tp") if meta.get("tp") is not None else parse_tp(serve_args),
        "precision": (meta.get("precision") or precision_from_model(model)).strip(),
        "model": model,
        # Only scheduled nightlies are in scope, so only they are expected.
        "nightly": data.get("nightly") is True,
        "accuracy_tasks": lm_eval_tasks(data),
    }
    return entry, configs


def _expected_workloads(workloads: dict[str, tuple[dict, dict]]):
    """The AMD nightly recipes, in workload order: the scope both expectations share."""
    for workload, (entry, configs) in sorted(workloads.items()):
        if not entry.get("nightly"):
            continue
        if not is_amd_workload(workload=workload, device=entry.get("device")):
            continue
        yield workload, entry, configs


def expected_accuracy(workloads: dict[str, tuple[dict, dict]]) -> list[dict]:
    """The AMD nightly accuracy results the recipes say should run.

    The perf side of coverage has always been recipe-derived; accuracy was
    inferred from the last WINDOW_DAYS of results instead, so a workload whose
    lm-eval step had been failing for longer than the window dropped out of its
    own denominator and stopped being reported missing. Reading it from
    ``lm_eval.tasks`` keeps a long outage visible, exactly as it does for perf.
    """
    out: list[dict] = []
    for workload, entry, _ in _expected_workloads(workloads):
        for task in entry.get("accuracy_tasks") or []:
            out.append(
                {
                    "workload": workload,
                    "model": entry.get("model") or "",
                    "device": entry.get("device") or "",
                    "task": task,
                }
            )
    return out


def expected_configs(workloads: dict[str, tuple[dict, dict]]) -> list[dict]:
    """The AMD nightly configs the recipes say should run, for the coverage card."""
    out: list[dict] = []
    for workload, entry, configs in _expected_workloads(workloads):
        for run_name, config in sorted(configs.items()):
            out.append(
                {
                    "workload": workload,
                    "run": run_name,
                    "model": entry.get("model") or "",
                    "device": entry.get("device") or "",
                    "precision": entry.get("precision") or "",
                    "tp": entry.get("tp"),
                    "isl": config.get("isl"),
                    "osl": config.get("osl"),
                    "conc": config.get("conc"),
                }
            )
    return out


def is_nightly_build(build: dict) -> bool:
    """Whether a build is a scheduled nightly: the nightly-run message, or on
    main, ``NIGHTLY=1`` or a scheduled build that says "nightly"."""
    branch = (build.get("branch") or "").strip()
    message = build.get("message") or ""
    env = build.get("env") or {}

    if _NIGHTLY_MSG_RE.search(message):
        return True
    if branch in {"main", "master"} and _truthy(env.get("NIGHTLY")):
        return True
    return (
        branch in {"main", "master"}
        and (build.get("source") or "") == "schedule"
        and bool(_NIGHTLY_WORD_RE.search(message))
    )


def nightly_info(build: dict) -> dict | None:
    """Return ``{vllm_commit, branch}`` for a nightly build, else None."""
    if not is_nightly_build(build):
        return None

    branch = (build.get("branch") or "").strip()
    env = build.get("env") or {}
    match = _NIGHTLY_MSG_RE.search(build.get("message") or "")
    commit = match.group(2) if match else None
    if not commit:
        commit = (env.get("VLLM_COMMIT") or "").strip() or commit_from_image(
            env.get("VLLM_IMAGE") or ""
        )
    return {"vllm_commit": commit, "branch": branch or "main"}


def amd_image(env: dict, vllm_commit: str) -> str:
    """Best-effort ROCm image URI for provenance / AMD detection."""
    image = (env.get("VLLM_IMAGE") or "").strip()
    if image and "rocm" in image.lower():
        return image
    if vllm_commit:
        return f"{AMD_IMAGE_REPO}:nightly-{vllm_commit}"
    return f"{AMD_IMAGE_REPO}:nightly"


def classify_artifact(path: str) -> tuple[str, str, str] | None:
    """Classify an artifact path.

    Returns ``("perf", workload, config)`` for a bench result,
    ``("accuracy", workload, task)`` for an lm-eval/bfcl result, or None for
    anything else (e.g. the large ``samples_*.jsonl`` we never download).
    """
    norm = (path or "").strip().lstrip("./")
    match = _PERF_ARTIFACT_RE.match(norm)
    if match:
        return "perf", match.group("wl"), match.group("cfg")
    match = _ACC_ARTIFACT_RE.match(norm)
    if match:
        return "accuracy", match.group("wl"), match.group("task")
    return None


def perf_event(
    raw: dict,
    *,
    entry: dict,
    config: dict,
    identity: dict,
) -> dict | None:
    """Build a canonical ``perf_result`` event from a raw bench artifact."""
    device = entry.get("device") or ""
    image = identity.get("image") or ""
    if not is_amd_workload(image=image, device=device, workload=entry.get("name")):
        return None
    # A crashed benchmark still writes a bench json; zero throughput would read
    # as a 100% regression.
    total = to_float(raw.get("total_token_throughput"))
    if total is None or total <= 0:
        log.warning(
            "Skipping bench result with no positive total_token_throughput (workload=%s build=%s)",
            entry.get("name"),
            identity.get("build_number"),
        )
        return None
    metrics = transform_perf(raw, tp=entry.get("tp") or 1)
    if not metrics:
        return None
    conc = config.get("conc")
    if conc is None:
        conc = raw.get("max_concurrency")
    return {
        "event": "perf_result",
        "received_at": utcnow_iso(),
        "nightly": True,
        "model": (raw.get("model_id") or entry.get("model") or "").strip(),
        "device": device,
        "precision": entry.get("precision") or "",
        "tp": entry.get("tp"),
        "isl": config.get("isl"),
        "osl": config.get("osl"),
        "conc": conc,
        "date": identity.get("date") or "",
        "build_number": identity.get("build_number"),
        "build_url": identity.get("build_url") or "",
        "build_commit": identity.get("build_commit") or "",
        "branch": identity.get("branch") or "main",
        "image": image,
        "vllm_commit": identity.get("vllm_commit") or "",
        # Floats: perf-eval takes the median across repetitions.
        "completed_requests": to_float(raw.get("completed")),
        "failed_requests": to_float(raw.get("failed")),
        "metrics": metrics,
    }


def accuracy_event(
    results_json: dict,
    *,
    workload: str,
    task: str,
    entry: dict,
    identity: dict,
) -> dict | None:
    """Build a canonical ``accuracy_result`` event from an lm-eval artifact."""
    payload = {
        "kind": "results",
        "workload": workload,
        "task": task,
        "device": entry.get("device") or "",
        "image": identity.get("image") or "",
        "vllm_commit": identity.get("vllm_commit") or "",
        "buildkite_build_number": identity.get("build_number"),
        "buildkite_build_url": identity.get("build_url") or "",
        "buildkite_commit": identity.get("build_commit") or "",
        "buildkite_branch": identity.get("branch") or "main",
        "nightly": True,
        "data": results_json,
    }
    event = normalize_eval_payload(payload)
    if event is None:
        return None
    event["nightly"] = True
    event["date"] = identity.get("date") or ""
    # The recipe's model id first: it is the same string the perf results for
    # this workload carry, so accuracy groups under the same model.
    event["model"] = (entry.get("model") or "").strip() or event.get("model") or workload
    return event


def event_key(event: dict) -> tuple:
    """Stable dedupe identity so re-runs never double-append the same result."""
    if event.get("event") == "perf_result":
        # With TP and precision: two recipes can run one model at one shape.
        return (
            "perf",
            event.get("build_number"),
            (event.get("model") or "").strip(),
            event.get("device"),
            event.get("tp"),
            event.get("precision") or "",
            event.get("isl"),
            event.get("osl"),
            event.get("conc"),
        )
    if event.get("event") == "accuracy_result":
        tasks = tuple(sorted((r.get("task"), r.get("metric")) for r in event.get("results") or []))
        return (
            "accuracy",
            event.get("build_number"),
            (event.get("workload") or "").strip(),
            tasks,
        )
    raise ValueError(f"not a result event: {event.get('event')!r}")


def artifact_provenance(artifact: dict, build_number: Any) -> dict:
    """The stable fields identifying an artifact; never its signed download URL."""
    return {
        "build_number": build_number,
        "buildkite_artifact_id": str(artifact.get("id") or "").strip(),
        "buildkite_artifact_job_id": str(artifact.get("job_id") or "").strip(),
        "buildkite_artifact_path": str(artifact.get("path") or "").strip().lstrip("./"),
        "buildkite_artifact_sha1": str(artifact.get("sha1sum") or "").strip().lower(),
    }


def artifact_marker(provenance: dict) -> dict:
    """Build a marker that compaction folds into the exact artifact index."""
    return {
        "event": ARTIFACT_MARKER_EVENT,
        "received_at": utcnow_iso(),
        **provenance,
    }


# ---------------------------------------------------------------------------
# I/O — Buildkite REST + GitHub raw (read-only)
# ---------------------------------------------------------------------------


def _bk_get(path: str, token: str, params: dict | None = None, budget: RequestBudget | None = None):
    url = f"{BUILDKITE_API_BASE}{path}"
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(1, BK_GET_MAX_ATTEMPTS + 1):
        try:
            if budget:
                budget.charge("listing")
            resp = requests.get(url, headers=headers, params=params, timeout=30)
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ):
            if attempt == BK_GET_MAX_ATTEMPTS:
                raise
            wait = BK_GET_RETRY_BACKOFF_SECONDS * attempt
            log.warning(
                "Buildkite request failed on %s, retry %d/%d in %ds",
                path,
                attempt,
                BK_GET_MAX_ATTEMPTS,
                wait,
            )
            time.sleep(wait)
            continue

        if resp.status_code in BK_GET_RETRYABLE_STATUS_CODES:
            if attempt == BK_GET_MAX_ATTEMPTS:
                # Fail closed after retry exhaustion; in particular, never
                # translate a 429 into an apparently complete empty page.
                resp.raise_for_status()
            try:
                retry_after = max(0, int(float(resp.headers.get("Retry-After", ""))))
            except (TypeError, ValueError):
                retry_after = BK_GET_RETRY_BACKOFF_SECONDS * attempt
            log.warning(
                "Buildkite returned HTTP %d on %s, retry %d/%d in %ds",
                resp.status_code,
                path,
                attempt,
                BK_GET_MAX_ATTEMPTS,
                retry_after,
            )
            time.sleep(retry_after)
            continue

        resp.raise_for_status()
        return resp.json()
    raise AssertionError("unreachable")


def _bk_paginate(
    path: str,
    token: str,
    params: dict | None = None,
    max_pages: int = 10,
    budget: RequestBudget | None = None,
):
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    params = dict(params or {})
    params.setdefault("per_page", 100)
    out: list = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        items = _bk_get(path, token, params, budget=budget)
        if not isinstance(items, list):
            raise RuntimeError(f"Buildkite returned a non-list page for {path}")
        if not items:
            return out
        out.extend(items)
        if len(items) < params["per_page"]:
            return out
        if page == max_pages:
            raise RuntimeError(
                f"Buildkite pagination safety cap reached for {path} after {max_pages} full pages"
            )
    raise AssertionError("unreachable")


def _bk_result_artifacts(
    build_number: Any, token: str, budget: RequestBudget | None = None
) -> list[dict]:
    """A build's result artifacts: one filtered listing per path pattern,
    sharing one page cap."""
    path = (
        f"/organizations/{BUILDKITE_ORG}/pipelines/{BUILDKITE_PIPELINE_SLUG}"
        f"/builds/{build_number}/artifacts"
    )
    remaining_pages = _RESULT_ARTIFACT_MAX_PAGES
    artifacts: list[dict] = []
    for path_filter in _RESULT_ARTIFACT_PATHS:
        if remaining_pages < 1:
            raise RuntimeError(
                f"Buildkite result artifact discovery safety cap reached for {path} "
                f"after {_RESULT_ARTIFACT_MAX_PAGES} pages"
            )
        rows = _bk_paginate(
            path,
            token,
            {"path": path_filter, "per_page": 100},
            max_pages=remaining_pages,
            budget=budget,
        )
        # A successful listing ends on one short (possibly empty) page. Count
        # that page as well as every full page across all the filters.
        remaining_pages -= len(rows) // 100 + 1
        artifacts.extend(rows)
    return artifacts


def _bk_download_json(
    download_url: str,
    token: str,
    budget: RequestBudget | None = None,
    *,
    label: str = "",
) -> dict | None:
    """Download a JSON artifact; None if it is permanently broken (4xx, not JSON).

    Transient failures raise once retries run out: an ingested build is not
    listed again, so skipping would lose the artifact. Logs use ``label``,
    never the URL, which is signed after the redirect.
    """
    for attempt in range(1, BK_GET_MAX_ATTEMPTS + 1):
        if budget:
            budget.charge("download")
        try:
            resp = requests.get(
                download_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=60,
                allow_redirects=True,
            )
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            if attempt == BK_GET_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Download of artifact {label or '?'} failed after "
                    f"{BK_GET_MAX_ATTEMPTS} attempts: {type(exc).__name__}"
                ) from None
            time.sleep(BK_GET_RETRY_BACKOFF_SECONDS * attempt)
            continue

        if resp.status_code in BK_GET_RETRYABLE_STATUS_CODES:
            if attempt == BK_GET_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Download of artifact {label or '?'} returned HTTP "
                    f"{resp.status_code} after {BK_GET_MAX_ATTEMPTS} attempts"
                )
            log.warning(
                "Artifact %s returned HTTP %d, retry %d/%d",
                label or "?",
                resp.status_code,
                attempt,
                BK_GET_MAX_ATTEMPTS,
            )
            time.sleep(BK_GET_RETRY_BACKOFF_SECONDS * attempt)
            continue

        if resp.status_code >= 400:
            log.warning("Skipping artifact %s: HTTP %d", label or "?", resp.status_code)
            return None
        try:
            return resp.json()
        except ValueError:
            log.warning("Skipping artifact %s: body is not JSON", label or "?")
            return None
    raise AssertionError("unreachable")


def fetch_workload_map(gh_token: str, ref: str) -> dict[str, tuple[dict, dict]]:
    """Every workload recipe at commit ``ref``, keyed by its ``name`` (which
    artifact paths use; the filename differs). A duplicated key is logged:
    YAML keeps only the last, as perf-eval does."""
    import yaml

    class RecipeLoader(yaml.SafeLoader):
        """SafeLoader that records duplicate mapping keys instead of hiding them."""

        def __init__(self, stream):
            super().__init__(stream)
            self.duplicate_keys: list[str] = []

        def construct_mapping(self, node, deep=False):
            seen: set = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in seen:
                    self.duplicate_keys.append(str(key))
                seen.add(key)
            return super().construct_mapping(node, deep=deep)

    def load_recipe(text: str, name: str):
        loader = RecipeLoader(text)
        try:
            data = loader.get_single_data()
            for key in loader.duplicate_keys:
                log.warning(
                    "Workload %s declares %r more than once; YAML keeps only the last, "
                    "so the earlier block never runs and is missing from coverage",
                    name,
                    key,
                )
            return data
        finally:
            loader.dispose()

    headers = {"Accept": "application/vnd.github+json"}
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
    listing = requests.get(
        f"https://api.github.com/repos/{WORKLOAD_REPO}/contents/workloads",
        headers=headers,
        params={"ref": ref},
        timeout=30,
    )
    listing.raise_for_status()
    out: dict[str, tuple[dict, dict]] = {}
    for item in listing.json():
        name = item.get("name") or ""
        if not name.endswith((".yaml", ".yml")):
            continue
        raw = requests.get(item["download_url"], headers=headers, timeout=30)
        raw.raise_for_status()
        try:
            data = load_recipe(raw.text, name)
        except yaml.YAMLError as exc:
            log.warning("Skipping unparseable workload %s: %s", name, exc)
            continue
        if not isinstance(data, dict) or not data.get("name"):
            continue
        entry, configs = workload_entry(data)
        out[entry["name"]] = (entry, configs)
    log.info("Loaded %d workload recipes", len(out))
    return out


def collect(
    store_path: Path,
    *,
    days: int,
    bk_token: str,
    gh_token: str,
    dry_run: bool = False,
    recheck_builds: int = DEFAULT_RECHECK_BUILDS,
    budget: RequestBudget | None = None,
) -> int:
    """Pull AMD nightly artifacts and append the new events; return how many.

    ``dry_run`` lists and reports the cost but downloads and writes nothing.
    """
    # No further back than the store keeps.
    if not 1 <= days <= WINDOW_DAYS:
        raise ValueError(f"perf-eval artifact lookback must be between 1 and {WINDOW_DAYS} days")
    budget = budget if budget is not None else RequestBudget()
    existing = read_events_strict(store_path)
    seen = {event_key(e) for e in existing if e.get("event") in RESULT_EVENTS}
    known_artifacts = {key for event in existing for key in artifact_keys_from_event(event)}
    ingested_builds = {
        str(event["build_number"])
        for event in existing
        if event.get("event") in RESULT_EVENTS and event.get("build_number") is not None
    }
    before = len(existing)

    # Builds are read against the recipes they ran, not main's: a recipe can
    # change TP after a build, and TP divides throughput into per-GPU numbers.
    recipes_at: dict[str, dict[str, tuple[dict, dict]]] = {}

    def recipes_for(commit: str) -> dict[str, tuple[dict, dict]]:
        if commit not in recipes_at:
            recipes_at[commit] = fetch_workload_map(gh_token, ref=commit)
        return recipes_at[commit]

    cutoff = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    builds = _bk_paginate(
        f"/organizations/{BUILDKITE_ORG}/pipelines/{BUILDKITE_PIPELINE_SLUG}/builds",
        bk_token,
        {"branch": "main", "state": "finished", "created_from": cutoff},
        budget=budget,
    )

    # Newest first, so the re-check window is the newest builds.
    nightlies = [(build, info) for build in builds if (info := nightly_info(build)) is not None]
    nightlies.sort(key=lambda pair: pair[0].get("number") or 0, reverse=True)
    log.info(
        "Examining %d finished builds since %s: %d nightlies",
        len(builds),
        cutoff,
        len(nightlies),
    )

    appended = 0
    markers_appended = 0
    discovered = 0
    would_download = 0
    pending_events: list[dict] = []
    for position, (build, night) in enumerate(nightlies):
        number = build.get("number")
        if position >= recheck_builds and str(number) in ingested_builds:
            budget.skipped_builds += 1
            continue
        if not build.get("commit"):
            log.warning("Build #%s has no perf-eval commit, so no recipes; skipping", number)
            continue
        identity = {
            "build_number": number,
            "build_url": build.get("web_url") or "",
            "build_commit": build["commit"],  # perf-eval repo commit
            "branch": night["branch"],
            "vllm_commit": night["vllm_commit"],
            "date": build.get("finished_at") or "",
            "image": amd_image(build.get("env") or {}, night["vllm_commit"]),
        }
        for artifact in _bk_result_artifacts(number, bk_token, budget=budget):
            kind = classify_artifact(artifact.get("path") or "")
            if kind is None:
                continue
            discovered += 1
            provenance = artifact_provenance(artifact, number)
            source_key = artifact_key(provenance)
            if source_key is not None and source_key in known_artifacts:
                continue
            _, workload, tail = kind
            if not is_amd_workload(workload=workload):
                continue
            recipe = recipes_for(identity["build_commit"]).get(workload)
            if recipe is None:
                log.warning("No recipe for workload %s (build #%s); skipping", workload, number)
                continue
            entry, configs = recipe
            if kind[0] == "perf" and tail not in configs:
                log.warning(
                    "No run %s in workload %s (build #%s); skipping", tail, workload, number
                )
                continue
            would_download += 1
            if dry_run:
                continue
            payload = _bk_download_json(
                artifact.get("download_url") or "",
                bk_token,
                budget=budget,
                label=f"#{number} {provenance['buildkite_artifact_path']}",
            )
            if payload is None:
                continue
            if kind[0] == "perf":
                event = perf_event(payload, entry=entry, config=configs[tail], identity=identity)
            else:
                event = accuracy_event(
                    payload, workload=workload, task=tail, entry=entry, identity=identity
                )
            if event is None:
                continue
            key = event_key(event)
            if key not in seen:
                event.update({field: provenance[field] for field in _ARTIFACT_PROVENANCE_FIELDS})
                pending_events.append(event)
                seen.add(key)
                appended += 1
            elif source_key is not None:
                # Already stored without its artifact ID; record just the ID.
                pending_events.append(artifact_marker(provenance))
                markers_appended += 1
            if source_key is not None:
                known_artifacts.add(source_key)

    if dry_run:
        log.info(
            "DRY RUN — nothing downloaded, nothing written.\n"
            "  nightlies in window ...... %d\n"
            "  re-listed ................ %d (newest %d always, plus any not yet ingested)\n"
            "  skipped as ingested ...... %d\n"
            "  result artifacts seen .... %d\n"
            "  would download ........... %d\n"
            "  cost so far .............. %s\n"
            "  a real run would cost .... %d requests total",
            len(nightlies),
            len(nightlies) - budget.skipped_builds,
            recheck_builds,
            budget.skipped_builds,
            discovered,
            would_download,
            budget.summary(),
            budget.total + would_download,
        )
        return 0

    # Coverage expects what the latest nightly's recipes define. Recipes at a
    # commit never change, so a snapshot is only taken for a new commit.
    epoch = datetime.min.replace(tzinfo=UTC)
    latest = max(
        (e for e in (*existing, *pending_events) if e.get("event") == "perf_result"),
        key=lambda e: finished_at(e) or epoch,
        default=None,
    )
    previous_expected = max(
        (e for e in existing if e.get("event") == EXPECTED_CONFIGS_EVENT),
        key=lambda e: received_at(e) or epoch,
        default=None,
    )
    recipe_commit = (latest or {}).get("build_commit") or ""
    new_commit = (previous_expected or {}).get("recipe_commit") != recipe_commit
    # A snapshot taken before the expectation covered accuracy has no
    # `accuracy` key. Refetch once rather than waiting for an unrelated recipe
    # change, which could be weeks away. Writing the key, even empty, settles it.
    lacks_accuracy = previous_expected is not None and "accuracy" not in previous_expected
    if recipe_commit and (new_commit or lacks_accuracy):
        recipes = recipes_for(recipe_commit)
        expected = expected_configs(recipes)
        if expected:
            pending_events.append(
                {
                    "event": EXPECTED_CONFIGS_EVENT,
                    "received_at": utcnow_iso(),
                    "recipe_commit": recipe_commit,
                    "configs": expected,
                    "accuracy": expected_accuracy(recipes),
                }
            )

    # Rewrite even with nothing new, so results that aged out are dropped.
    append_events(store_path, pending_events)
    log.info(
        "Appended %d new result events and %d artifact markers (%d existing records). "
        "Skipped re-listing %d already-ingested builds. Cost: %s",
        appended,
        markers_appended,
        before,
        budget.skipped_builds,
        budget.summary(),
    )
    return appended


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=WINDOW_DAYS,
        help=f"Lookback window in days, 1-{WINDOW_DAYS} (default: {WINDOW_DAYS})",
    )
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="Path to events.jsonl")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List and report the request cost without downloading or writing anything",
    )
    parser.add_argument(
        "--recheck-builds",
        type=int,
        default=DEFAULT_RECHECK_BUILDS,
        help=(
            "Always re-list the newest N nightlies even if already ingested, to catch "
            f"artifacts from a retried job (default: {DEFAULT_RECHECK_BUILDS})"
        ),
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS,
        help=(
            "Abort if outbound Buildkite requests would exceed this "
            f"(default: {DEFAULT_MAX_REQUESTS}; 0 disables the ceiling)"
        ),
    )
    args = parser.parse_args()

    # Before any connection is made.
    use_system_certificates()

    bk_token = os.getenv("BUILDKITE_TOKEN") or ""
    if not bk_token:
        log.error("BUILDKITE_TOKEN not set; cannot pull perf-eval artifacts")
        return 1
    gh_token = os.getenv("GITHUB_TOKEN") or ""

    collect(
        Path(args.store),
        days=args.days,
        bk_token=bk_token,
        gh_token=gh_token,
        dry_run=args.dry_run,
        recheck_builds=args.recheck_builds,
        budget=RequestBudget(args.max_requests or None),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
