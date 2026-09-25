# Perf Eval Dashboard

A static dashboard for AMD nightly performance and accuracy results from the
[`vllm/perf-eval`](https://buildkite.com/vllm/perf-eval) Buildkite pipeline.

> [!IMPORTANT]
> This is an experimental demo. It is not a supported or authoritative source
> of performance or accuracy results, and should not be relied on beyond
> demonstration purposes.

- **One page, no build step.** The whole frontend is `site/index.html`: inline
  CSS and vanilla JavaScript, plus a vendored copy of Chart.js.
- **One data file.** The page fetches a single `perf_eval.json` published next
  to it.
- **Nothing fetched at runtime** beyond that file. No CDN, no fonts, no
  analytics, no trackers.
- **Collected once a day** from Buildkite by a GitHub Actions workflow, and
  served from GitHub Pages.

## Contents

- [What this dashboard covers](#what-this-dashboard-covers)
- [Running it locally](#running-it-locally)
- [Deploying](#deploying)
- [Reading the dashboard](#reading-the-dashboard)
- [How regressions are detected](#how-regressions-are-detected)
- [How the data flows](#how-the-data-flows)
- [Development](#development)
- [Licence](#licence)

---

## What this dashboard covers

AMD nightlies only. If a run looks missing, check it is in scope first.

| | Included | Excluded |
|---|---|---|
| Hardware | AMD MI-series (`mi300x`, `mi355x`, …) | NVIDIA (H200, B200, A100) |
| Runs | Scheduled nightlies | Ad-hoc and pull-request builds |
| Branch | `main` | Every other branch |
| Build state | `finished` | Running, cancelled, failed-to-start |

- **NVIDIA** results are on [perf.vllm.ai](https://perf.vllm.ai).
- **Nightlies only**, because each runs the full matrix and so compares with
  the last. An ad-hoc build may cover one workload at one concurrency.

---

## Running it locally

### Set up

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -c constraints.txt -e ".[dev]"
```

`constraints.txt` pins every installed package to an exact version, including
dependencies of dependencies such as `urllib3` and `certifi`, so CI and your
machine run identical tools.

### Run the pipeline against real data

This is the same sequence the workflow runs, minus the branch commits. Ingest
appends to `data/events.jsonl`, so re-running only adds what is new, and it
is read-only against Buildkite, so it is safe to re-run as often as you like.

```bash
export BUILDKITE_TOKEN=bkua_...          # read-only: Read Builds + Read Artifacts
export GITHUB_TOKEN="$(gh auth token)"   # optional, but see the tip below

python scripts/perf_eval/collect_artifacts.py --days 14
python scripts/perf_eval/aggregate.py
python scripts/build_site.py
python -m http.server --directory _site 8000
```

Then open <http://localhost:8000>. `aggregate.py` and `build_site.py` only
read the local store, so the Buildkite token is needed for ingest and nothing
else.

> [!TIP]
> **Set `GITHUB_TOKEN` if you can.** It reads the public workload recipes from
> `vllm-project/perf-eval`, and GitHub rate-limits anonymous requests much more
> tightly. If the recipes fail to load you will see
> `No recipe for workload <name>; skipping` and an empty dashboard.

### Checks

```bash
pytest
ruff check . && ruff format --check .
pyright                      # type check, gating in CI
ty check scripts tests       # second opinion, advisory
python scripts/perf_eval/secrets_scan.py
```

---

## Deploying

### 1. Add the Buildkite token

Nothing is stored in this repository.

| Secret | Purpose | Scope needed |
|---|---|---|
| `BUILDKITE_TOKEN` | List builds, list and download artifacts | **Read-only**: Read Builds + Read Artifacts |
| `GITHUB_TOKEN` | Read the public workload recipes | The workflow's built-in token |

Add `BUILDKITE_TOKEN` as a **GitHub Actions repository secret** with that
exact name: Settings → Secrets and variables → Actions → New repository
secret, or `gh secret set BUILDKITE_TOKEN`. `GITHUB_TOKEN` needs nothing from
you; Actions provides it.

> [!WARNING]
> Use a **read-only** Buildkite token. The collector only issues GETs, and
> `tests/test_token_safety.py` asserts that, but a token with write scopes
> would still be more than it needs.

### 2. Point GitHub Pages at `gh-pages`

Settings → Pages → Deploy from a branch → `gh-pages` → `/root`.

The first workflow run creates the `dashboard-state` branch itself, as an
orphan, so there is no branch setup to do by hand.

### 3. Turn on push protection

Enable GitHub **secret scanning with push protection** on the repository. See
[Secret scanning](#secret-scanning) for why.

### How the token is kept safe

- It is injected as step-scoped `env:` on the single ingest step and read via
  `os.getenv` with a fail-closed check.
- `tests/test_token_safety.py` pins the Buildkite org to `vllm`, asserts only
  `collect_artifacts.py` can read the token, and asserts it never issues a
  write.
- Pushing checkouts use `persist-credentials: false`.
- Every push and pull request is secret-scanned.

---

## Reading the dashboard

### KPI cards

Each card answers "is tonight's build healthy?" Every card except *Latest
nightly* follows the filters.

| Card | Shows |
|---|---|
| Latest nightly | Date, vLLM commit, and a link to the build |
| Performance overnight | Median Output Throughput change vs the previous nightly |
| Regressions overnight | Config-metric pairs at least 0.5% worse |
| Improvements overnight | Config-metric pairs at least 0.5% better |
| Accuracy overnight | Models whose accuracy dropped at least 1 point |
| Coverage | What reported, against what the recipes expect |

> [!IMPORTANT]
> On *Performance overnight*, red just means the median went down. It is not
> a regression alarm.

### Tabs

Filters, the tab, the picked metric and the chart window are all kept in the
URL, so **Copy link** reproduces the view.

**Performance** (default)
- One bar chart per model and device, since per-GPU numbers don't compare
  across devices.
- Darker bar = higher concurrency. Red outline = at least 0.5% worse than
  last night. Faded = not in the newest nightly.
- Hover for the value, build, commit and change: `vs last night (#598)`, or
  `vs 8 nights ago (#586)` if the config skipped nights, or *No previous run*.
- Click a bar for its history. The table below expands per row.

**Throughput vs Latency**
- Interactivity (1 / TPOT) against Total Throughput, one curve per shape
  across concurrency, per model and device.
- A concurrency scaling chart, and a throughput heatmap by ISL/OSL ×
  concurrency.
- Red ring = throughput regression overnight.

**Trends**
- One chart per metric. Red line = regressed on that metric, newest point
  ringed.
- The chart window (1–14 days) changes the charts only, never the regression
  counts.

**Accuracy**
- lm-eval score per model, device and task, against the previous nightly.
- Flexible-extract is preferred over strict-match, which also grades format
  (gpt-oss-120b on gsm8k: 52% strict, 76% flexible).

**Configurations**
- Every metric's newest value per config. Grey = under the threshold, or
  spanning more than one night.

**Coverage panel** (bottom of the page)
- Configs the recipes expect but the newest build didn't report, grouped by
  workload.
- It matters because a workload that OOMs just stops reporting: it vanishes
  from the averages instead of showing as a regression.

### Colour

- **Red = regression, and nothing else.** The palette has no red or pink.
- **Per metric:** a config turns red only on the charts where it regressed.
- **Chips by size:** yellow up to 2.5%, orange up to 5%, red above 5%.
- **Green = improvement.**

> [!NOTE]
> Adding a colour? Add it to both `PALETTE` and `PALETTE_LIGHT`, in the same
> position, and keep it out of the 0–20° hue range.

### Our throughput numbers are not ATOM's

They are the same measurements, normalized differently. Check this before
comparing the two dashboards number for number.

| ATOM | Here | Relationship |
|---|---|---|
| Total Throughput (tok/s) | `tput_per_gpu`, "Total Throughput" | ours = ATOM ÷ TP |
| Output Throughput (tok/s) | `output_tput_per_gpu`, "Output Throughput" | ours = ATOM ÷ TP |
| Interac., `1000 / TPOT` (tok/s/user) | `mean_intvty`, "Interactivity" | identical |
| TPOT, TTFT (ms) | `mean_tpot`, `mean_ttft` (stored in s) | identical |

`transform_perf` divides by TP once at ingest, so every value here is already
per-GPU, which is what makes two devices comparable. ATOM reports what the
harness emitted and divides by GPU count only in its tradeoff charts. The
`/GPU` lives in the **unit**, not the label, and axis titles and tooltips read
both out of `metric_meta`, so they cannot go stale when a metric is renamed.

---

## How regressions are detected

**One rule: the newest nightly against the run before it.**

| | Counts as a change at |
|---|---|
| Performance metrics | **0.5%** or more |
| Accuracy scores | **1 point** or more (0.01 on the 0–1 scale) |

Smaller moves are neutral, and the page says how many smaller moves it did
not count wherever it reports a count. Tests pin both thresholds.

### Only overnight changes count

The overnight signals only count configs that reported in the newest nightly
**and** the one before it.

- A config that **skipped tonight** still has two earlier points, but that
  change happened on earlier nights. Counting it would pass an old change off
  as tonight's.
- A config that **ran tonight after skipping** is compared against whatever
  older run it has, so its change spans every night it missed. A config back
  from a two-week outage would otherwise report two weeks of drift as one
  night's regression.

Neither is counted. Configs that did not report show up in Coverage instead,
and a returning config is compared normally from its second consecutive night.
Every perf regression signal on the page goes through one predicate,
`comparedOvernight` in `site/index.html`. The Accuracy tab does not use it yet.

When nothing could be compared at all, the regression panel says *Nothing to
compare* rather than showing a green *No regressions*.

### Why these thresholds

**0.5% is chosen, not measured.** With no threshold, about half the
regressions flagged on a typical night were under 0.5%, the smallest 0.015%.
Once `repetitions: 3` lands on the AMD recipes, the spread across those
repetitions is a real noise floor and the threshold should be derived from it.

**Accuracy is not reproducible night to night**, despite the fixed dataset:
every AMD workload's gsm8k score moves every night. One gsm8k question out of
1,319 is worth 0.08 points and the run-to-run spread is about 0.7 points, so
the 1-point threshold sits just above it.

A run where some requests failed is not comparable to a clean one, so its
failed count shows in the chart tooltips and on its regression row.

### No smoothing, on purpose

Reducing measurement noise is the benchmark's job. perf-eval's
`lib/aggregate_perf.py` repeats a benchmark `repetitions` times on the same
warm server and median-aggregates every field before ingestion. Averaging
again here would blur the night-to-night change this dashboard exists to show.
If a metric is too noisy, the fix is a higher `repetitions` in the recipe.

> [!WARNING]
> **Known upstream gap.** Every NVIDIA workload in `vllm-project/perf-eval`
> sets `repetitions: 3`. None of the eleven AMD (`mi300x`/`mi355x`) workloads
> set it, so they default to `1`: every point on this dashboard is a single,
> unaggregated run. Raising `repetitions` on those recipes is the right way to
> make night-to-night comparison trustworthy.

### Time window

The view shows a trailing **14 days**, anchored to *now*. If the nightly stops
reporting, the dashboard goes empty and says so, with a link to the pipeline,
rather than presenting old numbers as current. The window can be narrowed,
never widened; it comes from `display_window_days` in the payload, which
`aggregate.py` owns.

Detection runs in the page rather than in `aggregate.py`, because the window
decides which run is "latest" and which is its predecessor. The per-metric
`status` in the payload applies the same rule over whole history, for machine
consumers of the JSON.

---

## How the data flows

```
Buildkite vllm/perf-eval            GitHub vllm-project/perf-eval
  artifact_paths: results/**/*        workloads/*.yaml (device, tp, precision)
            |                                     |
            +------------------+------------------+
                               |
                   collect_artifacts.py          read-only GETs only
                               |
                      data/events.jsonl          branch: dashboard-state
                               |
                        aggregate.py
                               |
                    data/perf_eval.json
                               |
                        build_site.py
                               |
                    site/ + payload -> _site/    branch: gh-pages
```

The upstream pipeline uploads its whole `results/` tree as Buildkite
artifacts, so the collector reads raw `bench-*.json` and `results_*.json`
files directly. Nothing has to be pushed to us, and no webhook receiver needs
hosting.

| Branch | Holds |
|---|---|
| `main` | Source only, so data commits never pollute its history |
| `dashboard-state` | `data/events.jsonl`, the unpublished event log, which churns on every collection |
| `gh-pages` | The built site and `perf_eval.json` |

> [!CAUTION]
> `dashboard-state` is unpublished (Pages serves only `gh-pages`), but it is an
> ordinary branch: anyone who can read the repository can read it.

### Collection schedule

`collect.yml` collects once a day at **17:17 UTC** (about noon US Central), and
on manual dispatch. Recent nightlies have mostly finished between about 10:30
and 13:00 UTC; one that finishes later is picked up the next day, or straight
away with a manual run. There is no build-finished trigger, since that would
need a hosted webhook endpoint.

A push that touches `site/`, `scripts/build_site.py` or the workflow only
rebuilds and redeploys the page from stored data; it never calls Buildkite.

Each run scans the last 14 days of finished `main` builds (1–14 via the
dispatch input). A build that already has results is not listed again once it
leaves the re-check window, so to **backfill** artifacts an older collector
missed, dispatch with `recheck_builds` (0–60); `30` re-lists every nightly in
the window, and results already stored are skipped.

### Failure handling

- A download that keeps failing with a timeout, 429 or gateway error **fails
  the run** after 3 attempts, since skipping it would lose that artifact for
  good.
- A 4xx, or a body that is not JSON, is skipped with a warning, so one broken
  artifact cannot block every later collection.
- A bench result with no positive `total_token_throughput` is a failed
  benchmark, and is skipped rather than published as zero throughput.
- The store is written only after the whole scan completes, so an abort leaves
  the previous state intact.

### What a run costs Buildkite

| | Requests |
|---|---|
| Builds listing | 1 |
| Artifact listing | 3 per nightly inside the re-check window (one per path filter) |
| Download | 1 per artifact not already ingested |

The re-check window defaults to the newest 3 nightlies, because a nightly can
finish with a failed workload that someone retries later, adding artifacts to
the same build. A steady-state run is `1 + 3×3 = 10` listings plus the new
nightly's artifacts.

> [!TIP]
> **See the cost before you spend it.** `--dry-run` does the listings, reports
> exactly what it would download, and stops, with no downloads and no writes:
>
> ```bash
> BUILDKITE_TOKEN=bkua_... python scripts/perf_eval/collect_artifacts.py --dry-run --days 14
> ```

<details>
<summary>How the cost is bounded</summary>

- `--max-requests` (default 1500) aborts the run rather than continuing. An
  unexpected request volume is a bug worth stopping on.
- Artifact listings use narrow path filters (`*results/*/bench-*.json`,
  `*results/*/*/results_*.json` and `*results/*/*/*/results_*.json`), so the
  pipeline's much larger sample and log tree is never enumerated. The deepest
  one exists because lm-eval writes its results into a subdirectory named
  after the model, under the task directory perf-eval gives it.
- Pagination is capped (10 pages for builds, a shared 10-page budget per build
  for artifacts) and raises rather than looping.
- Retries are capped at 3 attempts, honour `Retry-After`, and only apply to
  gateway-ish codes (429, 502, 503, 504, 520, 522, 524). A 500 is not retried,
  since it is usually persistent.
- Retry attempts are charged to the budget, so the reported total is what
  Buildkite actually saw.
- The workflow's `concurrency` group prevents two runs overlapping.

</details>

### Redundant deploys are skipped

`aggregate.py` restamps `generated_at` on every run, so the payload always
differs byte for byte even when no new nightly arrived, and Pages allows only
about ten builds an hour. `payload_changed.py` compares the fresh payload with
the one **live on `gh-pages`**, ignoring `generated_at`, and the deploy is
gated on the result.

It compares against the live copy, not one saved with the event store, so a
failed deploy cannot count as published. The skip applies **only to scheduled
runs**: a push to `site/` or a manual dispatch always republishes. A missing
or unreadable previous payload counts as changed, so the failure mode is one
redundant deploy, not a silently unpublished update.

### Retention

One number, `WINDOW_DAYS = 14` in `scripts/perf_eval/__init__.py`. The page
shows 14 days, so that is all anything keeps:

- `data/events.jsonl` keeps the last 14 days of nightly results, the newest
  recipe snapshot, and the IDs of artifacts downloaded in that time.
- `data/perf_eval.json` publishes the last 14 days of nightlies.
- The collector looks back at most 14 days, so it never re-lists a build whose
  results were already dropped.

Both files are written atomically (temp file, then rename). Fourteen days of
the log is about half a megabyte.

### Data identity

- **A series is what ran:** model, device, precision, TP, ISL/OSL and
  concurrency. Changing any of those starts a new line, and a regression is
  only ever measured within one line. Renaming a run with the same values
  continues it; a removed config's line stops and ages out, and is not
  reported missing.
- **Each build is read against the recipes at the perf-eval commit it ran,**
  never against `main`, so a recipe change landing after a nightly does not
  relabel it.
- **A nightly is identified by its vLLM commit**, falling back to build
  number. A nightly re-run on the same commit folds into one data point. When
  two observations share an identity, the newer timestamp wins, not the later
  position in the log.
- **Accuracy takes its model id from the workload recipe**, because lm-eval's
  `config.model` names the client backend (`local-completions`). Older events
  are repaired at aggregation. Bookkeeping keys such as `sample_len` are
  dropped.

<details>
<summary>The published payload, <code>perf_eval.json</code></summary>

This is the contract between the collectors and the page. `metric_meta`
carries `direction`, so the page colours a new metric correctly without a
frontend change, and `baselines` declares the comparison rule so its labels
and thresholds come from data.

```jsonc
{
  "generated_at": "2026-01-01T00:00:00Z",
  "scope":      { "hardware": "amd", "runs": "nightly", "description": "..." },
  "pipeline":   { "org": "vllm", "slug": "perf-eval", "url": "..." },
  "metric_meta": { "tput_per_gpu": { "label": "...", "unit": "tok/s/GPU", "direction": "higher" } },
  "thresholds": { "perf_rel": 0.005, "accuracy_abs": 0.01 },
  // What the recipes say should run, for Coverage. The page cannot reach
  // GitHub, so the collector snapshots it into the store.
  "expected": {
    "recorded_at": "2026-01-01T00:00:00Z",
    "configs":  [{ "workload": "wl-mi355x", "run": "8k-in-1k-out-conc-128", "model": "org/Model",
                   "device": "mi355x", "precision": "fp8", "tp": 8,
                   "isl": 8192, "osl": 1024, "conc": 128 }],
    "accuracy": [{ "workload": "wl-mi355x", "model": "org/Model",
                   "device": "mi355x", "task": "gsm8k" }]
  },
  "models": [{
    "model": "org/Model",
    "devices": ["mi355x"],
    "nightly_count": 12,
    "latest": { "date": "...", "vllm_commit": "...", "image": "...", "build_url": "..." },
    "perf_configs": [{
      "device": "mi355x", "isl": 8192, "osl": 1024, "conc": 128, "tp": 8,
      "label": "8K in / 1K out @ conc 128 (MI355X)",
      "metrics": {
        "tput_per_gpu": {
          "latest": 1234.5, "previous": 1200.0, "delta": 34.5, "delta_pct": 2.875,
          "direction": "higher", "status": "good", "label": "...", "unit": "tok/s/GPU",
          "series": [{ "date": "...", "value": 1200.0, "vllm_commit": "...", "build_url": "...",
                       "completed_requests": 512, "failed_requests": 0 }]
        }
      }
    }],
    "accuracy_tasks": [{
      "task": "gsm8k", "metric": "exact_match,strict-match", "primary": true,
      "latest": 0.81, "previous": 0.80, "status": "good", "series": [ /* ... */ ]
    }]
  }],
  "summary":   { "models": 1, "amd_devices": ["mi355x"], "nightlies": 12, "perf_points": 96, "accuracy_points": 12 },
  "retention": { "display_window_days": 14 }
}
```

</details>

---

## Development

### Layout

```
site/index.html              the entire dashboard
site/vendor/                 Chart.js and the AMD logo, with provenance
scripts/perf_eval/
  normalize.py               metric registry, AMD filter, event normalizers
  store.py                   events.jsonl: atomic writes, 14-day retention
  collect_artifacts.py       Buildkite REST -> canonical events
  aggregate.py               events.jsonl -> perf_eval.json
  merge_events.py            identity-based merge of two stores
  secrets_scan.py
scripts/build_site.py        site/ + perf_eval.json -> _site/
data/                        generated; lives on dashboard-state, gitignored here
tests/
.github/workflows/           collect.yml, ci.yml, secrets-scan.yml
```

The collect workflow installs only the three runtime packages (`requests`,
`PyYAML`, `truststore`), since it runs next to the Buildkite and write tokens.
Actions are pinned to commit SHAs.

### Chart.js is vendored

The page's one third-party runtime dependency, **Chart.js 4.4.1**, is
committed to `site/vendor/` rather than loaded from cdnjs. On a locked-down
network a blocked CDN gives a blank chart with no visible explanation; a local
copy also makes the dashboard work offline, including straight off disk.

The vendored file is byte-identical to what cdnjs serves: it matches the
Subresource Integrity digest cdnjs publishes,
`sha384-bs/nf9FbdNouRbMiFcrcZfLXYPKiPaGVGplVbv7dLGECccEXDW+S3zjqSKR5ZEaD`.
`site/vendor/README.md` records the provenance and verification commands, and
`tests/test_vendored_assets.py` asserts the digest, so a swapped or truncated
file fails CI.

<details>
<summary>Why the licence file sits beside it</summary>

Chart.js is MIT licensed, and MIT requires its notice to accompany every copy
that is distributed. Loading from a CDN distributed nothing; committing the
file and publishing it to `gh-pages` makes this project a redistributor. The
minified bundle carries no licence banner, so the notice lives in
`site/vendor/chart.umd.min.js.LICENSE.txt`, `build_site.py` publishes it with
the bundle, and the tests assert it is complete and reaches the built site.
See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

</details>

### Secret scanning

**`scripts/perf_eval/secrets_scan.py`** runs in CI on every push and pull
request (`secrets-scan.yml`), and is the same check you can run locally. It
detects **known token shapes only**: a fixed prefix, then at least N
characters from a known alphabet. That is all a GitHub or Buildkite token is,
so each provider is one line of data rather than a regular expression:

```python
TokenShape("Buildkite API token", "bkua_", 40, LOWER_HEX)
```

A test asserts the module contains no `re.compile` at all.

> [!IMPORTANT]
> The scan does not cover other providers' token formats, or git history: a
> credential committed and then deleted leaves a clean tree that passes.
> Enable GitHub **secret scanning with push protection** for those. It knows
> far more providers and rejects the push before a token lands. It is free on
> public repositories.

<details>
<summary>Why there is no generic hex rule, and no gitleaks</summary>

The scan used to flag any run of 40+ hex characters. In practice that caught
git commit SHAs, not credentials, and needed three suppression mechanisms to
stay usable: an unreadable pinned-action regex, a list of context hints, and
putting `data/` on the path allowlist. Deleting the rule removed all three,
and `data/` is now scanned for real. A gitleaks job was removed too, as an
unpinned binary download duplicating what push protection does.

</details>

### Type checking

`ruff` handles lint and formatting but does not type check. Both configured
checkers report zero errors on the current tree.

- **`pyright` is the gate.** Pylance's engine *is* pyright and reads the same
  `typeCheckingMode`, so configuring it in `[tool.pyright]` makes the editor's
  verdict reproducible on the command line and in CI.
- **`ty` is advisory.** It is fast and found a real bug here, but at `0.0.x`
  its diagnostics shift between releases, so it runs with
  `continue-on-error: true` until it reaches a stable release.

<details>
<summary>Why <code>standard</code> and not <code>strict</code></summary>

| Mode | Errors |
|---|---|
| `standard` | 0 |
| `strict` | 2307 |

About 95% of that gap is six rules (`reportUnknownMemberType`,
`reportUnknownVariableType`, `reportUnknownArgumentType`,
`reportUnknownParameterType`, `reportMissingParameterType`,
`reportMissingTypeArgument`), which fire because this domain is JSON-shaped.
Writing `dict[str, Any]` everywhere would add no safety. The useful fix is
`TypedDict` definitions for the event and payload shapes, which is worth doing
but is a project, not a config flag.

</details>

<details>
<summary>What the type checkers caught</summary>

One real defect in production code:

```python
# Before: the guard and the value are two separate lookups.
{e.get("device") for e in eval_events if e.get("device")}
```

Change one `.get()` and not the other and a `None` reaches `sorted()` as a
runtime `TypeError`. It is now a single bound lookup via a walrus.

The other ~20 findings were in tests that subscripted an `X | None` result
directly; adding `assert result is not None` gives a readable failure instead
of `TypeError: 'NoneType' object is not subscriptable`. Two signatures also
changed to match what they accept: `transform_perf(tp: int | None)`, and the
fixtures' `nightly: object`.

</details>

---

## Licence

MIT. See [LICENSE](LICENSE). Copyright © Advanced Micro Devices, Inc.,
matching [ROCm/ATOM](https://github.com/ROCm/ATOM).

Bundled third-party software and its required notices are recorded in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). The only bundled library is
Chart.js, also MIT, so there is no licence conflict.

## Relationship to `vllm-ci-dashboard`

This dashboard replaces the `Perf Eval` tab in `vllm-ci-dashboard`, where the
same feature lived inside a ~13k-line shared frontend module. The two run in
parallel for now; retiring the old tab is tracked separately.
