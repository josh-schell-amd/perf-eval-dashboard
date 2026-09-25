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

This is deliberately a narrow view. Read this before concluding that a run is
missing.

| | Included | Excluded |
|---|---|---|
| Hardware | AMD MI-series (`mi300x`, `mi355x`, …) | NVIDIA (H200, B200, A100) |
| Runs | Scheduled nightlies | Ad-hoc and pull-request builds |
| Branch | `main` | Every other branch |
| Build state | `finished` | Running, cancelled, failed-to-start |

**Why AMD only.** The upstream pipeline runs both vendors, but this dashboard
exists to track the AMD story. NVIDIA results are served by
[perf.vllm.ai](https://perf.vllm.ai) from the same pipeline's Databricks
ingest path.

**Why nightly only.** Nightlies run the full workload matrix on a fixed
cadence, which is what makes one run comparable to the next. An ad-hoc build
may cover a single workload at a single concurrency, so folding it into a
trend line would put an unrelated point next to a full sweep and make the
latest-versus-previous comparison meaningless.

> [!NOTE]
> The scope is enforced in code, not just stated. Two named predicates,
> `normalize.is_amd_workload` and `collect_artifacts.is_nightly_build`, filter
> at collection, and `aggregate._is_in_scope` re-applies them at aggregation so
> a hand-seeded or legacy event cannot widen what the page shows. The scope is
> also published inside `perf_eval.json` under `scope`, and stated in the
> dashboard header.

---

## Running it locally

### Set up

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -c constraints.txt -e ".[dev]"
```

`constraints.txt` pins every package to an exact version, so CI and your
machine run identical tools. A test fails if a direct dependency in
`pyproject.toml` has no pin there. To upgrade, change the pin, reinstall, and
run the checks.

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
> **Set `GITHUB_TOKEN` if you can.** It is only used to read the public
> workload recipes from `vllm-project/perf-eval`, but that takes around 29
> requests per recipe commit. Anonymous GitHub API access allows 60 an hour, so
> you get roughly two local runs before being throttled; with a token it is
> 5,000. If the recipes fail to load you will see
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

Every card answers "is tonight's build healthy?", which rules out headline
numbers like peak throughput: a maximum over every configuration reports which
config happens to be largest and barely moves night to night.

| Card | Signal |
|---|---|
| Latest nightly | Date, vLLM commit and build, linking to the build |
| Performance overnight | Median change in Output Throughput (tok/s/GPU) between the two newest nightlies, across configs that ran in both; neutral inside ±0.5% |
| Regressions overnight | Config-metric pairs that got at least 0.5% worse in the newest nightly; red when there are any |
| Improvements overnight | The same scan in the other direction, to confirm an optimization landed; green when there are any |
| Accuracy overnight | Models whose headline accuracy dropped at least 1 point in the newest nightly; opens the Accuracy tab |
| Coverage | Perf configs and accuracy results reporting in the newest build, against what that build's recipes define |

Every card except *Latest nightly* follows the Device, Model, Precision,
ISL/OSL and Concurrency filters. *Latest nightly* names the build the rest of
the row describes, so it ignores them. Accuracy has no shape, precision or
concurrency, so only the Device and Model filters narrow it.

> [!IMPORTANT]
> The *Performance overnight* card is the one place where red does **not**
> mean a regression. It outlines red when the median went down and green when
> it went up: a direction at a glance, not an alarm.

### Tabs

| Tab | What it shows |
|---|---|
| **Performance** | The default. One bar chart per model and device for a picked metric, then a detail table |
| **Throughput vs Latency** | ATOM's tradeoff view: interactivity against throughput, concurrency scaling, and a heatmap |
| **Trends** | One line chart per metric across the window, one line per configuration |
| **Accuracy** | lm-eval scores per model, device and task, against the previous nightly |
| **Configurations** | Every configuration's newest value for every metric, with its change |

The filters and the *Only regressed* toggle are kept in the URL, as are the
tab (`tab=`), the Performance metric (`metric=`) and the chart window
(`days=`), so **Copy link** always reproduces the view.

#### Performance

A metric picker drives one bar chart per **model and device**, since per-GPU
numbers do not compare across devices. Each bar is a configuration's newest
run in the window:

- **shaded darker** with concurrency;
- **outlined red** when that metric got 0.5% or more worse between the
  previous nightly and the newest one;
- **faded** when it did not run in the newest nightly.

A dashed line marks the chart's average, and clicking a bar opens its history.

Hovering a bar shows its value, build and vLLM commit, and the **change** with
what it is measured against: `vs last night (#598)` normally, or
`vs 8 nights ago (#586)` when the configuration skipped nights. Such a
catch-up change is never outlined, because it spans more than one night. When
there is no earlier run in the window, the card says *No previous run* rather
than leaving the change out.

Below the charts, a detail table lists every configuration's newest run:
throughput with an in-cell bar against the table's largest, TPOT and TTFT
heat-shaded from fastest to slowest, the change in the picked metric, and
links to the vLLM commit and the build. Clicking a row expands every metric
and the run behind it, including failed requests.

#### Throughput vs Latency

No extra ingest: each configuration already carries Total Throughput and Mean
TPOT, and Interactivity is 1 / TPOT. The page joins the two from the **same
nightly**, groups by model and device, and draws one curve per shape through
the concurrency sweep the recipes already run (`[1, 64, 128]` on most AMD
workloads).

Each model and device gets two charts: Interactivity against Total Throughput
(both higher-is-better), and concurrency scaling with throughput on the left
axis and TPOT on the right. Underneath, a heatmap shows the newest throughput
by ISL/OSL × concurrency; it hides until a model has two concurrency levels
and three cells. A red ring is a throughput regression in the newest nightly.

#### Trends

One chart per metric, so a regression that only moves TTFT is visible without
hunting through a dropdown. A red line means that configuration regressed on
*that* metric, and its newest point is ringed.

The **chart window** control (1, 3, 7 or 14 days, or a slider) changes only
what the charts draw. A narrowed window counts back from the **newest
nightly**, not from now, because the nightly lands mid-morning UTC and a
one-day window anchored to now would be empty most of the next day. Every
chart's x-axis is pinned to the window, so charts line up by date and can be
read down the stack.

#### Accuracy

Scores are lm-eval's, as the percentage of questions answered correctly, and
each is compared with **the previous nightly**, not with a reference score
for the model. The headline score per task is `exact_match,flexible-extract`
where present, then `exact_match,strict-match`, `acc_norm,none`, `acc,none`,
then the first score. Strict match also grades the answer format, which drags
gpt-oss-120b to about 52% against 76% flexible on gsm8k.

#### Configurations and Coverage

The Configurations tab shows every metric's newest value with its change;
changes under the threshold, or spanning more than one night, stay grey.

The Coverage panel at the bottom of the page lists configurations the recipes
expect but the newest build did not report, grouped by workload, because a
failed build step takes every config in that workload with it. This matters
because a workload that OOMs simply stops emitting rows: it disappears from
every average instead of showing up as a regression.

Coverage is measured against the recipes (`vllm_bench.configs` and
`lm_eval.tasks`), never inferred from what reported recently. An expectation
built from recent results forgets whatever has been absent long enough, so the
longer a workload stayed broken, the healthier the page would claim to be.

### Colour

Colour carries meaning, so it is allocated rather than picked.

- **Red means a regression, and nothing else.** The identity palette has no
  red, and no pink, which reads as light red on a 2px line.
- **Red is per metric, not per configuration.** A config keeps its identity
  colour on the charts where it is healthy and turns red only where it
  regressed. A chart where nothing regressed shows no red at all.
- **Regression chips are coloured by size:** yellow up to 2.5%, orange up to
  5%, red above 5%, judged on the value as displayed. A row's left border
  takes its worst chip's colour. **Green means improvement.**
- A red-outlined legend chip means "regressed on something, somewhere"; a red
  line means "regressed on *this* metric".

> [!NOTE]
> Light mode swaps in `PALETTE_LIGHT`: the same hues, darkened to read on
> white, in the same order so a series keeps its identity across themes. Add
> new colours to both lists together, and keep them out of the 0–20° hue
> range.

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
