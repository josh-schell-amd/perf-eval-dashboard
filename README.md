# Perf Eval Dashboard

A static dashboard for performance and accuracy results from the
[`vllm/perf-eval`](https://buildkite.com/vllm/perf-eval) Buildkite pipeline.

Zero build step: the whole frontend is one `site/index.html` with inline CSS
and vanilla JavaScript. Data arrives as a single `perf_eval.json` fetched
alongside the page.

### Why Chart.js is vendored

The page's one third-party runtime dependency is **Chart.js 4.4.1**, and it is
committed to `site/vendor/` rather than loaded from cdnjs:

```html
<script src="vendor/chart.umd.min.js"></script>
```

On a locked-down internal network a blocked CDN gives you a blank chart with
no visible explanation — the page looks broken and nothing says why. A local
copy also makes the dashboard work offline, including straight off disk from a
downloaded build artifact.

The vendored file is byte-identical to what cdnjs serves: it hashes to exactly
the Subresource Integrity digest cdnjs publishes for that URL,
`sha384-bs/nf9FbdNouRbMiFcrcZfLXYPKiPaGVGplVbv7dLGECccEXDW+S3zjqSKR5ZEaD`.
Vendoring changed where the file comes from, not what it is.
`site/vendor/README.md` records the provenance and the verification commands,
and `tests/test_vendored_assets.py` asserts the digest so a silently swapped
or truncated file fails CI.

**Attribution.** Chart.js is MIT licensed, and MIT requires its copyright and
permission notice to accompany every copy that is distributed. Loading from a
CDN distributed nothing — the browser fetched it from Cloudflare. Committing
the file and publishing it to `gh-pages` makes this project a redistributor,
which is what attaches the obligation. Because the minified bundle carries no
licence banner of its own, the notice lives beside it in
`site/vendor/chart.umd.min.js.LICENSE.txt`, `build_site.py` publishes it with
the bundle, and the tests assert both that it is complete and that it reaches
the built site. See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

Chart.js draws the trend charts and the history overlay. The sparklines on the
metric tiles are hand-drawn on a 2D canvas instead, because a wide model shows
dozens of tiles at once and dozens of Chart.js instances each running its own
animation loop is needlessly expensive.

Nothing is fetched at runtime. No CDN, no fonts, no analytics, no trackers.

## What this dashboard covers (and what it does not)

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

The scope is enforced by two named predicates —
`normalize.is_amd_workload` and `collect_artifacts.is_nightly_build` — and
re-applied at aggregation time by `aggregate._is_in_scope`, so a hand-seeded
or legacy event cannot widen what the page shows. It is also published inside
`perf_eval.json` under `scope`, and stated in the dashboard header.

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

Two branches, on purpose:

- **`dashboard-state`** holds `data/events.jsonl`, the private event log. It
  churns on every collection and is never published to the site.
- **`gh-pages`** holds the built site and `perf_eval.json`.
- **`main`** holds only source, so data commits never pollute its history.

The upstream pipeline uploads its entire `results/` tree as Buildkite
artifacts, so the collector reads raw `bench-*.json` and `results_*.json`
files directly. Nothing has to be pushed to us, and no webhook receiver needs
hosting.

## Credentials

Nothing is stored in this repository.

| Secret | Purpose | Scope needed |
|---|---|---|
| `BUILDKITE_TOKEN` | List builds, list and download artifacts | **Read-only**: Read Builds + Read Artifacts |
| `GITHUB_TOKEN` | Read the public workload recipes | The workflow's built-in token |

`BUILDKITE_TOKEN` goes in as a **GitHub Actions repository secret** with that
exact name — Settings → Secrets and variables → Actions → New repository
secret, or `gh secret set BUILDKITE_TOKEN`. It is injected as step-scoped
`env:` on the single ingest step and read via `os.getenv` with a fail-closed
check. The collector only issues GETs, so a read-only token is all it can use.
`GITHUB_TOKEN` needs nothing from you; Actions provides it.

Two other things before a live site shows real data: point Pages at the branch
(Settings → Pages → Deploy from a branch → `gh-pages` → `/root`), and note
that the first workflow run creates the `dashboard-state` branch itself as an
orphan, so there is no branch setup to do by hand.

Three guards keep it that way, all enforced in CI:

- **Secret scanning**, in two layers, on every push and pull request. See
  below.
- `tests/test_token_safety.py` pins the Buildkite org to `vllm`, asserts only
  `collect_artifacts.py` and `dev_build_times.py` can read the token, and
  asserts neither issues a write.
- Pushing checkouts use `persist-credentials: false`.

### Secret scanning

**`scripts/perf_eval/secrets_scan.py`** — instant, no install, and the same
check you can run locally before pushing. It detects **known token shapes
only**: a fixed prefix, then a run of at least N characters from a known
alphabet. That is all a GitHub or Buildkite token is, so it is a table of
those three facts rather than a set of regular expressions:

```python
TokenShape("Buildkite API token", "bkua_", 40, LOWER_HEX)
```

Adding a provider is one line of data, and there is no pattern to misread in
review. A test asserts the module contains no `re.compile` at all.

It used to also flag any run of 40+ hex characters. In practice that detected
git commit SHAs rather than credentials, and needed three separate suppression
mechanisms to stay usable — an unreadable pinned-action regex, a list of
context hints, and putting `data/` on the path allowlist. Deleting the one
rule removed all three, and `data/` is now scanned for real.

It runs in CI on every push and pull request (`secrets-scan.yml`).

**What it does not cover:** other providers' token formats, and git history —
a credential committed and then deleted leaves a clean tree the scan passes.
Enable GitHub **secret scanning with push protection** on the repository for
those: it knows far more providers and rejects the push before a token lands,
which beats finding it in history afterwards. It is free on public repos.
There used to be a gitleaks job here for the same purpose; it was removed as
an unpinned binary download duplicating what push protection does.

## Layout

```
site/index.html              the entire dashboard
scripts/perf_eval/
  normalize.py               metric registry, AMD filter, event normalizers
  store.py                   events.jsonl: atomic writes, retention, budgets
  collect_artifacts.py       Buildkite REST -> canonical events
  aggregate.py               events.jsonl -> perf_eval.json
  merge_events.py            identity-based merge of two stores
  secrets_scan.py
scripts/build_site.py        site/ + perf_eval.json -> _site/
scripts/dev_sample.py        synthetic store for local UI work
data/                        generated; lives on dashboard-state, gitignored here
tests/
.github/workflows/           collect.yml, ci.yml, secrets-scan.yml
```

## Running it locally

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -c constraints.txt -e ".[dev]"
```

`constraints.txt` pins the exact version of every package, the same way
`vllm-ci-dashboard` does, so CI and your machine run identical tools and a new
ruff or pyright release cannot break CI on its own. A test fails if a direct
dependency in `pyproject.toml` has no pin there. To upgrade, change the pin,
reinstall, and run the checks. The collect workflow installs only the three
runtime packages (`requests`, `PyYAML`, `truststore`), since it runs next to
the Buildkite and write tokens. Actions are pinned to commit SHAs, matching
`vllm-ci-dashboard`.

### Previewing the page without credentials

`dev_sample.py` writes a deterministic synthetic event store, so you can work
on the UI without a Buildkite token:

```bash
python scripts/dev_sample.py
python scripts/perf_eval/aggregate.py
python scripts/build_site.py
python -m http.server --directory _site 8000
```

The sample deliberately includes one NVIDIA run and one ad-hoc run. Neither
may appear on the rendered page — if either does, the scope filter has
regressed.

### Working with real data

You can run the entire pipeline locally, with no GitHub involved — it is the
same sequence the workflow runs, minus the branch commits.

```bash
# Clear any synthetic data first: real ingest APPENDS, so a leftover sample
# store would mix fake and real points in the same charts.
rm -f data/events.jsonl

export BUILDKITE_TOKEN=bkua_...          # read-only: Read Builds + Read Artifacts
export GITHUB_TOKEN="$(gh auth token)"   # optional, but see below

python scripts/perf_eval/collect_artifacts.py --days 14
python scripts/perf_eval/aggregate.py
python scripts/build_site.py
python -m http.server --directory _site 8000
```

Ingest is read-only against Buildkite, so it is safe to re-run as often as you
like.

**Set `GITHUB_TOKEN` too if you can.** It is only used to read the public
workload recipes from `vllm-project/perf-eval`, but that is one directory
listing plus one fetch per recipe — around 29 requests against the 28 recipes
currently in that repo. Anonymous GitHub API access allows 60 requests an
hour, so you would get roughly two local runs before being throttled; with a
token it is 5,000. If the recipes fail to load you will see
`No recipe for workload <name>; skipping` and an empty dashboard, which is the
symptom to watch for.

`aggregate.py` and `build_site.py` only read the local store, so the Buildkite
token is needed for ingest and nothing else.

### Checks

```bash
pytest
ruff check . && ruff format --check .
pyright                      # type check, gating in CI
ty check scripts tests       # second opinion, advisory
python scripts/perf_eval/secrets_scan.py
```

## Type checking

`ruff` handles lint and formatting but does **not** type check, so that is a
separate tool. Both of the ones configured here report zero errors on the
current tree.

**`pyright` is the gate.** Pylance is closed-source and editor-only, so it
cannot run in CI — but its engine *is* pyright, and it reads the same
`typeCheckingMode` setting. Configuring pyright in `[tool.pyright]` therefore
makes the verdict you see in the editor reproducible on the command line and
in CI, rather than two tools disagreeing. Mypy would have been the mature
alternative, but it applies different rules, so its output would not match
what the editor shows you.

**`ty` is advisory.** Astral's checker is fast and found a genuine bug here
(see below), but it is still `0.0.x`: its diagnostics shift between releases,
so gating CI on it would let an unrelated version bump break the build. It
runs with `continue-on-error: true` and should be promoted to gating once it
hits a stable release.

### Why `standard` and not `strict`

Measured on this tree:

| Mode | Errors |
|---|---|
| `standard` | 0 |
| `strict` | 2307 |

That gap is not 2307 latent bugs. About **95%** of it is six rules —
`reportUnknownMemberType`, `reportUnknownVariableType`,
`reportUnknownArgumentType`, `reportUnknownParameterType`,
`reportMissingParameterType`, `reportMissingTypeArgument` — which fire because
this domain is JSON-shaped: events and payloads are `dict`, and strict wants
`dict[str, Any]` plus a full annotation on every test function and pytest
fixture.

Satisfying it by writing `dict[str, Any]` everywhere would add no safety
whatsoever. Satisfying it *usefully* means `TypedDict` definitions for the
event and payload shapes — which is genuinely worth doing, because it would
catch real schema mistakes that tests currently have to cover by hand, but it
is a project rather than a config flag. `standard` is the honest gate until
then.

### What the type checkers caught

One real defect in production code, which is the argument for having them:

```python
# Before: the guard and the value are two separate lookups.
{e.get("device") for e in eval_events if e.get("device")}
```

Nothing tied those two `.get()` calls together. Change one and not the other
and a `None` reaches `sorted()` as a runtime `TypeError`. It is now a single
bound lookup via a walrus, so the guard and the value cannot drift apart.

The remaining ~20 findings were all in tests, calling a function that returns
`X | None` and immediately subscripting the result. Adding
`assert result is not None` first was worth doing on its own merits: the test
now fails with a readable assertion instead of
`TypeError: 'NoneType' object is not subscriptable`.

Two signatures also changed to tell the truth about what they accept:
`transform_perf(tp: int | None)`, because `tp` comes from a recipe that may
omit it and the body already handles that; and the test fixtures'
`nightly: object`, because events carry whatever JSON arrived and tests
deliberately pass non-boolean values to prove only a literal `True` counts.

## The published payload

`perf_eval.json` is the contract between the collectors and the page:

```jsonc
{
  "generated_at": "2026-01-01T00:00:00Z",
  "scope":      { "hardware": "amd", "runs": "nightly", "description": "..." },
  "pipeline":   { "org": "vllm", "slug": "perf-eval", "url": "..." },
  "metric_meta": { "tput_per_gpu": { "label": "...", "unit": "tok/s", "direction": "higher" } },
  "thresholds": { "perf_rel": 0.005, "accuracy_abs": 0.01 },
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
          "direction": "higher", "status": "good", "label": "...", "unit": "tok/s",
          "series": [{ "date": "...", "value": 1200.0, "vllm_commit": "...", "build_url": "..." }]
        }
      }
    }],
    "accuracy_tasks": [{
      "task": "gsm8k", "metric": "exact_match,strict-match", "primary": true,
      "latest": 0.81, "previous": 0.80, "status": "good", "series": [ /* ... */ ]
    }]
  }],
  "summary":   { "models": 1, "amd_devices": ["mi355x"], "nightlies": 12, "perf_points": 96, "accuracy_points": 12 },
  "retention": { "event_history_days": 180, "nightly_limit": 180, "adaptive": false }
}
```

`metric_meta` carries `direction` so the page renders a new metric with the
correct higher/lower-is-better colouring without a frontend change.

### Regression detection

One rule: **the newest nightly against the run before it**. A perf metric
counts as regressed or improved only when it moves by **0.5% or more**, and an
accuracy score only when it moves by **1 point or more** (0.01 on the 0–1
scale). Smaller moves are neutral.

**The 0.5% threshold is chosen, not measured.** With no threshold, about half
of the regressions flagged on a typical night were under 0.5% (the smallest
was 0.015%), and latencies stored at 0.1 ms resolution make a single rounding
step on a fast metric look like a regression. Because a threshold hides
movement, the page states it wherever it reports a count: the regression KPI
card, the regression panel (with how many smaller drops were not counted), the
trend-chart hint and the Configurations table.

Once `repetitions: 3` lands on the AMD recipes, the spread *across* those
repetitions is a real noise floor, and the threshold should be derived from it
instead.

**Accuracy is not reproducible night to night**, despite the fixed dataset:
every AMD workload's gsm8k score moves every night. One gsm8k question out of
1,319 is worth 0.08 points and the expected run-to-run spread is about 0.7
points, so the 1-point threshold sits just above it. Tests pin both thresholds.

### Accuracy results

lm-eval's own `config.model` names the client backend (`local-completions`),
not the model under test, so the collector takes the model id from the
workload recipe — the same string the perf results carry. Events collected
before that fix are repaired at aggregation, from the recipe expectation for
their workload or from lm-eval's output directory
(`results/<workload>/<task>/<org>__<name>/`).

Accuracy is a series per model, **device** and task, since one model can run
on several AMD devices. Bookkeeping keys such as `sample_len` (the question
count) are dropped. The headline score per task is `exact_match,flexible-extract`
where present, then `exact_match,strict-match`, `acc_norm,none`, `acc,none`,
then the first score: strict match also grades the answer format, which
drags gpt-oss-120b to about 52% against 76% flexible on gsm8k.

The rule is declared in `perf_eval.json` under `baselines`, so the labels and
thresholds in the UI come from data rather than being hard-coded in the page.
It is a list of one, which makes adding a second model later a data change
rather than a frontend change.

**No smoothing across nightlies, on purpose.** Reducing measurement noise is
the benchmark's job, not the dashboard's — and perf-eval already does it
properly. `lib/aggregate_perf.py` repeats a benchmark on the same warm server
`repetitions` times and median-aggregates every numeric field *before*
ingestion, so each point that reaches us is already an aggregate of several
measurements. Averaging again here would blur the night-to-night change this
dashboard exists to show, while only pretending to fix noise that belongs
upstream.

If a metric is too noisy to compare night to night, the fix is a higher
`repetitions` in the workload recipe, not a smoothing layer here.

> **Known upstream gap.** Every NVIDIA workload in `vllm-project/perf-eval`
> sets `repetitions: 3`. None of the eleven AMD (`mi300x`/`mi355x`) workloads
> set it, so they default to `1` — a single measurement with no aggregation.
> Since this dashboard is AMD-only, every point currently shown is one
> unaggregated run. Raising `repetitions` on those recipes is the right way to
> make night-to-night comparison trustworthy.

Detection runs in the page rather than in `aggregate.py`, because the time
window decides which run counts as "latest" and which as its predecessor. The
per-metric `status` published on each metric block applies the same rule over
whole history, and is kept for machine consumers of the JSON.

### Colour semantics

Colour carries meaning, so it is allocated rather than picked:

- **Red is reserved for regressions**, and there is exactly one red. The
  identity palette contains no red — and no pink, which reads as a light red
  on a 2px line — so a red line always means "this regressed" and can never be
  a healthy series that happened to be assigned red.
- **Red is applied per metric, not per configuration.** A config is rarely bad
  across the board: it keeps its identity colour on the charts where it is
  healthy and turns red only on the metrics it actually regressed on. A chart
  where nothing regressed shows no red at all.
- The regressed line is thickened and its **newest point ringed in white**,
  which says which build to blame.
- **Regression chips are coloured by size**, judged on the value as displayed:
  yellow up to 2.5%, orange up to 5%, red above 5%. A row's left border takes
  the colour of its worst chip. These colours are used on chips and row
  borders, never on a series line. **Green means improvement.**
- **One deliberate exception:** the *Performance overnight* KPI card outlines red
  when down and green when up. That card answers "is tonight faster or slower
  overall", which is directional rather than pass/fail, and the border should
  answer it at a glance without reading the number. It is symmetric on purpose
  — colour there reads as a direction, not as an alarm, so do not read its red
  as "something regressed". It stays neutral while the median moves less than
  the 0.5% threshold, since a median of +0.00% is not a direction.
- The *Accuracy overnight* card counts models whose headline score dropped at
  least 1 point in the latest nightly. Clicking it opens the Accuracy tab.
- The legend chip and the chart line answer different questions on purpose. A
  red-outlined chip means "this configuration regressed on something,
  somewhere"; a red line means "it regressed on *this* metric".

If you add palette colours, keep them out of the 0–20° hue range.

### Time window

The view shows a trailing **14 days**, anchored to *now* rather than to the
newest run. If the nightly stops reporting, the dashboard goes empty and says
how stale the data is rather than quietly presenting month-old numbers as
current — a dashboard that looks healthy because it is showing stale data is
worse than one that looks broken. The empty state names the problem and links
to the pipeline; it deliberately offers no way to widen the window.

**The window can be narrowed, never widened.** Two weeks is short enough that
everything on screen predates the same handful of image bumps. The window
comes from `display_window_days` in the payload, which `aggregate.py` owns.

The trend charts have a **chart window** control (1d, 3d, 7d and 14d presets,
plus a slider for any value from 1 to 14 days), kept in the URL as `#days=N`.
It changes only what the charts draw. Regressions, the KPI cards and coverage
always compare the newest nightly with the one before it, and a short window
would otherwise drop that previous run and quietly empty every count.

A narrowed window counts back from the **newest nightly**, not from now: the
nightly lands mid-morning UTC, so a one-day window anchored to now would be
empty for most of the next day. The axis still ends at now, so a nightly that
stopped reporting shows as a gap on the right. Spans of three days or less
tick every 6 hours with the UTC hour; longer ones tick once a day.

Each chart's x-axis is pinned to the chart window rather than auto-scaled to
the data it holds, so a chart with three points and one with fourteen still
line up by date and the stack can be read vertically.

### What the KPI cards mean

Every card answers "is tonight's build healthy", which rules out headline
numbers like peak throughput: those are a maximum over every configuration, so
they report which config happens to be largest and barely move night to night.

| Card | Signal |
|---|---|
| Latest nightly | Date, vLLM commit, build — links to the build |
| Performance overnight | Median change in output tok/s/GPU, newest nightly vs the one before, across configs that reported in both; neutral inside ±0.5% |
| Regressions overnight | Config-metric pairs that got worse by at least 0.5% in the newest nightly; red when there are any |
| Improvements overnight | The same scan in the other direction, to confirm an optimization landed; green when there are any |
| Accuracy overnight | Models whose headline accuracy dropped at least 1 point in the newest nightly; opens the Accuracy tab |
| Coverage | Perf configs reporting in the newest build vs those defined in the perf-eval recipes, and accuracy results reporting vs models with accuracy in the window |

Every card except *Latest nightly*, Coverage included, follows the Device,
Model, Precision, ISL/OSL and Concurrency filters. *Latest nightly* names the
build the rest of the row describes, so it ignores them. Accuracy has no shape,
precision or concurrency, so only the Device and Model filters narrow it.

When nothing could be compared — no configuration has both a run in the newest
nightly and an earlier one in the window — the regression panel says *Nothing
to compare* rather than showing a green *No regressions*. The legend selection
is kept in the URL as `show=`, so a copied link shows the same configurations.

The overnight cards only count configs that reported in the newest
nightly. A config that skipped tonight still has two earlier points, but its
change between them happened on earlier nights. Counting it would pass that old
change off as tonight's. Configs that did not report show up in Coverage
instead.

Coverage is the one that is easy to omit and expensive to miss: a workload that
OOMs simply stops emitting rows, so it silently disappears from every average
rather than showing up as a regression. Missing configurations are listed in a
panel under the KPI row, grouped by workload, because a failed build step
takes every config in that workload with it.

### Nightly identity

A nightly is identified by its vLLM commit, falling back to build number and
then date. The perf-eval repo's own commit (`build_commit`) is deliberately
not a fallback: it stays the same across many nightlies. That means **a nightly re-run on the same commit folds into one
data point** rather than appearing twice, which is the intended behaviour: it
is one nightly that happened to be executed twice.

When two observations share a nightly identity, the one with the newer
timestamp wins — resolved by timestamp rather than by position in the event
log, since the log is append-ordered and a re-ingested older observation would
otherwise silently override a newer one.

## Collection cadence

`collect.yml` runs on a schedule three times a day, at 01:17, 09:17 and 17:17
UTC, plus on a manual dispatch, on a push that touches `site/`, and on a
`repository_dispatch` of type `perf_eval_build_finished`.

**Those hours are a hedge, not a measurement.** The upstream nightly's
schedule lives in the Buildkite UI rather than in the `perf-eval` repo, so
there is nothing in code to derive it from, and a nightly sweeping a dozen
models across several GPU types runs for hours, so its finish time drifts.
Three passes is insurance against not knowing when that is.

Replace the guess with data once you have a token:

```bash
BUILDKITE_TOKEN=bkua_... python scripts/dev_build_times.py --days 30
```

It lists recent nightlies with their finish times and durations, summarises
the spread, and prints a suggested cron — likely one pass plus a retry rather
than three blind ones.

Each run scans the last 14 days of finished `main` builds (1–30 configurable
via the dispatch input). The window is a safety net for a nightly that landed
late, or for backfilling after a failed run.

**Backfilling a build that is already ingested** needs the `recheck_builds`
dispatch input (0–60). A build that has any results is never listed again once
it leaves the re-check window, so artifacts an older collector missed — the
accuracy results before lm-eval's model directory was recognised, for example
— only arrive if you re-list it. Running once with `recheck_builds: 30`
re-lists every nightly in the window; results already stored are skipped.

A download that keeps failing with a timeout, 429 or gateway error **fails the
run** after 3 attempts, for the same reason: skipping it would lose that
artifact for good. A 4xx or a body that is not JSON is skipped with a warning,
so one broken artifact cannot block every later collection. A bench result
with no positive `total_token_throughput` is a failed benchmark and is skipped
rather than published as zero throughput.

### What a run costs Buildkite

Every outbound request is counted and reported, and the arithmetic is fixed:

| | Requests |
|---|---|
| Builds listing | 1 |
| Artifact listing | 3 per nightly inside the re-check window (one per path filter) |
| Download | 1 per artifact not already ingested |

A build we already hold results for is **not re-listed** once it falls outside
the re-check window, which defaults to the newest 3 nightlies. Those three are
always re-listed because a nightly can finish with a failed workload that
someone retries later, adding artifacts to the same build number. So a steady
state run is `1 + 3×3 = 10` listings plus the new nightly's artifacts, not
one-plus-three-per-build across the whole 14 days.

**See the cost before you spend it.** `--dry-run` performs the listings, which
is what reveals how much work there is, then reports exactly what it would
download and stops — no downloads, no writes:

```bash
BUILDKITE_TOKEN=bkua_... python scripts/perf_eval/collect_artifacts.py --dry-run --days 14
```

Several properties bound the cost by construction rather than by good
intentions:

- `--max-requests` (default 1500) aborts the run rather than continuing. An
  unexpected request volume is a bug worth stopping on.
- Artifact listings use narrow path filters (`*results/*/bench-*.json`,
  `*results/*/*/results_*.json` and `*results/*/*/*/results_*.json`), so the
  pipeline's much larger sample and log tree is never enumerated. The deepest
  one exists because lm-eval writes its results into a subdirectory named
  after the model, under the task directory perf-eval gives it.
- Pagination is capped — 10 pages for builds, a shared 10-page budget per
  build for artifacts — and raises rather than looping.
- Retries are capped at 3 attempts, honour `Retry-After`, and only apply to
  gateway-ish codes (429, 502, 503, 504, 520, 522, 524). A 500 is not retried,
  since it is usually persistent.
- Retry attempts are charged to the budget, so the reported total is what
  Buildkite actually saw rather than what we intended.
- The workflow's `concurrency` group prevents two runs overlapping.
- The store is written only after the whole scan completes, so an abort leaves
  the previous state intact.

`repository_dispatch` is wired to receive but nothing currently sends it. In
`vllm-ci-dashboard` that came from a hosted Buildkite webhook bridge, which is
deliberately not ported here since it needs a running endpoint. So today cron
is the only automatic trigger, and worst case a nightly appears up to about
eight hours after it finishes.

### Redundant deploys are suppressed

`aggregate.py` restamps `generated_at` on every run, so the payload always
differs byte-wise even when no new nightly arrived. Publishing on that alone
would spend a Pages build republishing identical numbers, and Pages allows
only about ten builds an hour.

`payload_changed.py` compares the fresh payload against the one **live on
`gh-pages`** with `generated_at` excluded and everything else included, and the
deploy is gated on the result. It compares against the live copy rather than
one saved with the event store: that copy is written before the deploy runs,
so a failed or skipped deploy would count as published and leave the site
stale. The collector likewise records a new recipe snapshot only when the
recipes changed, since its timestamp is published and a fresh one every run
would make every payload look changed. The skip applies **only to scheduled runs** — a push to
`site/`, a manual dispatch or a build-finished dispatch always republishes,
because the page itself may have changed even though the data did not. That is
also why `collect.yml` triggers on pushes touching `site/`: without it a UI
edit would wait for the next scheduled run and then be skipped as unchanged.

A missing or unreadable previous payload counts as changed, so the failure
direction is one redundant deploy rather than a silently unpublished update.

## Retention

Both stores are byte-bounded and written atomically, so a crash mid-write
cannot leave a truncated file behind.

| Store | Budget | Why |
|---|---|---|
| `data/events.jsonl` | 24 MiB | Server-side only; CI reads it, nobody downloads it |
| `data/perf_eval.json` | 8 MiB | Downloaded by every browser that opens the page |

Retention targets 180 days. If an unusually wide workload set hits the byte
budget first, history is shed a **whole nightly at a time**, newest kept, so a
partial nightly is never presented as a complete comparison. When that
happens, `retention.adaptive` is `true` and the page says so in the footer.

## Licence

MIT — see [LICENSE](LICENSE). Copyright © Advanced Micro Devices, Inc.,
matching [ROCm/ATOM](https://github.com/ROCm/ATOM).

Bundled third-party software and its required notices are recorded in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). The only bundled dependency
is Chart.js, also MIT, so there is no licence conflict.

## Relationship to `vllm-ci-dashboard`

This dashboard replaces the `Perf Eval` tab in `vllm-ci-dashboard`, where the
same feature lived inside a ~13k-line shared frontend module and a
multi-surface collection workflow. The two run in parallel for now; retiring
the old tab is tracked separately.
