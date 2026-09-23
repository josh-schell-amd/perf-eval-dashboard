# Third-party notices

This project is MIT licensed (see [LICENSE](LICENSE)) and redistributes the
third-party software listed below. Each entry's licence requires its copyright
notice and permission notice to travel with the copy, so the full text is
committed alongside the file it covers and is published with the site.

Every bundled piece of software is MIT, so there is no conflict with this
project's own licence. The one non-software asset, the AMD logo, is covered
under "AMD logo" below.

Nothing else is bundled. The Python dependencies in `pyproject.toml`
(`requests`, `PyYAML`, and the dev extras) are installed at build time and are
not redistributed by this repository or by the published site.

---

## Chart.js 4.4.1

| | |
|---|---|
| Upstream | https://github.com/chartjs/Chart.js |
| Version | 4.4.1 (tag `v4.4.1`) |
| Licence | MIT |
| Copyright | Copyright (c) 2014-2022 Chart.js Contributors |
| Bundled as | `site/vendor/chart.umd.min.js` |
| Licence text | `site/vendor/chart.umd.min.js.LICENSE.txt` |
| Obtained from | `https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js` |
| SHA-384 | `sha384-bs/nf9FbdNouRbMiFcrcZfLXYPKiPaGVGplVbv7dLGECccEXDW+S3zjqSKR5ZEaD` |

Used for the trend charts and the metric-history chart.

**Why the licence text is a separate file.** The minified distribution carries
no licence banner — unlike many minified bundles, `chart.umd.min.js` contains
no `@license` comment, no copyright line and no reference to MIT anywhere in
the file. So the notice cannot travel inside it and has to sit beside it.

**Why this obligation exists at all.** MIT permits redistribution provided
"the above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software". While the library was
loaded from a CDN, the browser fetched it from Cloudflare and this project
distributed nothing. Committing the file and publishing it to `gh-pages` makes
this project a redistributor, which is what attaches the attribution
requirement. `tests/test_vendored_assets.py` asserts the notice exists, is
complete, and is copied into the built site, so a future change cannot quietly
drop it.

### Verifying

```bash
# Linux / macOS
printf 'sha384-%s\n' "$(openssl dgst -sha384 -binary site/vendor/chart.umd.min.js | base64)"
```

```powershell
# Windows
$bytes = [IO.File]::ReadAllBytes("site/vendor/chart.umd.min.js")
"sha384-" + [Convert]::ToBase64String([Security.Cryptography.SHA384]::Create().ComputeHash($bytes))
```

---

## AMD logo

| | |
|---|---|
| Bundled as | `site/vendor/amd.png` (favicon and header logo) |
| Obtained from | `https://raw.githubusercontent.com/buildkite/emojis/main/img-buildkite-64/amd.png` |
| Owner | Advanced Micro Devices, Inc. (trademark) |
| SHA-256 | `5da4382060eb39e1f5e9bf9d0f3c3c6b5ac7ebbef3e578f7a9f5bf3cbb1fe7e6` |

This is the image behind Buildkite's `:amd:` emoji, which perf-eval uses on its
AMD step labels. It is a trademark, not licensed software, so there is no
licence text to ship with it. Buildkite's emoji repository states that each
logo is owned by its creator. This is an AMD project using AMD's own mark. A
fork outside AMD should replace the file.

---

### If you upgrade or add a bundled asset

1. Commit the new file under `site/vendor/`.
2. Commit its licence text as `site/vendor/<filename>.LICENSE.txt`, copied
   verbatim from the upstream release — do not paraphrase it, and check
   whether the copyright years changed.
3. Add or update the entry in this file.
4. Update the digest and version in `tests/test_vendored_assets.py`.
