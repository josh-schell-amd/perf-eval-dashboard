# Vendored third-party assets

Committed on purpose rather than loaded from a CDN. See "Why Chart.js is
vendored" in the repository README for the reasoning.

## chart.umd.min.js

| | |
|---|---|
| Library | [Chart.js](https://www.chartjs.org/) |
| Version | 4.4.1 |
| Licence | MIT — `chart.umd.min.js.LICENSE.txt` |
| Copyright | Copyright (c) 2014-2022 Chart.js Contributors |
| Upstream | `https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js` |
| SHA-384 | `sha384-bs/nf9FbdNouRbMiFcrcZfLXYPKiPaGVGplVbv7dLGECccEXDW+S3zjqSKR5ZEaD` |

The digest above is the Subresource Integrity hash cdnjs publishes for that
URL, and this file matches it byte for byte — so vendoring changed where the
file comes from, not what it is.

### Attribution

MIT requires the copyright notice and permission notice to accompany every
copy, and redistributing this file is exactly what triggers that. The minified
bundle carries no licence banner of its own, so the notice lives beside it in
**`chart.umd.min.js.LICENSE.txt`**. `build_site.py` copies this whole directory
into the published site, so the notice ships with the bundle it covers — and
`tests/test_vendored_assets.py` asserts that it does.

Do not delete or rewrite that file, and keep it verbatim. See
[`THIRD-PARTY-NOTICES.md`](../../THIRD-PARTY-NOTICES.md) for the full record.

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

`tests/test_vendored_assets.py` asserts this automatically, so a silently
swapped or truncated file fails CI.

### Upgrading

1. Download the new `chart.umd.min.js` from cdnjs.
2. Replace `chart.umd.min.js.LICENSE.txt` with the licence text from the
   matching upstream tag, verbatim. Check whether the copyright years changed —
   they are part of the notice.
3. Update the version and SHA-384 in this file, in
   `tests/test_vendored_assets.py`, and in `THIRD-PARTY-NOTICES.md`.
4. Re-run `pytest` and open the page — confirm the charts still draw.

Keep the version aligned with
[ATOM's dashboard](https://github.com/ROCm/ATOM/blob/main/.github/dashboard/index.html)
unless there is a reason to diverge, so the two stay visually consistent.

## amd.png

The favicon and header logo. It is the image Buildkite renders for the `:amd:`
emoji, which perf-eval's `generate_pipeline.py` puts on every AMD step label,
so the dashboard carries the same mark as the builds it reports on.

| | |
|---|---|
| Source | [buildkite/emojis](https://github.com/buildkite/emojis), `img-buildkite-64/amd.png` |
| Upstream | `https://raw.githubusercontent.com/buildkite/emojis/main/img-buildkite-64/amd.png` |
| Size | 64×64 PNG, 1,381 bytes |
| Git blob | `797ab693c2ad8c9698f0d24139b9e22d658d9bcf` |
| SHA-256 | `5da4382060eb39e1f5e9bf9d0f3c3c6b5ac7ebbef3e578f7a9f5bf3cbb1fe7e6` |

The git blob hash matches the upstream file's, so the bytes are unchanged.

### Ownership

This file is not MIT, so it has no `.LICENSE.txt` beside it. Buildkite's repo
says only that "each logo is owned by their respective creators": the AMD logo
is an AMD trademark. This dashboard is an AMD project, so AMD is using its own
mark. Anyone forking it outside AMD should replace the file.
