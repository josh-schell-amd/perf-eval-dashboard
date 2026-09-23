#!/usr/bin/env python3
"""Generate a synthetic event store so the dashboard can be previewed locally.

Ingesting real data needs a Buildkite token. This produces a plausible store
instead, so the page and the payload schema can be worked on without
credentials:

    python scripts/dev_sample.py
    python scripts/perf_eval/aggregate.py
    python scripts/build_site.py
    python -m http.server --directory _site 8000

The data is deterministic (fixed seed) so a UI change produces a reviewable
diff rather than fresh noise.

It deliberately includes one NVIDIA run and one non-nightly run, which the
AMD-only, nightly-only scope filter must drop — if either shows up on the
page, the filter has regressed.

It also seeds three regression shapes so the Trends tab has something real to
render:

1. a clean 14% drop on the newest nightly — an unambiguous regression;
2. a marginal 3.5% dip, just over the 2% threshold — the smallest thing that
   still gets flagged;
3. a 12% spike on the *previous* nightly, so the newest run returning to
   normal reads as a regression against that one noisy night.

The third is the accepted cost of comparing against the previous run rather
than smoothing. The fix for it is upstream: raise `repetitions` in the
workload recipe so each nightly is a median of several measurements. Note that
the AMD recipes currently default to `repetitions: 1`.

This is a development tool. It never runs in CI and writes nowhere near the
real store unless you point --store at it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / "data" / "events.jsonl"

ROCM_IMAGE = "vllm/vllm-openai-rocm"
CUDA_IMAGE = "vllm/vllm-openai"

# (model, device, tensor-parallel degree)
MODELS = [
    ("deepseek-ai/DeepSeek-V4-Pro-FP8", "mi355x", 8),
    ("moonshotai/Kimi-K2-5-Instruct", "mi355x", 8),
    ("MiniMaxAI/MiniMax-M2-5", "mi300x", 4),
]
# (input length, output length, concurrency)
CONFIGS = [(8192, 1024, 128), (1024, 1024, 256)]
TASKS = [("gsm8k", "exact_match,strict-match"), ("bfcl_simple", "acc,none")]


def _build_url(build_number: int) -> str:
    return f"https://buildkite.com/vllm/perf-eval/builds/{build_number}"


def _perf_event(*, model, device, tp, isl, osl, conc, drift, identity) -> dict:
    base = 2400.0 if device == "mi355x" else 1500.0
    total = base * drift
    return {
        "event": "perf_result",
        "received_at": identity["received_at"],
        "nightly": True,
        "model": model,
        "device": device,
        "precision": "fp8",
        "tp": tp,
        "isl": isl,
        "osl": osl,
        "conc": conc,
        "date": identity["date"],
        "build_number": identity["build_number"],
        "build_url": _build_url(identity["build_number"]),
        "build_commit": "",
        "branch": "main",
        "image": f"{ROCM_IMAGE}:nightly-{identity['commit']}",
        "vllm_commit": identity["commit"],
        "metrics": {
            "tput_per_gpu": round(total, 4),
            "output_tput_per_gpu": round(total * 0.25, 4),
            "input_tput_per_gpu": round(total * 0.75, 4),
            "mean_ttft": round(0.21 / drift, 4),
            "p99_ttft": round(0.42 / drift, 4),
            "mean_tpot": round(0.018 / drift, 4),
            "mean_itl": round(0.019 / drift, 4),
            "mean_e2el": round(18.4 / drift, 4),
            "mean_intvty": round(55.0 * drift, 4),
        },
    }


def _accuracy_event(*, model, device, night, identity, rng) -> dict:
    rows = []
    for index, (task, metric) in enumerate(TASKS):
        score = 0.83 + 0.01 * math.sin(night / 4.0) + rng.uniform(-0.004, 0.004)
        rows.append(
            {
                "task": task,
                "metric": metric,
                "value": round(score, 4),
                "primary": index == 0,
            }
        )
    return {
        "event": "accuracy_result",
        "received_at": identity["received_at"],
        "nightly": True,
        "model": model,
        "workload": f"{model.split('/')[-1].lower()}_{device}",
        "task": "",
        "device": device,
        "date": identity["date"],
        "build_number": identity["build_number"],
        "build_url": _build_url(identity["build_number"]),
        "build_commit": "",
        "branch": "main",
        "image": f"{ROCM_IMAGE}:nightly-{identity['commit']}",
        "vllm_commit": identity["commit"],
        "results": rows,
    }


def generate(nightlies: int = 24, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    events: list[dict] = []

    # Anchored so the newest nightly is last night. The dashboard's time window
    # is anchored to now, so fixed calendar dates would put every run outside
    # the default window and the preview would show only the stale-data notice.
    newest = datetime.now(UTC).replace(hour=4, minute=0, second=0, microsecond=0)

    for night in range(nightlies):
        run_at = newest - timedelta(days=nightlies - 1 - night)
        identity = {
            "commit": f"{rng.getrandbits(160):040x}",
            "date": run_at.strftime("%Y-%m-%d %H:%M:%S"),
            "received_at": (run_at + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "build_number": 1000 + night,
        }
        # A slow sine drift plus small jitter, so some nightlies land inside the
        # noise band and render neutral while others trip good or bad.
        drift = 1.0 + 0.05 * math.sin(night / 3.0) + rng.uniform(-0.015, 0.015)
        for index, (model, device, tp) in enumerate(MODELS):
            for isl, osl, conc in CONFIGS:
                local = drift
                newest_night = night == nightlies - 1

                # Case 1 — a clean 14% regression on the newest nightly, on one
                # model and one shape only, so the per-metric red has something
                # real to show and most charts stay their identity colour.
                if index == 0 and conc == 128 and newest_night:
                    local = drift * 0.86

                # Case 2 — a marginal 3.5% dip on the newest nightly: just over
                # the 2% threshold, so it is the smallest move still flagged.
                if index == 1 and conc == 256 and newest_night:
                    local = drift * 0.965

                # Case 3 — the *previous* nightly spikes 12% and the newest
                # returns to normal, so a perfectly healthy run reads as a ~10%
                # regression against that one noisy night. This is the accepted
                # cost of comparing against the previous run; the fix is a
                # higher `repetitions` upstream, not smoothing here.
                if index == 2 and conc == 128 and night == nightlies - 2:
                    local = drift * 1.12
                events.append(
                    _perf_event(
                        model=model,
                        device=device,
                        tp=tp,
                        isl=isl,
                        osl=osl,
                        conc=conc,
                        drift=local,
                        identity=identity,
                    )
                )
            events.append(
                _accuracy_event(model=model, device=device, night=night, identity=identity, rng=rng)
            )

    # The recipe-derived expectation the coverage card measures against. It
    # deliberately includes one config that never reports, standing in for a
    # workload defined upstream that is failing or has never succeeded — the
    # case a data-derived expectation cannot see at all.
    expected = []
    for model, device, tp in MODELS:
        workload = f"{model.split('/')[-1].lower()}-{device}"
        for isl, osl, conc in CONFIGS:
            expected.append(
                {
                    "workload": workload,
                    "run": f"{isl // 1024}k-in-{osl // 1024}k-out-conc-{conc}",
                    "model": model,
                    "device": device,
                    "precision": "fp8",
                    "tp": tp,
                    "isl": isl,
                    "osl": osl,
                    "conc": conc,
                }
            )
    never_reported = MODELS[0]
    expected.append(
        {
            "workload": f"{never_reported[0].split('/')[-1].lower()}-{never_reported[1]}",
            "run": "8k-in-1k-out-conc-512",
            "model": never_reported[0],
            "device": never_reported[1],
            "precision": "fp8",
            "tp": never_reported[2],
            "isl": 8192,
            "osl": 1024,
            "conc": 512,
        }
    )
    events.append(
        {
            "event": "expected_configs",
            "received_at": newest.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "configs": expected,
        }
    )

    # Out-of-scope decoys. Neither may appear on the rendered page.
    decoy_date = newest.strftime("%Y-%m-%d %H:%M:%S")
    decoy_received = newest.strftime("%Y-%m-%dT%H:%M:%SZ")
    events.append(
        {
            "event": "perf_result",
            "received_at": decoy_received,
            "nightly": True,
            "model": "openai/gpt-oss-120b [NVIDIA - must not render]",
            "device": "h200",
            "precision": "mxfp4",
            "tp": 4,
            "isl": 8192,
            "osl": 1024,
            "conc": 128,
            "date": decoy_date,
            "build_number": 1024,
            "build_url": _build_url(1024),
            "branch": "main",
            "image": f"{CUDA_IMAGE}:nightly-beefcafebeefcafe",
            "vllm_commit": "beefcafebeefcafe",
            "metrics": {"tput_per_gpu": 9999.0},
        }
    )
    events.append(
        {
            "event": "perf_result",
            "received_at": decoy_received,
            "nightly": False,
            "model": "deepseek-ai/DeepSeek-V4-Pro-FP8 [ad-hoc - must not render]",
            "device": "mi355x",
            "precision": "fp8",
            "tp": 8,
            "isl": 8192,
            "osl": 1024,
            "conc": 128,
            "date": decoy_date,
            "build_number": 1025,
            "build_url": _build_url(1025),
            "branch": "main",
            "image": f"{ROCM_IMAGE}:nightly-cafebabecafebabe",
            "vllm_commit": "cafebabecafebabe",
            "metrics": {"tput_per_gpu": 1.0},
        }
    )
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help="Output events.jsonl")
    parser.add_argument("--nightlies", type=int, default=24, help="How many nightlies to fake")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a non-empty store (refused by default)",
    )
    args = parser.parse_args()

    if args.store.exists() and args.store.stat().st_size > 0 and not args.force:
        print(
            f"Refusing to overwrite the non-empty store at {args.store}. "
            "Pass --force if you really mean it.",
            file=sys.stderr,
        )
        return 1

    events = generate(nightlies=args.nightlies, seed=args.seed)
    args.store.parent.mkdir(parents=True, exist_ok=True)
    args.store.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(events)} synthetic events to {args.store}")
    print("Next: python scripts/perf_eval/aggregate.py && python scripts/build_site.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
