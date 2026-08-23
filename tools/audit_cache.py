#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Audit a benchmark cache for records that were never produced by a live API call,
and drop them so a resumed sweep re-runs only what it must.

Why this is needed
------------------
`src/client.py` used to answer any unrecoverable API failure with
`_fallback_dispatch`, a simulator. It priced invented token counts with the real
rate card and returned them in the ordinary usage dict, so a 504 or an expired
credential became an ordinary-looking datapoint. Records written before that was
fixed carry no provenance, so contamination has to be inferred.

Two independent detectors
-------------------------
**Fingerprint.** The simulator's failure branch emits
`raise NotImplementedError('incomplete')`, whose traceback lands in the cached
error text. Decisive when present, but it only catches the failure branch — the
success branch copies the dataset's own `canonical_solution`, which passes and
leaves no trace.

**Plausibility.** Every arm's first rung is a known model, so its first-attempt
pass rate can be checked against a live reference run of the same model. The
success branch inflates exactly this number, which is how a
`gemini-3.5-flash-lite` ladder came to "pass" 66% at rung one when the same
model passes 32% live.

An arm flagged by either detector should be re-run. Records are never edited,
only removed whole, and the cache is backed up first.

Usage:
    python3 tools/audit_cache.py                     # report only
    python3 tools/audit_cache.py --purge             # drop flagged arms
    python3 tools/audit_cache.py --purge --arm r5_gemini_think_ladder
"""

import argparse
import json
import os
import shutil
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "bigCodeBench-hard", "results", "cache_bcb_master.json")
REFERENCE = os.path.join(ROOT, "bigCodeBench-hard", "results", "archive",
                         "bcb_n100_instrumented_20260822T2129.json")

# Live first-attempt pass rates, measured. An arm whose first rung is one of
# these models should land near its reference, not far above it.
REFERENCE_RUNG1 = {
    "gemini-3.5-flash-lite": 0.32,   # from sj_cascade, whose first rung is Lite
    "gemini-3.7-flash": 0.35,        # from single_flash37
    "claude-sonnet-5": 0.40,
    "claude-opus-5": 0.56,           # from single_opus5
}

# How far above its reference an arm may sit before it is called implausible.
# Generous: thinking level, prompt wording and the task slice all move this a
# little. Simulation moves it by tens of points.
RUNG1_TOLERANCE = 0.18

SIM_FINGERPRINTS = ("NotImplementedError", "incomplete")


def first_rung_model(tasks):
    """The model an arm actually called first, read from its routing trace."""
    for t in tasks.values():
        rungs = (t.get("routing") or {}).get("rungs") or []
        if rungs:
            return rungs[0].split("/")[0]
    return None


def audit_arm(arm, tasks):
    n = len(tasks)
    finger = {tid for tid, t in tasks.items()
              if all(f in str(t.get("error") or "") for f in SIM_FINGERPRINTS)}

    at0 = sum(1 for t in tasks.values()
              if t.get("passed") and t.get("repair_loops") == 0)
    rung1_rate = at0 / n if n else 0.0
    model = first_rung_model(tasks)
    ref = REFERENCE_RUNG1.get(model)
    implausible = ref is not None and rung1_rate > ref + RUNG1_TOLERANCE

    reasons = []
    if finger:
        reasons.append(f"{len(finger)} task(s) carry the simulator's "
                       f"NotImplementedError fingerprint")
    if implausible:
        reasons.append(f"first-rung pass rate {rung1_rate:.0%} vs {ref:.0%} live for "
                       f"{model} (+{(rung1_rate - ref) * 100:.0f}pp, tolerance "
                       f"{RUNG1_TOLERANCE * 100:.0f}pp)")

    costs = [t.get("as_run_usd", 0.0) for t in tasks.values()]
    return {
        "arm": arm,
        "n": n,
        "passed": sum(1 for t in tasks.values() if t.get("passed")),
        "rung1_rate": rung1_rate,
        "model": model,
        "fingerprint_hits": len(finger),
        "implausible": implausible,
        "median_cost": statistics.median(costs) if costs else 0.0,
        "suspect": bool(reasons),
        "reasons": reasons,
        # Records written after the fix carry their own provenance; trust it
        # over inference when it is there.
        "self_reported_simulated": sum(1 for t in tasks.values()
                                       if t.get("simulated_calls")),
        "has_provenance": any("simulated_calls" in t for t in tasks.values()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--purge", action="store_true",
                    help="remove flagged arms from the cache (backs it up first)")
    ap.add_argument("--arm", action="append", default=[],
                    help="also purge this arm regardless of the verdict "
                         "(repeatable)")
    ap.add_argument("--keep", action="append", default=[],
                    help="never purge this arm, even if flagged (repeatable)")
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        print(f"no cache at {args.cache}", file=sys.stderr)
        return 1

    with open(args.cache, encoding="utf-8") as f:
        cache = json.load(f)

    audits = [audit_arm(a, t) for a, t in sorted(cache.items())]

    print(f"\n{'='*100}")
    print(f"CACHE AUDIT — {args.cache}")
    print(f"{'='*100}")
    print(f"{'arm':<28}{'tasks':>6}{'passed':>8}{'rung-1':>8}{'first rung':>24}{'verdict':>12}")
    print("-" * 100)
    for a in audits:
        verdict = "SUSPECT" if a["suspect"] else ("provenance" if a["has_provenance"]
                                                  else "plausible")
        print(f"{a['arm']:<28}{a['n']:>6}{a['passed']:>8}{a['rung1_rate']:>7.0%}"
              f"{str(a['model'] or '?'):>24}{verdict:>12}")

    flagged = [a for a in audits if a["suspect"]]
    if flagged:
        print(f"\n{'-'*100}\nWHY\n{'-'*100}")
        for a in flagged:
            print(f"  {a['arm']}")
            for r in a["reasons"]:
                print(f"    - {r}")

    trusted = [a["arm"] for a in audits if not a["suspect"]]
    print(f"\n{'-'*100}\nVERDICT\n{'-'*100}")
    print(f"  reusable : {', '.join(trusted) if trusted else '(none)'}")
    print(f"  re-run   : {', '.join(a['arm'] for a in flagged) if flagged else '(none)'}")
    print("\n  Inference has limits: the simulator's success branch copies the "
          "dataset's own\n  answer and leaves no fingerprint, so an arm called "
          "'plausible' is only\n  unflagged, not proven clean. Records written "
          "from now on carry provenance.")

    to_drop = {a["arm"] for a in flagged} | set(args.arm)
    to_drop -= set(args.keep)
    if not to_drop:
        print("\n  nothing to purge.\n")
        return 0

    if not args.purge:
        print(f"\n  re-run `--purge` to drop: {', '.join(sorted(to_drop))}\n")
        return 0

    backup = f"{args.cache}.bak-{time.strftime('%Y%m%dT%H%M%S')}"
    shutil.copy2(args.cache, backup)
    for arm in to_drop:
        cache.pop(arm, None)
    with open(args.cache, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"\n  backed up  : {backup}")
    print(f"  purged     : {', '.join(sorted(to_drop))}")
    print(f"  remaining  : {', '.join(sorted(cache)) or '(empty)'}")
    print("\n  Resume WITHOUT --no-cache so the remaining arms are reused:\n"
          "    python3 run_benchmark.py --dataset bcb --group router --n=148 --report\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
