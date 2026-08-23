#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Recompute the architecture-pattern findings in README section 1 from raw results.

Every number in "Why the cascade shape suits this dataset" comes out of this
script, so a reader can check the claim rather than trust the table.

What it measures
----------------
`repair_loops` in a result record is the turn that produced the final answer:
0 means the first attempt passed, 1 means the first repair turn rescued it, and
so on. Splitting an arm by that field separates two things the headline pass
rate fuses together -- how good the first attempt was, and how much the repair
budget recovered. The rescue rate is computed only over the tasks that arm
itself failed, which is what makes arms with different first-rung models
comparable.

The significance tests are two-proportion z-tests. They are here because most of
the pattern-level gaps do NOT clear significance at N=100, and a findings
section that quotes only the gaps that do would be cherry-picking.

Usage:
    python3 tools/analyze_patterns.py
    python3 tools/analyze_patterns.py --results <path to a results JSON>
"""

import argparse
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_RESULTS = os.path.join(ROOT, "bigCodeBench-hard", "results", "archive",
                               "bcb_n100_instrumented_20260822T2129.json")
N50_DIR = os.path.join(ROOT, "bigCodeBench-hard", "results")

# The rung each arm hands a failing task to, and whether that is a step up.
RUNGS = {
    "sj_escalation_shield": ("flash UP from lite", "claude-sonnet-5 UP"),
    "sj_cascade":           ("flash UP from lite", "flash again, flat"),
    "sj_hybrid":            ("flash UP from lite", "-"),
    "sj_smart_repair":      ("lite DOWN from flash", "flash medium UP"),
    "single_flash37":       ("flash, itself", "-"),
    "single_sonnet5":       ("sonnet, itself", "-"),
    "single_opus5":         ("opus, itself", "-"),
}


def two_proportion(a, na, b, nb):
    """z and two-sided p for a difference of proportions."""
    p1, p2 = a / na, b / nb
    pooled = (a + b) / (na + nb)
    se = math.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
    if se == 0:
        return p1, p2, float("nan"), 1.0
    z = (p1 - p2) / se
    return p1, p2, z, math.erfc(abs(z) / math.sqrt(2))


def by_turn(results):
    """{turn: (passed, total)} keyed by repair_loops."""
    out = {}
    for r in results:
        k = r.get("repair_loops", 0)
        p, t = out.get(k, (0, 0))
        out[k] = (p + (1 if r.get("passed") else 0), t + 1)
    return out


def rescue(results, turn):
    """Rescued-by-turn over everything that reached it, not over all tasks."""
    reached = [r for r in results if r.get("repair_loops", 0) >= turn]
    won = sum(1 for r in reached if r.get("passed") and r.get("repair_loops") == turn)
    return won, len(reached)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--n50", action="store_true", default=True,
                    help="also read the N=50 per-arm JSONs (default: on)")
    args = ap.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        data = json.load(f)
    arms = {s["id"]: s for s in data["summary"]}

    print(f"results: {args.results}")
    print(f"N = {data.get('n')}\n")

    print("PER-TURN BREAKDOWN")
    print(f"  {'arm':<22}{'1st try':>9}{'1st repair':>22}{'2nd repair':>22}   rungs")
    for aid, s in arms.items():
        R = s["results"]
        turns = by_turn(R)
        first = turns.get(0, (0, 0))[0]
        w1, n1 = rescue(R, 1)
        w2, n2 = rescue(R, 2)
        r1 = f"{w1}/{n1} = {w1 / n1:.0%}" if n1 else "-"
        r2 = f"{w2}/{n2} = {w2 / n2:.0%}" if n2 else "-"
        up = RUNGS.get(aid, ("?", "?"))
        print(f"  {aid:<22}{first:>4}/{len(R):<4}{r1:>22}{r2:>22}   {up[0]} | {up[1]}")

    # The escalate-up group is the two arms whose first repair steps up a tier.
    up_w = sum(rescue(arms[a]["results"], 1)[0] for a in ("sj_cascade", "sj_escalation_shield"))
    up_n = sum(rescue(arms[a]["results"], 1)[1] for a in ("sj_cascade", "sj_escalation_shield"))
    dn_w, dn_n = rescue(arms["sj_smart_repair"]["results"], 1)
    hy_w, hy_n = rescue(arms["sj_hybrid"]["results"], 1)
    fl_w, fl_n = rescue(arms["single_flash37"]["results"], 1)

    print("\nSIGNIFICANCE (two-proportion z-test, two-sided)")
    tests = [
        ("1st repair: escalate UP vs de-escalate", up_w, up_n, dn_w, dn_n),
        ("1st repair: escalate UP vs plan-anchored", up_w, up_n, hy_w, hy_n),
        ("1st repair: escalate UP vs same-model retry", up_w, up_n, fl_w, fl_n),
        ("1st try: plan+lite vs bare lite",
         by_turn(arms["sj_hybrid"]["results"]).get(0, (0, 0))[0], 100,
         by_turn(arms["sj_cascade"]["results"]).get(0, (0, 0))[0], 100),
        ("final: escalation shield vs plan & execute",
         arms["sj_escalation_shield"]["passed"], 100, arms["sj_hybrid"]["passed"], 100),
    ]
    for name, a, na, b, nb in tests:
        p1, p2, z, pv = two_proportion(a, na, b, nb)
        mark = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        print(f"  {name:<44}{p1:>6.0%} vs{p2:>5.0%}   z={z:+5.2f}  p={pv:.4f}  {mark}")

    # Premium calls: unconditional (planner, every task) vs conditional (only
    # after a test failure). This is the cost mechanism, not a proxy for it.
    hy = arms["sj_hybrid"]["results"]
    sh = arms["sj_escalation_shield"]["results"]
    hy_rep = sum(1 for r in hy if r.get("repair_loops", 0) >= 1)
    sh_r1 = sum(1 for r in sh if r.get("repair_loops", 0) >= 1)
    sh_r2 = sum(1 for r in sh if r.get("repair_loops", 0) >= 2)
    print("\nPREMIUM-MODEL CALLS")
    print(f"  sj_hybrid            {len(hy)} planner (unconditional) + {hy_rep} repair "
          f"= {len(hy) + hy_rep} flash calls -> {arms['sj_hybrid']['passed']}/100")
    print(f"  sj_escalation_shield {sh_r1} flash + {sh_r2} sonnet (both conditional) "
          f"= {sh_r1 + sh_r2} calls -> {arms['sj_escalation_shield']['passed']}/100")

    # Task shape: why a planner has little to decompose here.
    try:
        from src.datasets import load_dataset
        rows = list(load_dataset("bcb", max_tasks=data.get("n") or 100).values())
        tok = lambda t: len(t) // 4
        libs = lambda r: len(r["libs"] if isinstance(r["libs"], list) else eval(r["libs"]))
        shape = {
            "prompt tokens": [tok(r["complete_prompt"]) for r in rows],
            "gold solution": [tok(r["canonical_solution"]) for r in rows],
            "unit tests": [len(re.findall(r"def test", r["test"])) for r in rows],
            "libraries": [libs(r) for r in rows],
        }
        print("\nTASK SHAPE (why there is nothing to decompose)")
        for k, v in shape.items():
            print(f"  {k:<16} mean {statistics.mean(v):>6.0f}  median "
                  f"{statistics.median(v):>5.0f}  min {min(v):>5}  max {max(v):>5}")
    except Exception as e:
        print(f"\nTASK SHAPE: skipped ({e})")

    if args.n50:
        print("\nN=50 SWEEP (older harness, gemini-3.6-flash; ordering only)")
        for fn, label in [("n50_c2_claude_frontier_opus.json", "escalation (Sonnet->Opus)"),
                          ("n50_g2_smart_tiered_cascade.json", "cascade"),
                          ("n50_g4_dual_candidate_verifier.json", "collaboration"),
                          ("n50_g1_pure_lite_budget.json", "single model, repeated"),
                          ("n50_g3_advisor_executor.json", "planning & executing")]:
            path = os.path.join(N50_DIR, fn)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            turns = by_turn(d["results"])
            detail = " ".join(f"L{k}:{p}/{t}" for k, (p, t) in sorted(turns.items()))
            print(f"  {label:<28}{d['passed']:>3}/{d['n']} = {d['passed'] / d['n']:.0%}   {detail}")
        print("  (every gap here is inside binomial noise at N=50)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
