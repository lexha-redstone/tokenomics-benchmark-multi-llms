#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Turn a ClassEval sweep into a verdict on H1.

H1 (docs/pattern-dataset-selection.md): when one task contains sub-tasks of
unequal difficulty, routing them to models BY that difficulty should beat a
cascade at matched cost, because the cascade can only escalate the whole task
and so re-solves the easy methods at frontier prices.

Four things decide that, and this script prints all four whichever way they
fall:

1. **Cost per solved task.** H1 is a claim about cost at matched quality, not
   about pass rate. A routed arm that wins on pass rate by spending more has
   not supported it.
2. **The flat control.** `ce_route_flat` runs the same per-method loop with one
   model for every method. If the routed arm does not beat it, any advantage
   belongs to writing the class method-by-method, not to routing by difficulty
   -- a different claim.
3. **Per-tier delivery.** The routed arm has to show the cheap model actually
   passing the tiers it was routed. Winning overall while failing on
   `standalone` would mean the routing worked by accident.
4. **The integration gap.** Tasks where every method passed its own tests but
   the class-level suite still failed. That is the failure a planner is
   supposed to prevent, so it is the number that separates `ce_plan_route`
   from `ce_route_by_tier`.

Usage:
    python3 tools/analyze_classeval.py
    python3 tools/analyze_classeval.py --results classeval/results/classeval_all_results.json
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "classeval", "results")
# run_benchmark names the file after the --group it was given, so a sweep run
# with --group classeval and one run with explicit --variants land in different
# files. Prefer the group sweep, fall back to whatever exists.
_CANDIDATES = ["classeval_classeval_results.json", "classeval_all_results.json"]


def _default_results():
    for name in _CANDIDATES:
        path = os.path.join(RESULTS_DIR, name)
        if os.path.exists(path):
            return path
    return os.path.join(RESULTS_DIR, _CANDIDATES[0])


DEFAULT = None

TIER_ORDER = ["standalone", "lib_dep", "field_dep", "field_lib", "method_dep"]


def two_proportion(a, na, b, nb):
    if not na or not nb:
        return 0.0, 0.0, float("nan"), 1.0
    p1, p2 = a / na, b / nb
    pooled = (a + b) / (na + nb)
    se = math.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
    if se == 0:
        return p1, p2, float("nan"), 1.0
    z = (p1 - p2) / se
    return p1, p2, z, math.erfc(abs(z) / math.sqrt(2))


def mark(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def integration_gap(arm):
    """Tasks whose methods all passed but whose class-level suite did not."""
    gap = 0
    for r in arm.get("results", []):
        subs = r.get("subtasks") or []
        if subs and all(s.get("passed") for s in subs) and not r.get("passed"):
            gap += 1
    return gap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None,
                    help="results JSON (default: the newest ClassEval sweep)")
    args = ap.parse_args()
    args.results = args.results or _default_results()

    if not os.path.exists(args.results):
        print(f"no results at {args.results}\n"
              "run: python3 run_benchmark.py --dataset classeval --group classeval "
              "--n 91 --report")
        return 1

    with open(args.results, "r", encoding="utf-8") as f:
        data = json.load(f)
    arms = {s["id"]: s for s in data["summary"]}
    n = data.get("n", 0)
    print(f"results: {args.results}\nN = {n} tasks\n")

    print("HEADLINE")
    print(f"  {'arm':<20}{'class pass':>12}{'method pass':>14}{'total $':>10}"
          f"{'$/solved':>11}{'integ gap':>11}")
    for aid, s in arms.items():
        roll = s.get("subtask_rollup") or {}
        mp = (f"{roll.get('passed_subtasks', 0)}/{roll.get('n_subtasks', 0)}"
              if roll.get("n_subtasks") else "-")
        print(f"  {aid:<20}{s['passed']:>5}/{s['n']:<6}{mp:>14}"
              f"{s['total_as_run_usd']:>10.4f}{s['cost_per_solved_usd']:>11.4f}"
              f"{integration_gap(s):>11}")

    routed, cascade = arms.get("ce_route_by_tier"), arms.get("ce_cascade")
    flat, plan = arms.get("ce_route_flat"), arms.get("ce_plan_exec")

    print("\nH1 -- routed vs cascade")
    if not (routed and cascade):
        print("  need both ce_route_by_tier and ce_cascade in this sweep")
    else:
        p1, p2, z, pv = two_proportion(routed["passed"], routed["n"],
                                       cascade["passed"], cascade["n"])
        print(f"  pass rate     routed {p1:.0%}  vs cascade {p2:.0%}   "
              f"z={z:+.2f} p={pv:.4f} {mark(pv)}")
        rc, cc = routed["cost_per_solved_usd"], cascade["cost_per_solved_usd"]
        print(f"  $ per solved  routed ${rc:.4f} vs cascade ${cc:.4f}   "
              f"{'routed is ' + format(cc / rc, '.2f') + 'x cheaper' if rc and cc > rc else 'cascade is not beaten on cost'}")
        verdict = ("SUPPORTED" if (rc and cc > rc and routed["passed"] >= cascade["passed"])
                   else "NOT SUPPORTED")
        print(f"  --> H1 {verdict} on cost at matched-or-better pass rate")

    print("\nCONTROL -- routing by difficulty vs the same loop, one model")
    if not (routed and flat):
        print("  need both ce_route_by_tier and ce_route_flat; without the "
              "control no claim about difficulty routing is supported")
    else:
        p1, p2, z, pv = two_proportion(routed["passed"], routed["n"],
                                       flat["passed"], flat["n"])
        print(f"  pass rate     routed {p1:.0%}  vs flat {p2:.0%}   "
              f"z={z:+.2f} p={pv:.4f} {mark(pv)}")
        print(f"  $ per solved  routed ${routed['cost_per_solved_usd']:.4f} "
              f"vs flat ${flat['cost_per_solved_usd']:.4f}")
        if pv >= 0.05:
            print("  --> the advantage, if any, is per-method GENERATION, "
                  "not difficulty routing")

    print("\nPER-TIER DELIVERY (method-level pass rate; did the cheap rung deliver?)")
    header = "  " + f"{'arm':<20}" + "".join(f"{t:>14}" for t in TIER_ORDER)
    print(header)
    for aid, s in arms.items():
        roll = (s.get("subtask_rollup") or {}).get("by_tier") or {}
        cells = ""
        for t in TIER_ORDER:
            b = roll.get(t)
            cells += f"{(str(b['passed']) + '/' + str(b['n'])):>14}" if b else f"{'-':>14}"
        print(f"  {aid:<20}{cells}")

    print("\nPER-MODEL DELIVERY (who wrote it, and did it pass)")
    for aid, s in arms.items():
        roll = (s.get("subtask_rollup") or {}).get("by_model") or {}
        if not roll:
            continue
        parts = " ".join(f"{m}={b['passed']}/{b['n']}(${b['usd']:.4f})"
                         for m, b in sorted(roll.items()))
        print(f"  {aid:<20}{parts}")

    print("\nNOTE  `model_id` on a sub-task is the FINAL writer; a repaired "
          "method\n      carries `initial_model_id` too. Whole-class arms split "
          "their class\n      spend evenly across methods -- they buy the methods "
          "as a bundle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
