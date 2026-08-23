#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Answer the routing study's question: which combination is best, and best value?

Reads a sweep's consolidated JSON and reports, per arm:

  * pass rate and cost per solved task
  * how often the frontier model was actually invoked, and what it recovered
  * the Pareto frontier — arms that nothing else beats on both axes
  * the oracle ceiling, so a result can be read against what was reachable

Usage:
    python3 tools/analyze_router_study.py                     # newest bcb results
    python3 tools/analyze_router_study.py --results <path>
    python3 tools/analyze_router_study.py --baseline single_opus5
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_RESULTS = os.path.join(ROOT, "bigCodeBench-hard", "results", "bcb_all_results.json")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def arm_stats(row):
    tasks = row.get("results", [])
    n = row.get("n") or len(tasks) or 1
    passed = row.get("passed", sum(1 for t in tasks if t.get("passed")))
    usd = row.get("total_as_run_usd", sum(t.get("as_run_usd", 0.0) for t in tasks))

    frontier_tasks = [t for t in tasks if (t.get("routing") or {}).get("frontier_used")]
    frontier_solved = [t for t in frontier_tasks if t.get("passed")]
    # What the frontier tier actually bought: tasks it was handed and solved.
    rungs = [len((t.get("routing") or {}).get("rungs") or []) for t in tasks]

    return {
        "id": row.get("id", "?"),
        "name": row.get("name", row.get("id", "?")),
        "n": n,
        "passed": passed,
        "pass_rate": passed / n if n else 0.0,
        "usd": usd,
        "per_solved": usd / passed if passed else float("inf"),
        "frontier_calls": len(frontier_tasks),
        "frontier_rate": len(frontier_tasks) / n if n else 0.0,
        "frontier_solved": len(frontier_solved),
        "frontier_yield": (len(frontier_solved) / len(frontier_tasks)
                           if frontier_tasks else None),
        "avg_rungs": (sum(rungs) / len(rungs)) if rungs else 0.0,
        "solved_ids": {t.get("task_id") for t in tasks if t.get("passed")},
        "has_routing": any(t.get("routing") for t in tasks),
    }


def pareto(stats):
    """Arms that nothing else beats on BOTH pass rate and cost per solved task."""
    front = []
    for a in stats:
        dominated = any(
            b is not a
            and b["pass_rate"] >= a["pass_rate"]
            and b["per_solved"] <= a["per_solved"]
            and (b["pass_rate"] > a["pass_rate"] or b["per_solved"] < a["per_solved"])
            for b in stats
        )
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda a: -a["pass_rate"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--baseline", default=None,
                    help="arm to compare value against; defaults to the "
                         "same-budget frontier baseline r0b_opus5_solo, then "
                         "single_opus5")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        print(f"no results at {args.results}", file=sys.stderr)
        return 1

    data = load(args.results)
    stats = [arm_stats(r) for r in data.get("summary", [])]
    if not stats:
        print("no arms in this results file", file=sys.stderr)
        return 1

    n = stats[0]["n"]
    print(f"\n{'='*104}")
    print(f"ROUTING STUDY — {data.get('dataset_name', '?')} (N={n})")
    print(f"{'='*104}")
    print(f"{'arm':<26}{'pass':>10}{'$ total':>10}{'$/solved':>11}"
          f"{'frontier':>11}{'yield':>8}{'rungs':>7}")
    print("-" * 104)
    for a in sorted(stats, key=lambda x: -x["pass_rate"]):
        fr = f"{a['frontier_calls']}/{a['n']}" if a["has_routing"] else "-"
        fy = f"{a['frontier_yield']:.0%}" if a["frontier_yield"] is not None else "-"
        print(f"{a['id']:<26}{a['passed']:>4}/{a['n']:<5}{a['usd']:>10.4f}"
              f"{a['per_solved']:>11.4f}{fr:>11}{fy:>8}{a['avg_rungs']:>7.1f}")

    # --- reachability ----------------------------------------------------
    union = set().union(*[a["solved_ids"] for a in stats])
    print(f"\nOracle ceiling across these arms: {len(union)}/{n} "
          f"({len(union)/n:.0%}) — no single arm can exceed this on this slice.")

    # --- Pareto ----------------------------------------------------------
    front = pareto(stats)
    print(f"\n{'-'*104}\nPARETO FRONTIER (nothing beats these on both accuracy and value)\n{'-'*104}")
    for a in front:
        print(f"  {a['id']:<26} {a['pass_rate']:>5.0%}  ${a['per_solved']:.4f}/solved  "
              f"${a['usd']:.4f} total")

    # --- recommendation ---------------------------------------------------
    best_acc = max(stats, key=lambda a: (a["pass_rate"], -a["per_solved"]))
    best_val = min((a for a in stats if a["passed"]), key=lambda a: a["per_solved"])
    wanted = [args.baseline] if args.baseline else ["r0b_opus5_solo", "single_opus5"]
    base = next((a for w in wanted for a in stats if a["id"] == w), None)

    print(f"\n{'-'*104}\nREAD-OUT\n{'-'*104}")
    print(f"  Highest accuracy : {best_acc['id']} — {best_acc['pass_rate']:.0%} "
          f"at ${best_acc['per_solved']:.4f}/solved")
    print(f"  Best value       : {best_val['id']} — {best_val['pass_rate']:.0%} "
          f"at ${best_val['per_solved']:.4f}/solved")

    if base and base is not best_acc:
        for a in front:
            if a is base:
                continue
            acc_ratio = a["pass_rate"] / base["pass_rate"] if base["pass_rate"] else 0
            cost_ratio = a["per_solved"] / base["per_solved"] if base["per_solved"] else 0
            if acc_ratio >= 0.95 and cost_ratio < 1.0:
                print(f"  Sweet spot       : {a['id']} reaches {acc_ratio:.0%} of "
                      f"{base['id']}'s accuracy at {cost_ratio:.0%} of its cost per "
                      f"solved task")
                break

    frontier_arms = [a for a in stats if a["has_routing"] and a["frontier_calls"]]
    if frontier_arms:
        print("\n  Frontier-tier economics (was opus-5 worth calling?):")
        for a in sorted(frontier_arms, key=lambda x: -x["pass_rate"]):
            print(f"    {a['id']:<24} called on {a['frontier_rate']:>4.0%} of tasks, "
                  f"solved {a['frontier_solved']}/{a['frontier_calls']} of them "
                  f"({a['frontier_yield']:.0%} yield)")

    stale = [a["id"] for a in stats if not a["has_routing"]]
    if stale:
        print(f"\n  Note: no routing trace for {', '.join(stale)} — these are not "
              f"router arms, so frontier columns are blank rather than zero.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
