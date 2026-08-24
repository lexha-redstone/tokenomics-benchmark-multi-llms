#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Run the claude-opus-5 arm on ClassEval on its own, then merge it into the sweep.

`run_benchmark.py --dataset classeval --group classeval` deliberately does not
include an opus-5 arm: opus is priced ~2.5x Sonnet-5 and ~17x Gemini 3.7 Flash
per output token, so putting it in the standard group would reprice every sweep.
This script is the opt-in path -- it runs exactly one arm (`ce_single_opus`,
which is `run_ce_single` with model_id=claude-opus-5, the same shape as the
C0a/C0b/C0c singles) and then folds its row into the existing results file so
the comparison table shows opus-5 beside the arms it is being compared with.

The run itself goes through `src.sweep.run_arm`, the same loop
`run_benchmark.py` uses, so the row is scored by identical rules: same task
cache (`classeval/results/cache_classeval_master.json`, under key
`ce_single_opus`), same discard-and-retry policy for API failures, same summary
shape.

Usage:
  # run the opus-5 arm over the same 91 tasks, merge, and regenerate reports:
  python3 run_classeval_opus5.py --n 91

  # run only, do not touch the existing results file:
  python3 run_classeval_opus5.py --n 91 --no-merge

  # merge a previous opus-5 run into the sweep without re-running anything:
  python3 run_classeval_opus5.py --merge-only

Equivalent one-off through the master runner (no merge step):
  python3 run_benchmark.py --dataset classeval --variants ce_single_opus --n 91
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Same escape hatch as run_benchmark.py: point SJ_SRC at a source checkout of
# the harness instead of the installed `ctx-harness` package.
SJ_SRC = os.environ.get("SJ_SRC", "")
if SJ_SRC and os.path.isdir(SJ_SRC) and SJ_SRC not in sys.path:
    sys.path.insert(0, SJ_SRC)

from src.architectures import get_configurations
from src.datasets import load_dataset
from src.evaluator import straitjacket_status
from src.merge import (comparability_warnings, load_results, merge_into_file,
                       render_reports)
from src.sweep import load_cache, print_scoreboard, run_arm

ARM_ID = "ce_single_opus"
DS_KEY = "classeval"
DATASET_NAME = "ClassEval"
RESULTS_DIR = os.path.join(HERE, "classeval", "results")
CACHE_FILE = os.path.join(RESULTS_DIR, f"cache_{DS_KEY}_master.json")
BASE_RESULTS = os.path.join(RESULTS_DIR, f"{DS_KEY}_{DS_KEY}_results.json")
ARM_RESULTS = os.path.join(RESULTS_DIR, f"{DS_KEY}_opus5_results.json")

# Opus-5 is 2.5x Sonnet-5 per token in src/config.PRICING. Used only when the
# base results carry no Sonnet row to scale from.
_SONNET_ROW = "ce_single_sonnet"
_OPUS_OVER_SONNET = 2.5
_FALLBACK_USD_PER_TASK = 0.055


def _estimate_usd(n):
    """What this run is likely to cost, scaled from the measured Sonnet row.

    An estimate from this repo's own numbers beats a guess, and an arm that
    spends real money should say so before it starts, not after.
    """
    doc = load_results(BASE_RESULTS)
    for row in (doc or {}).get("summary", []):
        if row.get("id") == _SONNET_ROW and row.get("n"):
            per_task = row["total_as_run_usd"] / row["n"]
            return per_task * _OPUS_OVER_SONNET * n, (
                f"scaled from the measured `{_SONNET_ROW}` row "
                f"(${row['total_as_run_usd']:.4f} over {row['n']} tasks) at "
                f"{_OPUS_OVER_SONNET}x Opus/Sonnet pricing")
    return _FALLBACK_USD_PER_TASK * n, "rough per-task fallback (no Sonnet row to scale from)"


def _cached_count(task_ids):
    cache = load_cache(CACHE_FILE)
    done = cache.get(ARM_ID, {})
    return sum(1 for t in task_ids if t in done)


def _merge_and_report(arm_rows, base_path, do_report=True, tag="opus5"):
    """Fold `arm_rows` into the sweep results and rebuild the comparison."""
    print("\n" + "=" * 80)
    print("MERGING INTO EXISTING SWEEP RESULTS")
    print("=" * 80, flush=True)

    doc, path = merge_into_file(base_path, arm_rows)
    rows = doc["summary"]
    print(f"Merged results written to: {path}")
    if os.path.exists(path + ".bak"):
        print(f"Previous version kept at:  {path}.bak")

    for note in comparability_warnings(rows):
        print(f"  !! {note}", flush=True)

    md = html = None
    if do_report:
        md, html = render_reports(rows, DATASET_NAME, tag=tag)
        print(f"\nRegenerated Comparative Reports (all {len(rows)} arms):")
        print(f"  - Markdown: {md}")
        print(f"  - HTML:     {html}")

    print_scoreboard(rows, DATASET_NAME, doc.get("n", 0))
    return rows, md, html


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", "-n", type=int, default=91,
                        help="Number of ClassEval tasks (default: 91, matching the "
                             "existing sweep). Use the SAME n as the run you are "
                             "merging into or the pass rates are not comparable.")
    parser.add_argument("--split", "-s", default=None, help="Dataset split (default: standard)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore the task cache and re-run every task live")
    parser.add_argument("--task-retries", type=int, default=3,
                        help="Re-attempts for a task whose API calls failed (default: 3)")
    parser.add_argument("--allow-simulation", action="store_true",
                        help="Substitute SIMULATED output on an unrecoverable API failure "
                             "instead of discarding the task. Off by default.")
    parser.add_argument("--base", default=BASE_RESULTS,
                        help=f"Existing sweep results to merge into (default: {BASE_RESULTS})")
    parser.add_argument("--out", "-o", default=ARM_RESULTS,
                        help="Where to write this arm's standalone results JSON")
    parser.add_argument("--no-merge", action="store_true",
                        help="Run the arm but leave the existing results file untouched")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip the benchmark; merge a previous run's --out file "
                             "into --base and regenerate the reports")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip Markdown/HTML report generation after the merge")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Do not prompt for confirmation of the estimated API spend")
    args = parser.parse_args()

    do_report = not args.no_report

    # --- merge-only: no API calls, no dataset load -------------------------
    if args.merge_only:
        doc = load_results(args.out)
        if doc is None:
            print(f"ERROR: no standalone opus-5 results at {args.out}. Run without "
                  "--merge-only first.", file=sys.stderr)
            return 1
        _merge_and_report(doc.get("summary", []), args.base, do_report=do_report)
        return 0

    if args.allow_simulation:
        os.environ["ALLOW_SIMULATION"] = "1"
        print("WARNING: --allow-simulation is on. Failed API calls will be replaced by "
              "simulated output, marked `simulated: true` in the results.", flush=True)

    sj_state = straitjacket_status()
    print(f"straitjacket: backend={sj_state['backend']} ctx={sj_state['ctx_version']} "
          f"workspace={sj_state['workspace']}"
          + ("" if sj_state["available"] else f"\n  UNAVAILABLE: {sj_state['reason']}\n"
             "  This arm requires the harness and will refuse to run. "
             "`pip install ctx-harness` to enable it."), flush=True)

    configs = get_configurations(dataset=DS_KEY, variant_keys=[ARM_ID])
    if not configs:
        print(f"ERROR: variant `{ARM_ID}` is not registered.", file=sys.stderr)
        return 1
    cfg = configs[0]

    # This arm is @_arm(sj_required=True): it refuses to produce a row labelled
    # "Straitjacket ($0.00)" without the real harness. Stop here rather than on
    # the first task, so the failure is a message and not a traceback.
    if not sj_state["available"] and getattr(cfg["fn"], "sj_required", True):
        print(f"\nERROR: `{ARM_ID}` requires the straitjacket harness, which is "
              f"unavailable here ({sj_state['reason']}).\n"
              "       Install it with `pip install ctx-harness`, or point SJ_SRC at a "
              "source checkout.\n"
              "       Nothing was run and no file was modified.", file=sys.stderr)
        return 1

    print("=" * 80)
    print("CLASSEVAL SINGLE-ARM RUNNER: claude-opus-5")
    print(f"Arm: {cfg['name']} | Tasks (N): {args.n}")
    print("=" * 80, flush=True)

    problems = load_dataset(DS_KEY, split=args.split, max_tasks=args.n)
    task_ids = list(problems.keys())[:args.n]
    n = len(task_ids)

    cached = 0 if args.no_cache else _cached_count(task_ids)
    to_run = n - cached
    est_usd, basis = _estimate_usd(to_run)
    print(f"Loaded {n} tasks. {cached} already cached, {to_run} to run live.")
    print(f"Estimated API spend for this run: ~${est_usd:.2f} ({basis}).", flush=True)

    if to_run > 0 and not args.yes and sys.stdin.isatty():
        try:
            reply = input("Proceed with the live opus-5 run? [y/N] ").strip().lower()
        except EOFError:
            reply = ""
        if reply not in ("y", "yes"):
            print("Aborted. Nothing was run and no file was modified.")
            return 1

    # --- run the single arm through the shared sweep loop ------------------
    cache = load_cache(CACHE_FILE, no_cache=args.no_cache)
    summary = run_arm(cfg, problems, task_ids, cache=cache, cache_file=CACHE_FILE,
                      no_cache=args.no_cache, task_retries=args.task_retries,
                      sj_state=sj_state, label="[1/1] ")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"dataset": DS_KEY, "dataset_name": DATASET_NAME, "group": "opus5",
                   "n": n, "straitjacket": sj_state, "summary": [summary]}, f, indent=2)
    print(f"Saved this arm's standalone results to: {args.out}")

    # --- merge into the sweep so the row can be compared -------------------
    if args.no_merge:
        print("\n--no-merge: the existing results file was not modified. Merge later with:\n"
              f"  python3 {os.path.basename(__file__)} --merge-only")
        print_scoreboard([summary], DATASET_NAME, n)
        return 0

    try:
        _merge_and_report([summary], args.base, do_report=do_report)
    except FileNotFoundError as e:
        # The arm's own results are already on disk; only the comparison failed.
        print(f"\nWARNING: {e}", file=sys.stderr)
        print(f"This arm's results are safe at {args.out}; re-run with --merge-only "
              "once the base sweep exists.", file=sys.stderr)
        print_scoreboard([summary], DATASET_NAME, n)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
