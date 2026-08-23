#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Master Unified Benchmark Runner for Multi-LLM Tokenomics & Straitjacket Evaluation.

Runs end-to-end evaluation for any target dataset (BigCodeBench-Hard, SWE-bench Pro, WebDev)
and specified variant configurations/groups, with automatic Markdown report and HTML dashboard generation.

Usage Examples:
  # Run SWE-bench Pro on 30 tasks with all Straitjacket variants + generate reports:
  python3 run_benchmark.py --dataset swebench --group straitjacket --n 30 --report

  # Run BigCodeBench-Hard on 10 tasks with specific variants:
  python3 run_benchmark.py --dataset bcb --variants single_flash36,sj_hybrid,sj_smart_repair --n 10 --report

  # Run WebDev single-model baseline comparisons:
  python3 run_benchmark.py --dataset webdev --group single --n 10 --report
"""

import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The straitjacket harness is a normal dependency (`pip install ctx-harness`).
# SJ_SRC still lets a contributor point at a source checkout of the upstream
# repository instead, e.g. SJ_SRC=/path/to/straitjacket/src.
SJ_SRC = os.environ.get("SJ_SRC", "")
if SJ_SRC and os.path.isdir(SJ_SRC) and SJ_SRC not in sys.path:
    sys.path.insert(0, SJ_SRC)

from src.datasets import load_dataset
from src.client import (DispatchError, reset_simulated_calls,
                        simulated_calls, simulation_allowed)
from src.evaluator import (straitjacket_status,
                           aggregate_containment as _aggregate_containment,
                           classeval_subtask_summary as _classeval_subtask_summary)
from src.architectures import get_configurations, VARIANT_REGISTRY
from src.reporter import (allocate_report_paths, generate_markdown_report,
                          generate_html_dashboard)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", "-d", choices=["bcb", "bigcodebench", "bigcodebench-hard",
                                                    "swebench", "swebench_pro", "swe-bench",
                                                    "webdev", "web-dev",
                                                    "classeval", "class-eval", "ce"],
                        default="swebench", help="Dataset to evaluate (default: 'swebench')")
    parser.add_argument("--group", "-g", choices=["all", "single", "combo", "straitjacket", "sj",
                                                  "nextgen", "ablation", "router",
                                                  "classeval", "ce"],
                        default="all", help="Preset variant group to run (default: 'all')")
    parser.add_argument("--variants", "-v", default="",
                        help="Comma-separated variant IDs to run (e.g. 'single_flash36,sj_hybrid,sj_escalation_shield')")
    parser.add_argument("--n", "-n", type=int, default=30,
                        help="Number of tasks to evaluate (default: 30)")
    parser.add_argument("--split", "-s", default=None,
                        help="Dataset split (default: dataset standard)")
    parser.add_argument("--allow-simulation", action="store_true",
                        help="on an unrecoverable API failure, substitute SIMULATED output "
                             "instead of discarding and retrying the task. Off by default: a "
                             "504 or an expired credential must not become a datapoint.")
    parser.add_argument("--task-retries", type=int, default=3,
                        help="how many times to re-attempt a task whose API calls failed "
                             "(default: 3). The partial record is discarded each time.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore local task cache and force live re-evaluation")
    parser.add_argument("--out", "-o", default="",
                        help="Optional output path for JSON results")
    parser.add_argument("--report", "-r", action="store_true", default=True,
                        help="Automatically generate Markdown TCO report and HTML dashboard (default: True)")
    args = parser.parse_args()

    if args.allow_simulation:
        os.environ["ALLOW_SIMULATION"] = "1"
        print("WARNING: --allow-simulation is on. Failed API calls will be replaced by "
              "simulated output, marked `simulated: true` in the results.", flush=True)

    sj_state = straitjacket_status()
    print(f"straitjacket: backend={sj_state['backend']} ctx={sj_state['ctx_version']} "
          f"workspace={sj_state['workspace']}"
          + ("" if sj_state["available"] else f"\n  UNAVAILABLE: {sj_state['reason']}\n"
             "  straitjacket arms will refuse to run. `pip install ctx-harness` to enable them."),
          flush=True)

    # Normalize dataset name and subfolder paths
    d_norm = args.dataset.lower().replace("-", "_")
    if d_norm in ("bcb", "bigcodebench", "bigcodebench_hard"):
        dataset_name = "BigCodeBench-Hard"
        ds_key = "bcb"
        ds_folder = "bigCodeBench-hard"
    elif d_norm in ("swebench", "swebench_pro", "swe_bench"):
        dataset_name = "SWE-bench Pro"
        ds_key = "swebench"
        ds_folder = "swebench_pro"
    elif d_norm in ("classeval", "class_eval", "ce"):
        dataset_name = "ClassEval"
        ds_key = "classeval"
        ds_folder = "classeval"
    else:
        dataset_name = "WebDev"
        ds_key = "webdev"
        ds_folder = "webdev"

    print("=" * 80)
    print(f"STARTING MULTI-LLM BENCHMARK RUNNER")
    print(f"Dataset: {dataset_name} | Tasks (N): {args.n} | Group: {args.group.upper()}")
    print("=" * 80, flush=True)

    # 1. Load Dataset
    problems = load_dataset(ds_key, split=args.split, max_tasks=args.n)
    task_ids = list(problems.keys())[:args.n]
    n = len(task_ids)
    print(f"Successfully loaded {n} tasks for {dataset_name}.", flush=True)

    # 2. Resolve Configurations
    v_keys = [k.strip() for k in args.variants.split(",") if k.strip()] if args.variants else None
    configs = get_configurations(dataset=ds_key, group=args.group, variant_keys=v_keys)
    print(f"Resolved {len(configs)} architecture variant(s) for execution.\n", flush=True)

    # 3. Setup Dataset-Specific Results & Cache Directories
    results_dir = os.path.join(HERE, ds_folder, "results")
    os.makedirs(results_dir, exist_ok=True)
    reports_dir = os.path.join(HERE, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    cache_file = os.path.join(results_dir, f"cache_{ds_key}_master.json")
    cache = {}
    if not args.no_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as cf:
                cache = json.load(cf)
        except Exception:
            cache = {}

    summary_rows = []

    # 4. Execute Benchmark Loop
    for c_idx, cfg in enumerate(configs, start=1):
        v_id = cfg["id"]
        v_name = cfg["name"]
        fn = cfg["fn"]

        print(f"[{c_idx}/{len(configs)}] RUNNING: {v_name}")
        t0 = time.time()
        results = []
        failed_tasks = []
        passed_cnt = 0
        tot_usd = 0.0
        tot_triage_usd = 0.0
        tot_out_tok = 0

        if v_id not in cache:
            cache[v_id] = {}

        for t_idx, tid in enumerate(task_ids, start=1):
            prob = problems[tid]
            if not args.no_cache and tid in cache[v_id]:
                r = cache[v_id][tid]
                status_str = "PASS" if r.get("passed") else "FAIL"
                print(f"  [{t_idx}/{n}] {tid} ... [CACHED] {status_str} | cost=${r.get('as_run_usd', 0.0):.5f} | out_tok={r.get('output_tokens', 0)}", flush=True)
            else:
                # A task whose API calls failed is not a result. Drop the partial
                # record and re-attempt it; only persist what completed.
                raw_r, dispatch_err = None, None
                for attempt in range(1, args.task_retries + 1):
                    try:
                        reset_simulated_calls()
                        raw_r = fn(prob)
                        break
                    except DispatchError as e:
                        dispatch_err = e
                        if attempt < args.task_retries:
                            wait = min(15 * attempt, 60)
                            print(f"  [{t_idx}/{n}] {tid} ... {e.kind.upper()} FAILURE "
                                  f"(attempt {attempt}/{args.task_retries}); discarding the "
                                  f"partial record, retrying in {wait}s", flush=True)
                            time.sleep(wait)
                        else:
                            print(f"  [{t_idx}/{n}] {tid} ... GAVE UP after "
                                  f"{args.task_retries} attempts: {e}", flush=True)

                if raw_r is None:
                    # Recorded as an incomplete task, never as a pass or a fail,
                    # and deliberately NOT written to the cache.
                    failed_tasks.append({"task_id": tid, "kind": dispatch_err.kind,
                                         "error": str(dispatch_err)})
                    continue

                r = {
                    "task_id": tid,
                    "passed": raw_r.get("passed", False),
                    "as_run_usd": raw_r.get("as_run_usd", 0.0),
                    "triage_usd": raw_r.get("triage_usd", 0.0),
                    "output_tokens": raw_r.get("output_tokens", 0),
                    "total_tokens": raw_r.get("total_tokens", 0),
                    "repair_loops": raw_r.get("repair_loops", 0),
                    # The containment ledger is a measurement, not a detail:
                    # dropping it here is what made the reports show pass rate
                    # and dollars but never context residency.
                    # Provenance, so no later audit has to infer whether this
                    # record came from a live call.
                    "simulated_calls": simulated_calls(),
                    "containment": raw_r.get("containment"),
                    "retrievals": raw_r.get("retrievals"),
                    "routing": raw_r.get("routing"),
                    # ClassEval scores a task per method. Dropping these would
                    # leave only the class-level verdict, and the whole reason
                    # that dataset is here is that a pass can be attributed to
                    # the model that wrote the method.
                    "subtasks": raw_r.get("subtasks"),
                    "subtask_summary": raw_r.get("subtask_summary"),
                    "error": str(raw_r.get("error", ""))[:500]
                }
                cache[v_id][tid] = r
                with open(cache_file, "w", encoding="utf-8") as cf:
                    json.dump(cache, cf, indent=2)
                status_str = "PASS" if r["passed"] else "FAIL"
                sub = r.get("subtask_summary") or {}
                sub_str = (f" | methods={sub.get('passed_subtasks', 0)}/{sub.get('n_subtasks', 0)}"
                           if sub.get("n_subtasks") else "")
                print(f"  [{t_idx}/{n}] {tid} ... {status_str} | cost=${r['as_run_usd']:.5f} | out_tok={r['output_tokens']}{sub_str}", flush=True)

            results.append(r)
            if r["passed"]:
                passed_cnt += 1
            tot_usd += r["as_run_usd"]
            tot_out_tok += r["output_tokens"]

        dt = time.time() - t0
        # Rates are over tasks that actually completed. Counting a dropped task
        # as a failure would silently understate the arm.
        n_done = len(results) or n
        pass_rate = (passed_cnt / n_done) if n_done > 0 else 0.0
        cost_per_solved = (tot_usd / passed_cnt) if passed_cnt > 0 else 0.0
        avg_out = tot_out_tok / n_done if n_done > 0 else 0.0

        # Triage USD: prefer what the arm actually spent; the per-repair
        # estimate is only a fallback for arms that do not report it.
        measured = sum(r.get("triage_usd", 0.0) for r in results)
        if measured > 0:
            triage_usd = round(measured, 6)
        elif "$0.00" in cfg.get("triage_mode", ""):
            triage_usd = 0.0000
        else:
            repairs = sum(r.get("repair_loops", 0) for r in results)
            triage_usd = round(repairs * 0.0018, 5)

        # Roll the per-method records up once per arm, so a report can show
        # pass rate BY TIER without re-reading every task record.
        sub_records = [sr for r in results for sr in (r.get("subtasks") or [])]
        subtask_rollup = _classeval_subtask_summary(sub_records) if sub_records else None

        simulated = sum(1 for t in results if t.get("simulated_calls"))
        if failed_tasks:
            print(f"  !! {len(failed_tasks)} task(s) never completed and are EXCLUDED "
                  f"from this row: {', '.join(t['task_id'] for t in failed_tasks[:5])}"
                  + (" ..." if len(failed_tasks) > 5 else ""), flush=True)

        summary = {
            "id": v_id,
            "name": v_name,
            "straitjacket": sj_state,
            # Provenance for the row: how many tasks completed, how many were
            # dropped, and whether any datapoint is simulated rather than live.
            "completed": len(results),
            "incomplete_tasks": failed_tasks,
            "simulated_tasks": simulated,
            "simulation_allowed": simulation_allowed(),
            "containment": _aggregate_containment(results),
            "subtask_rollup": subtask_rollup,
            "category": cfg.get("category", ""),
            "models": cfg.get("models", "N/A"),
            "triage_mode": cfg.get("triage_mode", "$0.00"),
            "n": n_done,
            "passed": passed_cnt,
            "pass_rate": round(pass_rate, 3),
            "total_as_run_usd": round(tot_usd, 6),
            "total_triage_usd": round(triage_usd, 6),
            "cost_per_solved_usd": round(cost_per_solved, 6),
            "avg_output_tokens": round(avg_out, 1),
            "seconds": round(dt, 1),
            "results": results
        }
        summary_rows.append(summary)
        print(f"  -> Pass Rate: {passed_cnt}/{n} ({pass_rate:.1%}) | Cost: ${tot_usd:.4f} | $/Solved: ${cost_per_solved:.4f}\n", flush=True)

    # 5. Save JSON Output
    out_file = args.out or os.path.join(results_dir, f"{ds_key}_{args.group}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": ds_key,
            "dataset_name": dataset_name,
            "group": args.group,
            "n": n,
            "straitjacket": sj_state,
            "summary": summary_rows
        }, f, indent=2)
    print(f"Saved consolidated benchmark metrics to: {out_file}")

    # 6. Generate Reports
    if args.report:
        # One index for the sweep, shared by the markdown and the dashboard.
        md_path, html_path = allocate_report_paths(dataset_name, n)
        md_file = generate_markdown_report(summary_rows, dataset_name=dataset_name,
                                           output_path=md_path)
        html_file = generate_html_dashboard(summary_rows, dataset_name=dataset_name,
                                            output_path=html_path)
        print(f"\nGenerated Comparative Reports:")
        print(f"  - Markdown: {md_file}")
        print(f"  - HTML:     {html_file}")

    # 7. Print Final Scoreboard
    print("\n" + "=" * 95)
    print(f"FINAL COMPARATIVE TCO SCOREBOARD: {dataset_name.upper()} (N={n})")
    print("=" * 95)
    print(f"{'Configuration':<44} | {'Pass Rate':<10} | {'Total Cost ($)':<14} | {'$/Solved':<10} | {'Triage USD'}")
    print("-" * 95)
    for s in summary_rows:
        pr_str = f"{s['passed']}/{s['n']} ({s['pass_rate']:.0%})"
        cps_str = f"${s['cost_per_solved_usd']:.4f}" if s['passed'] > 0 else "N/A"
        print(f"{s['name']:<44} | {pr_str:<10} | ${s['total_as_run_usd']:<13.4f} | {cps_str:<10} | ${s['total_triage_usd']:.4f}")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    main()
