#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Master Unified Benchmark Runner for Multi-LLM Tokenomics & Straitjacket Evaluation.

Runs end-to-end evaluation for any target dataset (BigCodeBench-Hard, WebDev, ClassEval)
and specified variant configurations/groups, with automatic Markdown report and HTML dashboard generation.

Usage Examples:
  # Run BigCodeBench-Hard on 100 tasks with all Straitjacket variants + generate reports:
  python3 run_benchmark.py --dataset bcb --group straitjacket --n 100 --report

  # Run BigCodeBench-Hard on 10 tasks with specific variants:
  python3 run_benchmark.py --dataset bcb --variants single_flash37,sj_hybrid,sj_smart_repair --n 10 --report

  # Run the ClassEval sub-task routing comparison:
  python3 run_benchmark.py --dataset classeval --group classeval --n 91 --report

  # Run the FeatureBench expensive-oracle study (needs Docker; see docs/featurebench-setup.md):
  python3 run_benchmark.py --dataset featurebench --group featurebench --n 20 --report

  # Run WebDev single-model baseline comparisons:
  python3 run_benchmark.py --dataset webdev --group single --n 10 --report
"""

import os
import sys
import json
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
from src.evaluator import straitjacket_status
from src.architectures import get_configurations, VARIANT_REGISTRY
from src.reporter import (allocate_report_paths, generate_markdown_report,
                          generate_html_dashboard)
# The per-arm loop lives in src/sweep.py so single-arm runners
# (run_classeval_opus5.py) score their rows by exactly these rules.
from src.sweep import load_cache, run_arm, print_scoreboard

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", "-d", choices=["bcb", "bigcodebench", "bigcodebench-hard",
                                                    "webdev", "web-dev",
                                                    "classeval", "class-eval", "ce",
                                                    "featurebench", "feature-bench", "fb"],
                        default="bcb", help="Dataset to evaluate (default: 'bcb')")
    parser.add_argument("--group", "-g", choices=["all", "single", "combo", "straitjacket", "sj",
                                                  "nextgen", "ablation", "router",
                                                  "classeval", "ce",
                                                  "featurebench", "fb"],
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
    elif d_norm in ("classeval", "class_eval", "ce"):
        dataset_name = "ClassEval"
        ds_key = "classeval"
        ds_folder = "classeval"
    elif d_norm in ("featurebench", "feature_bench", "fb"):
        dataset_name = "FeatureBench"
        ds_key = "featurebench"
        ds_folder = "featurebench"
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
    cache = load_cache(cache_file, no_cache=args.no_cache)

    summary_rows = []

    # 4. Execute Benchmark Loop
    for c_idx, cfg in enumerate(configs, start=1):
        summary_rows.append(run_arm(
            cfg, problems, task_ids,
            cache=cache, cache_file=cache_file, no_cache=args.no_cache,
            task_retries=args.task_retries, sj_state=sj_state,
            label=f"[{c_idx}/{len(configs)}] "))

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
    print_scoreboard(summary_rows, dataset_name, n)

if __name__ == "__main__":
    main()
