#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
SWE-bench Pro (Public dataset) Master Sweet-Spot Evaluation Script.
Evaluates Single Models, Model Combinations, and Straitjacket Zero-Cost Local Triage variants.
"""

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from swebench_pro.dataset import load_problems
from swebench_pro.bench_runner import get_swebench_pro_configurations, run_swebench_pro_suite
from swebench_pro.report_generator import generate_markdown_report, generate_html_dashboard

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30, help="Number of SWE-bench Pro tasks to evaluate (default: 30)")
    parser.add_argument("--split", default="test", help="Dataset split (default: 'test')")
    parser.add_argument("--group", choices=["all", "single", "combo", "straitjacket", "nextgen"], default="all",
                        help="Filter to specific variant group (default: 'all')")
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing cache and rerun from scratch")
    args = parser.parse_args()

    print("=========================================================================================")
    print("STARTING SWE-BENCH PRO MULTI-LLM & STRAITJACKET EVALUATION")
    print(f"Tasks (N): {args.n} | Split: {args.split} | Group: {args.group.upper()} | No-Cache: {args.no_cache}")
    print("=========================================================================================", flush=True)

    problems = load_problems(split=args.split, max_tasks=args.n)
    task_ids = list(problems.keys())[:args.n]

    all_configs = get_swebench_pro_configurations()
    if args.group == "single":
        configs = [c for c in all_configs if "1. Single" in c["category"]]
    elif args.group == "combo":
        configs = [c for c in all_configs if "2. Combination" in c["category"]]
    elif args.group in ("straitjacket", "sj"):
        configs = [c for c in all_configs if "straitjacket" in c["category"].lower()]
    elif args.group == "nextgen":
        configs = [c for c in all_configs if "4. Next-Gen" in c["category"]]
    else:
        configs = all_configs

    results = run_swebench_pro_suite(task_ids, problems, configs, no_cache=args.no_cache)

    # Generate comparative reports
    md_path = generate_markdown_report(results, dataset_name="SWE-bench Pro")
    html_path = generate_html_dashboard(results, dataset_name="SWE-bench Pro")

    print("\n=========================================================================================")
    print("SWE-BENCH PRO COMPARATIVE EVALUATION COMPLETE")
    print(f"Reports saved:\n  - Markdown: {md_path}\n  - HTML:     {html_path}")
    print("=========================================================================================\n")

if __name__ == "__main__":
    main()
