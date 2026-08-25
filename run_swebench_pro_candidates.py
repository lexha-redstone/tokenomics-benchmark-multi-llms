#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Dedicated Runner for SWE-bench Pro Optimized Candidate Architectures.

Evaluates the 3 cost-effective (가성비) candidate architectures:
  1. sbp_grounded_contract     (Micro-Contract Localization Cascade)
  2. sbp_patch_health_router    (Patch-Health & Semantic Error-Class Router)
  3. sbp_sonnet_opus_sweetspot  (Cross-Provider Drafter + Escalator Sweet Spot)

Usage Examples:
  # 1. Run all 3 candidate architectures on ready local Docker tasks:
  python3 run_swebench_pro_candidates.py --tasks @/tmp/sbp_ready.txt --report

  # 2. Run on a sample of 10 tasks for Python repositories:
  SBP_LANGUAGES=python python3 run_swebench_pro_candidates.py --n 10 --report

  # 3. Run a specific candidate variant:
  python3 run_swebench_pro_candidates.py --variants sbp_grounded_contract --n 5 --report

  # 4. Or use the master benchmark runner:
  python3 run_benchmark.py --dataset swebench-pro --group sbp_candidates --tasks @/tmp/sbp_ready.txt --report
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SJ_SRC = os.environ.get("SJ_SRC", "")
if SJ_SRC and os.path.isdir(SJ_SRC) and SJ_SRC not in sys.path:
    sys.path.insert(0, SJ_SRC)

from src.architectures import get_configurations
from src.datasets import load_dataset
from src.evaluator import straitjacket_status
from src.reporter import (allocate_report_paths, generate_markdown_report,
                          generate_html_dashboard)
from src.sweep import load_cache, run_arm, print_scoreboard


def _requested_tasks(spec):
    spec = (spec or "").strip()
    if not spec:
        return []
    if spec.startswith("@"):
        with open(spec[1:], "r", encoding="utf-8") as f:
            items = [ln.strip() for ln in f]
    else:
        items = spec.split(",")
    seen, out = set(), []
    for t in (i.strip() for i in items):
        if t and not t.startswith("#") and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", "-v", default="",
                        help="Comma-separated candidate variants (default: all 3 candidates: "
                             "sbp_grounded_contract,sbp_patch_health_router,sbp_sonnet_opus_sweetspot)")
    parser.add_argument("--n", "-n", type=int, default=20,
                        help="Number of tasks to evaluate (default: 20)")
    parser.add_argument("--tasks", default="",
                        help="Specific task ids list or @path to file (e.g. @/tmp/sbp_ready.txt)")
    parser.add_argument("--languages", "-l", default="",
                        help="Filter by language, e.g. python, js, go (sets SBP_LANGUAGES)")
    parser.add_argument("--split", "-s", default=None,
                        help="SWE-bench Pro split")
    parser.add_argument("--cache-file", default="",
                        help="Path to dedicated cache file (default: swebench_pro/results/cache_swebench_pro_candidates.json)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-evaluation, bypass local task cache")
    parser.add_argument("--out", "-o", default="",
                        help="Output path for JSON results")
    parser.add_argument("--report", "-r", action="store_true", default=True,
                        help="Generate markdown report and HTML dashboard (default: True)")
    args = parser.parse_args()

    if args.languages:
        os.environ["SBP_LANGUAGES"] = args.languages

    sj_state = straitjacket_status()
    print("=" * 80)
    print("SWE-BENCH PRO OPTIMIZED CANDIDATE BENCHMARK RUNNER")
    print(f"Straitjacket: backend={sj_state['backend']} ctx={sj_state['ctx_version']}")
    print("=" * 80, flush=True)

    # 1. Load Dataset
    wanted = _requested_tasks(args.tasks)
    problems = load_dataset("swebench_pro", split=args.split,
                            max_tasks=None if wanted else args.n)
    if wanted:
        missing = [t for t in wanted if t not in problems]
        if missing:
            print(f"WARNING: {len(missing)} requested task id(s) not found in split (e.g. {missing[0]})",
                  file=sys.stderr)
        task_ids = [t for t in wanted if t in problems]
    else:
        task_ids = list(problems.keys())[:args.n]

    n = len(task_ids)
    print(f"Loaded {n} tasks for SWE-bench Pro.\n", flush=True)

    # 2. Resolve Configurations
    v_keys = [k.strip() for k in args.variants.split(",") if k.strip()] if args.variants else None
    if not v_keys:
        configs = get_configurations(dataset="swebench_pro", group="sbp_candidates")
    else:
        configs = get_configurations(dataset="swebench_pro", variant_keys=v_keys)

    print(f"Resolved {len(configs)} candidate variant(s) for execution:")
    for idx, c in enumerate(configs, 1):
        print(f"  [{idx}] {c['id']}: {c['name']}")
    print("", flush=True)

    # 3. Setup Results & Cache
    results_dir = os.path.join(HERE, "swebench_pro", "results")
    os.makedirs(results_dir, exist_ok=True)
    cache_file = args.cache_file or os.path.join(results_dir, "cache_swebench_pro_candidates.json")
    cache = load_cache(cache_file, no_cache=args.no_cache)

    summary_rows = []

    # 4. Execute Benchmark Loop
    for c_idx, cfg in enumerate(configs, start=1):
        summary_rows.append(run_arm(
            cfg, problems, task_ids,
            cache=cache, cache_file=cache_file, no_cache=args.no_cache,
            task_retries=1, sj_state=sj_state,
            label=f"[{c_idx}/{len(configs)}] "))

    # 5. Save Output
    out_file = args.out or os.path.join(results_dir, "swebench_pro_candidates_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "swebench_pro",
            "dataset_name": "SWE-bench Pro",
            "group": "sbp_candidates",
            "n": n,
            "straitjacket": sj_state,
            "summary": summary_rows
        }, f, indent=2)
    print(f"\nSaved consolidated metrics to: {out_file}")

    # 6. Generate Reports
    if args.report:
        md_path, html_path = allocate_report_paths("SWE-bench Pro Candidates", n)
        md_file = generate_markdown_report(summary_rows, dataset_name="SWE-bench Pro (Candidates)",
                                           output_path=md_path)
        html_file = generate_html_dashboard(summary_rows, dataset_name="SWE-bench Pro (Candidates)",
                                            output_path=html_path)
        print(f"\nGenerated Comparative Reports:")
        print(f"  - Markdown: {md_file}")
        print(f"  - HTML:     {html_file}")

    # 7. Print Final Scoreboard
    print_scoreboard(summary_rows, "SWE-bench Pro (Candidates)", n)


if __name__ == "__main__":
    sys.exit(main() or 0)
