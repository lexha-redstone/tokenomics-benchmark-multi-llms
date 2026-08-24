#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Comprehensive Multi-LLM Benchmark Evaluation Harness with Real Straitjacket Context Containment.
Evaluates Google Gemini (3.7-Flash & 3.5-Flash-Lite) Multi-LLM Architectures against
Claude Sonnet-5 and Single Gemini Baselines on BigCodeBench-Hard (N=50) and WebDev (N=50).

Usage:
  python3 run_straitjacket_benchmark.py --dataset all --n 50 --workers 4
"""

import os
import sys
import argparse
import time

# Ensure project root is in sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from straitjacket_eval.runner import run_full_benchmark, BENCHMARK_ARMS
from straitjacket_eval.reporter import generate_report

def main():
    parser = argparse.ArgumentParser(
        description="Run Straitjacket Multi-LLM Benchmark on BigCodeBench-Hard and WebDev suites."
    )
    parser.add_argument("--dataset", choices=["all", "bcb", "webdev"], default="all",
                        help="Dataset suite to evaluate (default: all)")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of tasks per dataset (default: 50)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent evaluation threads (default: 4)")
    parser.add_argument("--arm", type=str, default="all",
                        help="Specific architecture arm to run (default: all)")
    parser.add_argument("--report-only", action="store_true",
                        help="Generate report from existing results without running benchmarks")
    args = parser.parse_args()

    if args.report_only:
        print("Generating report from existing benchmark results...")
        rep = generate_report()
        print("\n--- Summary Report Preview ---")
        print(rep[:2000])
        return

    datasets = ["bcb", "webdev"] if args.dataset == "all" else [args.dataset]
    selected_arms = None if args.arm == "all" else [args.arm]

    print("=" * 80)
    print("STRAITJACKET MULTI-LLM TOKENOMICS BENCHMARK EVALUATION")
    print(f"Datasets: {datasets} | Sample Size: N={args.n} per suite | Workers: {args.workers}")
    print(f"Models: Google Gemini 3.7-Flash, Gemini 3.5-Flash-Lite, Claude Sonnet-5")
    print(f"Harness: Real Straitjacket Context Containment & $0.00 Deterministic Local Triage")
    print("=" * 80)

    t0 = time.time()
    summaries = run_full_benchmark(
        datasets=datasets,
        n=args.n,
        max_workers=args.workers,
        selected_arms=selected_arms
    )
    total_time = time.time() - t0

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE (Total time: {total_time:.1f}s)")
    print("Generating comprehensive tokenomics analytical report...")
    print("=" * 80)

    report_content = generate_report(summaries)
    print("\n" + report_content)

if __name__ == "__main__":
    main()
