#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Web-Dev Benchmark Runner (Adapter for unified src library).
"""

import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import (
    GEMINI_36_FLASH_ID, GEMINI_35_FLASH_LITE_ID,
    GEMINI_FLASH_ID, GEMINI_FLASH_LITE_ID, SONNET_ID, OPUS_5_ID, OPUS_ID,
    WEBDEV_SOLVER_ROLE, WEBDEV_ADVISOR_ROLE
)
from src.datasets import load_webdev_problems
from src.evaluator import straitjacket_status
from src.architectures import (
    run_single, run_read_write, run_cascade, run_hybrid,
    run_hybrid_straitjacket, run_cascade_straitjacket,
    run_escalation_shield_straitjacket, run_smart_repair_straitjacket,
    run_ultra_sweet_straitjacket
)

RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="hybrid",
                        choices=["single", "read-write", "cascade", "hybrid", "hybrid-straitjacket", "cascade-straitjacket",
                                 "escalation_shield_sj", "smart_repair_sj", "ultra_sweet_sj"])
    parser.add_argument("--n", type=int, default=10, help="Number of tasks to evaluate (default: 10)")
    parser.add_argument("--compare-all", action="store_true", help="Compare key sweet-spot architectures")
    args = parser.parse_args()

    sj = straitjacket_status()
    print(f"straitjacket: backend={sj['backend']} ctx={sj['ctx_version']}"
          + ("" if sj["available"] else f" -- {sj['reason']}"))

    problems = load_webdev_problems(max_tasks=args.n)
    task_ids = list(problems.keys())[:args.n]

    if args.compare_all:
        configs = [
            ("1. Single: gemini-3.5-flash-lite", "single", {"model": GEMINI_35_FLASH_LITE_ID}),
            ("2. Single: gemini-3.6-flash", "single", {"model": GEMINI_36_FLASH_ID}),
            ("3. Single: claude-sonnet-5", "single", {"model": SONNET_ID}),
            ("4. Single: claude-opus-5", "single", {"model": OPUS_5_ID}),
            ("5. Adv-Exec: 3.6-Flash + 3.5-Lite", "read-write", {"planner": GEMINI_36_FLASH_ID, "executor": GEMINI_35_FLASH_LITE_ID}),
            ("6. Cascade: 3.5-Lite -> 3.6-Flash", "cascade", {"gen_model": GEMINI_35_FLASH_LITE_ID, "esc_model": GEMINI_36_FLASH_ID}),
            ("7. Straitjacket Hybrid (Flash + Lite + Flash)", "hybrid-straitjacket", {"planner": GEMINI_36_FLASH_ID, "executor": GEMINI_35_FLASH_LITE_ID, "escalate": GEMINI_36_FLASH_ID}),
            ("8. Straitjacket Escalation Shield", "escalation_shield_sj", {}),
            ("9. Straitjacket Smart Repair (Pure Gemini)", "smart_repair_sj", {}),
            ("10. Straitjacket Ultra-Sweet Hybrid", "ultra_sweet_sj", {}),
        ]
        summaries = []
        for name, arch, kwargs in configs:
            print(f"\n=======================================================")
            print(f"CONFIG: {name}")
            print(f"=======================================================")
            results = []
            for tid in task_ids:
                prob = problems[tid]
                if arch == "single":
                    r = run_single(prob, model_id=kwargs.get("model", GEMINI_36_FLASH_ID))
                elif arch == "read-write":
                    r = run_read_write(prob, planner_model=kwargs.get("planner", GEMINI_36_FLASH_ID), executor_model=kwargs.get("executor", GEMINI_35_FLASH_LITE_ID))
                elif arch == "cascade":
                    r = run_cascade(prob, gen_model=kwargs.get("gen_model", GEMINI_35_FLASH_LITE_ID), esc_model=kwargs.get("esc_model", GEMINI_36_FLASH_ID))
                elif arch == "hybrid-straitjacket":
                    r = run_hybrid_straitjacket(prob, planner_model=kwargs.get("planner", GEMINI_36_FLASH_ID), executor_model=kwargs.get("executor", GEMINI_35_FLASH_LITE_ID), escalate_model=kwargs.get("escalate", GEMINI_36_FLASH_ID))
                elif arch == "escalation_shield_sj":
                    r = run_escalation_shield_straitjacket(prob)
                elif arch == "smart_repair_sj":
                    r = run_smart_repair_straitjacket(prob)
                elif arch == "ultra_sweet_sj":
                    r = run_ultra_sweet_straitjacket(prob)
                results.append(r)
            
            n = len(results)
            passed_cnt = sum(1 for r in results if r["passed"])
            tot_cost = sum(r["as_run_usd"] for r in results)
            avg_out = sum(r["output_tokens"] for r in results) / n if n else 0
            items = [r.get("containment") or {} for r in results]
            raw = sum(i.get("raw_tokens_est", 0) for i in items)
            dig = sum(i.get("digest_tokens_est", 0) for i in items)
            summaries.append({
                "name": name, "n": n, "passed": passed_cnt,
                "pass_rate": passed_cnt / n if n else 0,
                "total_usd": tot_cost,
                "cost_per_solved": tot_cost / passed_cnt if passed_cnt else -1.0,
                "avg_out_tok": avg_out,
                "raw_tokens_est": raw,
                "digest_tokens_est": dig,
                "tokens_kept_out": max(0, raw - dig),
                "containment_ratio": round(raw / dig, 2) if dig else None,
            })

        print("\n" + "=" * 95)
        print("OVERALL WEB-DEV SWEET-SPOT COMPARISON TABLE")
        print("=" * 95)
        print(f"{'Configuration':<44} | {'Pass Rate':<10} | {'Total Cost ($)':<14} | {'$/Solved':<10} | {'Avg Out':<8} | {'Kept out':<9}")
        print("-" * 95)
        for s in summaries:
            cps_str = f"${s['cost_per_solved']:.4f}" if s['cost_per_solved'] >= 0 else "N/A"
            kept = f"{s['tokens_kept_out']:,}" if s.get('tokens_kept_out') else "-"
            print(f"{s['name']:<44} | {s['passed']}/{s['n']} ({s['pass_rate']:.0%})  | ${s['total_usd']:<13.5f} | {cps_str:<10} | {s['avg_out_tok']:<8.0f} | {kept:<9}")
        print("=" * 95)
    else:
        print(f"Run custom arch: {args.arch} on {len(task_ids)} tasks")

if __name__ == "__main__":
    main()
