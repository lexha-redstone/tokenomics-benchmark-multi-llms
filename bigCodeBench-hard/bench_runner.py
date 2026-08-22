#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
BigCodeBench-Hard Benchmark Runner (Adapter for unified src library).
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
    GEMINI_37_FLASH_ID, GEMINI_35_FLASH_LITE_ID,
    GEMINI_FLASH_ID, GEMINI_FLASH_LITE_ID, SONNET_ID, OPUS_5_ID
)
from src.datasets import load_bcb_problems, BCB_DEFAULT_SPLIT
from src.architectures import (
    run_single, run_read_write, run_cascade, run_hybrid,
    run_hybrid_straitjacket, run_cascade_straitjacket,
    run_escalation_shield_straitjacket, run_smart_repair_straitjacket,
    run_ultra_sweet_straitjacket, run_contained_retrieval_cascade
)
from src.evaluator import straitjacket_status, aggregate_containment
from src.reporter import generate_markdown_report, generate_html_dashboard

RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_benchmark(arch, tasks, problems, **kwargs):
    results = []
    sj = straitjacket_status()
    print(f"\nRunning BigCodeBench-Hard Benchmark: Arch={arch.upper()} on {len(tasks)} tasks...")
    print(f"straitjacket: backend={sj['backend']} ctx={sj['ctx_version']} "
          f"raw_cap={sj['raw_cap_chars']}ch"
          + ("" if sj["available"] else f" -- {sj['reason']}"))
    for i, tid in enumerate(tasks, 1):
        problem = problems[tid]
        print(f"[{i}/{len(tasks)}] {tid} ({problem.get('entry_point', '')}) ... ", end="", flush=True)
        if arch == "single":
            r = run_single(problem, model_id=kwargs.get("model", GEMINI_37_FLASH_ID))
        elif arch == "read-write":
            r = run_read_write(problem, planner_model=kwargs.get("planner", GEMINI_37_FLASH_ID),
                               executor_model=kwargs.get("executor", GEMINI_35_FLASH_LITE_ID))
        elif arch == "cascade":
            r = run_cascade(problem, gen_model=kwargs.get("gen_model", GEMINI_35_FLASH_LITE_ID),
                            esc_model=kwargs.get("esc_model", GEMINI_37_FLASH_ID))
        elif arch == "hybrid":
            r = run_hybrid(problem, planner_model=kwargs.get("planner", GEMINI_37_FLASH_ID),
                           executor_model=kwargs.get("executor", GEMINI_35_FLASH_LITE_ID),
                           escalate_model=kwargs.get("escalate", GEMINI_37_FLASH_ID))
        elif arch in ("hybrid-straitjacket", "hybrid_sj"):
            r = run_hybrid_straitjacket(problem, planner_model=kwargs.get("planner", GEMINI_37_FLASH_ID),
                                        executor_model=kwargs.get("executor", GEMINI_35_FLASH_LITE_ID),
                                        escalate_model=kwargs.get("escalate", GEMINI_37_FLASH_ID))
        elif arch in ("cascade-straitjacket", "cascade_sj"):
            r = run_cascade_straitjacket(problem, gen_model=kwargs.get("gen_model", GEMINI_35_FLASH_LITE_ID),
                                         esc_model=kwargs.get("esc_model", GEMINI_37_FLASH_ID))
        elif arch in ("escalation_shield_sj", "shield_sj"):
            r = run_escalation_shield_straitjacket(problem)
        elif arch in ("smart_repair_sj", "smart_repair"):
            r = run_smart_repair_straitjacket(problem)
        elif arch in ("ultra_sweet_sj", "ultra_sweet"):
            r = run_ultra_sweet_straitjacket(problem)
        elif arch in ("contained_retrieval_sj", "retrieval_sj"):
            r = run_contained_retrieval_cascade(
                problem, gen_model=kwargs.get("gen_model", GEMINI_35_FLASH_LITE_ID),
                esc_model=kwargs.get("esc_model", GEMINI_37_FLASH_ID))
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{status} | cost=${r['as_run_usd']:.5f} | out_tok={r['output_tokens']}")

    n = len(results)
    passed_cnt = sum(1 for r in results if r["passed"])
    tot_cost = sum(r["as_run_usd"] for r in results)
    avg_out = sum(r["output_tokens"] for r in results) / n if n else 0

    containment = aggregate_containment(results)

    print("-" * 60)
    print(f"SUMMARY ({arch}): Pass Rate = {passed_cnt}/{n} ({passed_cnt/n:.1%}) | "
          f"Total Cost = ${tot_cost:.4f} | Avg Output Tokens = {avg_out:.1f}")
    if containment["captures"]:
        print(f"  containment: {containment['captures']} captures · "
              f"{containment['raw_tokens_est']:,} raw tokens -> "
              f"{containment['digest_tokens_est']:,} digest tokens "
              f"({containment['containment_ratio']}x, "
              f"{containment['tokens_kept_out']:,} kept out of context) · "
              f"profiles={','.join(containment['profiles']) or 'n/a'}")
    return {
        "arch": arch,
        "n": n,
        "straitjacket": sj,
        "containment": containment,
        "passed": passed_cnt,
        "pass_rate": round(passed_cnt / n, 3) if n else 0,
        "total_as_run_usd": round(tot_cost, 4),
        "avg_output_tokens": round(avg_out, 1),
        "results": results,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="hybrid-straitjacket",
                        choices=["single", "read-write", "cascade", "hybrid", "hybrid-straitjacket", "cascade-straitjacket",
                                 "escalation_shield_sj", "smart_repair_sj", "ultra_sweet_sj",
                                 "contained_retrieval_sj"])
    parser.add_argument("--n", type=int, default=10, help="Number of tasks (default: 10)")
    parser.add_argument("--model", default=GEMINI_37_FLASH_ID, help="Model for single arch")
    parser.add_argument("--planner", default=GEMINI_37_FLASH_ID, help="Planner model")
    parser.add_argument("--executor", default=GEMINI_35_FLASH_LITE_ID, help="Executor model")
    parser.add_argument("--escalate", default=GEMINI_37_FLASH_ID, help="Escalation model")
    parser.add_argument("--split", default=BCB_DEFAULT_SPLIT, help="Dataset split")
    parser.add_argument("--out", default="", help="Optional output JSON file path")
    parser.add_argument("--compare-all", action="store_true", help="Run comparison across key sweet-spot configs")
    args = parser.parse_args()

    problems = load_bcb_problems(split=args.split, max_tasks=args.n)
    task_ids = list(problems.keys())[:args.n]

    if args.compare_all:
        configs = [
            ("Single: gemini-3.5-flash-lite", "single", {"model": GEMINI_35_FLASH_LITE_ID}),
            ("Single: gemini-3.7-flash", "single", {"model": GEMINI_37_FLASH_ID}),
            ("Single: claude-sonnet-5", "single", {"model": SONNET_ID}),
            ("Read/Write: 3.7-Flash + 3.5-Lite", "read-write", {"planner": GEMINI_37_FLASH_ID, "executor": GEMINI_35_FLASH_LITE_ID}),
            ("Cascade: 3.5-Lite -> 3.7-Flash", "cascade", {"gen_model": GEMINI_35_FLASH_LITE_ID, "esc_model": GEMINI_37_FLASH_ID}),
            ("Straitjacket Hybrid: Flash + Lite + Flash (SJ)", "hybrid-straitjacket", {"planner": GEMINI_37_FLASH_ID, "executor": GEMINI_35_FLASH_LITE_ID, "escalate": GEMINI_37_FLASH_ID}),
            ("Straitjacket Smart Repair (Pure Gemini)", "smart_repair_sj", {}),
            ("Straitjacket Ultra-Sweet Hybrid", "ultra_sweet_sj", {}),
        ]
        summaries = []
        for name, arch, kwargs in configs:
            print(f"\n=======================================================")
            print(f"CONFIG: {name}")
            print(f"=======================================================")
            s = run_benchmark(arch, task_ids, problems, **kwargs)
            s["name"] = name
            summaries.append(s)

        print("\n" + "=" * 75)
        print(f"OVERALL SWEET-SPOT COMPARISON TABLE (N={args.n} BigCodeBench-Hard Tasks)")
        print("=" * 75)
        print(f"{'Configuration':<44} | {'Pass Rate':<10} | {'Total Cost ($)':<12} | {'Avg Out Tok':<10}")
        print("-" * 75)
        for s in summaries:
            print(f"{s['name']:<44} | {s['passed']}/{s['n']} ({s['pass_rate']:.0%})  | ${s['total_as_run_usd']:<11.4f} | {s['avg_output_tokens']:<10.1f}")
        print("=" * 75)
    else:
        res = run_benchmark(args.arch, task_ids, problems, model=args.model, planner=args.planner,
                            executor=args.executor, escalate=args.escalate)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Saved results to {args.out}")
