# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Comprehensive Multi-LLM Benchmark Runner for Straitjacket Evaluation.
Orchestrates parallel and sequential benchmark sweeps across datasets with persistent caching.
"""

import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    GEMINI_37_FLASH_ID, GEMINI_35_FLASH_LITE_ID, SONNET_ID, OPUS_5_ID
)
from .datasets import get_dataset
from .architectures import (
    run_single_gemini_37,
    run_single_claude_sonnet5,
    run_smart_tiered_cascade,
    run_straitjacket_smart_repair,
    run_straitjacket_escalation_shield,
    run_straitjacket_dag_wave,
    run_straitjacket_dual_consensus
)

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_HERE, "results")

# Complete Arm Registry
BENCHMARK_ARMS = [
    {
        "id": "single_gemini37",
        "name": "Control Baseline: Gemini 3.7-Flash Single",
        "category": "Baseline",
        "fn": run_single_gemini_37
    },
    {
        "id": "single_claudesonnet5",
        "name": "Control Baseline: Claude Sonnet-5 Single",
        "category": "Baseline",
        "fn": run_single_claude_sonnet5
    },
    {
        "id": "smart_tiered_cascade",
        "name": "Smart Tiered Cascade (2-Tiered Cascade: 3.5-Lite -> 3.7-Flash)",
        "category": "Core Architecture",
        "fn": run_smart_tiered_cascade
    },
    {
        "id": "straitjacket_smart_repair",
        "name": "Straitjacket Smart Repair (Advisor & Executor: 3.7-Flash -> 3.5-Lite)",
        "category": "Core Architecture",
        "fn": run_straitjacket_smart_repair
    },
    {
        "id": "straitjacket_escalation_shield",
        "name": "Straitjacket Escalation Shield (3-Tiered Cascade: Lite -> Lite -> 3.7-Flash)",
        "category": "Core Architecture",
        "fn": run_straitjacket_escalation_shield
    },
    {
        "id": "straitjacket_dag_wave",
        "name": "Straitjacket DAG Wave Orchestrator (ctx.route/v1 + CAS Checkpoint)",
        "category": "Advanced Architecture",
        "fn": run_straitjacket_dag_wave
    },
    {
        "id": "straitjacket_dual_consensus",
        "name": "Straitjacket Dual-Candidate Consensus Repair (Parallel 3.5-Lite + Diff)",
        "category": "Advanced Architecture",
        "fn": run_straitjacket_dual_consensus
    },
]

def run_arm_on_dataset(arm_def, dataset_name, n=50, max_workers=1, cache_dir=None):
    """
    Execute a single benchmark arm across N tasks in a dataset.
    """
    out_cache_dir = cache_dir or RESULTS_DIR
    os.makedirs(out_cache_dir, exist_ok=True)
    cache_file = os.path.join(out_cache_dir, f"cache_{dataset_name}.json")

    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    arm_id = arm_def["id"]
    if arm_id not in cache:
        cache[arm_id] = {}

    problems = get_dataset(dataset_name, n=n)
    task_ids = list(problems.keys())[:n]
    actual_n = len(task_ids)

    print(f"\n[{dataset_name.upper()}] Starting Arm: {arm_def['name']} (N={actual_n})", flush=True)
    t0 = time.time()
    results = []
    passed_count = 0
    tot_usd = 0.0
    tot_out_tok = 0
    tot_tokens_saved = 0

    def evaluate_task(tid):
        prob = problems[tid]
        if tid in cache[arm_id]:
            return tid, cache[arm_id][tid], True
        try:
            res = arm_def["fn"](prob)
            res["task_id"] = tid
        except Exception as e:
            res = {
                "task_id": tid,
                "passed": False,
                "as_run_usd": 0.0005,
                "output_tokens": 0,
                "total_tokens": 0,
                "seconds": 0.0,
                "repair_loops": 0,
                "tokens_saved": 0,
                "error": f"Evaluation Error: {e}"
            }
        return tid, res, False

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(evaluate_task, tid): tid for tid in task_ids}
            for fut in as_completed(futures):
                tid, res, was_cached = fut.result()
                if not was_cached:
                    cache[arm_id][tid] = res
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(cache, f, indent=2)
                results.append(res)
                if res["passed"]:
                    passed_count += 1
                tot_usd += res["as_run_usd"]
                tot_out_tok += res["output_tokens"]
                tot_tokens_saved += res.get("tokens_saved", 0)
                status_str = "PASS" if res["passed"] else "FAIL"
                cached_tag = " [CACHED]" if was_cached else ""
                print(f"  [{len(results)}/{actual_n}] {tid} -> {status_str}{cached_tag} | cost=${res['as_run_usd']:.5f} | tokens={res['output_tokens']}", flush=True)
    else:
        for idx, tid in enumerate(task_ids, start=1):
            tid, res, was_cached = evaluate_task(tid)
            if not was_cached:
                cache[arm_id][tid] = res
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
            results.append(res)
            if res["passed"]:
                passed_count += 1
            tot_usd += res["as_run_usd"]
            tot_out_tok += res["output_tokens"]
            tot_tokens_saved += res.get("tokens_saved", 0)
            status_str = "PASS" if res["passed"] else "FAIL"
            cached_tag = " [CACHED]" if was_cached else ""
            print(f"  [{idx}/{actual_n}] {tid} -> {status_str}{cached_tag} | cost=${res['as_run_usd']:.5f} | tokens={res['output_tokens']}", flush=True)

    dt = time.time() - t0
    pass_rate = (passed_count / actual_n) * 100.0 if actual_n > 0 else 0.0
    cost_per_solved = (tot_usd / passed_count) if passed_count > 0 else 0.0
    avg_output_tokens = (tot_out_tok / actual_n) if actual_n > 0 else 0.0

    arm_summary = {
        "arm_id": arm_id,
        "arm_name": arm_def["name"],
        "category": arm_def["category"],
        "dataset": dataset_name,
        "n": actual_n,
        "passed": passed_count,
        "pass_rate": round(pass_rate, 2),
        "total_as_run_usd": round(tot_usd, 6),
        "cost_per_solved_usd": round(cost_per_solved, 6),
        "avg_output_tokens": round(avg_output_tokens, 1),
        "total_tokens_saved": tot_tokens_saved,
        "duration_seconds": round(dt, 2),
        "results": results
    }

    # Save individual arm summary
    out_file = os.path.join(out_cache_dir, f"{dataset_name}_{arm_id}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(arm_summary, f, indent=2)

    print(f"  => SUMMARY: Pass Rate={pass_rate:.1f}% ({passed_count}/{actual_n}) | Total Cost=${tot_usd:.4f} | $/Solved=${cost_per_solved:.4f}", flush=True)
    return arm_summary

def run_full_benchmark(datasets=("bcb", "webdev"), n=50, max_workers=1, selected_arms=None):
    """
    Run complete benchmark across all requested datasets and architectures.
    """
    all_summaries = {}
    arms_to_run = BENCHMARK_ARMS
    if selected_arms:
        arms_to_run = [a for a in BENCHMARK_ARMS if a["id"] in selected_arms or a["id"] == selected_arms]

    for ds in datasets:
        all_summaries[ds] = []
        for arm in arms_to_run:
            summary = run_arm_on_dataset(arm, ds, n=n, max_workers=max_workers)
            all_summaries[ds].append(summary)

    # Save aggregate benchmark summary
    agg_file = os.path.join(RESULTS_DIR, "aggregate_benchmark_summary.json")
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nAll benchmark runs completed! Aggregate results written to {agg_file}", flush=True)
    return all_summaries
