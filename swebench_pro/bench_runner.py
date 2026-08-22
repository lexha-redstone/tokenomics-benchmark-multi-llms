#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
SWE-bench Pro Runner and Configuration Registry.
Provides programmatic access to SWE-bench Pro evaluations and delegates to the unified benchmark engine.
"""

import os
import sys
import json
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import (
    GEMINI_36_FLASH_ID, GEMINI_35_FLASH_LITE_ID,
    OPUS_5_ID, OPUS_48_ID, SONNET_ID
)
from src.datasets import load_swebench_pro_problems as load_problems
from src.evaluator import straitjacket_status, aggregate_containment as _aggregate_containment
from src.architectures import (
    run_single, run_read_write, run_cascade, run_hybrid,
    run_hybrid_straitjacket, run_cascade_straitjacket,
    run_escalation_shield_straitjacket, run_smart_repair_straitjacket,
    run_ultra_sweet_straitjacket, run_dual_verifier_cascade_straitjacket
)

RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CACHE_FILE = os.path.join(RESULTS_DIR, "cache_swebench_pro.json")

def get_swebench_pro_configurations():
    """Returns the standard SWE-bench Pro benchmark configuration matrix."""
    return [
        {
            "id": "single_opus5",
            "name": "1. Single: claude-opus-5",
            "category": "1. Single models",
            "fn": lambda p: run_single(p, OPUS_5_ID, None),
            "models": "Claude Opus-5",
            "triage_mode": "Straitjacket contained digest ($0.00)",
        },
        {
            "id": "single_sonnet5",
            "name": "2. Single: claude-sonnet-5",
            "category": "1. Single models",
            "fn": lambda p: run_single(p, SONNET_ID, None),
            "models": "Claude Sonnet-5",
            "triage_mode": "Straitjacket contained digest ($0.00)",
        },
        {
            "id": "single_gemini36_low",
            "name": "3. Single: gemini-3.6-flash (LOW)",
            "category": "1. Single models",
            "fn": lambda p: run_single(p, GEMINI_36_FLASH_ID, "low"),
            "models": "Gemini 3.6 Flash (LOW)",
            "triage_mode": "Straitjacket contained digest ($0.00)",
        },
        {
            "id": "single_gemini35_lite",
            "name": "4. Single: gemini-3.5-flash-lite",
            "category": "1. Single models",
            "fn": lambda p: run_single(p, GEMINI_35_FLASH_LITE_ID, None),
            "models": "Gemini 3.5 Lite",
            "triage_mode": "Straitjacket contained digest ($0.00)",
        },
        {
            "id": "combo_readwrite",
            "name": "5. Read/Write Split (3.6-Flash Planner + 3.5-Lite Executor)",
            "category": "2. Combination of models",
            "fn": lambda p: run_read_write(p, GEMINI_36_FLASH_ID, GEMINI_35_FLASH_LITE_ID),
            "models": "Gemini 3.6 Flash Plan + 3.5 Lite Exec",
            "triage_mode": "None / Direct",
        },
        {
            "id": "sj_escalation_shield",
            "name": "6. Straitjacket Escalation Shield (Gemini Lite -> Flash -> Sonnet-5)",
            "category": "3. Combination of models + straitjacket",
            "fn": lambda p: run_escalation_shield_straitjacket(p, GEMINI_35_FLASH_LITE_ID, GEMINI_36_FLASH_ID, SONNET_ID),
            "models": "Gemini Lite -> Flash -> Claude Sonnet-5",
            "triage_mode": "Straitjacket contained digest ($0.00)",
        },
        {
            "id": "sj_smart_repair",
            "name": "7. Straitjacket Smart Repair (Pure Gemini 3-Tier)",
            "category": "3. Combination of models + straitjacket",
            "fn": lambda p: run_smart_repair_straitjacket(p, GEMINI_36_FLASH_ID, GEMINI_35_FLASH_LITE_ID),
            "models": "Gemini 3.6 Flash -> 3.5 Lite -> Flash (Med)",
            "triage_mode": "Straitjacket contained digest ($0.00)",
        },
        {
            "id": "sj_ultra_sweet",
            "name": "8. Straitjacket Ultra-Sweet Hybrid (Claude Sonnet-5 -> Gemini Lite -> Opus-5)",
            "category": "3. Combination of models + straitjacket",
            "fn": lambda p: run_ultra_sweet_straitjacket(p, SONNET_ID, GEMINI_35_FLASH_LITE_ID, OPUS_5_ID),
            "models": "Claude Sonnet-5 -> Gemini Lite -> Claude Opus-5",
            "triage_mode": "Straitjacket contained digest ($0.00)",
        },
        {
            "id": "sj_dual_verifier",
            "name": "9. Straitjacket Dual-Verifier Cascade (4-Tier Synergy)",
            "category": "4. Next-Gen Multi-Provider + straitjacket",
            "fn": lambda p: run_dual_verifier_cascade_straitjacket(p, GEMINI_35_FLASH_LITE_ID, GEMINI_36_FLASH_ID, SONNET_ID, OPUS_5_ID),
            "models": "Gemini Lite -> Flash -> Sonnet-5 -> Opus-5",
            "triage_mode": "Straitjacket contained digest ($0.00)",
        },
    ]

def run_swebench_pro_suite(task_ids, problems, configs, no_cache=False):
    """Execute SWE-bench Pro benchmark suite across given task IDs and configurations."""
    cache = {}
    if not no_cache and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    summary_rows = []
    n = len(task_ids)

    for cfg in configs:
        cid = cfg["id"]
        cname = cfg["name"]
        fn = cfg["fn"]
        print(f"\n--- Running: {cname} ---", flush=True)

        if cid not in cache:
            cache[cid] = {}

        results = []
        passed_cnt = 0
        tot_usd = 0.0
        tot_out = 0
        t0 = time.time()

        for idx, tid in enumerate(task_ids, 1):
            prob = problems[tid]
            if not no_cache and tid in cache[cid]:
                res = cache[cid][tid]
            else:
                res = fn(prob)
                res["task_id"] = tid
                cache[cid][tid] = res
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)

            results.append(res)
            if res.get("passed"):
                passed_cnt += 1
            tot_usd += res.get("as_run_usd", 0.0)
            tot_out += res.get("output_tokens", 0)

            status = "PASS" if res.get("passed") else "FAIL"
            print(f"  [{idx}/{n}] {tid} ... {status} | cost=${res.get('as_run_usd', 0.0):.5f} | out_tok={res.get('output_tokens', 0)}", flush=True)

        dt = time.time() - t0
        pass_rate = passed_cnt / n if n > 0 else 0.0
        cost_per_solved = tot_usd / passed_cnt if passed_cnt > 0 else 0.0
        avg_out = tot_out / n if n > 0 else 0.0

        triage_usd = 0.0 if ("straitjacket" in cname.lower() or "$0.00" in cfg.get("triage_mode", "")) else round(passed_cnt * 0.0018, 5)

        containment = _aggregate_containment(results)

        summary = {
            "id": cid,
            "name": cname,
            "straitjacket": straitjacket_status(),
            "containment": containment,
            "category": cfg.get("category", ""),
            "models": cfg.get("models", "N/A"),
            "triage_mode": cfg.get("triage_mode", "$0.00"),
            "n": n,
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

        # Save individual config result JSON
        out_f = os.path.join(RESULTS_DIR, f"{cid}_results.json")
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary_rows
