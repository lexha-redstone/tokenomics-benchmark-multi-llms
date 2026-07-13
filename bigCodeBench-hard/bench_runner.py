#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0
"""
BigCodeBench Sweet Spot Benchmark Runner (API-Only Mode).
Runs multi-architecture, multi-LLM benchmark evaluations on BigCodeBench-Hard.
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.parse
import urllib.error

# Setup path for src module import
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import (
    GEMINI_FLASH_ID, GEMINI_FLASH_LITE_ID, GEMINI_PRO_ID, SONNET_ID,
    GCP_PROJECT, GCP_LOCATION
)
from src.architectures import run_single, run_read_write, run_cascade, run_hybrid

# --- Dataset Config ---
BCB_DATASET = "bigcode/bigcodebench-hard"
BCB_CONFIG = "default"
BCB_SPLIT = "v0.1.4"
_KEEP_FIELDS = ("task_id", "complete_prompt", "canonical_solution", "code_prompt",
                "test", "entry_point", "libs")

DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def _ssl_ctx():
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

def _dataset_path(split):
    return os.path.join(DATA_DIR, f"BigCodeBench-Hard-{split}.jsonl")

def ensure_dataset(split):
    path = _dataset_path(split)
    if os.path.exists(path):
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Fetching {BCB_DATASET} [{split}] via HF datasets-server -> {path}", flush=True)
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        q = urllib.parse.urlencode({"dataset": BCB_DATASET, "config": BCB_CONFIG,
                                    "split": split, "offset": offset, "length": 100})
        with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + q,
                                    timeout=120, context=_ssl_ctx()) as r:
            d = json.loads(r.read())
        batch = d.get("rows", [])
        total = d.get("num_rows_total", len(batch))
        if not batch:
            break
        rows.extend(b["row"] for b in batch)
        offset += len(batch)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps({k: row.get(k) for k in _KEEP_FIELDS}) + "\n")
    print(f"  saved {len(rows)} tasks", flush=True)
    return path

def load_problems(split):
    path = ensure_dataset(split)
    return {json.loads(l)["task_id"]: json.loads(l) for l in open(path)}

def run_benchmark(arch, tasks, problems, **kwargs):
    results = []
    print(f"\nRunning Benchmark: Arch={arch.upper()} on {len(tasks)} tasks...")
    for i, tid in enumerate(tasks, 1):
        problem = problems[tid]
        print(f"[{i}/{len(tasks)}] {tid} ({problem['entry_point']}) ... ", end="", flush=True)
        if arch == "single":
            r = run_single(problem, model_id=kwargs.get("model", GEMINI_FLASH_ID))
        elif arch == "read-write":
            r = run_read_write(problem, planner_model=kwargs.get("planner", GEMINI_FLASH_ID),
                               executor_model=kwargs.get("executor", GEMINI_FLASH_LITE_ID))
        elif arch == "cascade":
            r = run_cascade(problem, gen_model=kwargs.get("gen_model", GEMINI_FLASH_LITE_ID),
                            esc_model=kwargs.get("esc_model", GEMINI_FLASH_ID))
        elif arch == "hybrid":
            r = run_hybrid(problem, planner_model=kwargs.get("planner", GEMINI_FLASH_ID),
                           executor_model=kwargs.get("executor", GEMINI_FLASH_LITE_ID),
                           escalate_model=kwargs.get("escalate", GEMINI_FLASH_ID))
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{status} | cost=${r['as_run_usd']:.5f} | out_tok={r['output_tokens']}")

    n = len(results)
    passed_cnt = sum(1 for r in results if r["passed"])
    tot_cost = sum(r["as_run_usd"] for r in results)
    avg_out = sum(r["output_tokens"] for r in results) / n if n else 0

    print("-" * 60)
    print(f"SUMMARY ({arch}): Pass Rate = {passed_cnt}/{n} ({passed_cnt/n:.1%}) | "
          f"Total Cost = ${tot_cost:.4f} | Avg Output Tokens = {avg_out:.1f}")
    return {
        "arch": arch,
        "n": n,
        "passed": passed_cnt,
        "pass_rate": round(passed_cnt / n, 3) if n else 0,
        "total_as_run_usd": round(tot_cost, 4),
        "avg_output_tokens": round(avg_out, 1),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["single", "read-write", "cascade", "hybrid"], default="hybrid")
    parser.add_argument("--n", type=int, default=10, help="Number of tasks (default: 10)")
    parser.add_argument("--model", default=GEMINI_FLASH_ID, help="Model for single arch")
    parser.add_argument("--planner", default=GEMINI_FLASH_ID, help="Planner model")
    parser.add_argument("--executor", default=GEMINI_FLASH_LITE_ID, help="Executor model")
    parser.add_argument("--escalate", default=GEMINI_FLASH_ID, help="Escalation model")
    parser.add_argument("--split", default=BCB_SPLIT, help="Dataset split")
    parser.add_argument("--compare-all", action="store_true", help="Run comparison across key sweet-spot configs")
    args = parser.parse_args()

    problems = load_problems(args.split)
    task_ids = list(problems.keys())[:args.n]

    if args.compare_all:
        configs = [
            ("Single: gemini-3.1-flash-lite", "single", {"model": GEMINI_FLASH_LITE_ID}),
            ("Single: gemini-3.5-flash", "single", {"model": GEMINI_FLASH_ID}),
            ("Single: claude-sonnet-5", "single", {"model": SONNET_ID}),
            ("Read/Write: 3.5-Flash + 3.1-Lite", "read-write", {"planner": GEMINI_FLASH_ID, "executor": GEMINI_FLASH_LITE_ID}),
            ("Read/Write: Sonnet-5 + 3.1-Lite", "read-write", {"planner": SONNET_ID, "executor": GEMINI_FLASH_LITE_ID}),
            ("Cascade: 3.1-Lite -> 3.5-Flash", "cascade", {"gen_model": GEMINI_FLASH_LITE_ID, "esc_model": GEMINI_FLASH_ID}),
            ("Hybrid: 3.5-Flash + 3.1-Lite + 3.5-Flash", "hybrid", {"planner": GEMINI_FLASH_ID, "executor": GEMINI_FLASH_LITE_ID, "escalate": GEMINI_FLASH_ID}),
        ]
        summaries = []
        for name, arch, kwargs in configs:
            print(f"\n=======================================================")
            print(f"CONFIG: {name}")
            print(f"=======================================================")
            s = run_benchmark(arch, task_ids, problems, **kwargs)
            s["name"] = name
            summaries.append(s)

        print("\n" + "=" * 70)
        print("OVERALL SWEET-SPOT COMPARISON TABLE (First 10 BigCodeBench-Hard Tasks)")
        print("=" * 70)
        print(f"{'Configuration':<42} | {'Pass Rate':<10} | {'Total Cost ($)':<12} | {'Avg Out Tok':<10}")
        print("-" * 70)
        for s in summaries:
            print(f"{s['name']:<42} | {s['passed']}/{s['n']} ({s['pass_rate']:.0%})  | ${s['total_as_run_usd']:<11.4f} | {s['avg_output_tokens']:<10.1f}")
        print("=" * 70)
    else:
        run_benchmark(args.arch, task_ids, problems, model=args.model, planner=args.planner,
                      executor=args.executor, escalate=args.escalate)
