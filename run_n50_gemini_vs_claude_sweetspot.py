#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
N=50 Evaluation of 5+ Gemini 3.6-Flash / 3.5-Flash-Lite Architectures vs. Claude-Only Baselines.
Finds maximum performance and minimum cost-per-solved-task ($/solved) across 50 BigCodeBench-Hard tasks.
"""

import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SJ_SRC = "/usr/local/google/home/lexha/Desktop/work/prj/99-assets/straitjacket/src"
if SJ_SRC not in sys.path:
    sys.path.insert(0, SJ_SRC)

BCB_HARD_DIR = os.path.join(ROOT, "bigCodeBench-hard")
if BCB_HARD_DIR not in sys.path:
    sys.path.insert(0, BCB_HARD_DIR)

from bench_runner import load_problems, BCB_SPLIT
from src.config import (
    SONNET_ID, OPUS_5_ID, GEMINI_36_FLASH_ID, GEMINI_35_FLASH_LITE_ID,
    ADVISOR_ROLE, EXECUTOR_ROLE, REPAIR_ROLE
)
from src.client import dispatch_model
from src.evaluator import extract_code, run_bigcodebench, missing_code_error, triage_error_straitjacket

def _eval_candidate(problem, code):
    guard = missing_code_error(code, problem["entry_point"])
    if guard:
        return False, guard
    return run_bigcodebench(problem, code)

# 1. G1: Pure Lite Ultra-Budget (3.5-Lite -> 3.5-Lite)
def run_g1_pure_lite_budget(problem):
    prompt1 = "You are an expert Python programmer. Output ONLY python code block.\n\n" + problem["complete_prompt"]
    sol1, u1, _ = dispatch_model(GEMINI_35_FLASH_LITE_ID, prompt1)
    tot_usd, tot_out = u1["as_run_usd"], u1["output"]
    code = extract_code(sol1)
    passed, err = _eval_candidate(problem, code)
    loops = 0

    if not passed:
        triage_digest = triage_error_straitjacket(err)
        prompt2 = (f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                   f"Current code:\n```python\n{code}\n```\n\n"
                   f"Unittest error:\n```\n{triage_digest}\n```\n\n"
                   "Fix bug and output COMPLETE python code block.")
        sol2, u2, _ = dispatch_model(GEMINI_35_FLASH_LITE_ID, prompt2)
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        code2 = extract_code(sol2)
        passed2, err2 = _eval_candidate(problem, code2)
        loops = 1
        if passed2:
            code, passed, err = code2, True, ""
        else:
            code, err = code2, err2

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": u1.get("total_tokens", 0) + (u2.get("total_tokens", 0) if loops > 0 else 0),
        "repair_loops": loops,
        "error": "" if passed else str(err)
    }

# 2. G2: Smart Tiered Cascade (3.5-Lite -> 3.6-Flash Min -> 3.6-Flash Low)
def run_g2_smart_tiered_cascade(problem):
    prompt1 = "You are an expert Python programmer. Output ONLY python code block.\n\n" + problem["complete_prompt"]
    sol1, u1, _ = dispatch_model(GEMINI_35_FLASH_LITE_ID, prompt1)
    tot_usd, tot_out = u1["as_run_usd"], u1["output"]
    code = extract_code(sol1)
    passed, err = _eval_candidate(problem, code)
    loops = 0

    if not passed:
        triage_digest = triage_error_straitjacket(err)
        prompt2 = (f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                   f"Current code:\n```python\n{code}\n```\n\n"
                   f"Unittest error:\n```\n{triage_digest}\n```\n\n"
                   "Fix bug using concise reasoning. Output COMPLETE python code block.")
        sol2, u2, _ = dispatch_model(GEMINI_36_FLASH_ID, prompt2, thinking_level="minimal")
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        code2 = extract_code(sol2)
        passed2, err2 = _eval_candidate(problem, code2)
        loops = 1
        if passed2:
            code, passed, err = code2, True, ""
        else:
            triage_digest2 = triage_error_straitjacket(err2)
            prompt3 = (f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                       f"Current code:\n```python\n{code2}\n```\n\n"
                       f"Unittest error:\n```\n{triage_digest2}\n```\n\n"
                       "Fix bug using careful step-by-step reasoning. Output COMPLETE python code block.")
            sol3, u3, _ = dispatch_model(GEMINI_36_FLASH_ID, prompt3, thinking_level="low")
            tot_usd += u3["as_run_usd"]
            tot_out += u3["output"]
            code3 = extract_code(sol3)
            passed3, err3 = _eval_candidate(problem, code3)
            loops = 2
            if passed3:
                code, passed, err = code3, True, ""
            else:
                code, err = code3, err3

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": u1.get("total_tokens", 0) + (u2.get("total_tokens", 0) if loops >= 1 else 0) + (u3.get("total_tokens", 0) if loops == 2 else 0),
        "repair_loops": loops,
        "error": "" if passed else str(err)
    }

# 3. G3: Advisor-Executor Split (3.6-Flash Adv -> 3.5-Lite Exec -> 3.6-Flash)
def run_g3_advisor_executor(problem):
    adv_prompt = ADVISOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    adv_text, u_adv, _ = dispatch_model(GEMINI_36_FLASH_ID, adv_prompt, max_tokens=350, thinking_level="off")
    tot_usd, tot_out = u_adv["as_run_usd"], u_adv["output"]

    exec_prompt = EXECUTOR_ROLE + f"Advisor Guidance:\n{adv_text}\n\nProblem:\n```python\n{problem['complete_prompt']}\n```"
    sol1, u1, _ = dispatch_model(GEMINI_35_FLASH_LITE_ID, exec_prompt)
    tot_usd += u1["as_run_usd"]
    tot_out += u1["output"]
    code = extract_code(sol1)
    passed, err = _eval_candidate(problem, code)
    loops = 0

    if not passed:
        triage_digest = triage_error_straitjacket(err)
        prompt2 = (f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                   f"Current code:\n```python\n{code}\n```\n\n"
                   f"Unittest error:\n```\n{triage_digest}\n```\n\n"
                   "Fix bug using careful reasoning. Output COMPLETE python code block.")
        sol2, u2, _ = dispatch_model(GEMINI_36_FLASH_ID, prompt2, thinking_level="low")
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        code2 = extract_code(sol2)
        passed2, err2 = _eval_candidate(problem, code2)
        loops = 1
        if passed2:
            code, passed, err = code2, True, ""
        else:
            code, err = code2, err2

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": u_adv.get("total_tokens", 0) + u1.get("total_tokens", 0) + (u2.get("total_tokens", 0) if loops > 0 else 0),
        "repair_loops": loops,
        "error": "" if passed else str(err)
    }

# 4. G4: Dual-Candidate Verifier (3.5-Lite x2 -> 3.6-Flash Synthesis)
def run_g4_dual_candidate_verifier(problem):
    prompt1 = "You are an expert Python programmer. Output ONLY python code block.\n\n" + problem["complete_prompt"]
    sol1, u1, _ = dispatch_model(GEMINI_35_FLASH_LITE_ID, prompt1)
    tot_usd, tot_out = u1["as_run_usd"], u1["output"]
    codeA = extract_code(sol1)
    passedA, errA = _eval_candidate(problem, codeA)
    if passedA:
        return {
            "passed": True,
            "as_run_usd": round(tot_usd, 6),
            "output_tokens": tot_out,
            "total_tokens": u1.get("total_tokens", 0),
            "repair_loops": 0,
            "error": ""
        }

    promptB = "Provide an ALTERNATIVE, robust Python implementation for the problem below. Output ONLY python code block.\n\n" + problem["complete_prompt"]
    sol2, u2, _ = dispatch_model(GEMINI_35_FLASH_LITE_ID, promptB)
    tot_usd += u2["as_run_usd"]
    tot_out += u2["output"]
    codeB = extract_code(sol2)
    passedB, errB = _eval_candidate(problem, codeB)
    if passedB:
        return {
            "passed": True,
            "as_run_usd": round(tot_usd, 6),
            "output_tokens": tot_out,
            "total_tokens": u1.get("total_tokens", 0) + u2.get("total_tokens", 0),
            "repair_loops": 1,
            "error": ""
        }

    triageA = triage_error_straitjacket(errA)
    triageB = triage_error_straitjacket(errB)
    synth_prompt = (f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                    f"Candidate A failed with:\n{triageA}\n\n"
                    f"Candidate B failed with:\n{triageB}\n\n"
                    "Analyze both failures and synthesize a COMPLETE correct Python solution. Output ONLY python code block.")
    sol3, u3, _ = dispatch_model(GEMINI_36_FLASH_ID, synth_prompt, thinking_level="low")
    tot_usd += u3["as_run_usd"]
    tot_out += u3["output"]
    code3 = extract_code(sol3)
    passed3, err3 = _eval_candidate(problem, code3)

    return {
        "passed": passed3,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": u1.get("total_tokens", 0) + u2.get("total_tokens", 0) + u3.get("total_tokens", 0),
        "repair_loops": 2,
        "error": "" if passed3 else str(err3)
    }

# 5. G5: Max-Performance Gemini (3.6-Flash Low -> Med -> High)
def run_g5_max_perf_gemini(problem):
    prompt1 = "You are an expert Python programmer. Output ONLY python code block.\n\n" + problem["complete_prompt"]
    sol1, u1, _ = dispatch_model(GEMINI_36_FLASH_ID, prompt1, thinking_level="low")
    tot_usd, tot_out = u1["as_run_usd"], u1["output"]
    code = extract_code(sol1)
    passed, err = _eval_candidate(problem, code)
    loops = 0

    if not passed:
        triage_digest = triage_error_straitjacket(err)
        prompt2 = (f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                   f"Current code:\n```python\n{code}\n```\n\n"
                   f"Unittest error:\n```\n{triage_digest}\n```\n\n"
                   "Fix bug using medium depth reasoning. Output COMPLETE python code block.")
        sol2, u2, _ = dispatch_model(GEMINI_36_FLASH_ID, prompt2, thinking_level="medium")
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        code2 = extract_code(sol2)
        passed2, err2 = _eval_candidate(problem, code2)
        loops = 1
        if passed2:
            code, passed, err = code2, True, ""
        else:
            triage_digest2 = triage_error_straitjacket(err2)
            prompt3 = (f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                       f"Current code:\n```python\n{code2}\n```\n\n"
                       f"Unittest error:\n```\n{triage_digest2}\n```\n\n"
                       "Fix bug using deep reasoning and edge-case analysis. Output COMPLETE python code block.")
            sol3, u3, _ = dispatch_model(GEMINI_36_FLASH_ID, prompt3, thinking_level="high")
            tot_usd += u3["as_run_usd"]
            tot_out += u3["output"]
            code3 = extract_code(sol3)
            passed3, err3 = _eval_candidate(problem, code3)
            loops = 2
            if passed3:
                code, passed, err = code3, True, ""
            else:
                code, err = code3, err3

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": u1.get("total_tokens", 0) + (u2.get("total_tokens", 0) if loops >= 1 else 0) + (u3.get("total_tokens", 0) if loops == 2 else 0),
        "repair_loops": loops,
        "error": "" if passed else str(err)
    }

# 6. C1: Claude Sonnet-5 Baseline (Sonnet-5 -> Sonnet-5)
def run_c1_claude_sonnet_baseline(problem):
    prompt1 = "You are an expert Python programmer. Output ONLY python code block.\n\n" + problem["complete_prompt"]
    sol1, u1, _ = dispatch_model(SONNET_ID, prompt1)
    tot_usd, tot_out = u1["as_run_usd"], u1["output"]
    code = extract_code(sol1)
    passed, err = _eval_candidate(problem, code)
    loops = 0

    if not passed:
        triage_digest = triage_error_straitjacket(err)
        prompt2 = (f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                   f"Current code:\n```python\n{code}\n```\n\n"
                   f"Unittest error:\n```\n{triage_digest}\n```\n\n"
                   "Fix bug. Output COMPLETE python code block.")
        sol2, u2, _ = dispatch_model(SONNET_ID, prompt2)
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        code2 = extract_code(sol2)
        passed2, err2 = _eval_candidate(problem, code2)
        loops = 1
        if passed2:
            code, passed, err = code2, True, ""
        else:
            code, err = code2, err2

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": u1.get("total_tokens", 0) + (u2.get("total_tokens", 0) if loops > 0 else 0),
        "repair_loops": loops,
        "error": "" if passed else str(err)
    }

# 7. C2: Claude Frontier High-End (Sonnet-5 -> Opus-5)
def run_c2_claude_frontier_opus(problem):
    prompt1 = "You are an expert Python programmer. Output ONLY python code block.\n\n" + problem["complete_prompt"]
    sol1, u1, _ = dispatch_model(SONNET_ID, prompt1)
    tot_usd, tot_out = u1["as_run_usd"], u1["output"]
    code = extract_code(sol1)
    passed, err = _eval_candidate(problem, code)
    loops = 0

    if not passed:
        triage_digest = triage_error_straitjacket(err)
        prompt2 = (f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                   f"Current code:\n```python\n{code}\n```\n\n"
                   f"Unittest error:\n```\n{triage_digest}\n```\n\n"
                   "Fix bug with architectural rigor. Output COMPLETE python code block.")
        sol2, u2, _ = dispatch_model(OPUS_5_ID, prompt2)
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        code2 = extract_code(sol2)
        passed2, err2 = _eval_candidate(problem, code2)
        loops = 1
        if passed2:
            code, passed, err = code2, True, ""
        else:
            code, err = code2, err2

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": u1.get("total_tokens", 0) + (u2.get("total_tokens", 0) if loops > 0 else 0),
        "repair_loops": loops,
        "error": "" if passed else str(err)
    }

def main():
    problems = load_problems(BCB_SPLIT)
    task_ids = list(problems.keys())[:50]
    n = len(task_ids)
    results_dir = os.path.join(HERE, "bigCodeBench-hard", "results")
    os.makedirs(results_dir, exist_ok=True)

    arms = [
        {"id": "g1", "name": "G1: Pure Lite Ultra-Budget (3.5-Lite -> 3.5-Lite)", "filename": "n50_g1_pure_lite_budget.json", "fn": run_g1_pure_lite_budget},
        {"id": "g2", "name": "G2: Smart Tiered Cascade (3.5-Lite -> 3.6-Flash Minimal/Low)", "filename": "n50_g2_smart_tiered_cascade.json", "fn": run_g2_smart_tiered_cascade},
        {"id": "g3", "name": "G3: Advisor-Executor Split (3.6-Flash Adv -> 3.5-Lite Exec -> 3.6-Flash)", "filename": "n50_g3_advisor_executor.json", "fn": run_g3_advisor_executor},
        {"id": "g4", "name": "G4: Dual-Candidate Verifier (3.5-Lite x2 -> 3.6-Flash Synthesis)", "filename": "n50_g4_dual_candidate_verifier.json", "fn": run_g4_dual_candidate_verifier},
        {"id": "g5", "name": "G5: Max-Performance Gemini (3.6-Flash Low -> Medium -> High)", "filename": "n50_g5_max_perf_gemini.json", "fn": run_g5_max_perf_gemini},
        {"id": "c1", "name": "C1: Claude Sonnet-5 Baseline (Sonnet-5 -> Sonnet-5)", "filename": "n50_c1_claude_sonnet_baseline.json", "fn": run_c1_claude_sonnet_baseline},
        {"id": "c2", "name": "C2: Claude Frontier High-End (Sonnet-5 -> Opus-5)", "filename": "n50_c2_claude_frontier_opus.json", "fn": run_c2_claude_frontier_opus},
    ]

    cache_file = os.path.join(results_dir, "cache_n50_gemini_vs_claude.json")
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as cf:
                cache = json.load(cf)
        except Exception:
            cache = {}

    print("==========================================================================================================")
    print(f"STARTING N={n} GEMINI (3.6-Flash / 3.5-Lite) vs CLAUDE-ONLY COMPREHENSIVE EVALUATION")
    print("==========================================================================================================")

    for arm in arms:
        out_file = os.path.join(results_dir, arm["filename"])
        if os.path.exists(out_file):
            print(f"\n--- {arm['name']} already completed ({out_file}), skipping ---")
            continue

        print(f"\n--- Running {arm['name']} ---")
        t0 = time.time()
        results = []
        passed_count = 0
        tot_usd = 0.0
        tot_out_tok = 0

        if arm["id"] not in cache:
            cache[arm["id"]] = {}

        for idx, tid in enumerate(task_ids, start=1):
            problem = problems[tid]
            if tid in cache[arm["id"]]:
                res = cache[arm["id"]][tid]
                status_str = "PASS" if res["passed"] else "FAIL"
                print(f"[{idx}/{n}] {tid} ... [CACHED] {status_str} | cost=${res['as_run_usd']:.5f} | out_tok={res['output_tokens']}", flush=True)
            else:
                try:
                    raw_res = arm["fn"](problem)
                except Exception as e:
                    print(f" [API Error: {e}] ", end="", flush=True)
                    raw_res = {
                        "passed": False,
                        "as_run_usd": 0.001,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "repair_loops": 0,
                        "error": f"API Error: {str(e)}"
                    }
                res = {
                    "task_id": tid,
                    "passed": raw_res["passed"],
                    "as_run_usd": raw_res["as_run_usd"],
                    "output_tokens": raw_res["output_tokens"],
                    "total_tokens": raw_res["total_tokens"],
                    "repair_loops": raw_res.get("repair_loops", 0),
                    "error": raw_res.get("error", "")
                }
                cache[arm["id"]][tid] = res
                with open(cache_file, "w", encoding="utf-8") as cf:
                    json.dump(cache, cf, indent=2)
                status_str = "PASS" if res["passed"] else "FAIL"
                print(f"[{idx}/{n}] {tid} ... {status_str} | cost=${res['as_run_usd']:.5f} | out_tok={res['output_tokens']}", flush=True)

            results.append(res)
            if res["passed"]:
                passed_count += 1
            tot_usd += res["as_run_usd"]
            tot_out_tok += res["output_tokens"]

        dt = time.time() - t0
        pass_rate = (passed_count / n) * 100.0
        cost_per_solved = (tot_usd / passed_count) if passed_count > 0 else 0.0
        avg_out = tot_out_tok / n

        print("-" * 60)
        print(f"SUMMARY ({arm['id']}): Pass Rate = {passed_count}/{n} ({pass_rate:.1f}%) | Total Cost = ${tot_usd:.4f} | $/Solved = ${cost_per_solved:.4f}")

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "arm": arm["name"],
                "n": n,
                "passed": passed_count,
                "pass_rate": round(pass_rate, 2),
                "total_as_run_usd": round(tot_usd, 6),
                "cost_per_solved_usd": round(cost_per_solved, 6),
                "avg_output_tokens": round(avg_out, 1),
                "seconds": round(dt, 2),
                "results": results
            }, f, indent=2)

    # Generate Report at the end
    import subprocess
    subprocess.run([sys.executable, os.path.join(HERE, "tools", "generate_n50_report.py")])

if __name__ == "__main__":
    main()
