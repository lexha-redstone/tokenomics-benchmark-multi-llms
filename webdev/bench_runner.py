#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0
"""
Web-Dev Comprehensive Real Benchmark Runner (API-Only Mode).
Filters tasks using web-related libraries from BigCodeBench-Hard and runs real LLM evaluation.
"""

import os
import sys
import json
import time
import argparse
import ast

# Setup path for src module import
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import (
    GEMINI_FLASH_ID, GEMINI_FLASH_LITE_ID, SONNET_ID, OPUS_ID,
    SOLVER_ROLE as DEFAULT_SOLVER_ROLE,
    ADVISOR_ROLE as DEFAULT_ADVISOR_ROLE,
    EXECUTOR_ROLE as DEFAULT_EXECUTOR_ROLE,
    REPAIR_ROLE as DEFAULT_REPAIR_ROLE
)
from src.client import dispatch_model
from src.evaluator import extract_code, run_bigcodebench, missing_code_error, triage_error
from src.architectures import run_single, run_read_write, run_cascade, run_hybrid

DATA_PATH = os.path.join(HERE, "data", "BigCodeBench-Hard-WebDev.jsonl")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Web-Dev Specific Prompts (Web-centric) ---
SOLVER_ROLE = (
    "You are an expert Python programmer. Complete the function below. You are given its imports, "
    "signature, and docstring; several real web/networking libraries must be used correctly. Output the COMPLETE "
    "solution: all needed imports and the full function definition, handling edge cases and the "
    "documented return/exception behavior exactly. Output ONLY one ```python code block, no "
    "explanation.\n\n"
)

ADVISOR_ROLE = (
    "You are a senior ADVISOR in an advisor-executor coding system. You do NOT write code. Given "
    "the Python coding problem below (imports + function signature + docstring), which requires "
    "correctly using web/networking libraries, produce concise, precise implementation GUIDANCE for "
    "a separate executor model: which libraries/APIs to use and in what order, the intended "
    "algorithm, edge cases, and the EXACT documented return values and exceptions to honor. "
    "Under 200 words. Do NOT output any code.\n\n"
)

EXECUTOR_ROLE = DEFAULT_EXECUTOR_ROLE
REPAIR_ROLE = DEFAULT_REPAIR_ROLE

# --- Dataset Loader ---
def load_web_problems():
    web_libs = {"requests", "urllib", "flask", "flask_login", "flask_mail", "flask_wtf", 
                "werkzeug", "wtforms", "http", "ftplib", "smtplib", "bs4", "pyquery", "lxml", 
                "cgi", "socket"}
    
    problems = {}
    if not os.path.exists(DATA_PATH):
        print(f"Error: dataset file not found at {DATA_PATH}")
        sys.exit(1)
        
    for line in open(DATA_PATH):
        d = json.loads(line)
        try:
            lib_list = ast.literal_eval(d.get("libs", "[]"))
            if any(lib in web_libs for lib in lib_list):
                problems[d["task_id"]] = d
        except Exception:
            pass
            
    print(f"Loaded {len(problems)} Web-Dev tasks from BigCodeBench-Hard")
    return problems

# --- Web-Dev Specific Architectures ---

def run_dynamic_router(problem):
    router_prompt = (
        "You are an AI code complexity router. Analyze the Python problem below (libraries + entry point + docstring).\n"
        "Output ONLY one word: 'SIMPLE' if it is basic data manipulation or standard syntax, or 'COMPLEX' if it requires intricate algorithm or multi-library coordination.\n\n"
        f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    )
    r_text, r_usage, _ = dispatch_model(GEMINI_FLASH_ID, router_prompt, max_tokens=10, thinking_level="off")
    is_complex = "COMPLEX" in r_text.upper()
    tot_usd, tot_out, tot_tok = r_usage["as_run_usd"], r_usage["output"], r_usage["total_tokens"]

    if not is_complex:
        prompt = SOLVER_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\nWrite the complete solution."
        text, exec_usage, _ = dispatch_model(GEMINI_FLASH_LITE_ID, prompt)
        tot_usd += exec_usage["as_run_usd"]
        tot_out += exec_usage["output"]
        tot_tok += exec_usage["total_tokens"]
        code = extract_code(text)
        passed, err = run_bigcodebench(problem, code)
    else:
        adv_prompt = ADVISOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```"
        guidance, adv_usage, _ = dispatch_model(GEMINI_FLASH_ID, adv_prompt, thinking_level="low")
        tot_usd += adv_usage["as_run_usd"]
        tot_out += adv_usage["output"]
        tot_tok += adv_usage["total_tokens"]

        exec_prompt = EXECUTOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nGuidance:\n{guidance}\n\nWrite complete solution."
        sol, exec_usage, _ = dispatch_model(GEMINI_FLASH_LITE_ID, exec_prompt)
        tot_usd += exec_usage["as_run_usd"]
        tot_out += exec_usage["output"]
        tot_tok += exec_usage["total_tokens"]
        code = extract_code(sol)
        passed, err = run_bigcodebench(problem, code)

    return {"passed": passed, "usd": round(tot_usd, 6), "out_tok": tot_out, "tot_tok": tot_tok, "err": "" if passed else err}

def run_dual_advisor(problem):
    adv1_prompt = "Advisor 1 (Algorithm & Edge Cases): Identify core algorithm and 3 potential edge-case pitfalls. Under 150 words. No code.\n\n" + problem["complete_prompt"]
    g1, u1, _ = dispatch_model(GEMINI_FLASH_ID, adv1_prompt, max_tokens=512, thinking_level="low")

    adv2_prompt = "Advisor 2 (API Contract): List exact library functions, return types, and exception behavior required. Under 150 words. No code.\n\n" + problem["complete_prompt"]
    g2, u2, _ = dispatch_model(GEMINI_FLASH_ID, adv2_prompt, max_tokens=512, thinking_level="off")

    exec_prompt = (
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Algorithm Guidance:\n{g1}\n\n"
        f"API Contract Guidance:\n{g2}\n\n"
        "Write complete solution. Output ONLY one ```python code block."
    )
    sol, u_exec, _ = dispatch_model(GEMINI_FLASH_LITE_ID, exec_prompt)

    tot_usd = round(u1["as_run_usd"] + u2["as_run_usd"] + u_exec["as_run_usd"], 6)
    tot_out = u1["output"] + u2["output"] + u_exec["output"]
    tot_tok = u1["total_tokens"] + u2["total_tokens"] + u_exec["total_tokens"]
    code = extract_code(sol)
    passed, err = run_bigcodebench(problem, code)
    return {"passed": passed, "usd": tot_usd, "out_tok": tot_out, "tot_tok": tot_tok, "err": "" if passed else err}

def run_tdd_harness(problem):
    test_gen_prompt = (
        "Given the Python problem below, write 2 concise unittest assertion statements (e.g. `self.assertEqual(task_func(...), ...)` or `self.assertTrue(...)`) "
        "checking edge-case behavior. Output ONLY valid python assert statements.\n\n" + problem["complete_prompt"]
    )
    synth_tests, u_test, _ = dispatch_model(GEMINI_FLASH_ID, test_gen_prompt, max_tokens=512, thinking_level="low")

    exec_prompt = (
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Verify your code against these edge-case assertions before returning:\n```python\n{synth_tests}\n```\n\n"
        "Write complete solution. Output ONLY one ```python code block."
    )
    sol, u_exec, _ = dispatch_model(GEMINI_FLASH_LITE_ID, exec_prompt)

    tot_usd = round(u_test["as_run_usd"] + u_exec["as_run_usd"], 6)
    tot_out = u_test["output"] + u_exec["output"]
    tot_tok = u_test["total_tokens"] + u_exec["total_tokens"]
    code = extract_code(sol)
    passed, err = run_bigcodebench(problem, code)
    return {"passed": passed, "usd": tot_usd, "out_tok": tot_out, "tot_tok": tot_tok, "err": "" if passed else err}

def run_escalation_shield(problem, tier3_model):
    prompt = SOLVER_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\nWrite the complete solution."
    sol1, u1, _ = dispatch_model(GEMINI_FLASH_LITE_ID, prompt)
    tot_usd, tot_out, tot_tok = u1["as_run_usd"], u1["output"], u1["total_tokens"]
    code = extract_code(sol1)
    passed, err = run_bigcodebench(problem, code)

    if not passed:
        r1_prompt = (REPAIR_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\n"
                     f"Current code:\n```python\n{code}\n```\n\n"
                     f"Unittest error:\n```\n{err[-2000:]}\n```\n\nFix bug. Output complete python code.")
        sol2, u2, _ = dispatch_model(GEMINI_FLASH_ID, r1_prompt, thinking_level="low")
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        tot_tok += u2["total_tokens"]
        code2 = extract_code(sol2)
        guard = missing_code_error(code2, problem["entry_point"])
        if not guard:
            code = code2
            passed, err = run_bigcodebench(problem, code)

    if not passed:
        r2_prompt = (REPAIR_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\n"
                     f"Current code:\n```python\n{code}\n```\n\n"
                     f"Unittest error:\n```\n{err[-2000:]}\n```\n\nFix bug. Output complete python code.")
        sol3, u3, _ = dispatch_model(tier3_model, r2_prompt)
        tot_usd += u3["as_run_usd"]
        tot_out += u3["output"]
        tot_tok += u3["total_tokens"]
        code3 = extract_code(sol3)
        guard = missing_code_error(code3, problem["entry_point"])
        if not guard:
            passed, err = run_bigcodebench(problem, code3)

    return {"passed": passed, "usd": round(tot_usd, 6), "out_tok": tot_out, "tot_tok": tot_tok, "err": "" if passed else err}

def run_peer_reviewer(problem):
    prompt = SOLVER_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\nWrite complete solution."
    sol1, u1, _ = dispatch_model(SONNET_ID, prompt)
    code1 = extract_code(sol1)

    audit_prompt = (
        "You are a senior code reviewer. Review the candidate code below for subtle logic errors, missing imports, or incorrect return types.\n"
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Candidate Code:\n```python\n{code1}\n```\n\n"
        "If correct, output 'LGTM'. If flawed, output 2 concise bullet points describing the bug."
    )
    critique, u_audit, _ = dispatch_model(GEMINI_FLASH_ID, audit_prompt, thinking_level="medium")
    tot_usd = u1["as_run_usd"] + u_audit["as_run_usd"]
    tot_out = u1["output"] + u_audit["output"]
    tot_tok = u1["total_tokens"] + u_audit["total_tokens"]

    if "LGTM" in critique.upper():
        code = code1
        passed, err = run_bigcodebench(problem, code)
    else:
        rev_prompt = (
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Draft Code:\n```python\n{code1}\n```\n\n"
            f"Reviewer Critique:\n{critique}\n\nFix the code based on critique. Output ONLY one python code block."
        )
        sol2, u_rev, _ = dispatch_model(SONNET_ID, rev_prompt)
        tot_usd += u_rev["as_run_usd"]
        tot_out += u_rev["output"]
        tot_tok += u_rev["total_tokens"]
        code = extract_code(sol2)
        passed, err = run_bigcodebench(problem, code)

    return {"passed": passed, "usd": round(tot_usd, 6), "out_tok": tot_out, "tot_tok": tot_tok, "err": "" if passed else err}

def run_routed_shield_cascade(problem):
    router_prompt = (
        "You are an AI code complexity router. Analyze the Python problem below (libraries + entry point + docstring).\n"
        "Output ONLY one word: 'SIMPLE' if it is basic data manipulation or standard syntax, or 'COMPLEX' if it requires intricate algorithm or multi-library coordination.\n\n"
        f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    )
    r_text, r_usage, _ = dispatch_model(GEMINI_FLASH_ID, router_prompt, max_tokens=10, thinking_level="off")
    is_complex = "COMPLEX" in r_text.upper()
    tot_usd, tot_out, tot_tok = r_usage["as_run_usd"], r_usage["output"], r_usage["total_tokens"]

    if not is_complex:
        # Shield: Lite -> Flash Low -> Sonnet
        prompt = SOLVER_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\nWrite complete solution."
        sol1, u1, _ = dispatch_model(GEMINI_FLASH_LITE_ID, prompt)
        tot_usd += u1["as_run_usd"]
        tot_out += u1["output"]
        tot_tok += u1["total_tokens"]
        code = extract_code(sol1)
        passed, err = run_bigcodebench(problem, code)

        if not passed:
            r1_prompt = (REPAIR_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\n"
                         f"Current code:\n```python\n{code}\n```\n\n"
                         f"Unittest error:\n```\n{err[-2000:]}\n```\n\nFix bug. Output complete python code.")
            sol2, u2, _ = dispatch_model(GEMINI_FLASH_ID, r1_prompt, thinking_level="low")
            tot_usd += u2["as_run_usd"]
            tot_out += u2["output"]
            tot_tok += u2["total_tokens"]
            code2 = extract_code(sol2)
            guard = missing_code_error(code2, problem["entry_point"])
            if not guard:
                code = code2
                passed, err = run_bigcodebench(problem, code)

        if not passed:
            r2_prompt = (REPAIR_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\n"
                         f"Current code:\n```python\n{code}\n```\n\n"
                         f"Unittest error:\n```\n{err[-2000:]}\n```\n\nFix bug. Output complete python code.")
            sol3, u3, _ = dispatch_model(SONNET_ID, r2_prompt)
            tot_usd += u3["as_run_usd"]
            tot_out += u3["output"]
            tot_tok += u3["total_tokens"]
            code3 = extract_code(sol3)
            guard = missing_code_error(code3, problem["entry_point"])
            if not guard:
                passed, err = run_bigcodebench(problem, code3)
    else:
        # Complex: Flash Low -> Sonnet
        prompt = SOLVER_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\nWrite complete solution."
        sol1, u1, _ = dispatch_model(GEMINI_FLASH_ID, prompt, thinking_level="low")
        tot_usd += u1["as_run_usd"]
        tot_out += u1["output"]
        tot_tok += u1["total_tokens"]
        code = extract_code(sol1)
        passed, err = run_bigcodebench(problem, code)

        if not passed:
            r1_prompt = (REPAIR_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\n"
                         f"Current code:\n```python\n{code}\n```\n\n"
                         f"Unittest error:\n```\n{err[-2000:]}\n```\n\nFix bug. Output complete python code.")
            sol2, u2, _ = dispatch_model(SONNET_ID, r1_prompt)
            tot_usd += u2["as_run_usd"]
            tot_out += u2["output"]
            tot_tok += u2["total_tokens"]
            code2 = extract_code(sol2)
            guard = missing_code_error(code2, problem["entry_point"])
            if not guard:
                passed, err = run_bigcodebench(problem, code2)

    return {"passed": passed, "usd": round(tot_usd, 6), "out_tok": tot_out, "tot_tok": tot_tok, "err": "" if passed else err}

# --- DRIVER ---
def run_benchmark(arch, tasks, problems, **kwargs):
    results = []
    import_errors = []
    print(f"\nRunning Web-Dev Benchmark: Arch={arch.upper()} on {len(tasks)} tasks...")
    for i, tid in enumerate(tasks, 1):
        problem = problems[tid]
        print(f"[{i}/{len(tasks)}] {tid} ({problem['entry_point']}) ... ", end="", flush=True)
        t0 = time.time()
        
        try:
            # Overwrite default roles with Web-centric versions
            if arch == "single":
                r = run_single(problem, model_id=kwargs.get("model", GEMINI_FLASH_ID), solver_role=SOLVER_ROLE)
            elif arch == "read-write":
                r = run_read_write(problem, planner_model=kwargs.get("planner", GEMINI_FLASH_ID),
                               executor_model=kwargs.get("executor", GEMINI_FLASH_LITE_ID),
                               advisor_role=ADVISOR_ROLE, executor_role=EXECUTOR_ROLE)
            elif arch == "cascade":
                r = run_cascade(problem, gen_model=kwargs.get("gen_id", GEMINI_FLASH_LITE_ID),
                                esc_model=kwargs.get("esc_id", GEMINI_FLASH_ID),
                                solver_role=SOLVER_ROLE, repair_role=REPAIR_ROLE)
            elif arch == "hybrid":
                r = run_hybrid(problem, planner_model=kwargs.get("planner", GEMINI_FLASH_ID),
                               executor_model=kwargs.get("executor", GEMINI_FLASH_LITE_ID),
                               escalate_model=kwargs.get("escalate", GEMINI_FLASH_ID),
                               triage_model=kwargs.get("triage", GEMINI_FLASH_LITE_ID),
                               advisor_role=ADVISOR_ROLE, executor_role=EXECUTOR_ROLE, repair_role=REPAIR_ROLE)
            elif arch == "router":
                r = run_dynamic_router(problem)
            elif arch == "dual-advisor":
                r = run_dual_advisor(problem)
            elif arch == "tdd":
                r = run_tdd_harness(problem)
            elif arch == "shield":
                r = run_escalation_shield(problem, tier3_model=kwargs.get("tier3", SONNET_ID))
            elif arch == "peer-reviewer":
                r = run_peer_reviewer(problem)
            elif arch == "routed-shield":
                r = run_routed_shield_cascade(problem)
            else:
                raise ValueError(f"Unknown arch: {arch}")
            
            status = "PASS" if r["passed"] else "FAIL"
            dt = time.time() - t0
            # For backward compatibility with report formatting
            usd_val = r.get("usd") if "usd" in r else r.get("as_run_usd", 0.0)
            out_tok_val = r.get("out_tok") if "out_tok" in r else r.get("output_tokens", 0)
            err_val = r.get("err") if "err" in r else r.get("error", "")
            
            # Map standard keys if not present
            if "usd" not in r: r["usd"] = usd_val
            if "out_tok" not in r: r["out_tok"] = out_tok_val
            if "err" not in r: r["err"] = err_val
            
            print(f"{status} | cost=${usd_val:.5f} | out_tok={out_tok_val} | time={dt:.1f}s")
            
            if not r["passed"] and ("ImportError" in err_val or "ModuleNotFoundError" in err_val):
                print(f"  [IMPORT ERROR DECLARED] {tid}: {err_val.splitlines()[-1] if err_val else 'unknown'}")
                import_errors.append((arch, tid, err_val))
                
            results.append(r)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"passed": False, "usd": 0.0, "out_tok": 0, "tot_tok": 0, "err": str(e)})
            if "ImportError" in str(e) or "ModuleNotFoundError" in str(e):
                import_errors.append((arch, tid, str(e)))

    n = len(results)
    passed_cnt = sum(1 for r in results if r["passed"])
    tot_cost = sum(r["usd"] for r in results)
    avg_out = sum(r["out_tok"] for r in results) / n if n else 0

    print("-" * 60)
    print(f"SUMMARY ({arch}): Pass Rate = {passed_cnt}/{n} ({passed_cnt/n:.1%}) | "
          f"Total Cost = ${tot_cost:.4f} | Avg Output Tokens = {avg_out:.1f}")
    
    summary = {
        "arch": arch,
        "n": n,
        "passed": passed_cnt,
        "pass_rate": round(passed_cnt / n, 3) if n else 0,
        "total_usd": round(tot_cost, 5),
        "cost_per_solved": round(tot_cost / passed_cnt, 5) if passed_cnt > 0 else -1.0,
        "avg_out_tok": round(avg_out, 1)
    }
    return summary, import_errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["single", "read-write", "cascade", "hybrid", "router", "dual-advisor", "tdd", "shield", "peer-reviewer", "routed-shield"], default="hybrid")
    parser.add_argument("--n", type=int, default=10, help="Number of tasks to evaluate (default: 10)")
    parser.add_argument("--compare-all", action="store_true", help="Compare key sweet-spot architectures")
    args = parser.parse_args()

    problems = load_web_problems()
    task_ids = list(problems.keys())[:args.n]

    if args.compare_all:
        configs = [
            ("1. Single: gemini-3.1-flash-lite", "single", {"model": GEMINI_FLASH_LITE_ID}),
            ("2. Single: gemini-3.5-flash", "single", {"model": GEMINI_FLASH_ID}),
            ("3. Single: claude-sonnet-5", "single", {"model": SONNET_ID}),
            ("4. Single: claude-opus-4-8", "single", {"model": OPUS_ID}),
            ("5. Adv-Exec: 3.5-Flash + 3.1-Lite", "read-write", {"planner": GEMINI_FLASH_ID, "executor": GEMINI_FLASH_LITE_ID}),
            ("6. Adv-Exec: Sonnet-5 + 3.1-Lite", "read-write", {"planner": SONNET_ID, "executor": GEMINI_FLASH_LITE_ID}),
            ("7. Adv-Exec: Opus-4.8 + 3.1-Lite", "read-write", {"planner": OPUS_ID, "executor": GEMINI_FLASH_LITE_ID}),
            ("8. Cascade: 3.1-Lite -> 3.5-Flash Low", "cascade", {"gen_id": GEMINI_FLASH_LITE_ID, "esc_id": GEMINI_FLASH_ID}),
            ("9. Cascade: 3.1-Lite -> Sonnet-5", "cascade", {"gen_id": GEMINI_FLASH_LITE_ID, "esc_id": SONNET_ID}),
            ("10. Sweet-Spot Hybrid", "hybrid", {"planner": GEMINI_FLASH_ID, "executor": GEMINI_FLASH_LITE_ID, "escalate": GEMINI_FLASH_ID, "triage": GEMINI_FLASH_LITE_ID}),
            ("11. Dynamic Thinking Router", "router", {}),
            ("12. Dual-Perspective Advisor", "dual-advisor", {}),
            ("13. TDD Harness", "tdd", {}),
            ("14. Shield: 3.1-Lite -> 3.5-Low -> Sonnet-5", "shield", {"tier3": SONNET_ID}),
            ("17. Routed Shield Cascade (New Proposal)", "routed-shield", {}),
        ]
        summaries = []
        all_import_errors = []
        for name, arch, kwargs in configs:
            print(f"\n=======================================================")
            print(f"CONFIG: {name}")
            print(f"=======================================================")
            s, errs = run_benchmark(arch, task_ids, problems, **kwargs)
            s["name"] = name
            summaries.append(s)
            all_import_errors.extend(errs)

        print("\n" + "=" * 95)
        print("OVERALL REAL WEB-DEV SWEET-SPOT COMPARISON TABLE")
        print("=" * 95)
        print(f"{'Configuration':<44} | {'Pass Rate':<10} | {'Total Cost ($)':<14} | {'$/Solved':<10} | {'Avg Out':<8}")
        print("-" * 95)
        for s in summaries:
            cps_str = f"${s['cost_per_solved']:.4f}" if s['cost_per_solved'] >= 0 else "N/A"
            print(f"{s['name']:<44} | {s['passed']}/{s['n']} ({s['pass_rate']:.0%})  | ${s['total_usd']:<13.5f} | {cps_str:<10} | {s['avg_out_tok']:<8.0f}")
        print("=" * 95)

        # Print all detected import errors
        if all_import_errors:
            print("\n" + "!" * 95)
            print("DETECTED IMPORT ERRORS DURING BENCHMARK")
            print("!" * 95)
            for arch, tid, error_msg in all_import_errors:
                last_line = error_msg.splitlines()[-1] if error_msg else "unknown error"
                print(f"  - Arch: {arch:<15} | Task: {tid:<15} | {last_line}")
            print("!" * 95)
        else:
            print("\n>>> No ImportErrors or ModuleNotFoundErrors detected during runs.")

        # Save JSON results with a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(RESULTS_DIR, f"results_webdev_real_{timestamp}.json")
        with open(out_file, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nDetailed metrics saved to: {out_file}")
    else:
        s, errs = run_benchmark(args.arch, task_ids, problems)
        if errs:
            print("\nImport Errors Detected:")
            for arch, tid, error_msg in errs:
                print(f"  - Task: {tid} | {error_msg.splitlines()[-1]}")
