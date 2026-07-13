#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0
"""
Targeted Thinking Level Benchmark for gemini-3.5-flash on BigCodeBench-Hard.

Evaluates gemini-3.5-flash across thinking levels:
  1. OFF (budget=0)
  2. MINIMAL
  3. LOW
  4. MEDIUM
  5. HIGH

Across:
  - Part 1: Single-Model Direct Completion
  - Part 2: Escalated Repair on Failed Tasks

Outputs pass rates, thinking token consumption, output token counts, as-run cost, and latency.
"""

import os, sys, json, re, subprocess, tempfile, time, argparse, shutil, ssl
import urllib.request, urllib.parse, urllib.error

os.environ["MPLBACKEND"] = "Agg"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BCB_DATASET = "bigcode/bigcodebench-hard"
BCB_CONFIG = "default"
BCB_SPLIT = "v0.1.4"
_KEEP_FIELDS = ("task_id", "complete_prompt", "canonical_solution", "code_prompt", "test", "entry_point", "libs")

GEMINI_FLASH_ID = "gemini-3.5-flash"
GCP_PROJECT = os.environ.get("GCP_PROJECT", "my-argolis-prj")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")

# Pricing: Gemini 3.5 Flash: Input $1.50/1M, Output $9.00/1M (thoughts counted as output tokens)
PRICING_INPUT = 1.50
PRICING_OUTPUT = 9.00

SOLVER_ROLE = (
    "You are an expert Python programmer. Complete the function below. You are given its imports, "
    "signature, and docstring; several real libraries must be used correctly. Output the COMPLETE "
    "solution: all needed imports and the full function definition, handling edge cases and the "
    "documented return/exception behavior exactly. Output ONLY one ```python code block, no "
    "explanation.\n\n"
)

REPAIR_ROLE = (
    "You are an expert Python programmer. A candidate solution to the problem below FAILED its "
    "unit tests. Analyze the test error output, find the bug, and fix the code. Output the "
    "COMPLETE corrected solution: all needed imports and the full function definition. Output ONLY "
    "one ```python code block, no explanation.\n\n"
)

def _ssl_ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

def ensure_dataset(split=BCB_SPLIT):
    path = os.path.join(DATA_DIR, f"BigCodeBench-Hard-{split}.jsonl")
    if os.path.exists(path):
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
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
    return path

def load_problems(split=BCB_SPLIT):
    path = ensure_dataset(split)
    return {json.loads(l)["task_id"]: json.loads(l) for l in open(path)}

_gemini_client = None
def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    return _gemini_client

THINKING_LEVEL_HEADROOM = {
    "off": 0,
    "minimal": 2048,
    "low": 4096,
    "medium": 8192,
    "high": 16384
}

def gemini_call_thinking(prompt, thinking_level="off", base_max_tokens=2560):
    from google.genai import types
    level_lower = thinking_level.lower()

    if level_lower == "off":
        tc = types.ThinkingConfig(thinking_budget=0)
        max_tok = base_max_tokens
    else:
        tc = types.ThinkingConfig(thinking_level=level_lower.upper())
        max_tok = base_max_tokens + THINKING_LEVEL_HEADROOM.get(level_lower, 4096)

    cfg = types.GenerateContentConfig(
        max_output_tokens=max_tok,
        thinking_config=tc,
        http_options=types.HttpOptions(timeout=90000)
    )

    for attempt in range(3):
        try:
            t0 = time.time()
            resp = _gemini().models.generate_content(
                model=GEMINI_FLASH_ID,
                contents=prompt,
                config=cfg
            )
            dt = time.time() - t0
            m = resp.usage_metadata
            inp = m.prompt_token_count or 0
            candidate_tok = m.candidates_token_count or 0
            thoughts_tok = m.thoughts_token_count or 0
            total_out = candidate_tok + thoughts_tok

            cost = round(inp / 1e6 * PRICING_INPUT + total_out / 1e6 * PRICING_OUTPUT, 6)
            usage = {
                "input": inp,
                "candidate_output": candidate_tok,
                "thoughts_output": thoughts_tok,
                "total_output": total_out,
                "total_tokens": m.total_token_count or (inp + total_out),
                "as_run_usd": cost,
                "thinking_level": level_lower
            }
            return (resp.text or ""), usage, dt
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(2)


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()

_UNITTEST_RUNNER = (
    "\n\nimport unittest as _ut, sys as _sys\n"
    "_ut.TestCase.maxDiff = None\n"
    "_res = _ut.TextTestRunner(verbosity=0).run("
    "_ut.TestLoader().loadTestsFromTestCase(TestCases))\n"
    "_sys.exit(0 if _res.wasSuccessful() else 1)\n"
)

def run_bigcodebench(problem, solution_code):
    program = solution_code + "\n\n" + problem["test"] + _UNITTEST_RUNNER
    workdir = tempfile.mkdtemp(prefix="bcb_")
    path = os.path.join(workdir, "prog.py")
    with open(path, "w") as f:
        f.write(program)
    try:
        env = {**os.environ, "MPLBACKEND": "Agg"}
        r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=120, cwd=workdir, env=env)
        if r.returncode == 0:
            return True, ""
        return False, (r.stderr.strip() or "test failed")[-4000:]
    except subprocess.TimeoutExpired:
        return False, "timeout: execution exceeded 120s"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def missing_code_error(code, entry_point):
    if f"def {entry_point}" in code:
        return None
    return f"model response contains no `def {entry_point}` code block"

def benchmark_single_thinking_levels(tasks, problems):
    levels = ["off", "minimal", "low", "medium", "high"]
    results_by_level = {}

    for lvl in levels:
        print(f"\n=======================================================")
        print(f"EVALUATING: gemini-3.5-flash with thinking_level='{lvl.upper()}'")
        print(f"=======================================================")
        task_results = []
        for i, tid in enumerate(tasks, 1):
            p = problems[tid]
            prompt = SOLVER_ROLE + "Problem:\n```python\n" + p["complete_prompt"] + "\n```\n\nWrite the complete solution."
            text, usage, dt = gemini_call_thinking(prompt, thinking_level=lvl)
            code = extract_code(text)
            guard = missing_code_error(code, p["entry_point"])
            passed, err = (False, guard) if guard else run_bigcodebench(p, code)

            status = "PASS" if passed else "FAIL"
            print(f"[{i}/{len(tasks)}] {tid} ({p['entry_point']}) ... {status} | "
                  f"cost=${usage['as_run_usd']:.5f} | candidate_tok={usage['candidate_output']} | "
                  f"thought_tok={usage['thoughts_output']} | time={dt:.1f}s")

            task_results.append({
                "task_id": tid,
                "passed": passed,
                "usage": usage,
                "seconds": round(dt, 1),
                "error": "" if passed else err
            })

        n = len(task_results)
        passed_cnt = sum(1 for r in task_results if r["passed"])
        tot_cost = sum(r["usage"]["as_run_usd"] for r in task_results)
        avg_cand = sum(r["usage"]["candidate_output"] for r in task_results) / n
        avg_thought = sum(r["usage"]["thoughts_output"] for r in task_results) / n
        avg_dt = sum(r["seconds"] for r in task_results) / n

        results_by_level[lvl] = {
            "level": lvl,
            "passed": passed_cnt,
            "n": n,
            "pass_rate": round(passed_cnt / n, 3),
            "total_as_run_usd": round(tot_cost, 4),
            "avg_candidate_tokens": round(avg_cand, 1),
            "avg_thought_tokens": round(avg_thought, 1),
            "avg_seconds": round(avg_dt, 1),
            "tasks": task_results
        }

    return results_by_level

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="Number of tasks (default: 10)")
    args = parser.parse_args()

    problems = load_problems()
    task_ids = list(problems.keys())[:args.n]

    res = benchmark_single_thinking_levels(task_ids, problems)

    out_file = os.path.join(RESULTS_DIR, "results_thinking_levels_gemini35.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 90)
    print("GEMINI-3.5-FLASH THINKING LEVEL COMPARISON MATRIX (First 10 BigCodeBench-Hard Tasks)")
    print("=" * 90)
    print(f"{'Thinking Level':<16} | {'Pass Rate':<10} | {'Total Cost ($)':<12} | {'Avg Thought Tok':<16} | {'Avg Code Tok':<14} | {'Avg Time (s)':<10}")
    print("-" * 90)
    for lvl in ["off", "minimal", "low", "medium", "high"]:
        r = res[lvl]
        print(f"{r['level'].upper():<16} | {r['passed']}/{r['n']} ({r['pass_rate']:.0%})  | ${r['total_as_run_usd']:<11.4f} | {r['avg_thought_tokens']:<16.1f} | {r['avg_candidate_tokens']:<14.1f} | {r['avg_seconds']:<10.1f}")
    print("=" * 90)
    print(f"\nSaved detailed JSON results to: {out_file}")

if __name__ == "__main__":
    main()
