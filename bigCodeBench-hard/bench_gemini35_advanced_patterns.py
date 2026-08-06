#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0
"""
Advanced gemini-3.5-flash Multi-Model & Integration Pattern Benchmark.

Evaluates 6 NEW 3.5-Flash Integration Proposals on BigCodeBench-Hard (first N tasks):
  1. Pattern 1: Dynamic Thinking Router (3.5-Flash Router -> 3.1-Lite / 3.5-Flash Medium)
  2. Pattern 2: Dual-Perspective 3.5-Flash Advisor (Logic + API Contract -> 3.1-Lite Exec)
  3. Pattern 3: TDD Harness (3.5-Flash Synthetic Test Gen -> 3.1-Lite Exec & Local Test)
  4. Pattern 4: Frontier Escalation Shield (3.1-Lite -> 3.5-Flash Low -> Sonnet-5)
  5. Pattern 5: Peer Reviewer (Sonnet-5 Gen -> 3.5-Flash Medium Auditor -> Execution)
  6. Pattern 6: Tiered Thinking Ramping (3.5-Flash OFF -> LOW -> HIGH)

Outputs pass rate, total cost ($), cost per solved ($/solved), avg output tokens, and latency.
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

OPUS_ID = "claude-opus-4-8"
SONNET_ID = "claude-sonnet-5"
GEMINI_FLASH_ID = "gemini-3.5-flash"
GEMINI_FLASH_LITE_ID = "gemini-3.1-flash-lite"

GCP_PROJECT = os.environ.get("GCP_PROJECT", "my-argolis-prj")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")

PRICING = {
    OPUS_ID:              {"input": 5.00, "output": 25.00, "cache_read": 0.50,  "cache_write": 6.25},
    SONNET_ID:            {"input": 2.00, "output": 10.00, "cache_read": 0.20,  "cache_write": 2.50},
    GEMINI_FLASH_ID:      {"input": 1.50, "output": 9.00,  "cache_read": 0.15,  "cache_write": 0.00},
    GEMINI_FLASH_LITE_ID: {"input": 0.25, "output": 1.50,  "cache_read": 0.025, "cache_write": 0.00},
}

SOLVER_ROLE = (
    "You are an expert Python programmer. Complete the function below. You are given its imports, "
    "signature, and docstring; several real libraries must be used correctly. Output the COMPLETE "
    "solution: all needed imports and the full function definition, handling edge cases and the "
    "documented return/exception behavior exactly. Output ONLY one ```python code block, no "
    "explanation.\n\n"
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

# --- API Dispatchers ---
_vertex_token = {"tok": None, "exp": 0.0}
def _vertex_access_token():
    if _vertex_token["tok"] and _vertex_token["exp"] - 60 > time.time():
        return _vertex_token["tok"]
    import google.auth
    import google.auth.transport.requests
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    _vertex_token["tok"] = creds.token
    _vertex_token["exp"] = creds.expiry.timestamp() if creds.expiry else time.time() + 3000
    return creds.token

def claude_api_call(model_id, prompt, max_tokens=2560, thinking_level=None):
    host = ("aiplatform.googleapis.com" if GCP_LOCATION == "global"
            else f"{GCP_LOCATION}-aiplatform.googleapis.com")
    url = (f"https://{host}/v1/projects/{GCP_PROJECT}/locations/{GCP_LOCATION}"
           f"/publishers/anthropic/models/{model_id}:rawPredict")
    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": prompt}],
    }
    body = json.dumps(payload).encode()
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=body, headers={
                "Authorization": f"Bearer {_vertex_access_token()}",
                "Content-Type": "application/json",
            })
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=90, context=_ssl_ctx()) as r:
                d = json.loads(r.read())
            dt = time.time() - t0
            text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
            um = d.get("usage", {})
            inp, out = um.get("input_tokens", 0), um.get("output_tokens", 0)
            p = PRICING.get(model_id, PRICING[SONNET_ID])
            cost = round(inp / 1e6 * p["input"] + out / 1e6 * p["output"], 6)
            usage = {"input": inp, "output": out, "total_tokens": inp + out, "as_run_usd": cost}
            return text, usage, dt
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(2)

_gemini_client = None
def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    return _gemini_client

THINKING_LEVEL_HEADROOM = {"off": 0, "minimal": 2048, "low": 4096, "medium": 8192, "high": 16384}

def gemini_call(model_id, prompt, max_tokens=2560, thinking_level=None):
    from google.genai import types
    if thinking_level in ("off", "disabled"):
        thinking_level = None

    if thinking_level:
        tc = types.ThinkingConfig(thinking_level=thinking_level.upper())
        max_tokens = max_tokens + THINKING_LEVEL_HEADROOM.get(thinking_level.lower(), 4096)
    else:
        tc = types.ThinkingConfig(thinking_budget=0)

    cfg = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        thinking_config=tc,
        http_options=types.HttpOptions(timeout=90000)
    )

    for attempt in range(3):
        try:
            t0 = time.time()
            resp = _gemini().models.generate_content(
                model=model_id, contents=prompt, config=cfg
            )
            dt = time.time() - t0
            m = resp.usage_metadata
            inp = m.prompt_token_count or 0
            out = (m.candidates_token_count or 0) + (m.thoughts_token_count or 0)
            p = PRICING[model_id]
            cost = round(inp / 1e6 * p["input"] + out / 1e6 * p["output"], 6)
            usage = {"input": inp, "output": out, "total_tokens": m.total_token_count or (inp+out), "as_run_usd": cost}
            return (resp.text or ""), usage, dt
        except Exception as e:
            if "504" in str(e) or "DEADLINE_EXCEEDED" in str(e):
                print(f"\n[Warning] 504 DEADLINE_EXCEEDED on thinking_level={thinking_level}. Retrying with thinking_level='MEDIUM'...")
                cfg.thinking_config = types.ThinkingConfig(thinking_level="MEDIUM")
            if attempt == 2:
                raise e
            time.sleep(2)



def dispatch(model_id, prompt, max_tokens=2560, thinking_level=None):
    if model_id.startswith("gemini"):
        return gemini_call(model_id, prompt, max_tokens, thinking_level)
    return claude_api_call(model_id, prompt, max_tokens, thinking_level)

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

# --- PATTERN IMPLEMENTATIONS ---

def pattern_1_dynamic_router(problem):
    """
    Pattern 1: 3.5-Flash (thinking_level='OFF') acts as Router / Classifier.
    If classified SIMPLE -> 3.1-Flash-Lite executes directly.
    If classified COMPLEX -> 3.5-Flash (thinking_level='MEDIUM') plans and 3.1-Lite executes.
    """
    router_prompt = (
        "You are an AI code complexity router. Analyze the Python problem below (libraries + entry point + docstring).\n"
        "Output ONLY one word: 'SIMPLE' if it is basic data manipulation or standard syntax, or 'COMPLEX' if it requires intricate algorithm / scikit-learn / multi-library coordination.\n\n"
        f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    )
    r_text, r_usage, _ = dispatch(GEMINI_FLASH_ID, router_prompt, max_tokens=10, thinking_level="off")
    is_complex = "COMPLEX" in r_text.upper()

    tot_usd = r_usage["as_run_usd"]
    tot_out = r_usage["output"]

    if not is_complex:
        # Simple path: direct 3.1-Flash-Lite
        prompt = "You are an expert Python programmer. Output ONLY one ```python code block.\n\n" + problem["complete_prompt"]
        text, usage, _ = dispatch(GEMINI_FLASH_LITE_ID, prompt)
        tot_usd += usage["as_run_usd"]
        tot_out += usage["output"]
        code = extract_code(text)
    else:
        # Complex path: 3.5-Flash (MEDIUM) plans -> 3.1-Lite executes
        adv_prompt = "Provide concise implementation guidance (under 200 words). Do NOT output code.\n\n" + problem["complete_prompt"]
        guidance, adv_usage, _ = dispatch(GEMINI_FLASH_ID, adv_prompt, max_tokens=1024, thinking_level="medium")
        tot_usd += adv_usage["as_run_usd"]
        tot_out += adv_usage["output"]

        exec_prompt = f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nAdvisor guidance:\n{guidance}\n\nWrite complete solution. Output ONLY python code block."
        sol, exec_usage, _ = dispatch(GEMINI_FLASH_LITE_ID, exec_prompt)
        tot_usd += exec_usage["as_run_usd"]
        tot_out += exec_usage["output"]
        code = extract_code(sol)

    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)
    return {"passed": passed, "as_run_usd": round(tot_usd, 6), "output_tokens": tot_out, "code": code, "error": "" if passed else err}


def pattern_2_dual_perspective_advisors(problem):
    """
    Pattern 2: Dual 3.5-Flash Advisors:
      Advisor 1 (3.5-Flash LOW): Focuses on Algorithm & Edge Cases.
      Advisor 2 (3.5-Flash OFF): Focuses on API Contracts & Exact Return Types.
      Executor (3.1-Flash-Lite): Merges both and writes code.
    """
    adv1_prompt = "Advisor 1 (Algorithm & Edge Cases): Identify core algorithm and 3 potential edge-case pitfalls. Under 150 words. No code.\n\n" + problem["complete_prompt"]
    g1, u1, _ = dispatch(GEMINI_FLASH_ID, adv1_prompt, max_tokens=512, thinking_level="low")

    adv2_prompt = "Advisor 2 (API Contract): List exact library functions, return types, and exception behavior required. Under 150 words. No code.\n\n" + problem["complete_prompt"]
    g2, u2, _ = dispatch(GEMINI_FLASH_ID, adv2_prompt, max_tokens=512, thinking_level="off")

    exec_prompt = (
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Algorithm Guidance:\n{g1}\n\n"
        f"API Contract Guidance:\n{g2}\n\n"
        "Write complete solution. Output ONLY one ```python code block."
    )
    sol, u_exec, _ = dispatch(GEMINI_FLASH_LITE_ID, exec_prompt)

    tot_usd = round(u1["as_run_usd"] + u2["as_run_usd"] + u_exec["as_run_usd"], 6)
    tot_out = u1["output"] + u2["output"] + u_exec["output"]
    code = extract_code(sol)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)
    return {"passed": passed, "as_run_usd": tot_usd, "output_tokens": tot_out, "code": code, "error": "" if passed else err}


def pattern_3_tdd_harness(problem):
    """
    Pattern 3: 3.5-Flash (LOW) generates synthetic micro-unit-tests.
    3.1-Flash-Lite generates candidate code and runs against synthetic tests locally.
    """
    test_gen_prompt = (
        "Given the Python problem below, write 2 concise unittest assertion statements (e.g. `assert task_func(...) == ...`) "
        "checking edge-case behavior. Output ONLY valid python assert statements.\n\n" + problem["complete_prompt"]
    )
    synth_tests, u_test, _ = dispatch(GEMINI_FLASH_ID, test_gen_prompt, max_tokens=512, thinking_level="low")

    exec_prompt = (
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Verify your code against these edge-case assertions before returning:\n```python\n{synth_tests}\n```\n\n"
        "Write complete solution. Output ONLY one ```python code block."
    )
    sol, u_exec, _ = dispatch(GEMINI_FLASH_LITE_ID, exec_prompt)

    tot_usd = round(u_test["as_run_usd"] + u_exec["as_run_usd"], 6)
    tot_out = u_test["output"] + u_exec["output"]
    code = extract_code(sol)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)
    return {"passed": passed, "as_run_usd": tot_usd, "output_tokens": tot_out, "code": code, "error": "" if passed else err}


def pattern_4_frontier_escalation_shield(problem):
    """
    Pattern 4: 3.1-Lite -> 3.5-Flash (LOW) Cheap Repair -> Sonnet-5 Frontier Repair.
    3.5-Flash acts as a 'shield' for Sonnet-5.
    """
    # 1. 3.1-Lite initial
    prompt = SOLVER_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\nWrite complete solution."
    sol1, u1, _ = dispatch(GEMINI_FLASH_LITE_ID, prompt)
    tot_usd, tot_out = u1["as_run_usd"], u1["output"]
    code = extract_code(sol1)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    if not passed:
        # 2. Tier 2: 3.5-Flash (LOW) Repair Shield
        r1_prompt = (
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Current code:\n```python\n{code}\n```\n\n"
            f"Unittest error:\n```\n{err[-2000:]}\n```\n\nFix bug. Output complete python code."
        )
        sol2, u2, _ = dispatch(GEMINI_FLASH_ID, r1_prompt, thinking_level="low")
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        code2 = extract_code(sol2)
        if not missing_code_error(code2, problem["entry_point"]):
            code = code2
            passed, err = run_bigcodebench(problem, code)

    if not passed:
        # 3. Tier 3: Sonnet-5 Escalation (Only if 3.5-Flash failed)
        r2_prompt = (
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Current code:\n```python\n{code}\n```\n\n"
            f"Unittest error:\n```\n{err[-2000:]}\n```\n\nFix bug. Output complete python code."
        )
        sol3, u3, _ = dispatch(SONNET_ID, r2_prompt)
        tot_usd += u3["as_run_usd"]
        tot_out += u3["output"]
        code3 = extract_code(sol3)
        if not missing_code_error(code3, problem["entry_point"]):
            code = code3
            passed, err = run_bigcodebench(problem, code)

    return {"passed": passed, "as_run_usd": round(tot_usd, 6), "output_tokens": tot_out, "code": code, "error": "" if passed else err}


def pattern_5_peer_reviewer_auditor(problem):
    """
    Pattern 5: Sonnet-5 Generates -> 3.5-Flash (MEDIUM) Audits & Critiques -> Sonnet-5 Revises -> Execute.
    """
    # Step 1: Initial Gen (Sonnet-5)
    sol1, u1, _ = dispatch(SONNET_ID, SOLVER_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\nWrite complete solution.")
    code1 = extract_code(sol1)

    # Step 2: 3.5-Flash Code Auditor
    audit_prompt = (
        "You are a senior code reviewer. Review the candidate code below for subtle logic errors, missing imports, or incorrect return types.\n"
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Candidate Code:\n```python\n{code1}\n```\n\n"
        "If correct, output 'LGTM'. If flawed, output 2 concise bullet points describing the bug."
    )
    critique, u_audit, _ = dispatch(GEMINI_FLASH_ID, audit_prompt, thinking_level="medium")

    tot_usd = u1["as_run_usd"] + u_audit["as_run_usd"]
    tot_out = u1["output"] + u_audit["output"]

    if "LGTM" in critique.upper():
        code = code1
    else:
        # Step 3: Sonnet-5 Revision
        rev_prompt = (
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Draft Code:\n```python\n{code1}\n```\n\n"
            f"Reviewer Critique:\n{critique}\n\nFix the code based on critique. Output ONLY one python code block."
        )
        sol2, u_rev, _ = dispatch(SONNET_ID, rev_prompt)
        tot_usd += u_rev["as_run_usd"]
        tot_out += u_rev["output"]
        code = extract_code(sol2)

    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)
    return {"passed": passed, "as_run_usd": round(tot_usd, 6), "output_tokens": tot_out, "code": code, "error": "" if passed else err}


def pattern_6_tiered_thinking_ramping(problem):
    """
    Pattern 6: 3.5-Flash Single-Model Pipeline with Ramping Thinking:
      Attempt 1: 3.5-Flash (thinking_level='OFF')
      Attempt 2 (if fail): 3.5-Flash (thinking_level='LOW')
      Attempt 3 (if fail): 3.5-Flash (thinking_level='HIGH')
    """
    prompt = SOLVER_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\nWrite complete solution."

    # Attempt 1 (OFF)
    sol1, u1, _ = dispatch(GEMINI_FLASH_ID, prompt, thinking_level="off")
    tot_usd, tot_out = u1["as_run_usd"], u1["output"]
    code = extract_code(sol1)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    if not passed:
        # Attempt 2 (LOW)
        r1_prompt = (
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Current code:\n```python\n{code}\n```\n\n"
            f"Unittest error:\n```\n{err[-2000:]}\n```\n\nFix bug. Output complete python code."
        )
        sol2, u2, _ = dispatch(GEMINI_FLASH_ID, r1_prompt, thinking_level="low")
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        code2 = extract_code(sol2)
        if not missing_code_error(code2, problem["entry_point"]):
            code = code2
            passed, err = run_bigcodebench(problem, code)

    if not passed:
        # Attempt 3 (HIGH)
        r2_prompt = (
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Current code:\n```python\n{code}\n```\n\n"
            f"Unittest error:\n```\n{err[-2000:]}\n```\n\nFix bug with deep reasoning. Output complete python code."
        )
        sol3, u3, _ = dispatch(GEMINI_FLASH_ID, r2_prompt, thinking_level="medium")
        tot_usd += u3["as_run_usd"]
        tot_out += u3["output"]
        code3 = extract_code(sol3)
        if not missing_code_error(code3, problem["entry_point"]):
            code = code3
            passed, err = run_bigcodebench(problem, code)

    return {"passed": passed, "as_run_usd": round(tot_usd, 6), "output_tokens": tot_out, "code": code, "error": "" if passed else err}


# --- DRIVER ---
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="Number of tasks (default: 10)")
    args = parser.parse_args()

    problems = load_problems()
    task_ids = list(problems.keys())[:args.n]

    patterns = [
        ("Pattern 1: Dynamic Thinking Router (3.5-Flash OFF -> 3.1-Lite/3.5-Flash Medium)", pattern_1_dynamic_router),
        ("Pattern 2: Dual-Perspective 3.5-Flash Advisors (LOW + OFF -> 3.1-Lite)", pattern_2_dual_perspective_advisors),
        ("Pattern 3: TDD Harness (3.5-Flash Synthetic Tests -> 3.1-Lite)", pattern_3_tdd_harness),
        ("Pattern 4: Frontier Escalation Shield (3.1-Lite -> 3.5-Flash LOW -> Sonnet-5)", pattern_4_frontier_escalation_shield),
        ("Pattern 5: Peer Reviewer (Sonnet-5 Gen -> 3.5-Flash MEDIUM Auditor)", pattern_5_peer_reviewer_auditor),
        ("Pattern 6: Tiered Thinking Ramping (3.5-Flash OFF -> LOW -> HIGH)", pattern_6_tiered_thinking_ramping),
    ]

    summaries = []
    for name, fn in patterns:
        print(f"\n=======================================================")
        print(f"RUNNING PATTERN: {name}")
        print(f"=======================================================")
        task_results = []
        for i, tid in enumerate(task_ids, 1):
            p = problems[tid]
            print(f"[{i}/{len(task_ids)}] {tid} ({p['entry_point']}) ... ", end="", flush=True)
            r = fn(p)
            status = "PASS" if r["passed"] else "FAIL"
            print(f"{status} | cost=${r['as_run_usd']:.5f} | out_tok={r['output_tokens']}")
            task_results.append(r)

        n = len(task_results)
        passed_cnt = sum(1 for r in task_results if r["passed"])
        tot_cost = sum(r["as_run_usd"] for r in task_results)
        avg_out = sum(r["output_tokens"] for r in task_results) / n if n else 0
        cps = (tot_cost / passed_cnt) if passed_cnt > 0 else -1.0

        summaries.append({
            "name": name,
            "n": n,
            "passed": passed_cnt,
            "pass_rate": round(passed_cnt / n, 3),
            "total_as_run_usd": round(tot_cost, 4),
            "cost_per_solved_usd": round(cps, 4) if cps >= 0 else -1.0,
            "avg_output_tokens": round(avg_out, 1)
        })

    out_file = os.path.join(RESULTS_DIR, "results_gemini35_advanced_patterns.json")
    with open(out_file, "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n" + "=" * 92)
    print("ADVANCED 3.5-FLASH PATTERN COMPARISON TABLE (First 10 BigCodeBench-Hard Tasks)")
    print("=" * 92)
    print(f"{'Integration Pattern':<55} | {'Pass Rate':<10} | {'Total Cost ($)':<12} | {'$/Solved':<10}")
    print("-" * 92)
    for s in summaries:
        cps_str = f"${s['cost_per_solved_usd']:.4f}" if s['cost_per_solved_usd'] >= 0 else "N/A"
        print(f"{s['name']:<55} | {s['passed']}/{s['n']} ({s['pass_rate']:.0%})  | ${s['total_as_run_usd']:<11.4f} | {cps_str:<10}")
    print("=" * 92)
    print(f"\nSaved detailed JSON results to: {out_file}")

if __name__ == "__main__":
    main()
