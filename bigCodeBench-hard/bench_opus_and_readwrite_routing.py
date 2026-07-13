#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0
"""
Benchmark for:
1. Dynamic Read-Heavy vs Write-Heavy Task Classification & Routing Architectures.
2. 3-Level Escalation & Multi-Model Pipelines incorporating claude-opus-4-8.

Configurations Tested:
  - Arch 1: Read/Write Task Router (Input Prompt Length & Library Density Routing)
  - Arch 2: 3-Tier Escalation (3.1-Lite -> 3.5-Flash LOW -> Opus-4.8)
  - Arch 3: 3-Tier Escalation (3.1-Lite -> Sonnet-5 -> Opus-4.8)
  - Arch 4: Opus-4.8 Advisor + 3.1-Lite Executor + 3.5-Flash LOW Repair
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
    GEMINI_FLASH_ID:      {"input": 1.50, "output": 9.00,  "cache_read": 0.15,  "cache_write": 1.50},
    GEMINI_FLASH_LITE_ID: {"input": 0.25, "output": 1.50,  "cache_read": 0.025, "cache_write": 0.25},
}

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

# --- API Clients with Retry & Timeout ---
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
            p = PRICING[model_id]
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

# --- ARCHITECTURES TO TEST ---

def arch_readwrite_dynamic_router(problem):
    """
    Arch 1: Dynamic Read-Heavy vs Write-Heavy Task Router.
    Analyzes task input length and library complexity:
    - Read-Heavy (prompt len > 1200 chars or > 2 libs): Opus-4.8 Planner -> 3.1-Lite Executor.
    - Write-Heavy (prompt len <= 1200 chars): Direct 3.1-Lite Executor.
    """
    prompt = problem["complete_prompt"]
    libs = problem.get("libs") or []
    is_read_heavy = (len(prompt) > 1200) or (len(libs) >= 3)

    if is_read_heavy:
        adv_prompt = (
            "You are a senior ADVISOR. Provide concise implementation guidance (under 150 words) for an executor.\n"
            f"Problem:\n```python\n{prompt}\n```"
        )
        guidance, adv_usage, _ = dispatch(OPUS_ID, adv_prompt, max_tokens=512)
        tot_usd, tot_out = adv_usage["as_run_usd"], adv_usage["output"]

        exec_prompt = f"Problem:\n```python\n{prompt}\n```\n\nAdvisor guidance:\n{guidance}\n\nWrite complete solution. Output ONLY python code block."
        sol, exec_usage, _ = dispatch(GEMINI_FLASH_LITE_ID, exec_prompt)
        tot_usd += exec_usage["as_run_usd"]
        tot_out += exec_usage["output"]
        code = extract_code(sol)
    else:
        exec_prompt = "You are an expert Python programmer. Output ONLY python code block.\n\n" + prompt
        sol, exec_usage, _ = dispatch(GEMINI_FLASH_LITE_ID, exec_prompt)
        tot_usd, tot_out = exec_usage["as_run_usd"], exec_usage["output"]
        code = extract_code(sol)

    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)
    return {"passed": passed, "as_run_usd": round(tot_usd, 6), "output_tokens": tot_out, "is_read_heavy": is_read_heavy, "code": code, "error": "" if passed else err}


def arch_3level_escalation_flash_opus(problem):
    """
    Arch 2: 3-Level Escalation:
      Level 1: 3.1-Flash-Lite (Initial Gen)
      Level 2: 3.5-Flash (LOW Thinking) (Cheap Repair)
      Level 3: claude-opus-4-8 (Frontier Repair)
    """
    prompt = "You are an expert Python programmer. Output ONLY python code block.\n\n" + problem["complete_prompt"]
    sol1, u1, _ = dispatch(GEMINI_FLASH_LITE_ID, prompt)
    tot_usd, tot_out = u1["as_run_usd"], u1["output"]
    code = extract_code(sol1)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    if not passed:
        # Level 2: 3.5-Flash LOW
        r1_prompt = f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nCurrent code:\n```python\n{code}\n```\n\nUnittest error:\n```\n{err[-2000:]}\n```\n\nFix bug. Output complete python code."
        sol2, u2, _ = dispatch(GEMINI_FLASH_ID, r1_prompt, thinking_level="low")
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        code2 = extract_code(sol2)
        if not missing_code_error(code2, problem["entry_point"]):
            code = code2
            passed, err = run_bigcodebench(problem, code)

    if not passed:
        # Level 3: claude-opus-4-8
        r2_prompt = f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nCurrent code:\n```python\n{code}\n```\n\nUnittest error:\n```\n{err[-2000:]}\n```\n\nFix bug with deep reasoning. Output complete python code."
        sol3, u3, _ = dispatch(OPUS_ID, r2_prompt)
        tot_usd += u3["as_run_usd"]
        tot_out += u3["output"]
        code3 = extract_code(sol3)
        if not missing_code_error(code3, problem["entry_point"]):
            code = code3
            passed, err = run_bigcodebench(problem, code)

    return {"passed": passed, "as_run_usd": round(tot_usd, 6), "output_tokens": tot_out, "code": code, "error": "" if passed else err}


def arch_3level_escalation_sonnet_opus(problem):
    """
    Arch 3: 3-Level Escalation:
      Level 1: 3.1-Flash-Lite
      Level 2: claude-sonnet-5
      Level 3: claude-opus-4-8
    """
    prompt = "You are an expert Python programmer. Output ONLY python code block.\n\n" + problem["complete_prompt"]
    sol1, u1, _ = dispatch(GEMINI_FLASH_LITE_ID, prompt)
    tot_usd, tot_out = u1["as_run_usd"], u1["output"]
    code = extract_code(sol1)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    if not passed:
        # Level 2: Sonnet-5
        r1_prompt = f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nCurrent code:\n```python\n{code}\n```\n\nUnittest error:\n```\n{err[-2000:]}\n```\n\nFix bug. Output complete python code."
        sol2, u2, _ = dispatch(SONNET_ID, r1_prompt)
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        code2 = extract_code(sol2)
        if not missing_code_error(code2, problem["entry_point"]):
            code = code2
            passed, err = run_bigcodebench(problem, code)

    if not passed:
        # Level 3: Opus-4.8
        r2_prompt = f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nCurrent code:\n```python\n{code}\n```\n\nUnittest error:\n```\n{err[-2000:]}\n```\n\nFix bug with deep reasoning. Output complete python code."
        sol3, u3, _ = dispatch(OPUS_ID, r2_prompt)
        tot_usd += u3["as_run_usd"]
        tot_out += u3["output"]
        code3 = extract_code(sol3)
        if not missing_code_error(code3, problem["entry_point"]):
            code = code3
            passed, err = run_bigcodebench(problem, code)

    return {"passed": passed, "as_run_usd": round(tot_usd, 6), "output_tokens": tot_out, "code": code, "error": "" if passed else err}


def arch_opus_advisor_35flash_repair(problem):
    """
    Arch 4: Opus-4.8 Advisor + 3.1-Lite Executor + 3.5-Flash (LOW) Repair.
    Opus-4.8 handles Read-heavy planning, 3.1-Lite handles Write-heavy gen, 3.5-Flash handles Repair.
    """
    adv_prompt = (
        "You are a senior ADVISOR. Provide concise implementation guidance (under 150 words) for an executor.\n"
        f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    )
    guidance, adv_usage, _ = dispatch(OPUS_ID, adv_prompt, max_tokens=512)
    tot_usd, tot_out = adv_usage["as_run_usd"], adv_usage["output"]

    exec_prompt = f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nAdvisor guidance:\n{guidance}\n\nWrite complete solution. Output ONLY python code block."
    sol, exec_usage, _ = dispatch(GEMINI_FLASH_LITE_ID, exec_prompt)
    tot_usd += exec_usage["as_run_usd"]
    tot_out += exec_usage["output"]
    code = extract_code(sol)

    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    if not passed:
        r1_prompt = f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nCurrent code:\n```python\n{code}\n```\n\nUnittest error:\n```\n{err[-2000:]}\n```\n\nFix bug. Output complete python code."
        sol2, u2, _ = dispatch(GEMINI_FLASH_ID, r1_prompt, thinking_level="low")
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        code2 = extract_code(sol2)
        if not missing_code_error(code2, problem["entry_point"]):
            code = code2
            passed, err = run_bigcodebench(problem, code)

    return {"passed": passed, "as_run_usd": round(tot_usd, 6), "output_tokens": tot_out, "code": code, "error": "" if passed else err}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="Number of tasks (default: 10)")
    args = parser.parse_args()

    problems = load_problems()
    task_ids = list(problems.keys())[:args.n]

    all_configs = [
        ("Arch 1: Read/Write Task Router (Opus-4.8/3.1-Lite)", arch_readwrite_dynamic_router),
        ("Arch 2: 3-Tier Escalation (3.1-Lite -> 3.5-Flash LOW -> Opus-4.8)", arch_3level_escalation_flash_opus),
        ("Arch 3: 3-Tier Escalation (3.1-Lite -> Sonnet-5 -> Opus-4.8)", arch_3level_escalation_sonnet_opus),
        ("Arch 4: Opus-4.8 Advisor + 3.1-Lite Exec + 3.5-Flash LOW Repair", arch_opus_advisor_35flash_repair),
    ]

    summaries = []
    for name, fn in all_configs:
        print(f"\n=======================================================")
        print(f"RUNNING CONFIG: {name}")
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

    out_file = os.path.join(RESULTS_DIR, "results_opus_and_readwrite_routing.json")
    with open(out_file, "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n" + "=" * 90)
    print("OPUS-4.8 & READ/WRITE ROUTING COMPARISON TABLE (First 10 BigCodeBench-Hard Tasks)")
    print("=" * 90)
    print(f"{'Configuration':<58} | {'Pass Rate':<10} | {'Total Cost ($)':<12} | {'$/Solved':<10}")
    print("-" * 90)
    for s in summaries:
        cps_str = f"${s['cost_per_solved_usd']:.4f}" if s['cost_per_solved_usd'] >= 0 else "N/A"
        print(f"{s['name']:<58} | {s['passed']}/{s['n']} ({s['pass_rate']:.0%})  | ${s['total_as_run_usd']:<11.4f} | {cps_str:<10}")
    print("=" * 90)
    print(f"\nSaved detailed JSON results to: {out_file}")

if __name__ == "__main__":
    main()
