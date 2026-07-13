#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0
"""
Comprehensive BigCodeBench-Hard Sweet Spot & Architectural Search Harness.

Evaluates:
1. Baseline Single-Model Direct Completion (3.1-Flash-Lite, 3.5-Flash, Sonnet-5, Opus-4.8)
2. Read/Write Split (Advisor-Executor) with various Planners + 3.1-Flash-Lite
3. Generation Offload Cascade (Cheap Gen + Escalation)
4. Error Log Triage Offload
5. Sweet-Spot Hybrid (Read/Write + Triage + Thinking Escalation)
6. NEW Arch F: Dual-Candidate Parallel Execution (Local 0-Cost Selection)
7. NEW Arch G: Contract-Guided Planning (TDSG - Test-Driven Spec Generation)
8. NEW Arch H: Category-Aware Adaptive Repair Routing (Smart Repair)
9. NEW Arch I: Ultra-Sweet Hybrid (Contract-Guided + Dual-Candidate + Smart Repair)

Outputs full per-task tracebacks, token usage, cost accounting ($/solved), and summary matrix.
"""

import os, sys, json, re, subprocess, tempfile, time, argparse, shutil, ssl
import urllib.request, urllib.parse, urllib.error

os.environ["MPLBACKEND"] = "Agg"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Dataset Config ---
BCB_DATASET = "bigcode/bigcodebench-hard"
BCB_CONFIG = "default"
BCB_SPLIT = "v0.1.4"
_KEEP_FIELDS = ("task_id", "complete_prompt", "canonical_solution", "code_prompt", "test", "entry_point", "libs")

# --- Model IDs ---
OPUS_ID = "claude-opus-4-8"
SONNET_ID = "claude-sonnet-5"
GEMINI_FLASH_ID = "gemini-3.5-flash"
GEMINI_FLASH_LITE_ID = "gemini-3.1-flash-lite"
GEMINI_PRO_ID = "gemini-3.1-pro-preview"

GCP_PROJECT = os.environ.get("GCP_PROJECT", "my-argolis-prj")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")

PRICING = {
    OPUS_ID:              {"input": 5.00, "output": 25.00, "cache_read": 0.50,  "cache_write": 6.25},
    SONNET_ID:            {"input": 2.00, "output": 10.00, "cache_read": 0.20,  "cache_write": 2.50},
    GEMINI_FLASH_ID:      {"input": 1.50, "output": 9.00,  "cache_read": 0.15,  "cache_write": 1.50},
    GEMINI_FLASH_LITE_ID: {"input": 0.25, "output": 1.50,  "cache_read": 0.025, "cache_write": 0.25},
    GEMINI_PRO_ID:        {"input": 2.00, "output": 12.00, "cache_read": 0.20,  "cache_write": 2.00},
}

# --- Prompts ---
SOLVER_ROLE = (
    "You are an expert Python programmer. Complete the function below. You are given its imports, "
    "signature, and docstring; several real libraries must be used correctly. Output the COMPLETE "
    "solution: all needed imports and the full function definition, handling edge cases and the "
    "documented return/exception behavior exactly. Output ONLY one ```python code block, no "
    "explanation.\n\n"
)

ADVISOR_ROLE = (
    "You are a senior ADVISOR in an advisor-executor coding system. You do NOT write code. Given "
    "the Python coding problem below (imports + function signature + docstring), which requires "
    "correctly using several real libraries, produce concise, precise implementation GUIDANCE for "
    "a separate executor model: which libraries/APIs to use and in what order, the intended "
    "algorithm, edge cases, and the EXACT documented return values and exceptions to honor. "
    "Under 200 words. Do NOT output any code.\n\n"
)

CONTRACT_ADVISOR_ROLE = (
    "You are a senior SOFTWARE ARCHITECT. Given the Python problem below, produce a structured, "
    "contract-guided specification for a code generator. Output format must be strictly Markdown:\n"
    "1. **APIs & Library Imports**: exact list of functions/classes from pandas/numpy/sklearn/etc.\n"
    "2. **Algorithm Steps**: step-by-step logic\n"
    "3. **Preconditions & Return Contracts**: return types, null handling, empty collection behavior\n"
    "4. **Edge-case Invariants**: explicit conditions (e.g. zero-division, missing keys)\n"
    "Keep total guidance under 250 words. Do NOT output code.\n\n"
)

EXECUTOR_ROLE = (
    "You are an EXECUTOR in an advisor-executor coding system. Given a Python problem and an "
    "advisor's guidance, output the COMPLETE Python solution: all needed imports and the full "
    "function definition, honoring the documented return/exception behavior exactly. Output ONLY "
    "one ```python code block, no explanation.\n\n"
)

REPAIR_ROLE = (
    "You are an expert Python programmer. A candidate solution to the problem below FAILED its "
    "unit tests. Analyze the test error output, find the bug, and fix the code. Output the "
    "COMPLETE corrected solution: all needed imports and the full function definition. Do not "
    "output a diff or a fragment. Output ONLY one ```python code block, no explanation.\n\n"
)

TRIAGE_ROLE = (
    "You are a test-failure triage tool. Compress the Python unittest stderr below into a SHORT "
    "digest (max 12 lines) preserving EXACTLY: (1) each failing test method name, (2) the "
    "exception type and message, (3) assertion diffs with expected vs actual values (truncate "
    "values longer than ~120 chars), and (4) traceback lines inside the candidate solution "
    "(file \"prog.py\") with their line numbers. DROP unittest boilerplate, separators, and "
    "library-internal frames. Copy identifiers and values VERBATIM -- never paraphrase numbers. "
    "Output plain text only, no code fences.\n\nStderr:\n"
)

def _ssl_ctx():
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

# --- API Clients ---
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

_CLAUDE_THINKING_BUDGET = {"low": 2048, "medium": 4096, "high": 8192}

def claude_api_call(model_id, prompt, max_tokens=2560, thinking_level=None, temperature=0.0):
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
    if thinking_level:
        add_b = _CLAUDE_THINKING_BUDGET.get(thinking_level, 4096)
        payload["max_tokens"] = max_tokens + add_b
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": thinking_level}

    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {_vertex_access_token()}",
        "Content-Type": "application/json",
    })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180, context=_ssl_ctx()) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
    um = d.get("usage", {})
    inp, out = um.get("input_tokens", 0), um.get("output_tokens", 0)
    cr, cc = um.get("cache_read_input_tokens", 0), um.get("cache_creation_input_tokens", 0)
    p = PRICING.get(model_id, PRICING[SONNET_ID])
    cost = round(inp / 1e6 * p["input"] + cr / 1e6 * p["cache_read"]
                 + cc / 1e6 * p["cache_write"] + out / 1e6 * p["output"], 6)
    usage = {"input_raw": inp, "output": out, "cache_read": cr, "cache_creation": cc,
             "prompt_tokens": inp + cr + cc, "total_tokens": inp + cr + cc + out,
             "as_run_usd": cost}
    return text, usage, dt

_gemini_client = None

def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    return _gemini_client

_GEMINI_THINKING_HEADROOM = {"minimal": 2048, "low": 4096, "medium": 8192, "high": 16384}

def gemini_call(model_id, prompt, max_tokens=2560, thinking_level=None, temperature=0.0):
    from google.genai import types
    if thinking_level in ("off", "disabled"):
        thinking_level = None
    if thinking_level is None and model_id == GEMINI_FLASH_ID:
        thinking_level = "minimal"

    if thinking_level:
        tc = types.ThinkingConfig(thinking_level=thinking_level.upper())
        max_tokens = max_tokens + _GEMINI_THINKING_HEADROOM.get(thinking_level.lower(), 0)
    else:
        tc = types.ThinkingConfig(thinking_budget=0)

    cfg = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        thinking_config=tc,
        temperature=temperature if temperature > 0 else None
    )

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
    usage = {"input_raw": inp, "output": out, "cache_read": 0, "cache_creation": 0,
             "prompt_tokens": inp, "total_tokens": m.total_token_count or 0,
             "as_run_usd": cost}
    return (resp.text or ""), usage, dt

def dispatch_model(model_id, prompt, max_tokens=2560, thinking_level=None, temperature=0.0):
    if model_id.startswith("gemini"):
        return gemini_call(model_id, prompt, max_tokens, thinking_level, temperature)
    return claude_api_call(model_id, prompt, max_tokens, thinking_level, temperature)

# --- Code Extraction & Unittest Runner ---
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

def triage_error(raw_err, model_id=GEMINI_FLASH_LITE_ID):
    prompt = TRIAGE_ROLE + "```\n" + raw_err + "\n```"
    text, usage, dt = dispatch_model(model_id, prompt, max_tokens=768)
    digest = text.strip() or raw_err[-1200:]
    return digest[:1200], usage, dt

def classify_error(err):
    """Categorize error into Syntax/Import vs Assertion vs Exception/Timeout."""
    if "SyntaxError:" in err or "IndentationError:" in err or "ImportError:" in err or "ModuleNotFoundError:" in err or "NameError:" in err:
        return "SYNTAX_IMPORT"
    elif "AssertionError:" in err or "FAILED (" in err:
        return "ASSERTION"
    elif "timeout:" in err:
        return "TIMEOUT"
    else:
        return "EXCEPTION"

# --- Architecture Handlers ---

def run_arch_single(problem, model_id):
    """Arch A: Single Model Direct Completion."""
    prompt = SOLVER_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\nWrite the complete solution."
    text, usage, dt = dispatch_model(model_id, prompt)
    code = extract_code(text)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)
    return {
        "passed": passed,
        "as_run_usd": usage["as_run_usd"],
        "output_tokens": usage["output"],
        "total_tokens": usage["total_tokens"],
        "seconds": round(dt, 1),
        "code": code,
        "error": "" if passed else err,
    }

def run_arch_read_write(problem, planner_model, executor_model):
    """Arch B: Read/Write Split (Advisor-Executor)."""
    adv_prompt = ADVISOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    guidance, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024)

    exec_prompt = (EXECUTOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                   f"Advisor guidance:\n{guidance}\n\nWrite the complete solution.")
    sol, exec_usage, exec_dt = dispatch_model(executor_model, exec_prompt, max_tokens=2560)
    code = extract_code(sol)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    tot_usd = round(adv_usage["as_run_usd"] + exec_usage["as_run_usd"], 6)
    tot_out = adv_usage["output"] + exec_usage["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage["total_tokens"]

    return {
        "passed": passed,
        "as_run_usd": tot_usd,
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(adv_dt + exec_dt, 1),
        "code": code,
        "guidance": guidance,
        "error": "" if passed else err,
    }

def run_arch_cascade(problem, gen_model, esc_model, max_repairs=2):
    """Arch C: Generation Offload Cascade."""
    prompt = SOLVER_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\nWrite the complete solution."
    text, usage, dt = dispatch_model(gen_model, prompt)
    code = extract_code(text)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]

    loop = 0
    while not passed and loop < max_repairs:
        loop += 1
        target_model = esc_model
        think_level = "low" if target_model == GEMINI_FLASH_ID else None

        repair_prompt = (REPAIR_ROLE + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\n"
                         f"Current solution:\n```python\n{code}\n```\n\n"
                         f"Unit test error:\n```\n{err[-2500:]}\n```\n\n"
                         "Write the complete corrected solution.")
        r_text, r_usage, r_dt = dispatch_model(target_model, repair_prompt, thinking_level=think_level)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]

        new_code = extract_code(r_text)
        guard = missing_code_error(new_code, problem["entry_point"])
        if guard:
            passed, err = False, guard
        else:
            code = new_code
            passed, err = run_bigcodebench(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "code": code,
        "error": "" if passed else err,
    }

def run_arch_hybrid(problem, planner_model=GEMINI_FLASH_ID, executor_model=GEMINI_FLASH_LITE_ID, escalate_model=GEMINI_FLASH_ID):
    """Arch E: Sweet Spot Hybrid (Read/Write + Cheap Repair + Triage + Thinking Escalation)."""
    adv_prompt = ADVISOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    guidance, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024)

    exec_prompt = (EXECUTOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                   f"Advisor guidance:\n{guidance}\n\nWrite the complete solution.")
    sol, exec_usage, exec_dt = dispatch_model(executor_model, exec_prompt, max_tokens=2560)
    code = extract_code(sol)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    tot_usd = adv_usage["as_run_usd"] + exec_usage["as_run_usd"]
    tot_out = adv_usage["output"] + exec_usage["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage["total_tokens"]

    if not passed:
        # Step 3a: Cheap repair by executor
        r1_prompt = (REPAIR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                     f"Current solution:\n```python\n{code}\n```\n\n"
                     f"Unit test error:\n```\n{err[-2500:]}\n```\n\n"
                     "Write the complete corrected solution.")
        r1_text, r1_usage, _ = dispatch_model(executor_model, r1_prompt)
        tot_usd += r1_usage["as_run_usd"]
        tot_out += r1_usage["output"]
        tot_tok += r1_usage["total_tokens"]

        new_code = extract_code(r1_text)
        guard = missing_code_error(new_code, problem["entry_point"])
        if not guard:
            code = new_code
            passed, err = run_bigcodebench(problem, code)

    if not passed:
        # Step 3b: Log triage + Escalated Thinking repair
        digest, tr_usage, _ = triage_error(err, model_id=executor_model)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        esc_prompt = (REPAIR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                      f"Current solution:\n```python\n{code}\n```\n\n"
                      f"Triaged Error Digest:\n```\n{digest}\n```\n\n"
                      "Write the complete corrected solution.")
        esc_text, esc_usage, _ = dispatch_model(escalate_model, esc_prompt, thinking_level="low")
        tot_usd += esc_usage["as_run_usd"]
        tot_out += esc_usage["output"]
        tot_tok += esc_usage["total_tokens"]

        new_code = extract_code(esc_text)
        guard = missing_code_error(new_code, problem["entry_point"])
        if not guard:
            code = new_code
            passed, err = run_bigcodebench(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "code": code,
        "error": "" if passed else err,
    }

# --- NEW ARCHITECTURES ---

def run_arch_f_dual_candidate(problem, planner_model=GEMINI_FLASH_ID, executor_model=GEMINI_FLASH_LITE_ID):
    """
    NEW Arch F: Dual-Candidate Parallel Execution (Local 0-Cost Selection).
    Generates 2 candidate solutions with slightly different temperature/prompting from 3.1-Flash-Lite.
    Executes both against local unittests (0 cost). If either passes, return PASS.
    """
    adv_prompt = ADVISOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    guidance, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024)

    exec_prompt = (EXECUTOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                   f"Advisor guidance:\n{guidance}\n\nWrite the complete solution.")

    # Candidate 1 (t=0.0)
    sol1, exec_usage1, exec_dt1 = dispatch_model(executor_model, exec_prompt, max_tokens=2560, temperature=0.0)
    code1 = extract_code(sol1)
    guard1 = missing_code_error(code1, problem["entry_point"])
    passed1, err1 = (False, guard1) if guard1 else run_bigcodebench(problem, code1)

    tot_usd = adv_usage["as_run_usd"] + exec_usage1["as_run_usd"]
    tot_out = adv_usage["output"] + exec_usage1["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage1["total_tokens"]

    if passed1:
        return {
            "passed": True,
            "as_run_usd": round(tot_usd, 6),
            "output_tokens": tot_out,
            "total_tokens": tot_tok,
            "seconds": round(adv_dt + exec_dt1, 1),
            "code": code1,
            "error": "",
            "candidate_used": 1,
        }

    # Candidate 2 (t=0.3 with explicit edge-case emphasis)
    exec_prompt2 = (exec_prompt + "\n\nPay EXTRA attention to edge cases, empty input handling, and exact return types.")
    sol2, exec_usage2, exec_dt2 = dispatch_model(executor_model, exec_prompt2, max_tokens=2560, temperature=0.3)
    code2 = extract_code(sol2)
    guard2 = missing_code_error(code2, problem["entry_point"])
    passed2, err2 = (False, guard2) if guard2 else run_bigcodebench(problem, code2)

    tot_usd += exec_usage2["as_run_usd"]
    tot_out += exec_usage2["output"]
    tot_tok += exec_usage2["total_tokens"]

    return {
        "passed": passed2,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(adv_dt + exec_dt1 + exec_dt2, 1),
        "code": code2 if passed2 else code1,
        "error": "" if passed2 else err1,
        "candidate_used": 2 if passed2 else 1,
    }

def run_arch_g_contract_guided(problem, planner_model=GEMINI_FLASH_ID, executor_model=GEMINI_FLASH_LITE_ID):
    """
    NEW Arch G: Contract-Guided / Test-Driven Spec Generation (TDSG).
    Planner outputs structured Markdown contract (imports, preconditions, return types, edge-case invariants).
    Executor generates code adhering strictly to contract.
    """
    adv_prompt = CONTRACT_ADVISOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    contract, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024)

    exec_prompt = (
        "You are an EXECUTOR in a contract-guided software engineering system.\n"
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Target Implementation Contract:\n{contract}\n\n"
        "Implement the solution adhering strictly to the contract and documented behavior. Output ONLY one ```python code block."
    )
    sol, exec_usage, exec_dt = dispatch_model(executor_model, exec_prompt, max_tokens=2560)
    code = extract_code(sol)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    tot_usd = round(adv_usage["as_run_usd"] + exec_usage["as_run_usd"], 6)
    tot_out = adv_usage["output"] + exec_usage["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage["total_tokens"]

    return {
        "passed": passed,
        "as_run_usd": tot_usd,
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(adv_dt + exec_dt, 1),
        "code": code,
        "contract": contract,
        "error": "" if passed else err,
    }

def run_arch_h_smart_repair(problem, planner_model=GEMINI_FLASH_ID, executor_model=GEMINI_FLASH_LITE_ID, escalate_model=GEMINI_FLASH_ID):
    """
    NEW Arch H: Category-Aware Adaptive Repair Routing.
    Initial gen via Read/Write. On failure, classifies error:
      - SYNTAX_IMPORT -> Cheap 3.1-Flash-Lite repair (no thinking)
      - ASSERTION -> Log Triage -> 3.5-Flash (thinking="low") or Sonnet-5
      - TIMEOUT/EXCEPTION -> Re-plan guidance from Planner -> 3.1-Flash-Lite re-exec
    """
    adv_prompt = ADVISOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    guidance, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024)

    exec_prompt = (EXECUTOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                   f"Advisor guidance:\n{guidance}\n\nWrite the complete solution.")
    sol, exec_usage, exec_dt = dispatch_model(executor_model, exec_prompt, max_tokens=2560)
    code = extract_code(sol)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    tot_usd = adv_usage["as_run_usd"] + exec_usage["as_run_usd"]
    tot_out = adv_usage["output"] + exec_usage["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage["total_tokens"]

    if not passed:
        cat = classify_error(err)
        if cat == "SYNTAX_IMPORT":
            # Quick cheap repair
            r_prompt = (REPAIR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                        f"Current solution:\n```python\n{code}\n```\n\n"
                        f"Syntax/Import Error:\n```\n{err[-1500:]}\n```\n\n"
                        "Fix the syntax/import error. Output complete corrected solution.")
            r_text, r_usage, _ = dispatch_model(executor_model, r_prompt)
            tot_usd += r_usage["as_run_usd"]
            tot_out += r_usage["output"]
            tot_tok += r_usage["total_tokens"]
            new_code = extract_code(r_text)
            if not missing_code_error(new_code, problem["entry_point"]):
                code = new_code
                passed, err = run_bigcodebench(problem, code)
        elif cat == "ASSERTION":
            # Triage + Thinking Escalation
            digest, tr_usage, _ = triage_error(err, model_id=executor_model)
            tot_usd += tr_usage["as_run_usd"]
            tot_out += tr_usage["output"]
            tot_tok += tr_usage["total_tokens"]

            r_prompt = (REPAIR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                        f"Current solution:\n```python\n{code}\n```\n\n"
                        f"Assertion Failure Digest:\n```\n{digest}\n```\n\n"
                        "Correct the logic failure. Output complete corrected solution.")
            r_text, r_usage, _ = dispatch_model(escalate_model, r_prompt, thinking_level="low")
            tot_usd += r_usage["as_run_usd"]
            tot_out += r_usage["output"]
            tot_tok += r_usage["total_tokens"]
            new_code = extract_code(r_text)
            if not missing_code_error(new_code, problem["entry_point"]):
                code = new_code
                passed, err = run_bigcodebench(problem, code)
        else: # TIMEOUT or EXCEPTION
            # Re-plan with error context
            re_adv_prompt = (ADVISOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                            f"Previous failed code resulted in {cat}:\n```\n{err[-1500:]}\n```\n"
                            "Provide NEW revised guidance to avoid this exception/timeout.")
            re_guidance, re_adv_usage, _ = dispatch_model(planner_model, re_adv_prompt, max_tokens=1024)
            tot_usd += re_adv_usage["as_run_usd"]
            tot_out += re_adv_usage["output"]
            tot_tok += re_adv_usage["total_tokens"]

            r_prompt = (EXECUTOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                        f"Revised Guidance:\n{re_guidance}\n\nWrite the complete solution.")
            r_text, r_usage, _ = dispatch_model(executor_model, r_prompt, max_tokens=2560)
            tot_usd += r_usage["as_run_usd"]
            tot_out += r_usage["output"]
            tot_tok += r_usage["total_tokens"]
            new_code = extract_code(r_text)
            if not missing_code_error(new_code, problem["entry_point"]):
                code = new_code
                passed, err = run_bigcodebench(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "code": code,
        "error": "" if passed else err,
    }

def run_arch_i_ultra_sweet(problem, planner_model=GEMINI_FLASH_ID, executor_model=GEMINI_FLASH_LITE_ID, escalate_model=GEMINI_FLASH_ID):
    """
    NEW Arch I: Ultra-Sweet Hybrid (Contract-Guided + Dual-Candidate + Category-Aware Smart Repair).
    Integrates Contract-Guided Planning (Arch G) with Dual-Candidate Parallel Execution (Arch F)
    and Category-Aware Smart Repair (Arch H).
    """
    # 1. Contract-Guided Planning
    adv_prompt = CONTRACT_ADVISOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    contract, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024)

    exec_prompt1 = (
        "You are an EXECUTOR in a contract-guided software engineering system.\n"
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Target Implementation Contract:\n{contract}\n\n"
        "Implement the solution adhering strictly to the contract and documented behavior. Output ONLY one ```python code block."
    )

    # 2a. Candidate 1 (t=0.0)
    sol1, exec_usage1, _ = dispatch_model(executor_model, exec_prompt1, max_tokens=2560, temperature=0.0)
    code1 = extract_code(sol1)
    guard1 = missing_code_error(code1, problem["entry_point"])
    passed1, err1 = (False, guard1) if guard1 else run_bigcodebench(problem, code1)

    tot_usd = adv_usage["as_run_usd"] + exec_usage1["as_run_usd"]
    tot_out = adv_usage["output"] + exec_usage1["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage1["total_tokens"]

    if passed1:
        return {
            "passed": True,
            "as_run_usd": round(tot_usd, 6),
            "output_tokens": tot_out,
            "total_tokens": tot_tok,
            "code": code1,
            "error": "",
        }

    # 2b. Candidate 2 (t=0.2 with edge case focus)
    exec_prompt2 = exec_prompt1 + "\n\nPay extra attention to boundary conditions, zero inputs, and exact exception types."
    sol2, exec_usage2, _ = dispatch_model(executor_model, exec_prompt2, max_tokens=2560, temperature=0.2)
    code2 = extract_code(sol2)
    guard2 = missing_code_error(code2, problem["entry_point"])
    passed2, err2 = (False, guard2) if guard2 else run_bigcodebench(problem, code2)

    tot_usd += exec_usage2["as_run_usd"]
    tot_out += exec_usage2["output"]
    tot_tok += exec_usage2["total_tokens"]

    if passed2:
        return {
            "passed": True,
            "as_run_usd": round(tot_usd, 6),
            "output_tokens": tot_out,
            "total_tokens": tot_tok,
            "code": code2,
            "error": "",
        }

    # If both failed, pick candidate with cleaner error
    code = code2 if "SyntaxError" not in err2 else code1
    err = err2 if "SyntaxError" not in err2 else err1

    # 3. Category-Aware Smart Repair
    cat = classify_error(err)
    if cat == "SYNTAX_IMPORT":
        r_prompt = (REPAIR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                    f"Current solution:\n```python\n{code}\n```\n\n"
                    f"Syntax/Import Error:\n```\n{err[-1500:]}\n```\n\n"
                    "Fix the syntax/import error. Output complete corrected solution.")
        r_text, r_usage, _ = dispatch_model(executor_model, r_prompt)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        new_code = extract_code(r_text)
        if not missing_code_error(new_code, problem["entry_point"]):
            code = new_code
            passed, err = run_bigcodebench(problem, code)
    else:
        # Triage + Thinking Escalation
        digest, tr_usage, _ = triage_error(err, model_id=executor_model)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        r_prompt = (REPAIR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                    f"Current solution:\n```python\n{code}\n```\n\n"
                    f"Triaged Error Digest:\n```\n{digest}\n```\n\n"
                    "Correct the failure. Output complete corrected solution.")
        r_text, r_usage, _ = dispatch_model(escalate_model, r_prompt, thinking_level="low")
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        new_code = extract_code(r_text)
        if not missing_code_error(new_code, problem["entry_point"]):
            code = new_code
            passed, err = run_bigcodebench(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "code": code,
        "error": "" if passed else err,
    }


# --- Driver ---
def run_config(cfg_name, arch, tasks, problems, **kwargs):
    print(f"\n=======================================================")
    print(f"RUNNING CONFIG: {cfg_name} (Arch={arch.upper()})")
    print(f"=======================================================")
    task_results = []
    for i, tid in enumerate(tasks, 1):
        p = problems[tid]
        print(f"[{i}/{len(tasks)}] {tid} ({p['entry_point']}) ...", end="", flush=True)

        if arch == "single":
            r = run_arch_single(p, model_id=kwargs.get("model", GEMINI_FLASH_ID))
        elif arch == "read-write":
            r = run_arch_read_write(p, planner_model=kwargs.get("planner", GEMINI_FLASH_ID), executor_model=kwargs.get("executor", GEMINI_FLASH_LITE_ID))
        elif arch == "cascade":
            r = run_arch_cascade(p, gen_model=kwargs.get("gen_model", GEMINI_FLASH_LITE_ID), esc_model=kwargs.get("esc_model", GEMINI_FLASH_ID))
        elif arch == "hybrid":
            r = run_arch_hybrid(p, planner_model=kwargs.get("planner", GEMINI_FLASH_ID), executor_model=kwargs.get("executor", GEMINI_FLASH_LITE_ID), escalate_model=kwargs.get("escalate", GEMINI_FLASH_ID))
        elif arch == "dual-candidate":
            r = run_arch_f_dual_candidate(p, planner_model=kwargs.get("planner", GEMINI_FLASH_ID), executor_model=kwargs.get("executor", GEMINI_FLASH_LITE_ID))
        elif arch == "contract":
            r = run_arch_g_contract_guided(p, planner_model=kwargs.get("planner", GEMINI_FLASH_ID), executor_model=kwargs.get("executor", GEMINI_FLASH_LITE_ID))
        elif arch == "smart-repair":
            r = run_arch_h_smart_repair(p, planner_model=kwargs.get("planner", GEMINI_FLASH_ID), executor_model=kwargs.get("executor", GEMINI_FLASH_LITE_ID), escalate_model=kwargs.get("escalate", GEMINI_FLASH_ID))
        elif arch == "ultra-sweet":
            r = run_arch_i_ultra_sweet(p, planner_model=kwargs.get("planner", GEMINI_FLASH_ID), executor_model=kwargs.get("executor", GEMINI_FLASH_LITE_ID), escalate_model=kwargs.get("escalate", GEMINI_FLASH_ID))
        else:
            raise ValueError(f"Unknown arch: {arch}")

        r["task_id"] = tid
        task_results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f" {status} | cost=${r['as_run_usd']:.5f} | out_tok={r['output_tokens']}")

    n = len(task_results)
    passed_cnt = sum(1 for r in task_results if r["passed"])
    tot_cost = sum(r["as_run_usd"] for r in task_results)
    avg_out = sum(r["output_tokens"] for r in task_results) / n if n else 0
    cost_per_solved = (tot_cost / passed_cnt) if passed_cnt > 0 else float("inf")

    summary = {
        "name": cfg_name,
        "arch": arch,
        "n": n,
        "passed": passed_cnt,
        "pass_rate": round(passed_cnt / n, 3) if n else 0,
        "total_as_run_usd": round(tot_cost, 4),
        "cost_per_solved_usd": round(cost_per_solved, 4) if cost_per_solved != float("inf") else -1.0,
        "avg_output_tokens": round(avg_out, 1),
        "tasks": task_results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="Number of tasks (default: 10)")
    parser.add_argument("--split", default=BCB_SPLIT, help="Dataset split")
    args = parser.parse_args()

    problems = load_problems(args.split)
    task_ids = list(problems.keys())[:args.n]

    all_configs = [
        # --- Baselines ---
        ("Single: gemini-3.1-flash-lite", "single", {"model": GEMINI_FLASH_LITE_ID}),
        ("Single: gemini-3.5-flash", "single", {"model": GEMINI_FLASH_ID}),
        ("Single: claude-sonnet-5", "single", {"model": SONNET_ID}),
        ("Single: claude-opus-4-8", "single", {"model": OPUS_ID}),
        ("Read/Write: 3.5-Flash + 3.1-Lite", "read-write", {"planner": GEMINI_FLASH_ID, "executor": GEMINI_FLASH_LITE_ID}),
        ("Read/Write: Sonnet-5 + 3.1-Lite", "read-write", {"planner": SONNET_ID, "executor": GEMINI_FLASH_LITE_ID}),
        ("Read/Write: Opus-4.8 + 3.1-Lite", "read-write", {"planner": OPUS_ID, "executor": GEMINI_FLASH_LITE_ID}),
        ("Cascade: 3.1-Lite -> 3.5-Flash", "cascade", {"gen_model": GEMINI_FLASH_LITE_ID, "esc_model": GEMINI_FLASH_ID}),
        ("Cascade: 3.1-Lite -> Sonnet-5", "cascade", {"gen_model": GEMINI_FLASH_LITE_ID, "esc_model": SONNET_ID}),
        ("Hybrid: 3.5-Flash + 3.1-Lite + 3.5-Flash", "hybrid", {"planner": GEMINI_FLASH_ID, "executor": GEMINI_FLASH_LITE_ID, "escalate": GEMINI_FLASH_ID}),

        # --- New Architectures ---
        ("Arch F: Dual-Candidate (3.5-Flash + 2x 3.1-Lite)", "dual-candidate", {"planner": GEMINI_FLASH_ID, "executor": GEMINI_FLASH_LITE_ID}),
        ("Arch G: Contract-Guided (3.5-Flash + 3.1-Lite)", "contract", {"planner": GEMINI_FLASH_ID, "executor": GEMINI_FLASH_LITE_ID}),
        ("Arch G: Contract-Guided (Sonnet-5 + 3.1-Lite)", "contract", {"planner": SONNET_ID, "executor": GEMINI_FLASH_LITE_ID}),
        ("Arch H: Smart-Repair (3.5-Flash + 3.1-Lite + Smart Esc)", "smart-repair", {"planner": GEMINI_FLASH_ID, "executor": GEMINI_FLASH_LITE_ID, "escalate": GEMINI_FLASH_ID}),
        ("Arch I: Ultra-Sweet Hybrid (Contract + Dual + Smart)", "ultra-sweet", {"planner": GEMINI_FLASH_ID, "executor": GEMINI_FLASH_LITE_ID, "escalate": GEMINI_FLASH_ID}),
        ("Arch I: Ultra-Sweet Hybrid (Sonnet-5 + Dual + Smart)", "ultra-sweet", {"planner": SONNET_ID, "executor": GEMINI_FLASH_LITE_ID, "escalate": SONNET_ID}),
    ]

    results = []
    for name, arch, kwargs in all_configs:
        res = run_config(name, arch, task_ids, problems, **kwargs)
        results.append(res)

    out_file = os.path.join(RESULTS_DIR, "results_sweetspot_comprehensive.json")
    with open(out_file, "w") as f:
        json.dump({"tasks": task_ids, "split": args.split, "results": results}, f, indent=2)

    print("\n" + "=" * 88)
    print("ALL-ARCHITECTURE SWEET-SPOT COMPARISON TABLE (First 10 BigCodeBench-Hard Tasks)")
    print("=" * 88)
    print(f"{'Configuration':<44} | {'Pass Rate':<10} | {'Total Cost ($)':<12} | {'$/Solved':<10} | {'Avg Out Tok':<10}")
    print("-" * 88)
    for r in results:
        ps_str = f"${r['cost_per_solved_usd']:.4f}" if r['cost_per_solved_usd'] >= 0 else "N/A"
        print(f"{r['name']:<44} | {r['passed']}/{r['n']} ({r['pass_rate']:.0%})  | ${r['total_as_run_usd']:<11.4f} | {ps_str:<10} | {r['avg_output_tokens']:<10.1f}")
    print("=" * 88)
    print(f"\nSaved detailed JSON results to: {out_file}")

if __name__ == "__main__":
    main()
