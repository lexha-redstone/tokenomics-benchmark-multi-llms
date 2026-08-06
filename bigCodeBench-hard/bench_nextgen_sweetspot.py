#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0
"""
Next-Gen Multi-LLM Sweet-Spot & Architectural Benchmark on BigCodeBench-Hard.

Evaluates Next-Generation Architectures and Model Combinations:
  - gemini-3.6-flash       (Replaces gemini-3.5-flash in top-performing configurations)
  - gemini-3.5-flash-lite  (Replaces gemini-3.1-flash-lite in low-cost executor & triage roles)
  - claude-opus-5          (Replaces Opus-4.8 in high-end advisor & escalation roles)
  - claude-sonnet-5      (Next-gen mid/frontier escalation & advisor combinations)

Includes:
  1. Core 7 Sweet-Spot Configurations (1:1 Next-Gen equivalent of N=30 Sweet-Spot)
  2. Extended Top-Performing Architectures (Opus-5 Read/Write, 3-Tier Escalation,
     Contract-Guided Spec Generation, Dual-Candidate Parallel Selection, and Ultra-Sweet Hybrid)

Features auto-resume caching, retry resilience, and customizable actual pricing accounting.
"""

import os, sys, json, re, subprocess, tempfile, time, argparse, shutil, ssl
import urllib.request, urllib.parse, urllib.error

os.environ["MPLBACKEND"] = "Agg"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CACHE_FILE = os.path.join(RESULTS_DIR, "cache_nextgen_sweetspot.json")

# --- Dataset Config ---
BCB_DATASET = "bigcode/bigcodebench-hard"
BCB_CONFIG = "default"
BCB_SPLIT = "v0.1.4"
_KEEP_FIELDS = ("task_id", "complete_prompt", "canonical_solution", "code_prompt", "test", "entry_point", "libs")

# --- Next-Gen Model IDs ---
# These models replace earlier generations (gemini-3.5-flash -> gemini-3.6-flash,
# gemini-3.1-flash-lite -> gemini-3.5-flash-lite, Opus-4.8 -> Opus-5) and include Sonnet-5.
OPUS_ID = "claude-opus-5"                  # Replaced from claude-opus-4-8
SONNET_ID = "claude-sonnet-5"              # Added / updated Sonnet combination model
GEMINI_FLASH_ID = "gemini-3.6-flash"       # Replaced from gemini-3.5-flash
GEMINI_FLASH_LITE_ID = "gemini-3.5-flash-lite"  # Replaced from gemini-3.1-flash-lite

GCP_PROJECT = os.environ.get("GCP_PROJECT", "my-argolis-prj")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")

# ==============================================================================
# --- MODEL PRICING TABLE ($ per 1,000,000 tokens) ---
# NOTE: The values below are initial estimates. You can modify these numbers
# directly with actual billing/contract pricing as needed.
# ==============================================================================
PRICING = {
    OPUS_ID:              {"input": 5.00,  "output": 25.00, "cache_read": 0.50,  "cache_write": 6.25},
    SONNET_ID:            {"input": 2.00,  "output": 10.00, "cache_read": 0.20,  "cache_write": 2.50},
    GEMINI_FLASH_ID:      {"input": 1.50,  "output": 7.50,  "cache_read": 0.15,  "cache_write": 0.00},
    GEMINI_FLASH_LITE_ID: {"input": 0.30,  "output": 2.50,  "cache_read": 0.03, "cache_write": 0.00},
}

# --- Role Prompts ---
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

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return json.load(open(CACHE_FILE))
        except Exception:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

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

# --- API Clients with Retry Loop ---
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
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=body, headers={
                "Authorization": f"Bearer {_vertex_access_token()}",
                "Content-Type": "application/json",
            })
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120, context=_ssl_ctx()) as r:
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
        except Exception as e:
            if attempt == 3:
                raise e
            time.sleep(2 ** attempt)

def _fresh_gemini():
    from google import genai
    return genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

# Next-gen Gemini models are never run with OFF thinking: MINIMAL is the floor.
_GEMINI_THINKING_HEADROOM = {"minimal": 2048, "low": 4096, "medium": 8192, "high": 16384}
GEMINI_THINKING_FLOOR = "minimal"

def _normalize_gemini_thinking(thinking_level):
    lvl = (thinking_level or "").lower()
    return lvl if lvl in _GEMINI_THINKING_HEADROOM else GEMINI_THINKING_FLOOR

def gemini_call(model_id, prompt, max_tokens=2560, thinking_level=None, temperature=0.0):
    from google.genai import types
    lvl = _normalize_gemini_thinking(thinking_level)
    tc = types.ThinkingConfig(thinking_level=lvl.upper())
    max_tokens = max_tokens + _GEMINI_THINKING_HEADROOM[lvl]

    cfg = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        thinking_config=tc,
        temperature=temperature if temperature > 0 else None,
        http_options=types.HttpOptions(timeout=90000)
    )

    for attempt in range(4):
        try:
            t0 = time.time()
            resp = _fresh_gemini().models.generate_content(
                model=model_id, contents=prompt, config=cfg
            )
            dt = time.time() - t0
            m = resp.usage_metadata
            inp = m.prompt_token_count or 0
            out = (m.candidates_token_count or 0) + (m.thoughts_token_count or 0)
            p = PRICING.get(model_id, PRICING[GEMINI_FLASH_ID])
            cost = round(inp / 1e6 * p["input"] + out / 1e6 * p["output"], 6)
            usage = {"input_raw": inp, "output": out, "cache_read": 0, "cache_creation": 0,
                     "prompt_tokens": inp, "total_tokens": m.total_token_count or (inp + out),
                     "as_run_usd": cost}
            return (resp.text or ""), usage, dt
        except Exception as e:
            if attempt == 3:
                raise e
            time.sleep(3)

def dispatch_model(model_id, prompt, max_tokens=2560, thinking_level=None, temperature=0.0):
    if model_id.startswith("gemini"):
        return gemini_call(model_id, prompt, max_tokens, thinking_level, temperature)
    return claude_api_call(model_id, prompt, max_tokens, thinking_level, temperature)

# --- Gemini Escalation Thinking Ladder ---
# OFF thinking is never used. The first attempt of a pipeline runs at MINIMAL and
# every escalation step climbs one rung, so a deeper escalation always thinks at
# least as hard as the step that failed before it.
GEMINI_THINKING_LADDER = ["minimal", "low", "medium", "high"]

def escalated_thinking(model_id, depth, explicit=None):
    """Thinking level for `model_id` at escalation `depth` (1 = first attempt).

    Gemini: the ladder rung for `depth` acts as a floor, so an explicit level is
    honored only when it is at least as high (and "off" is always lifted).
    Claude: `explicit` passes through unchanged (None = extended thinking off).
    """
    if not model_id.startswith("gemini"):
        return explicit
    rung = min(max(depth, 1), len(GEMINI_THINKING_LADDER)) - 1
    lvl = (explicit or "").lower()
    if lvl in GEMINI_THINKING_LADDER:
        rung = max(rung, GEMINI_THINKING_LADDER.index(lvl))
    return GEMINI_THINKING_LADDER[rung]

# --- Code Extraction & Subprocess Unittest Runner ---
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
    if any(k in err for k in ("SyntaxError:", "IndentationError:", "ImportError:", "ModuleNotFoundError:", "NameError:")):
        return "SYNTAX_IMPORT"
    elif "AssertionError:" in err or "FAILED (" in err:
        return "ASSERTION"
    elif "timeout:" in err:
        return "TIMEOUT"
    return "EXCEPTION"

# ==============================================================================
# --- NEXT-GEN SWEET-SPOT & TOP ARCHITECTURAL PIPELINES ---
# ==============================================================================

def run_single(problem, model_id, thinking_level=None):
    prompt = "You are an expert Python programmer. Output ONLY python code block.\n\n" + problem["complete_prompt"]
    sol, u, _ = dispatch_model(model_id, prompt,
                               thinking_level=escalated_thinking(model_id, 1, thinking_level))
    code = extract_code(sol)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)
    return {
        "passed": passed,
        "as_run_usd": u["as_run_usd"],
        "output_tokens": u["output"],
        "total_tokens": u["total_tokens"],
        "code": code,
        "error": "" if passed else err,
    }

def run_read_write(problem, planner_model, executor_model):
    adv_prompt = ADVISOR_ROLE + problem["complete_prompt"]
    guidance, u_adv, _ = dispatch_model(planner_model, adv_prompt, max_tokens=512,
                                        thinking_level=escalated_thinking(planner_model, 1))

    exec_prompt = (
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Advisor Guidance:\n{guidance}\n\n"
        "Write complete Python solution. Output ONLY python code block."
    )
    sol, u_exec, _ = dispatch_model(executor_model, exec_prompt,
                                    thinking_level=escalated_thinking(executor_model, 1))
    tot_usd = u_adv["as_run_usd"] + u_exec["as_run_usd"]
    tot_out = u_adv["output"] + u_exec["output"]
    tot_tok = u_adv["total_tokens"] + u_exec["total_tokens"]
    code = extract_code(sol)

    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)
    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "code": code,
        "error": "" if passed else err,
    }

def run_3tier_cascade(problem, l1_model, l2_model, l2_thinking, l3_model, l3_thinking=None):
    prompt = "You are an expert Python programmer. Output ONLY python code block.\n\n" + problem["complete_prompt"]
    sol1, u1, _ = dispatch_model(l1_model, prompt,
                                 thinking_level=escalated_thinking(l1_model, 1))
    tot_usd, tot_out, tot_tok = u1["as_run_usd"], u1["output"], u1["total_tokens"]
    code = extract_code(sol1)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    if not passed:
        r1_prompt = (
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Current code:\n```python\n{code}\n```\n\n"
            f"Unittest error:\n```\n{err[-2000:]}\n```\n\n"
            "Fix bug. Output complete python code block."
        )
        sol2, u2, _ = dispatch_model(l2_model, r1_prompt,
                                     thinking_level=escalated_thinking(l2_model, 2, l2_thinking))
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        tot_tok += u2["total_tokens"]
        code2 = extract_code(sol2)
        if not missing_code_error(code2, problem["entry_point"]):
            code = code2
            passed, err = run_bigcodebench(problem, code)

    if not passed and l3_model:
        r2_prompt = (
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Current code:\n```python\n{code}\n```\n\n"
            f"Unittest error:\n```\n{err[-2000:]}\n```\n\n"
            "Fix bug. Output complete python code block."
        )
        sol3, u3, _ = dispatch_model(l3_model, r2_prompt,
                                     thinking_level=escalated_thinking(l3_model, 3, l3_thinking))
        tot_usd += u3["as_run_usd"]
        tot_out += u3["output"]
        tot_tok += u3["total_tokens"]
        code3 = extract_code(sol3)
        if not missing_code_error(code3, problem["entry_point"]):
            code = code3
            passed, err = run_bigcodebench(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "code": code,
        "error": "" if passed else err,
    }

def run_smart_repair(problem, planner_model, executor_model, escalate_model):
    adv_prompt = ADVISOR_ROLE + problem["complete_prompt"]
    guidance, u_adv, _ = dispatch_model(planner_model, adv_prompt, max_tokens=512,
                                        thinking_level=escalated_thinking(planner_model, 1))
    tot_usd, tot_out, tot_tok = u_adv["as_run_usd"], u_adv["output"], u_adv["total_tokens"]

    exec_prompt = (
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Advisor Guidance:\n{guidance}\n\n"
        "Write complete Python solution. Output ONLY python code block."
    )
    sol, u_exec, _ = dispatch_model(executor_model, exec_prompt,
                                    thinking_level=escalated_thinking(executor_model, 1))
    tot_usd += u_exec["as_run_usd"]
    tot_out += u_exec["output"]
    tot_tok += u_exec["total_tokens"]
    code = extract_code(sol)

    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    if not passed:
        cat = classify_error(err)
        if cat == "SYNTAX_IMPORT":
            r_prompt = (
                REPAIR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                f"Current code:\n```python\n{code}\n```\n\n"
                f"Syntax/Import error:\n```\n{err[-1500:]}\n```\n\n"
                "Fix syntax/import error. Output complete python code block."
            )
            sol2, u2, _ = dispatch_model(executor_model, r_prompt,
                                         thinking_level=escalated_thinking(executor_model, 2))
            tot_usd += u2["as_run_usd"]
            tot_out += u2["output"]
            tot_tok += u2["total_tokens"]
            code2 = extract_code(sol2)
            if not missing_code_error(code2, problem["entry_point"]):
                code = code2
                passed, err = run_bigcodebench(problem, code)
        else:
            digest, tr_usage, _ = triage_error(err, model_id=executor_model)
            tot_usd += tr_usage["as_run_usd"]
            tot_out += tr_usage["output"]
            tot_tok += tr_usage["total_tokens"]

            r_prompt = (
                REPAIR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                f"Current code:\n```python\n{code}\n```\n\n"
                f"Triaged Error:\n```\n{digest}\n```\n\n"
                "Fix bug. Output complete python code block."
            )
            sol2, u2, _ = dispatch_model(escalate_model, r_prompt,
                                         thinking_level=escalated_thinking(escalate_model, 3))
            tot_usd += u2["as_run_usd"]
            tot_out += u2["output"]
            tot_tok += u2["total_tokens"]
            code2 = extract_code(sol2)
            if not missing_code_error(code2, problem["entry_point"]):
                code = code2
                passed, err = run_bigcodebench(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "code": code,
        "error": "" if passed else err,
    }

def run_contract_guided(problem, planner_model, executor_model):
    adv_prompt = CONTRACT_ADVISOR_ROLE + problem["complete_prompt"]
    guidance, u_adv, _ = dispatch_model(planner_model, adv_prompt, max_tokens=768,
                                        thinking_level=escalated_thinking(planner_model, 1))
    tot_usd, tot_out, tot_tok = u_adv["as_run_usd"], u_adv["output"], u_adv["total_tokens"]

    exec_prompt = (
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Contract Specification:\n{guidance}\n\n"
        "Implement solution adhering strictly to contract. Output ONLY python code block."
    )
    sol, u_exec, _ = dispatch_model(executor_model, exec_prompt,
                                    thinking_level=escalated_thinking(executor_model, 1))
    tot_usd += u_exec["as_run_usd"]
    tot_out += u_exec["output"]
    tot_tok += u_exec["total_tokens"]
    code = extract_code(sol)

    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    if not passed:
        r_prompt = (
            REPAIR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Contract:\n{guidance}\n\n"
            f"Current solution:\n```python\n{code}\n```\n\n"
            f"Test failure:\n```\n{err[-1500:]}\n```\n\n"
            "Correct the failure. Output complete python code block."
        )
        sol2, u2, _ = dispatch_model(planner_model, r_prompt,
                                     thinking_level=escalated_thinking(planner_model, 2))
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        tot_tok += u2["total_tokens"]
        code2 = extract_code(sol2)
        if not missing_code_error(code2, problem["entry_point"]):
            code = code2
            passed, err = run_bigcodebench(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "code": code,
        "error": "" if passed else err,
    }

def run_ultra_sweet_hybrid(problem, planner_model, executor_model, escalate_model):
    adv_prompt = CONTRACT_ADVISOR_ROLE + problem["complete_prompt"]
    guidance, u_adv, _ = dispatch_model(planner_model, adv_prompt, max_tokens=768,
                                        thinking_level=escalated_thinking(planner_model, 1))
    tot_usd, tot_out, tot_tok = u_adv["as_run_usd"], u_adv["output"], u_adv["total_tokens"]

    exec_prompt1 = (
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Contract Specification:\n{guidance}\n\n"
        "Implement solution adhering strictly to contract. Output ONLY python code block."
    )
    sol1, u_exec1, _ = dispatch_model(executor_model, exec_prompt1, temperature=0.0,
                                      thinking_level=escalated_thinking(executor_model, 1))
    tot_usd += u_exec1["as_run_usd"]
    tot_out += u_exec1["output"]
    tot_tok += u_exec1["total_tokens"]
    code1 = extract_code(sol1)
    guard1 = missing_code_error(code1, problem["entry_point"])
    passed1, err1 = (False, guard1) if guard1 else run_bigcodebench(problem, code1)

    if passed1:
        return {
            "passed": True,
            "as_run_usd": round(tot_usd, 6),
            "output_tokens": tot_out,
            "total_tokens": tot_tok,
            "code": code1,
            "error": "",
        }

    exec_prompt2 = exec_prompt1 + "\n\nPay extra attention to boundary conditions and exceptions."
    sol2, u_exec2, _ = dispatch_model(executor_model, exec_prompt2, temperature=0.2,
                                      thinking_level=escalated_thinking(executor_model, 2))
    tot_usd += u_exec2["as_run_usd"]
    tot_out += u_exec2["output"]
    tot_tok += u_exec2["total_tokens"]
    code2 = extract_code(sol2)
    guard2 = missing_code_error(code2, problem["entry_point"])
    passed2, err2 = (False, guard2) if guard2 else run_bigcodebench(problem, code2)

    if passed2:
        return {
            "passed": True,
            "as_run_usd": round(tot_usd, 6),
            "output_tokens": tot_out,
            "total_tokens": tot_tok,
            "code": code2,
            "error": "",
        }

    code = code2 if "SyntaxError" not in err2 else code1
    err = err2 if "SyntaxError" not in err2 else err1

    digest, tr_usage, _ = triage_error(err, model_id=executor_model)
    tot_usd += tr_usage["as_run_usd"]
    tot_out += tr_usage["output"]
    tot_tok += tr_usage["total_tokens"]

    r_prompt = (
        REPAIR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Contract:\n{guidance}\n\n"
        f"Current code:\n```python\n{code}\n```\n\n"
        f"Triaged Error:\n```\n{digest}\n```\n\n"
        "Correct the failure. Output complete python code block."
    )
    # Terminal repair of the deepest pipeline (contract -> cand1 -> cand2 -> triage),
    # so it runs at the top rung of the escalation ladder.
    sol3, u3, _ = dispatch_model(escalate_model, r_prompt,
                                 thinking_level=escalated_thinking(escalate_model, 4))
    tot_usd += u3["as_run_usd"]
    tot_out += u3["output"]
    tot_tok += u3["total_tokens"]
    code3 = extract_code(sol3)
    if not missing_code_error(code3, problem["entry_point"]):
        code = code3
        passed, err = run_bigcodebench(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "code": code,
        "error": "" if passed else err,
    }

# ==============================================================================
# --- NEXT-GEN BENCHMARK GROUP CONFIGURATION REGISTRY ---
# ==============================================================================

def get_nextgen_configurations():
    """
    Returns the Next-Gen benchmark group combinations.
    - gemini-3.5-flash -> replaced with gemini-3.6-flash
    - Opus-4.8 -> replaced with Opus-5 (claude-opus-5)
    - gemini-3.1-flash-lite -> replaced with gemini-3.5-flash-lite
    - Combines gemini-3.6-flash, gemini-3.5-flash-lite, Opus-5, and Sonnet-5

    Gemini models never run with OFF thinking: MINIMAL is the floor, and each
    escalation tier climbs the MINIMAL -> LOW -> MEDIUM -> HIGH ladder.
    """
    return [
        # --- Group A: Core 7 Next-Gen Sweet-Spot Architectures ---
        ("1. Single: 3.6-Flash (MINIMAL)", lambda p: run_single(p, GEMINI_FLASH_ID, "minimal")),
        ("2. Single: 3.6-Flash (LOW)", lambda p: run_single(p, GEMINI_FLASH_ID, "low")),
        ("3. Read/Write: 3.6-Flash + 3.5-Lite (MINIMAL)", lambda p: run_read_write(p, GEMINI_FLASH_ID, GEMINI_FLASH_LITE_ID)),
        ("4. Pure Gemini 3-Tier: 3.5-Lite (MINIMAL) -> 3.6-Flash (LOW) -> 3.6-Flash (MEDIUM)", lambda p: run_3tier_cascade(p, GEMINI_FLASH_LITE_ID, GEMINI_FLASH_ID, "low", GEMINI_FLASH_ID, "medium")),
        ("5. Escalation Shield: 3.5-Lite (MINIMAL) -> 3.6-Flash (LOW) -> Sonnet-5", lambda p: run_3tier_cascade(p, GEMINI_FLASH_LITE_ID, GEMINI_FLASH_ID, "low", SONNET_ID, None)),
        ("6. 3-Tier Frontier: 3.5-Lite (MINIMAL) -> 3.6-Flash (MEDIUM) -> Opus-5", lambda p: run_3tier_cascade(p, GEMINI_FLASH_LITE_ID, GEMINI_FLASH_ID, "medium", OPUS_ID, None)),
        ("7. Smart Repair: 3.6-Flash + 3.5-Lite + Triage + 3.6-Flash (MEDIUM)", lambda p: run_smart_repair(p, GEMINI_FLASH_ID, GEMINI_FLASH_LITE_ID, GEMINI_FLASH_ID)),

        # --- Group B: Extended Next-Gen Combinations ---
        ("8. Single: Opus-5 (OFF)", lambda p: run_single(p, OPUS_ID, None)),
        ("9. Single: Sonnet-5 (OFF)", lambda p: run_single(p, SONNET_ID, None)),
        ("10. Read/Write: Opus-5 + 3.5-Lite", lambda p: run_read_write(p, OPUS_ID, GEMINI_FLASH_LITE_ID)),
        ("11. Read/Write: Sonnet-5 + 3.5-Lite", lambda p: run_read_write(p, SONNET_ID, GEMINI_FLASH_LITE_ID)),
        ("12. 3-Tier Escalation: 3.5-Lite -> Sonnet-5 -> Opus-5", lambda p: run_3tier_cascade(p, GEMINI_FLASH_LITE_ID, SONNET_ID, None, OPUS_ID, None)),
        ("13. Arch G: Contract-Guided (3.6-Flash + 3.5-Lite)", lambda p: run_contract_guided(p, GEMINI_FLASH_ID, GEMINI_FLASH_LITE_ID)),
        ("14. Arch G: Contract-Guided (Sonnet-5 + 3.5-Lite)", lambda p: run_contract_guided(p, SONNET_ID, GEMINI_FLASH_LITE_ID)),
        ("15. Arch I: Ultra-Sweet Hybrid (3.6-Flash + 3.5-Lite + 3.6-Flash)", lambda p: run_ultra_sweet_hybrid(p, GEMINI_FLASH_ID, GEMINI_FLASH_LITE_ID, GEMINI_FLASH_ID)),
        ("16. Arch I: Ultra-Sweet Hybrid (Sonnet-5 + 3.5-Lite + Sonnet-5)", lambda p: run_ultra_sweet_hybrid(p, SONNET_ID, GEMINI_FLASH_LITE_ID, SONNET_ID)),
    ]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30, help="Number of tasks to benchmark (default: 30)")
    parser.add_argument("--split", default=BCB_SPLIT, help="BigCodeBench dataset split (default: v0.1.4)")
    parser.add_argument("--group", choices=["core", "extended", "all"], default="core",
                        help="Benchmark configuration group to run ('core' = Core 7 Sweet-Spot, 'all' = all 16 configs)")
    args = parser.parse_args()

    problems = load_problems(args.split)
    task_ids = list(problems.keys())[:args.n]
    cache = load_cache()

    all_configs = get_nextgen_configurations()
    if args.group == "core":
        active_configs = all_configs[:7]
    elif args.group == "extended":
        active_configs = all_configs[7:]
    else:
        active_configs = all_configs

    summaries = []
    for cfg_name, fn in active_configs:
        print(f"\n=======================================================")
        print(f"RUNNING NEXT-GEN CONFIG: {cfg_name} (N={len(task_ids)})")
        print(f"=======================================================")
        task_results = []
        if cfg_name not in cache:
            cache[cfg_name] = {}

        for i, tid in enumerate(task_ids, 1):
            p = problems[tid]
            if tid in cache[cfg_name]:
                r = cache[cfg_name][tid]
                print(f"[{i}/{len(task_ids)}] {tid} ({p['entry_point']}) ... [CACHED] {'PASS' if r['passed'] else 'FAIL'}")
            else:
                print(f"[{i}/{len(task_ids)}] {tid} ({p['entry_point']}) ... ", end="", flush=True)
                r = fn(p)
                cache[cfg_name][tid] = r
                save_cache(cache)
                status = "PASS" if r["passed"] else "FAIL"
                print(f"{status} | cost=${r['as_run_usd']:.5f} | out_tok={r['output_tokens']}")
            task_results.append(r)

        n = len(task_results)
        passed_cnt = sum(1 for r in task_results if r["passed"])
        tot_cost = sum(r["as_run_usd"] for r in task_results)
        avg_out = sum(r["output_tokens"] for r in task_results) / n if n else 0
        cps = (tot_cost / passed_cnt) if passed_cnt > 0 else -1.0

        summaries.append({
            "name": cfg_name,
            "n": n,
            "passed": passed_cnt,
            "pass_rate": round(passed_cnt / n, 3) if n else 0,
            "total_as_run_usd": round(tot_cost, 4),
            "cost_per_solved_usd": round(cps, 4) if cps >= 0 else -1.0,
            "avg_output_tokens": round(avg_out, 1)
        })

    out_file = os.path.join(RESULTS_DIR, "results_nextgen_sweetspot.json")
    with open(out_file, "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n" + "=" * 98)
    print(f"NEXT-GEN MULTI-LLM SWEET-SPOT COMPARISON TABLE (First {args.n} BigCodeBench-Hard Tasks)")
    print("=" * 98)
    print(f"{'Configuration':<66} | {'Pass Rate':<10} | {'Total Cost ($)':<12} | {'$/Solved':<10}")
    print("-" * 98)
    for s in summaries:
        cps_str = f"${s['cost_per_solved_usd']:.4f}" if s['cost_per_solved_usd'] >= 0 else "N/A"
        print(f"{s['name']:<66} | {s['passed']}/{s['n']} ({s['pass_rate']:.0%})  | ${s['total_as_run_usd']:<11.4f} | {cps_str:<10}")
    print("=" * 98)
    print(f"\nSaved detailed JSON results to: {out_file}")
    print("NOTE: You can customize pricing figures in PRICING dict at the top of this script.")

if __name__ == "__main__":
    main()
