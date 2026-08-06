#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0
"""
Single-Model Benchmark on BigCodeBench-Hard.

Evaluates standalone single-pass (single mode) generation across:
  1. gemini-3.6-flash (thinking_level: "medium")
  2. claude-sonnet-5 (thinking: OFF)
  3. claude-opus-5 (thinking: OFF)
  4. claude-opus-4-8 (thinking: OFF)

Features auto-resume caching, retry resilience, and customizable pricing accounting.
"""

import os, sys, json, re, subprocess, tempfile, time, argparse, shutil, ssl
import urllib.request, urllib.parse, urllib.error

os.environ["MPLBACKEND"] = "Agg"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CACHE_FILE = os.path.join(RESULTS_DIR, "cache_single_models.json")

# --- Dataset Config ---
BCB_DATASET = "bigcode/bigcodebench-hard"
BCB_CONFIG = "default"
BCB_SPLIT = "v0.1.4"
_KEEP_FIELDS = ("task_id", "complete_prompt", "canonical_solution", "code_prompt", "test", "entry_point", "libs")

# --- Model IDs ---
GEMINI_36_FLASH_ID = "gemini-3.6-flash"
SONNET_5_ID = "claude-sonnet-5"
OPUS_5_ID = "claude-opus-5"
OPUS_4_8_ID = "claude-opus-4-8"

GCP_PROJECT = os.environ.get("GCP_PROJECT", "my-argolis-prj")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")

# ==============================================================================
# --- MODEL PRICING TABLE ($ per 1,000,000 tokens) ---
# ==============================================================================
PRICING = {
    GEMINI_36_FLASH_ID: {"input": 1.50,  "output": 7.50,  "cache_read": 0.15,  "cache_write": 0.00},
    SONNET_5_ID:        {"input": 2.00,  "output": 10.00, "cache_read": 0.20,  "cache_write": 2.50},
    OPUS_5_ID:          {"input": 5.00,  "output": 25.00, "cache_read": 0.50,  "cache_write": 6.25},
    OPUS_4_8_ID:        {"input": 5.00,  "output": 25.00, "cache_read": 0.50,  "cache_write": 6.25},
}

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
            p = PRICING.get(model_id, PRICING[SONNET_5_ID])
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
            p = PRICING.get(model_id, PRICING[GEMINI_36_FLASH_ID])
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

# ==============================================================================
# --- SINGLE MODE BENCHMARK RUNNER ---
# ==============================================================================

def run_single(problem, model_id, thinking_level=None):
    prompt = "You are an expert Python programmer. Output ONLY python code block.\n\n" + problem["complete_prompt"]
    sol, u, _ = dispatch_model(model_id, prompt, thinking_level=thinking_level)
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

def get_single_configurations():
    """
    Returns the 4 single mode benchmark configurations requested:
      1. gemini-3.6-flash & medium
      2. sonnet-5
      3. opus-5
      4. opus-4.8
    """
    return [
        ("1. Single: Gemini 3.6-Flash (MEDIUM)", lambda p: run_single(p, GEMINI_36_FLASH_ID, "medium")),
        ("2. Single: Claude Sonnet-5 (OFF)", lambda p: run_single(p, SONNET_5_ID, None)),
        ("3. Single: Claude Opus-5 (OFF)", lambda p: run_single(p, OPUS_5_ID, None)),
        ("4. Single: Claude Opus-4.8 (OFF)", lambda p: run_single(p, OPUS_4_8_ID, None)),
    ]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30, help="Number of tasks to benchmark (default: 30)")
    parser.add_argument("--split", default=BCB_SPLIT, help="BigCodeBench dataset split (default: v0.1.4)")
    parser.add_argument("--model", choices=["all", "gemini-3.6-flash", "sonnet-5", "opus-5", "opus-4.8"],
                        default="all", help="Which model to run in single mode (default: all)")
    args = parser.parse_args()

    problems = load_problems(args.split)
    task_ids = list(problems.keys())[:args.n]
    cache = load_cache()

    all_configs = get_single_configurations()
    if args.model == "gemini-3.6-flash":
        active_configs = [all_configs[0]]
    elif args.model == "sonnet-5":
        active_configs = [all_configs[1]]
    elif args.model == "opus-5":
        active_configs = [all_configs[2]]
    elif args.model == "opus-4.8":
        active_configs = [all_configs[3]]
    else:
        active_configs = all_configs

    summaries = []
    for cfg_name, fn in active_configs:
        print(f"\n=======================================================")
        print(f"RUNNING SINGLE MODE CONFIG: {cfg_name} (N={len(task_ids)})")
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

    out_file = os.path.join(RESULTS_DIR, "results_single_models.json")
    with open(out_file, "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n" + "=" * 98)
    print(f"SINGLE MODE BENCHMARK COMPARISON TABLE (First {args.n} BigCodeBench-Hard Tasks)")
    print("=" * 98)
    print(f"{'Configuration':<50} | {'Pass Rate':<10} | {'Total Cost ($)':<14} | {'$/Solved':<10}")
    print("-" * 98)
    for s in summaries:
        cps_str = f"${s['cost_per_solved_usd']:.4f}" if s['cost_per_solved_usd'] >= 0 else "N/A"
        print(f"{s['name']:<50} | {s['passed']}/{s['n']} ({s['pass_rate']:.0%})  | ${s['total_as_run_usd']:<13.4f} | {cps_str:<10}")
    print("=" * 98)
    print(f"\nSaved detailed JSON results to: {out_file}")

if __name__ == "__main__":
    main()
