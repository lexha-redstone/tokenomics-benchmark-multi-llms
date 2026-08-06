# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Unified Dataset Loader for BigCodeBench-Hard, SWE-bench Pro, and WebDev Benchmarks.
Handles dataset loading, HuggingFace dataset caching, and problem dictionary creation.
"""

import os
import sys
import json
import ast
import urllib.request
import urllib.parse
import urllib.error

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(_HERE)

# ==============================================================================
# --- BIGCODEBENCH-HARD DATASET LOADER ---
# ==============================================================================

BCB_DATASET = "bigcode/bigcodebench-hard"
BCB_CONFIG = "default"
BCB_DEFAULT_SPLIT = "v0.1.4"
_BCB_KEEP_FIELDS = ("task_id", "complete_prompt", "canonical_solution", "code_prompt",
                    "test", "entry_point", "libs")

def _ssl_ctx():
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

def ensure_bcb_dataset(split=BCB_DEFAULT_SPLIT):
    """Ensure BigCodeBench-Hard split file is present locally, fetching from HF if needed."""
    data_dir = os.path.join(ROOT_DIR, "bigCodeBench-hard", "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"BigCodeBench-Hard-{split}.jsonl")
    if os.path.exists(path):
        return path

    print(f"Fetching {BCB_DATASET} [{split}] from HuggingFace -> {path}", flush=True)
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        q = urllib.parse.urlencode({
            "dataset": BCB_DATASET, "config": BCB_CONFIG,
            "split": split, "offset": offset, "length": 100
        })
        with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + q,
                                    timeout=120, context=_ssl_ctx()) as r:
            d = json.loads(r.read())
        batch = d.get("rows", [])
        total = d.get("num_rows_total", len(batch))
        if not batch:
            break
        rows.extend(b["row"] for b in batch)
        offset += len(batch)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: row.get(k) for k in _BCB_KEEP_FIELDS}) + "\n")
    print(f"  saved {len(rows)} tasks to {path}", flush=True)
    return path

def load_bcb_problems(split=BCB_DEFAULT_SPLIT, max_tasks=None):
    """Load BigCodeBench-Hard tasks as a dictionary keyed by task_id."""
    path = ensure_bcb_dataset(split)
    problems = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["dataset_type"] = "bcb"
                problems[row["task_id"]] = row
                if max_tasks and len(problems) >= max_tasks:
                    break
    return problems

# ==============================================================================
# --- SWE-BENCH PRO DATASET LOADER ---
# ==============================================================================

SWEBENCH_DATASET = "SWE-bench/SWE-bench_Pro"
SWEBENCH_PUBLIC_FILE = "SWE-bench_Pro-public-test.jsonl"

def load_swebench_pro_problems(split="test", max_tasks=None):
    """Load SWE-bench Pro (Public dataset) tasks as a dictionary keyed by instance_id."""
    data_dir = os.path.join(ROOT_DIR, "swebench_pro", "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, SWEBENCH_PUBLIC_FILE)
    
    if not os.path.exists(path):
        # Check alternative locations
        for alt in [
            os.path.join(ROOT_DIR, "swebench-pro", "data", SWEBENCH_PUBLIC_FILE),
            os.path.join(ROOT_DIR, "data", SWEBENCH_PUBLIC_FILE),
        ]:
            if os.path.exists(alt):
                path = alt
                break

    problems = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["dataset_type"] = "swebench"
                    iid = row.get("instance_id") or row.get("task_id")
                    problems[iid] = row
                    if max_tasks and len(problems) >= max_tasks:
                        break
        print(f"Loaded {len(problems)} SWE-bench Pro tasks from {path}")
    else:
        print(f"[Notice] SWE-bench Pro dataset file not found at {path}.")
        
    return problems

# ==============================================================================
# --- WEB-DEV DATASET LOADER ---
# ==============================================================================

def load_webdev_problems(max_tasks=None):
    """Load Web-Dev tasks filtered by web/networking libraries from BigCodeBench-Hard."""
    web_libs = {
        "requests", "urllib", "flask", "flask_login", "flask_mail", "flask_wtf", 
        "werkzeug", "wtforms", "http", "ftplib", "smtplib", "bs4", "pyquery", "lxml", 
        "cgi", "socket"
    }
    
    path = os.path.join(ROOT_DIR, "webdev", "data", "BigCodeBench-Hard-WebDev.jsonl")
    if not os.path.exists(path):
        # Generate from BCB dataset if needed
        bcb_path = ensure_bcb_dataset(BCB_DEFAULT_SPLIT)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        web_rows = []
        with open(bcb_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                try:
                    lib_list = ast.literal_eval(d.get("libs", "[]"))
                    if any(lib in web_libs for lib in lib_list):
                        web_rows.append(d)
                except Exception:
                    pass
        with open(path, "w", encoding="utf-8") as f:
            for row in web_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Filtered and saved {len(web_rows)} Web-Dev tasks to {path}")

    problems = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                d["dataset_type"] = "webdev"
                problems[d["task_id"]] = d
                if max_tasks and len(problems) >= max_tasks:
                    break
    print(f"Loaded {len(problems)} Web-Dev tasks from {path}")
    return problems

# ==============================================================================
# --- UNIFIED DATASET DISPATCHER ---
# ==============================================================================

def load_dataset(dataset_name, split=None, max_tasks=None):
    """
    Unified dataset dispatcher.
    Supports:
      - 'bcb', 'bigcodebench', 'bigcodebench-hard'
      - 'swebench', 'swebench_pro', 'swe-bench'
      - 'webdev', 'web-dev'
    """
    name = dataset_name.lower().replace("-", "_")
    if name in ("bcb", "bigcodebench", "bigcodebench_hard"):
        s = split or BCB_DEFAULT_SPLIT
        return load_bcb_problems(split=s, max_tasks=max_tasks)
    elif name in ("swebench", "swebench_pro", "swe_bench", "swe_bench_pro"):
        s = split or "test"
        return load_swebench_pro_problems(split=s, max_tasks=max_tasks)
    elif name in ("webdev", "web_dev"):
        return load_webdev_problems(max_tasks=max_tasks)
    else:
        raise ValueError(f"Unknown dataset name: '{dataset_name}'. Supported: 'bcb', 'swebench', 'webdev'.")
