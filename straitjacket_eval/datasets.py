# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Unified Dataset Loader for BigCodeBench-Hard (N=50) and WebDev (N=50) Evaluation Suites.
Ensures local caching, deterministic ordering, and complete problem extraction.
"""

import os
import sys
import json
import ast
import urllib.request
import urllib.parse

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)

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
    """Ensure BigCodeBench-Hard dataset file exists locally, downloading if necessary."""
    data_dir = os.path.join(PROJECT_ROOT, "bigCodeBench-hard", "data")
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
    return path

def load_bcb_hard(max_tasks=50):
    """Load exactly `max_tasks` (default N=50) BigCodeBench-Hard tasks."""
    path = ensure_bcb_dataset(BCB_DEFAULT_SPLIT)
    problems = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["dataset_name"] = "bigcodebench-hard"
                problems[row["task_id"]] = row
                if max_tasks and len(problems) >= max_tasks:
                    break
    return problems

def load_webdev(max_tasks=50):
    """Load exactly `max_tasks` (default N=50) WebDev & Networking tasks."""
    web_networking_libs = {
        "requests", "urllib", "flask", "flask_login", "flask_mail", "flask_wtf",
        "werkzeug", "wtforms", "http", "ftplib", "smtplib", "bs4", "pyquery", "lxml",
        "cgi", "socket", "email", "json", "base64", "hashlib", "cryptography", "Crypto",
        "jwt", "ssl", "html", "xml", "xmlrpc", "asyncio", "aiohttp", "fastapi",
        "tornado", "paramiko", "mechanize", "scrapy", "selenium", "chardet", "zipfile",
        "tarfile", "csv", "yaml"
    }

    bcb_path = ensure_bcb_dataset(BCB_DEFAULT_SPLIT)
    problems = {}
    with open(bcb_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            is_web = False
            try:
                lib_list = ast.literal_eval(d.get("libs", "[]"))
                if any(l in web_networking_libs for l in lib_list):
                    is_web = True
            except Exception:
                pass

            if not is_web:
                prompt_lower = d.get("complete_prompt", "").lower()
                if any(k in prompt_lower for k in [
                    "http", "url", "html", "json", "api", "web", "socket", "request",
                    "response", "flask", "server", "client", "email", "parse", "encode"
                ]):
                    is_web = True

            if is_web:
                d["dataset_name"] = "webdev"
                problems[d["task_id"]] = d
                if max_tasks and len(problems) >= max_tasks:
                    break

    return problems

def get_dataset(name="bcb", n=50):
    """Unified dataset accessor for 'bcb' and 'webdev'."""
    name_clean = name.lower().replace("-", "").replace("_", "")
    if "web" in name_clean:
        return load_webdev(max_tasks=n)
    return load_bcb_hard(max_tasks=n)
