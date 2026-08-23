#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Report the tasks a sweep never completed, and resume only those.

Why a task goes missing
-----------------------
`src/client.py` retries a transient API failure (429, 504, 503, ...)
DISPATCH_MAX_ATTEMPTS times, and `run_benchmark.py` then re-attempts the whole
task `--task-retries` times, discarding the partial record each time. Only when
both budgets are exhausted is the task dropped: it is written to
`incomplete_tasks` in the results JSON and deliberately NOT written to the
cache. So a 504 that was absorbed by a retry costs time, not data — only a gap
in the cache is a lost datapoint.

Two sources, deliberately independent:
  * cache gaps        -- authoritative, written incrementally, survives a crash
  * incomplete_tasks  -- carries the error text, so 429 / 504 / auth are
                         distinguishable; written only when the sweep finishes

Resuming is the ordinary runner WITHOUT `--no-cache`: every task already in the
cache prints [CACHED] and costs nothing, so only the gaps hit the API.

Usage:
    python3 tools/rerun_incomplete.py --dataset bcb --group router --n 148
    python3 tools/rerun_incomplete.py --dataset bcb --group router --n 148 --rerun
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.architectures import get_configurations
from src.datasets import load_dataset

DS = {
    "bcb": ("bcb", "bigCodeBench-hard"),
    "bigcodebench": ("bcb", "bigCodeBench-hard"),
    "swebench": ("swebench", "swebench_pro"),
    "swebench_pro": ("swebench", "swebench_pro"),
    "webdev": ("webdev", "webdev"),
}

# Ordered: the first pattern that matches the error text names the class.
ERROR_CLASSES = [
    ("429", re.compile(r"\b429\b|rate limit|resource exhausted", re.I)),
    ("504", re.compile(r"\b504\b|gateway time|deadline", re.I)),
    ("503/500", re.compile(r"\b50[023]\b|unavailable|internal error", re.I)),
    ("timeout", re.compile(r"timed out|timeout", re.I)),
    ("auth", re.compile(r"\b40[13]\b|credential|unauthenticated|permission denied", re.I)),
]


def classify(text):
    for name, pat in ERROR_CLASSES:
        if pat.search(text or ""):
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", "-d", default="bcb")
    ap.add_argument("--group", "-g", default="router")
    ap.add_argument("--variants", "-v", default="")
    ap.add_argument("--n", "-n", type=int, default=148)
    ap.add_argument("--results", default="",
                    help="results JSON to read incomplete_tasks from "
                         "(default: <results_dir>/<ds>_<group>_results.json)")
    ap.add_argument("--only", default="",
                    help="comma-separated error classes to resume, e.g. '429,504'. "
                         "Needs the results JSON; without it every gap is resumed.")
    ap.add_argument("--rerun", action="store_true",
                    help="run the resume sweep instead of only printing it")
    args = ap.parse_args()

    ds_key, ds_folder = DS[args.dataset.lower().replace("-", "_")]
    results_dir = os.path.join(ROOT, ds_folder, "results")
    cache_file = os.path.join(results_dir, f"cache_{ds_key}_master.json")
    results_file = args.results or os.path.join(results_dir, f"{ds_key}_{args.group}_results.json")

    v_keys = [k.strip() for k in args.variants.split(",") if k.strip()] or None
    configs = get_configurations(dataset=ds_key, group=args.group, variant_keys=v_keys)
    variants = [c["id"] for c in configs]

    task_ids = list(load_dataset(ds_key, max_tasks=args.n).keys())[:args.n]

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)

    # Error text per (variant, task), when the sweep got far enough to write it.
    errors = {}
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("summary", []):
            for t in (s.get("incomplete_tasks") or []):
                errors[(s["id"], t["task_id"])] = t.get("error", "")

    wanted = {c.strip() for c in args.only.split(",") if c.strip()}
    gaps, skipped = {}, {}
    for v in variants:
        have = cache.get(v, {})
        miss = [t for t in task_ids if t not in have]
        if not miss:
            continue
        if wanted:
            keep = [t for t in miss if classify(errors.get((v, t), "")) in wanted]
            drop = [t for t in miss if t not in keep]
            if drop:
                skipped[v] = drop
            miss = keep
        if miss:
            gaps[v] = miss

    print(f"cache:   {cache_file}")
    print(f"results: {results_file}" + ("" if os.path.exists(results_file) else "  (absent)"))
    print(f"tasks:   {len(task_ids)} | variants: {len(variants)}")
    print("note:    run this AFTER the sweep exits -- a variant the sweep has "
          "not reached\n         yet is indistinguishable here from one that "
          "failed.\n")

    total = 0
    for v in variants:
        have = len(cache.get(v, {}))
        miss = [t for t in task_ids if t not in cache.get(v, {})]
        total += len(miss)
        if not miss:
            print(f"  {v:<26} {have:>4}/{len(task_ids)}  complete")
            continue
        buckets = {}
        for t in miss:
            buckets.setdefault(classify(errors.get((v, t), "")), []).append(t)
        detail = ", ".join(f"{k}={len(x)}" for k, x in sorted(buckets.items()))
        print(f"  {v:<26} {have:>4}/{len(task_ids)}  MISSING {len(miss)}  [{detail}]")
        for t in miss[:8]:
            print(f"      - {t}  ({classify(errors.get((v, t), ''))})")
        if len(miss) > 8:
            print(f"      ... and {len(miss) - 8} more")

    if skipped:
        n_sk = sum(len(x) for x in skipped.values())
        print(f"\n  --only {args.only}: {n_sk} gap(s) left alone "
              f"(other error class, or no recorded error).")

    if not gaps:
        print(f"\nNothing to resume ({total} gap(s) total, none selected).")
        return 0

    cmd = [sys.executable, "run_benchmark.py", "--dataset", args.dataset,
           "--group", args.group, f"--n={args.n}", "--report",
           "--variants", ",".join(gaps)]
    print(f"\nResume {sum(len(x) for x in gaps.values())} task(s) across "
          f"{len(gaps)} variant(s). No --no-cache: everything already cached "
          f"is replayed for free.\n\n  " + " ".join(cmd) + "\n")

    if args.rerun:
        return subprocess.call(cmd, cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
