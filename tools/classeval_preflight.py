#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Run ClassEval's own gold solutions and quarantine the tasks the environment
cannot pass.

Why this has to exist
---------------------
A benchmark task whose *reference solution* fails is not measuring the model.
On a clean checkout, 6 of ClassEval's 100 classes fail with their own
`solution_code`, for reasons that have nothing to do with any arm under test:

  ModuleNotFoundError   an optional third-party import the environment lacks
                        (PyPDF2, reportlab, ...)
  missing corpus        nltk_data and friends, absent until downloaded
  dataset rot           gold code written against an older library -- e.g.
                        `np.mat`, removed in NumPy 2.0
  environment coupling  a test that resolves a hostname, or reads the clock

Left alone, each of those becomes a silent ~1-point penalty applied equally to
every arm, and a cheap model gets blamed for a missing pip package. Worse, the
penalty is not equal in practice: an arm that escalates spends real money
re-solving a task that was never solvable.

So gold is run first, and anything it cannot pass is written to a quarantine
file that the loader honours. Tasks are never edited, only excluded, and the
exclusion is recorded with its reason so the count is auditable.

Usage:
    python3 tools/classeval_preflight.py                 # report only
    python3 tools/classeval_preflight.py --write         # write the quarantine file
    python3 tools/classeval_preflight.py --write --repeat 3   # also catch flaky tasks
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.classeval import assemble_class
from src.datasets import CLASSEVAL_DEFAULT_SPLIT, load_classeval_problems, quarantine_path
from src.evaluator import run_classeval_class, run_classeval_method

# Ordered: first match names the reason.
REASONS = [
    ("missing_module", re.compile(r"ModuleNotFoundError|ImportError|No module named", re.I)),
    ("missing_corpus", re.compile(r"nltk_data|LookupError|Resource .* not found", re.I)),
    ("library_drift", re.compile(r"was removed in|is deprecated|AttributeError: module", re.I)),
    ("network", re.compile(r"gaierror|getaddrinfo|Name or service not known|URLError|"
                           r"socket\.timeout|ConnectionError|hostname", re.I)),
    ("timeout", re.compile(r"timeout: execution exceeded", re.I)),
    ("missing_test_class", re.compile(r"MissingTestClass", re.I)),
]

# A task can be scorable for a whole-class arm and unscorable for a per-method
# arm. `methods_info` omits a method the class actually needs on at least two
# rows (CookiesUtil.set_cookies, JobMarketplace.matches_requirements), so a
# class assembled method-by-method is missing it and fails through no fault of
# the model. Left in, that penalises exactly the arms the experiment is about,
# which is a bias between arms rather than a constant subtracted from all of
# them -- far more damaging than the gold failures above.
ASSEMBLY_REASON = "assembly_gap"


def classify(evidence_text):
    for name, pat in REASONS:
        if pat.search(evidence_text or ""):
            return name
    return "gold_fails"          # real, but cause not auto-identifiable


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=CLASSEVAL_DEFAULT_SPLIT)
    ap.add_argument("--n", type=int, default=0, help="limit tasks (0 = all)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run gold this many times; a task that does not pass "
                         "every time is quarantined as flaky (default: 1)")
    ap.add_argument("--skip-assembly", action="store_true",
                    help="only check whole-class gold, not gold re-assembled "
                         "from its own methods (the per-method arms need both)")
    ap.add_argument("--write", action="store_true",
                    help="write the quarantine file the loader reads")
    args = ap.parse_args()

    problems = load_classeval_problems(split=args.split,
                                       max_tasks=args.n or None,
                                       apply_quarantine=False)
    tasks = list(problems.values())
    print(f"preflight: {len(tasks)} ClassEval tasks, gold solution, "
          f"{args.repeat} run(s) each\n")

    quarantine, flaky, t0 = {}, [], time.time()
    for i, prob in enumerate(tasks, start=1):
        gold = prob["solution_code"]
        verdicts, evidence = [], ""
        for _ in range(max(1, args.repeat)):
            ok, ev = run_classeval_class(prob, gold)
            verdicts.append(ok)
            if not ok and not evidence:
                evidence = str(ev)

        # Second gate: can the class even be rebuilt from its own methods?
        asm_ok, asm_ev = True, ""
        if all(verdicts) and not args.skip_assembly:
            sources = {s["name"]: s["solution_code"] for s in prob["subtasks"]}
            asm_ok, ev2 = run_classeval_class(prob, assemble_class(prob, sources))
            asm_ev = "" if asm_ok else str(ev2)

        if all(verdicts) and asm_ok:
            print(f"  [{i}/{len(tasks)}] {prob['task_id']:<14} gold OK")
            continue

        if all(verdicts) and not asm_ok:
            reason, evidence = ASSEMBLY_REASON, asm_ev
        else:
            reason = "flaky" if any(verdicts) else classify(evidence)
        if reason == "flaky":
            flaky.append(prob["task_id"])

        # Locate which method(s) gold cannot pass, so the exclusion is specific.
        broken = []
        for sub in prob["subtasks"]:
            m_ok, _ = run_classeval_method(prob, gold, sub)
            if not m_ok:
                broken.append(sub["name"])

        gold_defs = set(re.findall(r"^\s*def\s+(\w+)", prob["solution_code"], re.M))
        known = {s["name"] for s in prob["subtasks"]} | {"__init__"}
        quarantine[prob["task_id"]] = {
            "reason": reason,
            "methods": broken,
            "methods_missing_from_dataset": sorted(gold_defs - known),
            "integration_only": not broken,
            "evidence": (evidence or "").strip()[:400],
        }
        print(f"  [{i}/{len(tasks)}] {prob['task_id']:<14} QUARANTINE {reason}"
              + (f" (methods: {', '.join(broken)})" if broken else " (integration test only)"))

    kept = len(tasks) - len(quarantine)
    print(f"\n{kept}/{len(tasks)} tasks usable; {len(quarantine)} quarantined "
          f"in {time.time() - t0:.0f}s")
    by_reason = {}
    for v in quarantine.values():
        by_reason[v["reason"]] = by_reason.get(v["reason"], 0) + 1
    for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<20}{v}")
    if flaky:
        print(f"   flaky task ids: {', '.join(flaky)}")

    path = quarantine_path(args.split)
    if not args.write:
        print(f"\nreport only. Pass --write to record this at {path}")
        return 0

    payload = {
        "split": args.split,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "repeat": args.repeat,
        "n_tasks": len(tasks),
        "n_quarantined": len(quarantine),
        "tasks": quarantine,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {path}")
    print("load_classeval_problems() now excludes these unless "
          "apply_quarantine=False.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
