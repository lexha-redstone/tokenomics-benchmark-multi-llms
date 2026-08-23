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
from src.datasets import (CLASSEVAL_DEFAULT_SPLIT, CLASSEVAL_NLTK_CORPORA,
                          classeval_install_hint, classeval_missing_modules,
                          classeval_required_modules, load_classeval_problems,
                          quarantine_path)
from src.evaluator import run_classeval_class, run_classeval_method
from src.paths import display as rel, scrub

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


_MODULE_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")


def classify(evidence_text):
    """Name the cause, and for a missing import name the MODULE too.

    "missing_module" on its own is not actionable -- the reader cannot tell
    whether to install something or to accept the loss. The module name turns
    it into a one-line fix.
    """
    text = evidence_text or ""
    for name, pat in REASONS:
        if pat.search(text):
            if name == "missing_module":
                m = _MODULE_RE.search(text)
                return f"missing_module:{m.group(1)}" if m else name
            return name
    return "gold_fails"          # real, but cause not auto-identifiable


def _missing_corpora():
    """nltk corpora that pip does not install and `nltk.data.find` cannot see."""
    try:
        import nltk
    except Exception:
        return list(CLASSEVAL_NLTK_CORPORA)
    probes = {"punkt": "tokenizers/punkt",
              "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
              "wordnet": "corpora/wordnet",
              "omw-1.4": "corpora/omw-1.4"}
    gap = []
    for name in CLASSEVAL_NLTK_CORPORA:
        try:
            nltk.data.find(probes.get(name, f"corpora/{name}"))
        except Exception:
            gap.append(name)
    return gap


def check_dependencies(args):
    """Report the dataset's third-party imports. True when all are importable.

    Run BEFORE the gold loop, because discovering a missing package one task at
    a time costs a full execution pass and then reports it as a quarantined
    task, which is the wrong conclusion: the task is fine, the machine is not.
    """
    problems = load_classeval_problems(split=args.split, apply_quarantine=False)
    required = classeval_required_modules(problems)
    missing = classeval_missing_modules(problems)

    print(f"dependencies: {len(required)} third-party module(s) used by the split")
    for name, tasks in required.items():
        state = "MISSING" if name in missing else "ok"
        print(f"  {name:<14}{len(tasks):>3} task(s)  {state}")

    corpora_gap = _missing_corpora() if "nltk" in required else []
    if not missing and not corpora_gap:
        print("  all present\n")
        return True

    if corpora_gap and not missing:
        print(f"\n  nltk is installed but {len(corpora_gap)} corpus/corpora are not: "
              f"{', '.join(corpora_gap)}")
        print("  fix:  python3 -c \"import nltk; "
              f"[nltk.download(c) for c in {tuple(corpora_gap)!r}]\"")
        print("\n  Same class of problem as a missing package: the task is fine, "
              "this\n  machine cannot run it.\n")
        return False

    blocked = sorted({t for tasks in missing.values() for t in tasks})
    print(f"\n  {len(missing)} module(s) missing, blocking {len(blocked)} task(s): "
          f"{', '.join(blocked)}")
    print(f"  fix:  {classeval_install_hint(missing)}")
    print(f"  or:   pip install -r {rel(os.path.join(ROOT, 'classeval', 'requirements.txt'))}")
    if corpora_gap:
        print(f"  nltk corpora also missing: {', '.join(corpora_gap)}")
    if "nltk" in missing or "nltk" in required:
        corpora = ",".join(repr(c) for c in CLASSEVAL_NLTK_CORPORA)
        print("  nltk also needs its corpora:\n"
              f"        python3 -c \"import nltk; [nltk.download(c) for c in ({corpora})]\"")
    print("\n  These tasks are NOT broken -- this machine cannot run them. Quarantining"
          "\n  them would shrink the benchmark and make these numbers incomparable with"
          "\n  a fully provisioned machine. Install and re-run, or pass"
          "\n  --accept-missing-deps to proceed anyway.\n")
    return False


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
    ap.add_argument("--deps-only", action="store_true",
                    help="report the dataset's third-party requirements and stop")
    ap.add_argument("--accept-missing-deps", action="store_true",
                    help="run anyway with packages missing. The tasks that need "
                         "them are quarantined, which shrinks the benchmark and "
                         "makes this machine's numbers incomparable with a fully "
                         "provisioned one.")
    args = ap.parse_args()

    if not check_dependencies(args) and not args.accept_missing_deps:
        return 2
    if args.deps_only:
        return 0

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
            "evidence": scrub((evidence or "").strip())[:400],
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
        print(f"   {k:<28}{v}")
    if flaky:
        print(f"   flaky task ids: {', '.join(flaky)}")

    path = quarantine_path(args.split)
    if not args.write:
        print(f"\nreport only. Pass --write to record this at {rel(path)}")
        return 0

    payload = {
        "_comment": ("Environment-specific. Regenerate on each machine; do not "
                     "copy between them. A `missing_module:*` reason means a "
                     "package is absent, not that the task is broken -- see "
                     "classeval/requirements.txt."),
        "split": args.split,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "repeat": args.repeat,
        "n_tasks": len(tasks),
        "n_quarantined": len(quarantine),
        "missing_modules": sorted(classeval_missing_modules(split=args.split)),
        "tasks": quarantine,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {rel(path)}")
    print("load_classeval_problems() now excludes these unless "
          "apply_quarantine=False.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
