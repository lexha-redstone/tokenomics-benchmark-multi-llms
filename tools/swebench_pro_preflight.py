#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Run SWE-bench Pro's own gold patches before spending anything on models.

Why this exists
---------------
The same reason FeatureBench has one, learned the same way: if a row cannot be
scored on this machine, it cannot be scored for *any* arm, and finding that out
mid-sweep means two machines have measured different task sets while reporting
comparable-looking pass rates. SWE-bench Pro removes the biggest of those
failure modes -- nothing has to be rebuilt locally -- but not the rest: an
image that will not pull, a run script whose dependency install needs network
the sandbox denied, a suite too slow for the timeout.

So gold runs first. Every row whose own reference patch does not resolve is
written to `swebench_pro/data/quarantine-<split>.json` with the reason, and the
loader honours it. Tasks are excluded, never edited.

**The quarantine file is environment-specific. Regenerate it on each machine;
do not copy it between them.**

A gold run is also the only honest smoke test of this harness. It exercises
every step an arm's attempt takes -- container start, reset, `git apply`, the
graded-test restore, upstream's run script, upstream's parser, the resolution
rule -- against a patch that is known to be correct. If gold does not pass, the
harness is wrong, not the model.

Usage
-----
    python3 tools/swebench_pro_preflight.py --list                # rows by language
    python3 tools/swebench_pro_preflight.py --ready               # images already local
    python3 tools/swebench_pro_preflight.py --languages python --pull --n 5
    python3 tools/swebench_pro_preflight.py --gold 3              # the real smoke test
    python3 tools/swebench_pro_preflight.py --gold 0 --write      # whole split, quarantine
"""

import argparse
import collections
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.datasets import (SWEBENCH_PRO_DEFAULT_SPLIT, ensure_swebench_pro_scripts,
                          load_swebench_pro_problems, sbp_required_tests,
                          swebench_pro_quarantine_path)
from src.evaluator import SWEBenchProEnv, docker_available, _sbp_default_platform
from src import straitjacket as sj


def classify(evidence_text, setup_error=""):
    """Name the failure so a quarantine entry can be fixed rather than accepted."""
    t = (str(setup_error) + " " + str(evidence_text)).lower()
    if "no such image" in t or "manifest unknown" in t or "pull access denied" in t:
        return "missing_image"
    if "docker run failed" in t or "container unavailable" in t:
        return "container_start_failed"
    if "no git repository" in t or "base commit" in t:
        return "image_tree_mismatch"
    if "did not apply" in t:
        return "gold_patch_does_not_apply"
    if "restore the graded test files" in t:
        return "test_restore_failed"
    if "run_script" in t and "not" in t:
        return "missing_run_script"
    if "timed out" in t or "timeout" in t:
        return "timeout"
    if "parser failed" in t:
        return "parser_failed"
    return "gold_fails_tests"


def _image_present(image):
    p = subprocess.run(["docker", "image", "inspect", image],
                       capture_output=True, text=True, timeout=120)
    return p.returncode == 0


def report_list(problems):
    by_lang = collections.Counter(str(p.get("repo_language") or "?")
                                  for p in problems.values())
    by_repo = collections.Counter(str(p.get("repo") or "?") for p in problems.values())
    print(f"{len(problems)} rows")
    print("\nby language:")
    for lang, n in by_lang.most_common():
        print(f"  {lang:<6} {n:>4}")
    print("\nby repository (top 15):")
    for repo, n in by_repo.most_common(15):
        print(f"  {repo:<44} {n:>4}")
    tests = [len(sbp_required_tests(p)) for p in problems.values()]
    if tests:
        tests.sort()
        print(f"\nrequired tests per row: min {tests[0]}  median "
              f"{tests[len(tests) // 2]}  max {tests[-1]}")


def report_ready(problems, out_path=""):
    """Which rows can run right now, with no pull. Feeds `--tasks @file`."""
    ok, missing = [], []
    seen = {}
    for iid, p in problems.items():
        image = p.get("image_name") or ""
        if image not in seen:
            seen[image] = _image_present(image)
        (ok if seen[image] else missing).append(iid)
    print(f"{len(ok)} row(s) runnable now; {len(missing)} need a pull "
          f"({len(seen)} distinct images, {sum(seen.values())} local)")
    if out_path and ok:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ok) + "\n")
        print(f"wrote {len(ok)} ids to {os.path.relpath(out_path, ROOT)}")
    return ok


def pull_images(problems, timeout=7200):
    """Pre-pull, so the first attempt's clock is not a download."""
    platform = os.environ.get("SBP_PLATFORM") or _sbp_default_platform()
    images = []
    for p in problems.values():
        image = p.get("image_name") or ""
        if image and image not in images:
            images.append(image)
    failed = {}
    for i, image in enumerate(images, 1):
        if _image_present(image):
            print(f"[{i}/{len(images)}] {image}  (already local)")
            continue
        argv = ["docker", "pull"] + (["--platform", platform] if platform else []) + [image]
        print(f"[{i}/{len(images)}] {' '.join(argv)}", flush=True)
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            failed[image] = (r.stderr or "").strip()[-200:]
            print(f"    FAILED: {failed[image]}")
    print(f"\n{len(images) - len(failed)}/{len(images)} images available")
    return failed


def run_gold(problems, limit=0, only=None):
    """Apply each row's own `patch` and run its tests. Returns per-task verdicts."""
    if only:
        items = [(t, problems[t]) for t in only if t in problems]
    else:
        items = list(problems.items())
    if limit:
        items = items[:limit]
    verdicts = {}
    for i, (iid, problem) in enumerate(items, 1):
        gold = problem.get("patch") or ""
        head = f"[{i}/{len(items)}] {iid}"
        if not gold.strip():
            print(f"{head}  QUARANTINE (no gold patch)")
            verdicts[iid] = {"ok": False, "reason": "no_gold_patch", "evidence": ""}
            continue
        if not sbp_required_tests(problem):
            print(f"{head}  QUARANTINE (names no required tests)")
            verdicts[iid] = {"ok": False, "reason": "no_required_tests", "evidence": ""}
            continue
        try:
            ensure_swebench_pro_scripts(iid)
        except Exception as e:                               # noqa: BLE001
            print(f"{head}  QUARANTINE (run script unavailable: {e})")
            verdicts[iid] = {"ok": False, "reason": "missing_run_script",
                             "evidence": str(e)[:300]}
            continue

        with SWEBenchProEnv(problem) as env:
            passed, evidence = env.score(gold)
            report, setup_error = env.last_report, env.setup_error
        if passed:
            print(f"{head}  ok  ({report.get('required')} required tests passed)")
            verdicts[iid] = {"ok": True, "reason": "gold_passes",
                             "required": report.get("required")}
        else:
            reason = classify(evidence, setup_error)
            ratio = report.get("test_pass_ratio")
            print(f"{head}  QUARANTINE ({reason})"
                  + (f"  test_pass_ratio={ratio}" if ratio is not None else "")
                  + (f"  missing={report.get('missing_names')}" if report.get("missing_names") else ""))
            verdicts[iid] = {
                "ok": False, "reason": reason,
                "test_pass_ratio": ratio,
                "missing": report.get("missing"),
                "missing_names": report.get("missing_names"),
                "reported": report.get("reported"),
                "evidence": str(evidence)[:600],
                "setup_error": str(setup_error)[:300],
            }
    return verdicts


def write_quarantine(split, verdicts, n_total):
    bad = {iid: v for iid, v in verdicts.items() if not v["ok"]}
    path = swebench_pro_quarantine_path(split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "_comment": ("Environment-specific. Regenerate on each machine; do not copy "
                     "between them. A `missing_image` reason means `docker pull` "
                     "failed, not that the task is broken."),
        "split": split,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tasks": n_total,
        "n_checked": len(verdicts),
        "partial": len(verdicts) < n_total,
        "n_quarantined": len(bad),
        "tasks": bad,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {os.path.relpath(path, ROOT)} -- {len(bad)} quarantined "
          f"of {len(verdicts)} checked")


def _requested_tasks(spec):
    spec = (spec or "").strip()
    if not spec:
        return []
    if spec.startswith("@"):
        with open(spec[1:], "r", encoding="utf-8") as f:
            items = [ln.strip() for ln in f]
    else:
        items = spec.split(",")
    seen, out = set(), []
    for t in (i.strip() for i in items):
        if t and not t.startswith("#") and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=SWEBENCH_PRO_DEFAULT_SPLIT)
    ap.add_argument("--languages", default="",
                    help="comma-separated repo_language filter: python, go, js, ts")
    ap.add_argument("--n", type=int, default=0,
                    help="limit how many rows are loaded (0 = the whole split)")
    ap.add_argument("--tasks", default="",
                    help="operate on exactly these instance ids (comma list or @file)")
    ap.add_argument("--list", action="store_true",
                    help="print what the split holds, then stop")
    ap.add_argument("--ready", action="store_true",
                    help="list rows whose image is already local")
    ap.add_argument("--ready-out", default="", metavar="PATH",
                    help="write the ready ids one per line, for `--tasks @PATH`")
    ap.add_argument("--pull", action="store_true", help="pre-pull the images")
    ap.add_argument("--pull-timeout", type=int, default=7200)
    ap.add_argument("--gold", type=int, default=-1, metavar="N",
                    help="run N rows' gold patches (0 = all). The real smoke test.")
    ap.add_argument("--write", action="store_true",
                    help="write the quarantine file from the gold verdicts")
    args = ap.parse_args()

    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    problems = load_swebench_pro_problems(
        split=args.split, max_tasks=args.n or None,
        # The point of a preflight is to decide what goes in the quarantine
        # file, so it must see the rows the file already excludes.
        apply_quarantine=False, languages=langs)
    only = _requested_tasks(args.tasks)
    if only:
        problems = {k: v for k, v in problems.items() if k in only}

    if args.list:
        report_list(problems)
        return 0

    needs_docker = args.ready or args.pull or args.gold >= 0
    if needs_docker:
        ok, why = docker_available()
        print(f"docker: {why}")
        if not ok:
            print("\nThis harness needs a working Docker daemon. See "
                  "docs/swebench-pro-setup.md.", file=sys.stderr)
            return 2

    if args.gold >= 0:
        # The executor captures every test run through the harness, so a gold
        # run without it fails with "execution error" on every row and looks
        # like a broken dataset. Say which it is, before spending the pulls.
        st = sj.status()
        print(f"straitjacket: backend={st['backend']} ctx={st['ctx_version']}")
        if st["backend"] == "off":
            print("\nThe harness captures every test run, so gold cannot be scored "
                  "without it: `pip install ctx-harness`.", file=sys.stderr)
            return 2
        if st["backend"] != "library":
            print("\nWARNING: sbp_evidence_gate needs the library backend to read "
                  "typed evidence. Under any other backend it degrades into "
                  "sbp_cascade and its row must not be quoted as an evidence-gate "
                  "result.", file=sys.stderr)

    if args.ready:
        report_ready(problems, args.ready_out)
    if args.pull:
        pull_images(problems, timeout=args.pull_timeout)
    if args.gold >= 0:
        verdicts = run_gold(problems, limit=args.gold, only=only)
        n_ok = sum(1 for v in verdicts.values() if v["ok"])
        print(f"\ngold: {n_ok}/{len(verdicts)} rows resolved")
        if args.write:
            write_quarantine(args.split, verdicts, len(problems))
        elif n_ok < len(verdicts):
            print("re-run with --write to quarantine the failures")
        # A gold run that resolves nothing is a broken harness, not a hard
        # dataset: say so with an exit code a CI step can read.
        return 0 if n_ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
