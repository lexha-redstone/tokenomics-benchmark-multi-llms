#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Run FeatureBench's own gold patches before spending anything on models.

Why this exists
---------------
ClassEval taught the lesson the cheap way: 8 of its 100 classes cannot be
scored on a provisioned machine, and *twelve more* on a bare one -- and if you
find that out during a sweep, two machines have silently measured different
task sets while reporting comparable-looking pass rates. FeatureBench has more
ways to fail, not fewer: an image that will not pull, a `test_patch` that will
not apply, a container whose pytest cannot collect the named files, a workdir
that is not where this repository guessed it is.

So gold runs first. Every row whose own reference patch does not resolve is
written to `featurebench/data/quarantine-<split>.json` with the reason, and the
loader honours it. Tasks are excluded, never edited.

**The quarantine file is environment-specific. Regenerate it on each machine;
do not copy it between them.**

Usage
-----
    python3 tools/featurebench_preflight.py --settings     # what repo_settings holds
    python3 tools/featurebench_preflight.py --disk         # how much --pull will download
    python3 tools/featurebench_preflight.py --pull         # pre-pull the images
    python3 tools/featurebench_preflight.py --n 5          # try 5 rows, report
    python3 tools/featurebench_preflight.py --write        # quarantine what gold fails
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

from src.datasets import (FEATUREBENCH_DEFAULT_SPLIT, featurebench_quarantine_path,
                          load_featurebench_problems)
from src.evaluator import (FeatureBenchEnv, docker_available,
                           featurebench_test_files, featurebench_test_ratio)
from src import straitjacket as sj


def classify(evidence_text, setup_error=""):
    """Name the failure so a quarantine entry can be fixed rather than accepted."""
    t = (str(setup_error) + " " + str(evidence_text)).lower()
    if "no such image" in t or "manifest unknown" in t or "pull access denied" in t:
        return "missing_image"
    if "docker run failed" in t or "container unavailable" in t:
        return "container_start_failed"
    if "test_patch did not apply" in t:
        return "test_patch_conflict"
    if "patch did not apply" in t:
        return "gold_patch_conflict"
    if "no such file or directory" in t or "file or directory not found" in t:
        return "test_file_missing"
    if "modulenotfounderror" in t or "importerror" in t:
        return "missing_module"
    if "error" in t and "collect" in t:
        return "collection_error"
    return "gold_fails"


def show_settings(problems):
    """Print the keys `repo_settings` actually carries.

    `src/datasets._fb_workdir` reads the repository's in-image location from
    this blob, and a wrong binding would silently change what every arm is
    scored on -- so the keys are inspected rather than assumed.
    """
    keys = collections.Counter()
    workdirs = collections.Counter()
    for p in problems.values():
        keys.update((p.get("settings") or {}).keys())
        workdirs[p.get("repo_workdir")] += 1
    print(f"\nrepo_settings keys across {len(problems)} rows:")
    for k, n in keys.most_common():
        print(f"  {k:28} {n}")
    print("\nresolved repo_workdir (what the executor will use):")
    for w, n in workdirs.most_common(10):
        print(f"  {w:40} {n}")
    # The keys the executor actually depends on. Printed with their values
    # because binding a wrong shape here silently changes what every arm is
    # scored on -- the same reason the key list above is inspected rather than
    # assumed.
    print("\nexecution-relevant values (first row that has them):")
    for key in ("test_cmd", "timeout_run", "timeout_one", "timeout_collect",
                "library_name", "repository", "install", "pip_packages"):
        for p in problems.values():
            v = (p.get("settings") or {}).get(key)
            if v not in (None, "", [], {}):
                print(f"  {key:20} {str(v)[:220]!r}")
                break
        else:
            print(f"  {key:20} (absent or empty on every row)")

    sample = next(iter(problems.values()), None)
    if sample:
        print(f"\nsample row `{sample.get('instance_id')}`:")
        print(f"  image_name    {sample.get('image_name')}")
        print(f"  repo          {sample.get('repo')}")
        print(f"  FAIL_TO_PASS  {sample.get('FAIL_TO_PASS')}")
        print(f"  PASS_TO_PASS  {sample.get('PASS_TO_PASS')}")
        print(f"  statement     {len(sample.get('problem_statement') or '')} chars")
        print(f"  gold patch    {len(sample.get('patch') or '')} chars")
    print("\n`repo_workdir` above is only the fallback. At run time the executor "
          "reads the image's own WORKDIR (`docker inspect`) and falls back to the "
          "git root, so the printed guess is used only if both are unavailable.")


def _hub_size(image):
    """Compressed size of one image from the Docker Hub API, or None.

    `full_size` is the sum of the *compressed* layer sizes -- what crosses the
    network. What lands on disk is larger, and layers shared between images are
    stored once, so this is an upper bound on download and a rough lower bound
    on disk. `docker system df` after the pull is the only exact answer.
    """
    import urllib.request

    from src.datasets import _ssl_ctx     # same CA handling as the dataset fetch

    ref = str(image or "").strip()
    if not ref:
        return None
    path, _, tag = ref.partition(":")
    tag = tag or "latest"
    url = f"https://hub.docker.com/v2/repositories/{path}/tags/{tag}"
    try:
        with urllib.request.urlopen(url, timeout=30, context=_ssl_ctx()) as r:
            return json.load(r).get("full_size")
    except Exception:
        return None


def report_disk(problems):
    """Ask the registry what `--pull` is about to download, before it starts."""
    counts = collections.Counter(p.get("image_name") for p in problems.values()
                                 if p.get("image_name"))
    print(f"\n{len(counts)} distinct image(s) for {len(problems)} instances")
    known, unknown, total = 0, 0, 0
    for image, n in counts.most_common():
        size = _hub_size(image)
        if size is None:
            unknown += 1
            print(f"  {'?':>9}  x{n:<3} {image}")
            continue
        known += 1
        total += size
        print(f"  {size / 1e9:7.2f} GB  x{n:<3} {image}")

    if not known:
        print("\ncould not reach the registry; run with network access, or just "
              "`--pull` and watch `docker system df`.")
        return
    mean = total / known
    projected = mean * len(counts)
    print(f"\ncompressed download:  {total / 1e9:.1f} GB measured over {known} "
          f"image(s)" + (f", ~{projected / 1e9:.0f} GB projected for all "
                         f"{len(counts)}" if unknown else ""))
    print(f"on disk after pull:   roughly {projected / 1e9 * 1.4:.0f}-"
          f"{projected / 1e9 * 2.5:.0f} GB")
    print("\nTwo things move that number, in opposite directions:")
    print("  - images are decompressed on disk (up ~1.4-2.5x)")
    print("  - layers shared between images are stored ONCE (down, possibly a lot,")
    print("    since these images are built from a common base)")
    print("`docker system df` after the pull is the only exact answer.")


def _image_present(image):
    """Is this image already in the local store? Cheap, and offline."""
    return subprocess.run(["docker", "image", "inspect", image],
                          capture_output=True, timeout=60).returncode == 0


def pull_images(problems, timeout=7200):
    """Pre-pull every distinct image, resumably.

    Safe to interrupt and re-run. Two things make that true:

      * images already in the local store are skipped outright, so a re-run
        goes straight to where it stopped instead of re-checking the registry
        for everything;
      * `docker pull` itself keeps whatever layers it finished. An interrupted
        pull resumes at **layer** granularity, not byte -- a layer that was
        mid-flight when you stopped restarts, completed layers do not.

    Docker's progress output is deliberately NOT captured. These images are
    ~10 GB each; swallowing the progress bars leaves the terminal silent for
    many minutes and makes a working pull look hung.
    """
    images = sorted({p.get("image_name") for p in problems.values() if p.get("image_name")})
    have = [i for i in images if _image_present(i)]
    todo = [i for i in images if i not in set(have)]
    print(f"\n{len(images)} distinct image(s) for {len(problems)} instances")
    if have:
        print(f"  {len(have)} already present locally -- skipping")
    if not todo:
        print("  nothing to pull.")
        return []

    print(f"  {len(todo)} to pull. Safe to interrupt: finished images are kept "
          f"and a re-run continues from here.\n")
    failed = []
    for i, img in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] docker pull {img}", flush=True)
        try:
            # Inherit stdout/stderr so Docker's own progress is visible.
            rc = subprocess.run(["docker", "pull", img], timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc, note = 1, f"exceeded --pull-timeout ({timeout}s)"
            print(f"    TIMEOUT: {note}")
        except KeyboardInterrupt:
            print(f"\ninterrupted at {i}/{len(todo)}. Completed images are kept; "
                  f"re-run --pull to continue.")
            raise
        if rc != 0:
            failed.append(img)
            print(f"    FAILED (see output above): {img}")
    if failed:
        print(f"\n{len(failed)} image(s) could not be pulled; their instances will "
              "quarantine as `missing_image`.")
    return failed


def run_gold(problems, limit=0):
    """Apply each row's own `patch` and run its tests. Returns per-task verdicts."""
    items = list(problems.items())
    if limit:
        items = items[:limit]
    verdicts = {}
    for i, (iid, problem) in enumerate(items, 1):
        gold = problem.get("patch") or ""
        files = featurebench_test_files(problem)
        head = f"[{i}/{len(items)}] {iid}"
        if not gold.strip():
            # Level 2 rows ship no reference patch; there is nothing to verify
            # and nothing to blame on the environment, so they are left in.
            print(f"{head}  SKIP (no gold patch -- Level 2 row)")
            verdicts[iid] = {"ok": True, "reason": "no_gold_patch", "skipped": True}
            continue
        if not files:
            print(f"{head}  QUARANTINE (names no test file)")
            verdicts[iid] = {"ok": False, "reason": "no_test_files", "evidence": ""}
            continue

        with FeatureBenchEnv(problem) as env:
            passed, evidence = env.score(gold)
        ratio = featurebench_test_ratio(evidence)
        if passed:
            print(f"{head}  ok")
            verdicts[iid] = {"ok": True, "reason": "gold_passes"}
        else:
            reason = classify(evidence, env.setup_error)
            print(f"{head}  QUARANTINE ({reason})"
                  + (f"  test_pass_ratio={ratio}" if ratio is not None else ""))
            verdicts[iid] = {
                "ok": False, "reason": reason,
                "test_pass_ratio": ratio,
                "evidence": str(evidence)[:600],
                "setup_error": str(env.setup_error)[:300],
            }
    return verdicts


def write_quarantine(split, problems, verdicts, n_total):
    bad = {iid: v for iid, v in verdicts.items() if not v["ok"]}
    path = featurebench_quarantine_path(split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "_comment": ("Environment-specific. Regenerate on each machine; do not copy "
                     "between them. A `missing_image` reason means `docker pull` "
                     "failed, not that the task is broken."),
        "split": split,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tasks": n_total,
        "n_checked": len(verdicts),
        "n_quarantined": len(bad),
        "tasks": bad,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {os.path.relpath(path, ROOT)} -- {len(bad)} quarantined "
          f"of {len(verdicts)} checked")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=FEATUREBENCH_DEFAULT_SPLIT)
    ap.add_argument("--n", type=int, default=0, help="limit rows checked (0 = all)")
    ap.add_argument("--settings", action="store_true",
                    help="print what repo_settings holds, then stop")
    ap.add_argument("--disk", action="store_true",
                    help="ask the registry how much --pull will download, then stop")
    ap.add_argument("--pull", action="store_true",
                    help="pre-pull every distinct image before checking "
                         "(resumable: already-present images are skipped)")
    ap.add_argument("--pull-timeout", type=int, default=7200,
                    help="seconds allowed per image (default 7200); these are "
                         "~10 GB each, so a short cap kills a healthy pull")
    ap.add_argument("--write", action="store_true",
                    help="write the quarantine file the loader honours")
    args = ap.parse_args()

    problems = load_featurebench_problems(split=args.split, apply_quarantine=False)
    if not problems:
        print("no FeatureBench rows loaded", file=sys.stderr)
        return 2

    if args.settings:
        show_settings(problems)
        return 0

    if args.disk:
        report_disk(problems)
        return 0

    ok, why = docker_available()
    print(f"docker: {why}")
    if not ok:
        print("\nFeatureBench cannot run without Docker. See docs/featurebench-setup.md.",
              file=sys.stderr)
        return 2

    st = sj.status()
    print(f"straitjacket: backend={st['backend']} ctx={st['ctx_version']}")
    if st["backend"] != "library":
        print("\nWARNING: fb_evidence_gate needs the library backend to read typed "
              "evidence. Under any other backend it degrades into fb_cascade and "
              "its row must not be quoted as an evidence-gate result.", file=sys.stderr)

    if args.pull:
        pull_images(problems, timeout=args.pull_timeout)

    verdicts = run_gold(problems, limit=args.n)
    counts = collections.Counter(v["reason"] for v in verdicts.values() if not v["ok"])
    good = sum(1 for v in verdicts.values() if v["ok"])
    print(f"\ngold passes on {good}/{len(verdicts)} checked")
    for reason, n in counts.most_common():
        print(f"  {reason:24} {n}")

    if args.write:
        write_quarantine(args.split, problems, verdicts, len(problems))
    elif counts:
        print("\nre-run with --write to record these so the loader skips them.")

    # Exit 2 while rows are unscorable and unrecorded: running a sweep now would
    # charge the models for the environment, exactly as ClassEval's preflight
    # refuses to.
    return 0 if (good == len(verdicts) or args.write) else 2


if __name__ == "__main__":
    sys.exit(main())
