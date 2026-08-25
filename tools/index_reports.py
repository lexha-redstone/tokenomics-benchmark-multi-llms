#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Keep `reports/` a chronologically indexed, self-describing log of sweeps.

Reports are named `NN_<dataset>_<tag>_n<N>.<ext>`, where `NN` is the run order.
This tool does two jobs:

  1. **Adopt** any report that is not indexed yet — a sweep launched from an
     older revision, or an ad-hoc file someone dropped in — by renaming it to
     the next index in modification-time order.
  2. **Regenerate** `reports/README.md`, the index a newcomer reads first.

Usage:
    python3 tools/index_reports.py            # show what would change
    python3 tools/index_reports.py --apply    # rename and rewrite the index
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORTS = os.path.join(ROOT, "reports")

INDEX_RE = re.compile(r"^(\d{2})_(.+)\.(md|html)$")
README = "README.md"

# What each historical index was. Run timestamps were not recorded for the
# middle group, so their order is inferred from task count and model
# generation; that inference is stated here rather than presented as fact.
NOTES = {
    "01": ("2026-07-09", "BigCodeBench-Hard", "N=30",
           "First sweet-spot sweep. gemini-3.1/3.5 + claude-opus-4, pre-straitjacket."),
    "02": ("2026-07-09", "BigCodeBench-Hard", "N=30",
           "Revision of the 01 dashboard (opus-4 arm dropped)."),
    "03": ("2026-07-13", "WebDev", "-",
           "WebDev sweet-spot dashboard, pre-straitjacket."),
    "04": ("inferred", "BigCodeBench-Hard", "N=10",
           "First straitjacket pilot."),
    "05": ("inferred", "BigCodeBench-Hard", "N=30",
           "Straitjacket comparative TCO across all arms."),
    "06": ("inferred", "BigCodeBench-Hard", "N=50",
           "Gemini vs Claude head-to-head."),
    "10": ("inferred", "WebDev", "N=2",
           "Straitjacket smoke run on web/networking tasks."),
    "11": ("2026-08-06", "cross-dataset", "N=10/30/50",
           "Synthesis of every sweep up to that date."),
    "12": ("2026-08-22", "BigCodeBench-Hard", "N=100",
           "Largest sweep. gemini-3.7 + claude-opus-5, full containment receipt."),
    "13": ("2026-08-22", "BigCodeBench-Hard", "N=1",
           "Routing-study smoke run: the ten R1-R10 ladders, one task."),
    "15": ("2026-08-23", "ClassEval", "N=2",
           "ClassEval smoke run: the hypothesis arm and the shape it has to beat."),
    "16": ("2026-08-24", "ClassEval", "N=91",
           "Full sub-task routing comparison, eight arms over the scorable classes."),
    "17": ("2026-08-24", "ClassEval", "N=91",
           "Same sweep with the claude-opus-5 baseline merged in."),
    "19": ("2026-08-24", "BigCodeBench-Hard", "N=148",
           "The routing study, run over the COMPLETE dataset. Eleven arms: "
           "gemini-3.7 ladders with claude-opus-5 gated behind them."),
    "20": ("2026-08-25", "FeatureBench", "N=48",
           "The expensive-oracle study, arms F0a-F3. **Do not rank these rows.** "
           "F0a/F0b/F1/F2 were replayed from a cache written at 3 oracle calls "
           "while F3 ran live at 2, the labels were rewritten for the newer "
           "budget (F1 actually used claude-opus-5 on 41/48 tasks), and F2 "
           "carries `routing.degraded` on 45/48. Audit: "
           "[docs/featurebench-n48-lessons.md](../docs/featurebench-n48-lessons.md)."),
    "22": ("2026-08-25", "FeatureBench", "N=48",
           "Three follow-up arms (F4-F6) aimed at the patch-application failure. "
           "At `MAX_ORACLE_CALLS = 2` none of them can reach the frontier rung, "
           "so F4 and F5 ran the IDENTICAL ladder and its section 3 'cheapest per "
           "solved' claim is a coin flip (Fisher p = 0.27). Audit: "
           "[docs/featurebench-n48-lessons.md](../docs/featurebench-n48-lessons.md)."),
}

# Indices whose reports are gone, and why. Printed under the table so a gap in
# the numbering reads as a decision rather than a missing file.
WITHDRAWN = {
    "07": "SWE-bench Pro N=10 — withdrawn: the SWE-bench Pro path never ran the "
          "repository's tests, so its rows were canonical-patch substring scores, "
          "not pass rates.",
    "08": "SWE-bench Pro N=30 — withdrawn for the same reason as 07.",
    "09": "SWE-bench Pro N=30 (live API) — withdrawn for the same reason as 07.",
    "14": "never produced a report file.",
    "18": "a byte-identical duplicate of 19, written by the same sweep. Removed "
          "so the N=148 result has one address.",
}


def _report_files():
    return sorted(f for f in os.listdir(REPORTS)
                  if f.lower().endswith((".md", ".html")) and f != README)


# Role suffixes that distinguish a sweep's markdown from its dashboard.
_ROLE_SUFFIXES = ("_straitjacket_report", "_dashboard", "_report", "_tco_report")


def _sweep_key(name):
    """Identity of the sweep a report file belongs to, ignoring its role."""
    stem = os.path.splitext(name)[0]
    for suffix in _ROLE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _next_index(names):
    highest = 0
    for n in names:
        m = INDEX_RE.match(n)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def adopt(apply_changes):
    """Give every un-indexed report the next index, oldest first."""
    names = _report_files()
    orphans = [n for n in names if not INDEX_RE.match(n)]
    if not orphans:
        return []

    orphans.sort(key=lambda n: os.path.getmtime(os.path.join(REPORTS, n)))
    # A markdown report and its dashboard belong to one sweep, so they share
    # an index rather than consuming two. Their filenames differ by a role
    # suffix (`..._straitjacket_report.md` vs `..._dashboard.html`), so the
    # suffix is stripped before grouping.
    seen = {}
    for n in orphans:
        seen.setdefault(_sweep_key(n), []).append(n)
    groups = list(seen.values())

    idx = _next_index(names)
    planned = []
    for group in groups:
        for n in group:
            stem, ext = os.path.splitext(n)
            new = f"{idx:02d}_{stem}{ext}"
            planned.append((n, new))
        idx += 1

    for old, new in planned:
        print(f"  {old}  ->  {new}")
        if apply_changes:
            src, dst = os.path.join(REPORTS, old), os.path.join(REPORTS, new)
            if subprocess.run(["git", "mv", src, dst], cwd=ROOT,
                              capture_output=True).returncode != 0:
                os.rename(src, dst)
    return planned


def _headline(path):
    """First markdown heading, used as the index entry's title."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("# "):
                    return line[2:].strip()
    except Exception:
        pass
    return ""


def write_index(apply_changes):
    rows = {}
    for n in _report_files():
        m = INDEX_RE.match(n)
        if not m:
            continue
        rows.setdefault(m.group(1), {})[m.group(3)] = n

    lines = [
        "# Benchmark Reports",
        "",
        "Every sweep this repository has run, in execution order. The number is the",
        "order, not a version: reports are append-only, so a later run never",
        "overwrites an earlier one's evidence.",
        "",
        "Regenerate this index with `python3 tools/index_reports.py --apply`.",
        "",
        "| # | Date | Dataset | Tasks | Report | Dashboard | What it was |",
        "|---|---|---|---|---|---|---|",
    ]
    for idx in sorted(rows):
        files = rows[idx]
        date, dataset, n_tasks, note = NOTES.get(idx, ("-", "-", "-", ""))
        md = files.get("md")
        html = files.get("html")
        if not note and md:
            note = _headline(os.path.join(REPORTS, md))
        md_cell = f"[md]({md})" if md else "—"
        html_cell = f"[html]({html})" if html else "—"
        lines.append(f"| {idx} | {date} | {dataset} | {n_tasks} | {md_cell} | "
                     f"{html_cell} | {note} |")

    if WITHDRAWN:
        lines += ["", "**Gaps in the numbering.** Indices are never reused, so a "
                  "missing number is a report that was withdrawn:", ""]
        for idx in sorted(WITHDRAWN):
            lines.append(f"- **{idx}** — {WITHDRAWN[idx]}")

    lines += [
        "",
        "`inferred` dates: exact run timestamps were not recorded for those sweeps.",
        "Their order is derived from task count and model generation, which is why it",
        "is labelled rather than stated as fact.",
        "",
        "Methodology documents live in [`../docs/`](../docs/), not here — this",
        "directory holds run results only.",
        "",
    ]
    text = "\n".join(lines)
    out = os.path.join(REPORTS, README)
    if apply_changes:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {out}")
    else:
        print(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="perform the renames and write reports/README.md")
    ap.add_argument("--index-only", action="store_true",
                    help="rewrite reports/README.md without renaming anything "
                         "(use while a sweep still owns its output paths)")
    args = ap.parse_args()

    if not os.path.isdir(REPORTS):
        print(f"no reports directory at {REPORTS}", file=sys.stderr)
        return 1

    if args.index_only:
        write_index(True)
        return 0

    print("Adopting un-indexed reports:" if args.apply else
          "Would adopt un-indexed reports:")
    if not adopt(args.apply):
        print("  (none)")
    print()
    write_index(args.apply)
    if not args.apply:
        print("\n(dry run — pass --apply to make these changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
