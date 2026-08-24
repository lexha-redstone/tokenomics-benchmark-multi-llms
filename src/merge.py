# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Fold a single-arm run's summary row back into an existing sweep's results.

Why this module exists
----------------------
A sweep writes `<ds>/results/<ds>_<group>_results.json` and a report pair whose
tables are built straight from `summary`. When one extra arm is run on its own
afterwards (`run_classeval_opus5.py`), the comparison the run was for only
exists once its row sits in the *same* list as the arms it is being compared
against.

Merging is by variant `id`: a re-run replaces its previous row in place rather
than appending a second one, so running the opus arm twice cannot produce a
results file that scores it twice. The original file is never edited blind --
`merge_into_file` writes a `.bak` of what it replaced.

Comparability is checked, not assumed. Rows that were scored over a different
number of completed tasks are still merged (dropping a row silently would be
worse) but reported loudly, because a pass rate over 91 tasks and one over 40
do not belong in the same column without the reader being told.
"""

import json
import os
import shutil

from .reporter import allocate_report_paths, generate_markdown_report, generate_html_dashboard


def load_results(path):
    """A sweep results document, or None when it does not exist yet."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_summary_rows(base_rows, new_rows):
    """Base rows with `new_rows` folded in, matched on variant `id`.

    Order is preserved: a replaced row keeps its original position, a genuinely
    new arm is appended. Returns (rows, replaced_ids, added_ids).
    """
    merged = list(base_rows or [])
    index = {r.get("id"): i for i, r in enumerate(merged) if r.get("id")}
    replaced, added = [], []

    for row in new_rows or []:
        rid = row.get("id")
        if rid is not None and rid in index:
            merged[index[rid]] = row
            replaced.append(rid)
        else:
            if rid is not None:
                index[rid] = len(merged)
            merged.append(row)
            added.append(rid)
    return merged, replaced, added


def comparability_warnings(rows):
    """Human-readable notes about rows that are not scored on equal footing."""
    notes = []
    counts = {}
    for r in rows:
        counts.setdefault(r.get("n"), []).append(r.get("name", r.get("id")))
    if len(counts) > 1:
        detail = "; ".join(f"N={k}: " + ", ".join(v) for k, v in sorted(
            counts.items(), key=lambda kv: -len(kv[1])))
        notes.append("arms were scored over different task counts -- pass rates are "
                     f"NOT directly comparable ({detail})")
    sim = [r.get("name", r.get("id")) for r in rows if r.get("simulated_tasks")]
    if sim:
        notes.append("rows containing SIMULATED tasks: " + ", ".join(sim))
    incomplete = [f"{r.get('name', r.get('id'))} ({len(r['incomplete_tasks'])} dropped)"
                  for r in rows if r.get("incomplete_tasks")]
    if incomplete:
        notes.append("rows with tasks that never completed: " + ", ".join(incomplete))
    return notes


def merge_into_file(base_path, new_rows, out_path=None, backup=True):
    """Merge `new_rows` into the sweep document at `base_path`.

    Writes to `out_path` (default: in place) and returns (document, path).
    """
    doc = load_results(base_path)
    if doc is None:
        raise FileNotFoundError(
            f"no existing results to merge into at {base_path}. Run the full sweep "
            "first (e.g. `python3 run_benchmark.py --dataset classeval "
            "--group classeval --n 91 --report`), or pass --base to point at the "
            "results file you meant.")

    merged, replaced, added = merge_summary_rows(doc.get("summary", []), new_rows)
    doc["summary"] = merged
    doc["n"] = max([r.get("n", 0) for r in merged] or [doc.get("n", 0)])

    target = out_path or base_path
    if backup and os.path.exists(target) and os.path.abspath(target) == os.path.abspath(base_path):
        shutil.copyfile(target, target + ".bak")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    for rid in replaced:
        print(f"  merged: replaced existing row `{rid}`")
    for rid in added:
        print(f"  merged: added new row `{rid}`")
    return doc, target


def render_reports(rows, dataset_name, tag="straitjacket", n_tasks=None):
    """Markdown + HTML for a merged row set, under one new report index."""
    n = n_tasks or max([r.get("n", 0) for r in rows] or [0])
    md_path, html_path = allocate_report_paths(dataset_name, n, tag=tag)
    md = generate_markdown_report(rows, dataset_name=dataset_name, output_path=md_path)
    html = generate_html_dashboard(rows, dataset_name=dataset_name, output_path=html_path)
    return md, html
