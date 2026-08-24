# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Contract tests for merging a single-arm run back into a sweep (`src/merge.py`).

`run_classeval_opus5.py` exists so one expensive arm can be run apart from the
group and then compared against it. The merge is where that comparison is
actually built, so these pin the three ways it could quietly produce a results
file that lies:

  * **double-counting.** Re-running the arm must replace its row, not append a
    second one. A results file holding two `ce_single_opus` rows would show the
    same arm twice in every table with no indication which run is current.
  * **position.** A replaced row keeps its slot and an untouched row keeps its
    numbers, so merging cannot silently reorder or perturb the arms that were
    not re-run.
  * **comparability.** Rows scored over different task counts are still merged
    -- dropping one would be worse -- but the mismatch has to be reported. A
    62% over 91 tasks and an 86% over 12 do not belong in one column unread.

No test here calls a model or touches the repository's own results files.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.merge import (comparability_warnings, load_results,  # noqa: E402
                       merge_into_file, merge_summary_rows)


def _row(rid, n=91, passed=60, usd=1.0, **kw):
    row = {"id": rid, "name": f"row {rid}", "n": n, "passed": passed,
           "pass_rate": round(passed / n, 3), "total_as_run_usd": usd,
           "total_triage_usd": 0.0, "cost_per_solved_usd": round(usd / passed, 6),
           "avg_output_tokens": 800.0, "category": "test", "models": "test",
           "results": []}
    row.update(kw)
    return row


def _doc(rows):
    return {"dataset": "classeval", "dataset_name": "ClassEval", "group": "classeval",
            "n": 91, "summary": rows}


def test_rerunning_an_arm_replaces_its_row_rather_than_appending():
    base = [_row("a"), _row("b"), _row("opus", passed=70)]
    merged, replaced, added = merge_summary_rows(base, [_row("opus", passed=78)])

    assert [r["id"] for r in merged] == ["a", "b", "opus"]
    assert replaced == ["opus"] and added == []
    assert merged[2]["passed"] == 78


def test_a_new_arm_is_appended_and_the_existing_rows_are_untouched():
    base = [_row("a"), _row("b")]
    merged, replaced, added = merge_summary_rows(base, [_row("opus", passed=78)])

    assert [r["id"] for r in merged] == ["a", "b", "opus"]
    assert (replaced, added) == ([], ["opus"])
    assert merged[:2] == base


def test_merging_is_idempotent():
    base = [_row("a"), _row("opus")]
    once, _, _ = merge_summary_rows(base, [_row("opus", passed=78)])
    twice, _, _ = merge_summary_rows(once, [_row("opus", passed=78)])
    assert once == twice


def test_mismatched_task_counts_are_reported_not_hidden():
    rows = [_row("a", n=91), _row("opus", n=12, passed=10)]
    notes = " ".join(comparability_warnings(rows))
    assert "NOT directly comparable" in notes
    assert "N=91" in notes and "N=12" in notes

    assert comparability_warnings([_row("a"), _row("b")]) == []


def test_simulated_and_dropped_tasks_are_reported():
    rows = [_row("a", simulated_tasks=2),
            _row("b", incomplete_tasks=[{"task_id": "t1", "kind": "auth"}])]
    notes = " ".join(comparability_warnings(rows))
    assert "SIMULATED" in notes
    assert "never completed" in notes


def test_merge_into_file_backs_up_what_it_replaced(tmp_path):
    base_path = tmp_path / "sweep.json"
    base_path.write_text(json.dumps(_doc([_row("a"), _row("b")])), encoding="utf-8")

    doc, path = merge_into_file(str(base_path), [_row("opus", passed=78)])

    assert path == str(base_path)
    assert [r["id"] for r in doc["summary"]] == ["a", "b", "opus"]
    # The pre-merge file is recoverable, so a bad merge is never a lost sweep.
    backup = load_results(str(base_path) + ".bak")
    assert [r["id"] for r in backup["summary"]] == ["a", "b"]
    assert load_results(str(base_path)) == doc


def test_merge_into_a_missing_sweep_says_how_to_produce_one(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        merge_into_file(str(tmp_path / "nope.json"), [_row("opus")])
    assert "run_benchmark.py" in str(excinfo.value)
