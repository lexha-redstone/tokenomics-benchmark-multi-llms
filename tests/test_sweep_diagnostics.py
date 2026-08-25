# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Contract tests for the measurements a sweep must not drop on the floor.

Why this exists
---------------
`src/swebench_pro.py` computes `test_pass_ratio` per task, with a comment
saying exactly why: *"on a dataset where frontier agents resolve 20-40%, a
binary verdict makes every cheap arm read as an undifferentiated zero."* It
then handed the value to `src/sweep.py`, which rebuilt the record from a
whitelist of keys that did not include it. The partial credit was computed,
dropped before the cache write, and never reached a report — so a sweep whose
arms all scored 0/50 published a table of zeroes while the one number that
would have distinguished them was thrown away in transit.

`routing.degraded` had the same fate one layer later: carried into the results
file and never rendered by the reporter.

Pinned here:

  * every diagnostic a task record produces survives into the arm's row;
  * `suite_reach_rate` counts *attempts*, not tasks, because that is the
    quantity that separates "hard dataset" from "broken harness";
  * the task ids each arm was actually scored on are recorded, so two rows with
    different denominators cannot be compared without somebody noticing.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sweep import diagnostics_rollup, run_arm                    # noqa: E402


def _task(**kw):
    base = {"passed": False, "as_run_usd": 0.01, "output_tokens": 10,
            "test_pass_ratio": None, "guard_reasons": [], "routing": None,
            "grounding": None}
    base.update(kw)
    return base


# ==============================================================================
# --- THE ROLLUP ---
# ==============================================================================

def test_suite_reach_rate_counts_attempts_not_tasks():
    """A task with three attempts of which one was graded is not '100% reached'.
    Reports 21 and 23 could only be diagnosed by dividing the containment
    receipt's capture count by the attempt count by hand."""
    d = diagnostics_rollup([
        _task(guard_reasons=["apply_failed", "apply_failed", ""]),
        _task(guard_reasons=["apply_failed", "apply_failed", "apply_failed"]),
    ])
    assert d["attempts"] == 6
    assert d["suite_reached"] == 1
    assert d["suite_reach_rate"] == pytest.approx(1 / 6, abs=1e-4)


def test_guard_reasons_are_tallied_dominant_first():
    """The report prints the head of this dict, so the ordering is the finding."""
    d = diagnostics_rollup([
        _task(guard_reasons=["apply_failed", "no_patch"]),
        _task(guard_reasons=["apply_failed", "apply_failed"]),
    ])
    assert list(d["guard_reasons"]) == ["apply_failed", "no_patch"]
    assert d["guard_reasons"]["apply_failed"] == 3


def test_partial_credit_is_averaged_over_the_tasks_that_have_it():
    """A task that never reached a suite has no ratio, and counting it as 0
    would understate the arms that did get graded."""
    d = diagnostics_rollup([
        _task(test_pass_ratio=0.5), _task(test_pass_ratio=1.0), _task(),
    ])
    assert d["test_pass_ratio_avg"] == 0.75
    assert d["test_pass_ratio_n"] == 2


def test_a_ratio_of_zero_is_averaged_in_rather_than_treated_as_missing():
    d = diagnostics_rollup([_task(test_pass_ratio=0.0), _task(test_pass_ratio=1.0)])
    assert d["test_pass_ratio_avg"] == 0.5 and d["test_pass_ratio_n"] == 2


def test_routing_provenance_is_counted():
    d = diagnostics_rollup([
        _task(routing={"frontier_used": True, "degraded": False}),
        _task(routing={"frontier_used": False, "degraded": True}),
        _task(routing={"frontier_used": False, "degraded": False}),
    ])
    assert d["routed"] == 3 and d["frontier_used"] == 1 and d["degraded"] == 1


def test_grounded_tasks_are_counted():
    d = diagnostics_rollup([
        _task(grounding={"read": ["a.py"]}),
        _task(grounding={"read": []}),
        _task(grounding=None),
    ])
    assert d["grounded"] == 1


def test_a_dataset_that_reports_no_attempts_yields_a_null_rate_not_a_zero():
    """ClassEval and BigCodeBench-Hard produce no guard records. A `0%` reach
    rate for them would be a false alarm in every report they appear in."""
    d = diagnostics_rollup([_task(guard_reasons=None), _task(guard_reasons=None)])
    assert d["attempts"] == 0 and d["suite_reach_rate"] is None


def test_an_empty_arm_does_not_raise():
    d = diagnostics_rollup([])
    assert d["attempts"] == 0 and d["test_pass_ratio_avg"] is None


# ==============================================================================
# --- WHAT SURVIVES INTO THE ARM'S ROW ---
# ==============================================================================

@pytest.fixture
def sweep_inputs():
    problems = {"t1": {"task_id": "t1"}, "t2": {"task_id": "t2"}}
    return problems, ["t1", "t2"]


def _cfg(fn):
    return {"id": "arm", "name": "Arm", "fn": fn, "models": "m",
            "triage_mode": "Straitjacket contained digest ($0.00)"}


def test_the_partial_credit_metric_reaches_the_arm_row(sweep_inputs, tmp_path):
    """The regression this file exists for: computed per task, dropped by the
    whitelist in `run_arm`, absent from every report."""
    problems, ids = sweep_inputs

    def fn(problem):
        return {"passed": False, "as_run_usd": 0.01, "output_tokens": 5,
                "test_pass_ratio": 0.25, "sbp": {"required": 4, "missing": 3},
                "guard_reason": "", "guard_reasons": ["", ""],
                "suite_reached": 2, "attempts": 2, "repair_loops": 1,
                "grounding": {"read": ["a.py"]},
                "routing": {"frontier_used": True, "degraded": False}}

    row = run_arm(_cfg(fn), problems, ids,
                  cache_file=str(tmp_path / "cache.json"))
    for task in row["results"]:
        assert task["test_pass_ratio"] == 0.25
        assert task["sbp"] == {"required": 4, "missing": 3}
        assert task["guard_reasons"] == ["", ""]
        assert task["grounding"] == {"read": ["a.py"]}
    assert row["diagnostics"]["test_pass_ratio_avg"] == 0.25
    assert row["diagnostics"]["suite_reach_rate"] == 1.0
    assert row["diagnostics"]["frontier_used"] == 2


def test_the_guard_reason_of_the_last_attempt_is_recorded(sweep_inputs, tmp_path):
    problems, ids = sweep_inputs

    def fn(problem):
        return {"passed": False, "as_run_usd": 0.0, "output_tokens": 0,
                "guard_reason": "apply_failed",
                "guard_reasons": ["apply_failed", "apply_failed"]}

    row = run_arm(_cfg(fn), problems, ids, cache_file=str(tmp_path / "c.json"))
    assert all(t["guard_reason"] == "apply_failed" for t in row["results"])
    assert row["diagnostics"]["guard_reasons"] == {"apply_failed": 4}
    assert row["diagnostics"]["suite_reach_rate"] == 0.0


def test_the_task_ids_the_arm_was_scored_on_are_recorded(sweep_inputs, tmp_path):
    """Two arms in one report can have different denominators — 49, 50, 47, 40
    in reports 21 and 23 — and their pass rates were printed side by side."""
    problems, ids = sweep_inputs
    row = run_arm(_cfg(lambda p: {"passed": True, "as_run_usd": 0.0,
                                  "output_tokens": 0}),
                  problems, ids, cache_file=str(tmp_path / "c.json"))
    assert row["task_ids"] == ["t1", "t2"]


def test_a_dropped_task_is_absent_from_task_ids_not_recorded_as_a_failure(
        sweep_inputs, tmp_path, capsys):
    from src.client import DispatchError
    problems, ids = sweep_inputs
    seen = []

    def fn(problem):
        seen.append(problem["task_id"])
        if problem["task_id"] == "t2":
            raise DispatchError("503", model_id="m", kind="transient")
        return {"passed": True, "as_run_usd": 0.0, "output_tokens": 0}

    row = run_arm(_cfg(fn), problems, ids, cache_file=str(tmp_path / "c.json"))
    assert row["task_ids"] == ["t1"]
    assert row["n"] == 1 and row["completed"] == 1
    assert [t["task_id"] for t in row["incomplete_tasks"]] == ["t2"]


def test_a_low_suite_reach_rate_is_announced_on_the_console(
        sweep_inputs, tmp_path, capsys):
    """A sweep that spends real money for six hours should say, while it is
    running, that nothing it produced was graded."""
    problems, ids = sweep_inputs
    run_arm(_cfg(lambda p: {"passed": False, "as_run_usd": 0.0, "output_tokens": 0,
                            "guard_reasons": ["apply_failed", "apply_failed"]}),
            problems, ids, cache_file=str(tmp_path / "c.json"))
    out = capsys.readouterr().out
    assert "reached the test suite" in out
    assert "apply_failed=4" in out
    assert "not the models" in out


def test_a_fully_graded_arm_gets_no_warning(sweep_inputs, tmp_path, capsys):
    problems, ids = sweep_inputs
    run_arm(_cfg(lambda p: {"passed": True, "as_run_usd": 0.0, "output_tokens": 0,
                            "guard_reasons": ["", ""]}),
            problems, ids, cache_file=str(tmp_path / "c.json"))
    assert "reached the test suite" not in capsys.readouterr().out
