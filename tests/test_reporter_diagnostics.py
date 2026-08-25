# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Contract tests for the two things a report must refuse to leave unsaid.

Why this exists
---------------
Reports 21 and 23 are correct in every number they print and misleading in
what they omit. They print five arms at 0-2.5% and three "candidate
architectures" at 0/44, 0/37 and 0/50, under names promising an Opus-5
escalation. What is not printed:

  * roughly 89% of attempts never reached a test — recoverable only by
    dividing the containment receipt's `Captures` column by the attempt count
    by hand;
  * Opus-5 was never called once in any of the eight arms;
  * the arms were scored on 49, 50, 47, 40, 44, 37 and 50 tasks respectively,
    printed side by side as though the denominators matched;
  * `test_pass_ratio` and `routing.degraded` existed for every row and were
    rendered nowhere.

A report is the deliverable here, so these are pinned as contract rather than
left to review.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reporter import generate_markdown_report                    # noqa: E402


def _row(name, **kw):
    row = {
        "id": name, "name": name, "models": "Gemini 3.7 Flash", "n": 10,
        "passed": 0, "pass_rate": 0.0, "total_as_run_usd": 1.0,
        "total_triage_usd": 0.0, "cost_per_solved_usd": 0.0,
        "avg_output_tokens": 100.0, "triage_mode": "Straitjacket ($0.00)",
        "task_ids": [f"t{i}" for i in range(10)],
        "diagnostics": {
            "attempts": 30, "suite_reached": 3, "suite_reach_rate": 0.1,
            "guard_reasons": {"apply_failed": 25, "no_patch": 2},
            "test_pass_ratio_avg": 0.25, "test_pass_ratio_n": 3,
            "frontier_used": 0, "degraded": 0, "routed": 10, "grounded": 10,
        },
    }
    row.update(kw)
    return row


def _render(rows, tmp_path, name="SWE-bench Pro"):
    out = tmp_path / "r.md"
    generate_markdown_report(rows, dataset_name=name, output_path=str(out))
    return out.read_text(encoding="utf-8")


# ==============================================================================
# --- THE DIAGNOSTICS TABLE ---
# ==============================================================================

def test_the_report_states_how_many_attempts_were_ever_graded(tmp_path):
    md = _render([_row("S0a")], tmp_path)
    assert "Attempt Diagnostics" in md
    assert "3/30 (10%)" in md


def test_a_low_reach_rate_is_flagged_as_a_harness_finding_not_a_model_one(tmp_path):
    md = _render([_row("S0a")], tmp_path)
    assert "Most attempts were never graded" in md
    assert "not whether it resolved the issue" in md


def test_the_dominant_guard_failure_is_named(tmp_path):
    md = _render([_row("S0a")], tmp_path)
    assert "`apply_failed` × 25" in md


def test_partial_credit_is_shown_with_the_count_it_was_averaged_over(tmp_path):
    """`0.250` over three graded attempts and `0.250` over fifty are different
    claims; printing the mean alone hides which one this is."""
    md = _render([_row("S0a")], tmp_path)
    assert "`0.250` (n=3)" in md


def test_an_arm_that_grades_everything_gets_no_warning(tmp_path):
    md = _render([_row("Good", diagnostics=dict(
        attempts=30, suite_reached=30, suite_reach_rate=1.0, guard_reasons={},
        test_pass_ratio_avg=0.8, test_pass_ratio_n=10, frontier_used=4,
        degraded=0, routed=10, grounded=10))], tmp_path)
    assert "Most attempts were never graded" not in md
    assert "all attempts graded" in md


def test_an_arm_naming_a_frontier_model_that_never_called_it_is_flagged(tmp_path):
    """The exact defect in reports 21 and 23: five arms whose model column
    promised Opus-5 and whose routing record shows it was never invoked."""
    md = _render([_row("S2", models="Gemini 3.7 Flash -> Claude Opus-5")], tmp_path)
    assert "Frontier rung never invoked" in md
    assert "did not test the architecture they are named after" in md


def test_an_arm_that_did_reach_the_frontier_is_not_flagged(tmp_path):
    row = _row("S2", models="Gemini 3.7 Flash -> Claude Opus-5")
    row["diagnostics"]["frontier_used"] = 3
    assert "Frontier rung never invoked" not in _render([row], tmp_path)


def test_an_arm_that_never_claimed_a_frontier_model_is_not_flagged(tmp_path):
    """`sbp_single_flash` calls no frontier model by design."""
    assert "Frontier rung never invoked" not in _render([_row("S0a")], tmp_path)


def test_degraded_routing_is_rendered(tmp_path):
    """A gate that needed typed evidence and did not get it did not test what
    the arm's name says. It was recorded per task and printed nowhere."""
    row = _row("S2")
    row["diagnostics"]["degraded"] = 7
    md = _render([row], tmp_path)
    assert "Degraded" in md and "| 7 |" in md


# ==============================================================================
# --- THE TASK-SET AUDIT ---
# ==============================================================================

def test_arms_scored_on_different_task_sets_are_flagged(tmp_path):
    a = _row("A", task_ids=[f"t{i}" for i in range(10)])
    b = _row("B", task_ids=[f"t{i}" for i in range(7)], n=7)
    md = _render([a, b], tmp_path)
    assert "scored on different task sets" in md
    assert "only 7 were completed by every arm" in md
    assert "`A`=10" in md and "`B`=7" in md


def test_identical_task_sets_produce_no_audit_warning(tmp_path):
    a, b = _row("A"), _row("B")
    assert "scored on different task sets" not in _render([a, b], tmp_path)


def test_a_single_arm_report_produces_no_audit_warning(tmp_path):
    assert "scored on different task sets" not in _render([_row("A")], tmp_path)


def test_the_audit_sits_above_the_performance_table_it_qualifies(tmp_path):
    a = _row("A")
    b = _row("B", task_ids=["t0"], n=1)
    md = _render([a, b], tmp_path)
    assert md.index("scored on different task sets") < md.index("## 1.")


# ==============================================================================
# --- SECTION NUMBERING ---
# ==============================================================================

def test_sections_are_numbered_without_gaps_when_diagnostics_are_present(tmp_path):
    md = _render([_row("A")], tmp_path)
    assert "## 1. Comparative TCO" in md
    assert "## 2. Attempt Diagnostics" in md
    assert "## 3. Key TCO & Architectural Insights" in md


def test_a_dataset_without_diagnostics_keeps_the_historical_layout(tmp_path):
    """BigCodeBench-Hard and ClassEval produce no guard records; their reports
    must not sprout an empty section or renumber the ones readers cite."""
    row = _row("A")
    row.pop("diagnostics")
    row.pop("task_ids")
    md = _render([row], tmp_path, name="BigCodeBench-Hard")
    assert "Attempt Diagnostics" not in md
    assert "## 2. Key TCO & Architectural Insights" in md


def test_the_containment_receipt_still_renders_alongside_diagnostics(tmp_path):
    row = _row("A", containment={
        "captures": 5, "raw_tokens_est": 1000, "digest_tokens_est": 100,
        "evidence_sent_tokens_est": 90, "native_baseline_tokens_est": 120,
        "profiles": ["pytest/v1"], "treatments": ["straitjacket"]})
    md = _render([row], tmp_path)
    assert "## 2. Attempt Diagnostics" in md
    assert "## 3. Context Containment Receipt" in md
    assert "## 4. Key TCO & Architectural Insights" in md
