# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Contract tests for routing over failures that happened before the suite ran.

Why this exists
---------------
`src/routing.py` was written for failures with a typed fact tier: a profile
recognised the test output, the evidence graph named the failing identities,
and the gate reasoned over that census. Everything else fell into one branch —
"no typed evidence" — and classified as `shallow`.

On SWE-bench Pro that branch is the *common* case, not the fallback. Roughly
89% of attempts in reports 21 and 23 died before a test ran, almost all of them
at `git apply`. Every one of those arrived as an untyped string, classified as
`shallow`, and the evidence gate declined to escalate — so an "evidence gate"
arm was reading a constant and behaving identically to an attempt counter.

Two failures also need to be told apart from each other, which `shallow` could
not do:

  * `apply_failed` — the model produced a diff that does not fit the tree.
    Once is a local defect; twice running is the clearest "hand this over"
    signal in the sweep.
  * `container_unavailable` / `restore_failed` — the *environment* failed. No
    model fixes this, and escalating spends the study's dearest resource on a
    Docker problem.

The other half of the file pins the arithmetic that made the frontier rung
unreachable: with `MAX_ORACLE_CALLS = 2` over a two-rung ladder there is
exactly one gate evaluation, at `attempt == 1`, and no gate escalates on it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import swebench_pro as sbp                                   # noqa: E402
from src.evaluator import _guard_evidence                             # noqa: E402
from src.routing import (GATES, Difficulty, classify, frontier_is_reachable,
                         gate_after_ladder, LEVELS)                   # noqa: E402


def guard(reason):
    return _guard_evidence(f"synthetic {reason}", reason)


# ==============================================================================
# --- TYPING A PRE-EXECUTION FAILURE ---
# ==============================================================================

def test_environment_is_a_level_of_its_own():
    assert "environment" in LEVELS


@pytest.mark.parametrize("reason", ["container_unavailable", "restore_failed",
                                    "row_no_test_files", "execution_error",
                                    "harness_error"])
def test_environment_failures_are_typed_as_environment(reason):
    d = classify(guard(reason))
    assert d.level == "environment"
    assert d.is_environment is True
    assert d.is_hard is False, "no model fixes a container that would not start"
    assert d.guard == reason


@pytest.mark.parametrize("reason", ["no_patch", "not_a_diff", "no_hunk",
                                    "truncated_output"])
def test_malformed_responses_are_shallow(reason):
    """Prose instead of a diff, or a diff cut off at the output cap. The next
    cheap rung fixes these as readily as an expensive one."""
    d = classify(guard(reason))
    assert d.level == "shallow" and d.is_hard is False


def test_a_first_apply_failure_is_local_not_shallow():
    """`shallow` said 'a cheap model will fix this and there is nothing to
    reason over'. Neither half was true here, and saying it is what made the
    evidence gate a no-op on the dominant failure of the sweep."""
    d = classify(guard("apply_failed"))
    assert d.level == "local"
    assert d.is_hard is False


def test_the_same_apply_failure_twice_running_is_a_stall():
    first = classify(guard("apply_failed"))
    second = classify(guard("apply_failed"), previous=first)
    assert second.level == "stalled" and second.is_hard is True


def test_a_different_guard_the_second_time_is_not_a_stall():
    """Moving from 'the diff does not apply' to 'there is no diff' is a change
    of failure, not a model that has stopped converging."""
    first = classify(guard("apply_failed"))
    second = classify(guard("no_patch"), previous=first)
    assert second.level == "shallow"


def test_a_guard_failure_is_typed_so_it_does_not_mark_the_run_degraded():
    """`degraded` means the gate wanted typed evidence and got none. A guard
    failure is fully known — pretending otherwise would flag every SWE-bench Pro
    row as unroutable."""
    d = classify(guard("apply_failed"))
    assert d.typed is True and d.profile == "guard/v1"


def test_the_guard_slug_survives_into_the_audit_record():
    assert classify(guard("restore_failed")).as_dict()["guard"] == "restore_failed"


def test_a_plain_string_with_no_guard_still_falls_back_to_untyped_shallow():
    """Evidence captured outside the harness keeps the old behaviour rather
    than being silently reclassified."""
    d = classify("some unstructured error text")
    assert d.level == "shallow" and d.typed is False and d.guard == ""


# ==============================================================================
# --- GATES REFUSE TO BUY A FRONTIER MODEL FOR A DOCKER FAILURE ---
# ==============================================================================

@pytest.mark.parametrize("gate_name", ["after_ladder", "evidence",
                                       "evidence_immediate"])
@pytest.mark.parametrize("attempt", [1, 2, 5])
def test_no_gate_escalates_on_an_environment_failure(gate_name, attempt):
    """Checked *before* the 'cheap rungs exhausted' fallback: a container that
    never started does not become worth frontier pricing because the ladder ran
    out of rungs."""
    d = classify(guard("container_unavailable"))
    escalate, why = GATES[gate_name](d, attempt, 2)
    assert escalate is False
    assert "environment failure" in why


def test_the_evidence_gate_still_escalates_on_a_stalled_apply_failure():
    """The behaviour the whole change exists to produce."""
    first = classify(guard("apply_failed"))
    second = classify(guard("apply_failed"), previous=first)
    escalate, why = GATES["evidence"](second, 2, 2)
    assert escalate is True
    assert "stalled" in why


def test_the_counter_gate_still_ignores_evidence_when_it_is_not_environmental():
    d = classify(guard("apply_failed"))
    assert gate_after_ladder(d, 2, 2)[0] is True
    assert gate_after_ladder(d, 1, 2)[0] is False


# ==============================================================================
# --- FRONTIER REACHABILITY ---
# ==============================================================================

def test_a_counter_gate_cannot_reach_the_frontier_when_the_budget_equals_the_ladder():
    """The arithmetic behind reports 21 and 23: `MAX_ORACLE_CALLS = 2` over a
    two-rung ladder produces one gate evaluation, at `attempt == 1`, and
    `gate_after_ladder` answers 'cheap rungs remain'."""
    assert frontier_is_reachable(GATES["after_ladder"], 2, 2) is False
    assert frontier_is_reachable(GATES["after_ladder"], 2, 3) is True


def test_gate_never_is_reported_unreachable_because_that_is_its_job():
    assert frontier_is_reachable(GATES["never"], 2, 9) is False


def test_reachability_is_answered_against_the_friendliest_evidence():
    """A `False` has to mean structurally impossible, not merely unlikely, or
    the check cries wolf on every arm whose gate is conservative."""
    assert frontier_is_reachable(GATES["evidence"], 2, 2) is True


def test_the_registry_has_no_unreachable_frontier_arms_at_the_shipped_budget():
    assert sbp.unreachable_frontier_arms() == []


def test_the_check_catches_the_budget_that_shipped_reports_21_and_23():
    assert "sbp_cascade" in sbp.unreachable_frontier_arms(2)


def test_every_arm_advertising_the_frontier_model_is_covered_by_the_check():
    """A new arm must not be able to opt out of the invariant by not being
    listed in `_ARM_SHAPES`."""
    advertised = {vid for vid, cfg in sbp.SWEBENCH_PRO_VARIANTS.items()
                  if "Opus" in cfg["models"] and "plan" not in vid
                  and vid != "sbp_single_opus"}
    assert advertised <= set(sbp._ARM_SHAPES)


def test_arm_names_state_the_rung_count_that_will_actually_be_spent():
    """A report quotes these names verbatim. One that says '2 rungs' beside a
    run that made three is the same class of error as an unreachable frontier,
    just quieter."""
    for cfg in sbp.SWEBENCH_PRO_VARIANTS.values():
        assert "{rungs}" not in cfg["name"]
        assert f"{sbp.MAX_ORACLE_CALLS} rungs" in cfg["name"]


def test_the_registry_refuses_to_finalise_at_an_unrunnable_budget(monkeypatch):
    """The failure mode is silent by nature — unreachable code raises nothing —
    so it has to be asserted at import time or it ships again."""
    monkeypatch.setattr(sbp, "MAX_ORACLE_CALLS", 2)
    with pytest.raises(RuntimeError, match="cannot be reached"):
        sbp._finalise_registry()


# ==============================================================================
# --- TRUNCATED OUTPUT IS DIAGNOSED, NOT BLAMED ON THE DIFF ---
# ==============================================================================

def test_a_truncated_response_is_relabelled_from_apply_failed():
    """A diff cut off at the output cap still carries `---`, `+++` and `@@`, so
    it passes the birth gate and dies at `git apply` — filed, before this, as
    the same failure as a diff that was simply wrong."""
    out = sbp._relabel_truncated(guard("apply_failed"), {"truncated": True})
    assert out.reason == "truncated_output"
    assert "output token cap" in str(out)
    assert "synthetic apply_failed" in str(out), "the original failure is kept"


def test_relabelling_leaves_an_untruncated_response_alone():
    ev = guard("apply_failed")
    assert sbp._relabel_truncated(ev, {"truncated": False}) is ev
    assert sbp._relabel_truncated(ev, {}) is ev


def test_relabelling_never_overwrites_an_environment_failure():
    """A container that would not start is not explained by the output cap."""
    ev = guard("container_unavailable")
    assert sbp._relabel_truncated(ev, {"truncated": True}) is ev


def test_relabelling_leaves_a_real_test_failure_alone():
    """The suite ran; the response length is not the story."""
    ev = _guard_evidence("", "")
    assert sbp._relabel_truncated(ev, {"truncated": True}) is ev


def test_the_oracle_budget_exceeds_the_rung_count():
    """The invariant behind every unreachable-frontier bug in this repository:
    K oracle calls give K-1 gate evaluations, and every gate compares `attempt`
    against `len(TIERS)`."""
    assert sbp.MAX_ORACLE_CALLS > len(sbp.TIERS)
