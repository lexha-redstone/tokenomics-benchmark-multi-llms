# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Contract tests for the FeatureBench arms.

These do not touch Docker or an API. They pin the things that would silently
produce a wrong row: which rung each gate escalates to, that the oracle-call
budget is actually held constant across arms (the comparison H2 rests on), and
that a patch which cannot apply is fed back as evidence rather than scored as a
plain failure.
"""

import types

import pytest

from src import featurebench as fb
from src.evaluator import (extract_patch, missing_patch_error,
                           featurebench_test_files, featurebench_test_ratio)


PROBLEM = {
    "instance_id": "demo__repo-1",
    "repo": "demo/repo",
    "base_commit": "0" * 40,
    "problem_statement": "Add a widget registry.",
    "FAIL_TO_PASS": ["tests/test_widget.py"],
    "PASS_TO_PASS": ["tests/test_core.py"],
    "image_name": "demo/img:1",
    "repo_workdir": "/workspace/repo",
}

GOOD_PATCH = ("```diff\n"
              "diff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n"
              "@@ -1 +1 @@\n-old\n+new\n```")


class FakeEvidence(str):
    """Stands in for a harness capture: `typed` controls the gate's view."""

    def __new__(cls, text, typed=True, failing=5):
        obj = super().__new__(cls, text)
        obj._typed, obj._failing = typed, failing
        return obj

    @property
    def run(self):
        if not self._typed:
            return types.SimpleNamespace(evidence_graph=lambda: None, profile="text/v1")
        items = [types.SimpleNamespace(id=f"t{i}", failure_class="AssertionError")
                 for i in range(self._failing)]
        graph = types.SimpleNamespace(items=items,
                                      aggregate={"failing": self._failing})
        return types.SimpleNamespace(evidence_graph=lambda: graph, profile="unittest/v1")


class FakeEnv:
    """Records every scoring call; never resolves, so the ladder runs to its cap."""

    instances = []

    def __init__(self, problem, timeout=900.0):
        self.problem = problem
        self.started = True
        self.setup_error = ""
        self.calls = []
        FakeEnv.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def score(self, patch):
        self.calls.append(patch)
        return False, FakeEvidence("FAILED tests/test_widget.py::test_a", typed=True)


@pytest.fixture
def wired(monkeypatch):
    """Replace the container and the API with recorders."""
    FakeEnv.instances = []
    calls = []

    def fake_dispatch(model_id, prompt, max_tokens=2560, thinking_level=None, problem=None):
        calls.append({"model": model_id, "thinking": thinking_level, "prompt": prompt})
        return GOOD_PATCH, {"as_run_usd": 0.01, "output": 10, "total_tokens": 20,
                            "input_raw": 10, "prompt_tokens": 10}, 0.1

    monkeypatch.setattr(fb, "FeatureBenchEnv", FakeEnv)
    monkeypatch.setattr(fb, "dispatch_model", fake_dispatch)
    monkeypatch.setattr(fb, "_treat_error",
                        lambda err, t, problem=None: ("DIGEST",
                                                      {"as_run_usd": 0.0, "output": 0,
                                                       "total_tokens": 0}, 0.0))
    # The arms are `sj_required`; the harness is not installed in CI.
    monkeypatch.setattr("src.architectures.sj.require", lambda: None)
    return calls


# -- patch handling --------------------------------------------------------

def test_extract_patch_from_fenced_diff():
    assert extract_patch(GOOD_PATCH).startswith("diff --git")


def test_extract_patch_from_unfenced_diff():
    raw = "Here you go:\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    assert extract_patch(raw).startswith("diff --git")


def test_missing_patch_error_rejects_prose_before_the_container_starts():
    assert missing_patch_error("I would change the widget module.") is not None
    assert missing_patch_error("") is not None
    # Headers but no hunk is still nothing to apply.
    assert missing_patch_error("--- a/x\n+++ b/x\n") is not None
    assert missing_patch_error(extract_patch(GOOD_PATCH)) is None


def test_test_files_are_fail_to_pass_first_and_deduplicated():
    files = featurebench_test_files(
        {"FAIL_TO_PASS": ["a.py"], "PASS_TO_PASS": ["b.py", "a.py"]})
    assert files == ["a.py", "b.py"]


def test_test_pass_ratio_reads_pytest_summary():
    assert featurebench_test_ratio("=== 3 passed, 1 failed in 2s ===") == 0.75
    assert featurebench_test_ratio("2 passed, 2 errors in 1s") == 0.5
    assert featurebench_test_ratio("collection crashed") is None


# -- the budget the H2 comparison rests on ---------------------------------

@pytest.mark.parametrize("arm", [
    lambda: fb.run_fb_single(PROBLEM, model_id=fb.GEMINI_37_FLASH_ID, thinking_level="low"),
    fb.run_fb_cascade,
    fb.run_fb_evidence_gate,
    fb.run_fb_plan_exec,
])
def test_every_arm_makes_at_most_three_oracle_calls(wired, arm):
    arm() if not callable(getattr(arm, "__wrapped__", None)) else arm(PROBLEM)
    env = FakeEnv.instances[-1]
    assert len(env.calls) == fb.MAX_ORACLE_CALLS, (
        "arms must be matched on oracle calls -- that is the scarce resource "
        "H2 makes a claim about")


# -- who holds each rung ---------------------------------------------------

def test_cascade_escalates_one_rung_per_failure_ending_at_the_frontier(wired):
    out = fb.run_fb_cascade(PROBLEM)
    assert out["routing"]["rungs"] == [
        f"{fb.GEMINI_37_FLASH_ID}/low", f"{fb.SONNET_ID}/off", f"{fb.FRONTIER}/off"]
    assert out["routing"]["frontier_used"] is True


def test_evidence_gate_jumps_to_the_frontier_when_the_digest_says_broad(wired):
    out = fb.run_fb_evidence_gate(PROBLEM)
    rungs = out["routing"]["rungs"]
    # Five failing identities classifies as `broad`, so the gate fires on the
    # first repair instead of walking to sonnet -- this is the whole difference
    # from fb_cascade, and it is what r9 did at N=148.
    assert rungs[1] == f"{fb.FRONTIER}/off"
    assert out["routing"]["frontier_rung"] == 2


def test_evidence_gate_flags_itself_degraded_without_a_typed_fact_tier(wired,
                                                                       monkeypatch):
    def untyped_score(self, patch):
        self.calls.append(patch)
        return False, FakeEvidence("boom", typed=False)

    monkeypatch.setattr(FakeEnv, "score", untyped_score)
    out = fb.run_fb_evidence_gate(PROBLEM)
    assert out["routing"]["degraded"] is True, (
        "an evidence gate with no evidence is a counter gate wearing the wrong "
        "label; the row must say so")


RANK = {f"{fb.GEMINI_37_FLASH_ID}/low": 0, f"{fb.SONNET_ID}/off": 1,
        f"{fb.FRONTIER}/off": 2}


def test_the_ladder_never_de_escalates(wired):
    """The strongest finding in the repo, enforced rather than hoped for.

    N=100 measured a repair turn that de-escalates rescuing 16% of failures
    against 41% for one that escalates (z = +3.55, p = 0.0004). Escalation here
    is a one-way ratchet: once an arm is at a rung it never drops below it, and
    a spare oracle call is spent re-running the rung it holds.
    """
    for arm in (fb.run_fb_cascade, fb.run_fb_evidence_gate):
        ranks = [RANK[r] for r in arm(PROBLEM)["routing"]["rungs"]]
        assert ranks == sorted(ranks), f"{arm.__name__} de-escalated: {ranks}"


def test_the_gate_escalates_at_most_once(wired):
    """It may *hold* the frontier for a spare attempt, but never re-decides."""
    out = fb.run_fb_evidence_gate(PROBLEM)
    assert sum(1 for d in out["routing"]["decisions"] if d["escalate"]) == 1


# -- the H2 contrast -------------------------------------------------------

def test_plan_exec_spends_the_frontier_model_before_the_first_oracle_call(wired):
    out = fb.run_fb_plan_exec(PROBLEM)
    assert wired[0]["model"] == fb.OPUS_5_ID, "the planner runs first"
    assert fb.PLANNER_ROLE[:40] in wired[0]["prompt"]
    # ...and never again: the executor and both repairs are the cheap model.
    assert [c["model"] for c in wired[1:]] == [fb.GEMINI_37_FLASH_ID] * 3
    assert out["routing"]["frontier_used"] is False
    # The plan reaches the executor, or the arm is just a flash single.
    assert "implementation plan" in wired[1]["prompt"].lower()


def test_single_arm_never_escalates(wired):
    out = fb.run_fb_single(PROBLEM, model_id=fb.SONNET_ID)
    assert set(out["routing"]["rungs"]) == {f"{fb.SONNET_ID}/off"}
    assert out["routing"]["frontier_used"] is False


# -- result shape ----------------------------------------------------------

def test_result_carries_the_partial_credit_metric(wired):
    out = fb.run_fb_cascade(PROBLEM)
    assert "test_pass_ratio" in out
    assert out["passed"] is False
    assert out["repair_loops"] == fb.MAX_ORACLE_CALLS - 1
    assert out["as_run_usd"] > 0


def test_registry_excludes_the_frontier_single_from_the_default_group():
    from src.architectures import get_configurations
    ids = {c["id"] for c in get_configurations("featurebench", "featurebench")}
    assert "fb_evidence_gate" in ids
    assert "fb_single_opus" not in ids, (
        "opus is priced far above the rest; including it would silently reprice "
        "every routine sweep, as ClassEval's baseline does not")
