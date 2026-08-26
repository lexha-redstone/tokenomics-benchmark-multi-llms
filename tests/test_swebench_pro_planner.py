# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Contract tests for the models bought *before* the first oracle call.

Why this exists
---------------
Two arms spend a model ahead of any test run: `sbp_plan_exec`, the H2
challenger, and `sbp_grounded_contract`, whose locator writes a file contract.
Both used to make that call *outside* `_ladder`, and `_ladder` is where the
container is opened and the repository is read. So the architect and the
locator were the only models in the study working blind, while every rung they
were briefing could see the files — and both of their roles ask, in so many
words, for real paths:

    PLANNER_ROLE           "...which files to create or modify ...
                            Be concrete and name real paths."
    CONTRACT_LOCATOR_ROLE  "TARGET_FILES: Exact repository file paths that
                            must be modified or created"

Measured on the running split, text alone locates 21% of the files a reference
patch touches, and `apply_failed` accounts for 44-56% of all attempts in report
30. A planner that cannot see the tree is being asked for exactly the thing it
has no way to know, and losing on that would have been read as "front-loaded
planning loses to fail->escalate" — the H2 verdict this dataset was adopted to
settle.

Pinned here:

  * the planner runs inside the container, after grounding, before the first
    solve, and its prompt carries the quoted source;
  * its cost lands in the arm's ledger;
  * it is an LLM call, not an oracle call, so the budget H2 holds constant is
    untouched;
  * an arm that passes no planner takes a byte-identical path — which is what
    makes the cached S0b/S1/S2 records from report 30 still valid.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import swebench_pro as sbp                                   # noqa: E402
from src.config import GEMINI_37_FLASH_ID, SONNET_ID, OPUS_5_ID       # noqa: E402


PROBLEM = {
    "instance_id": "instance_demo__repo-abc123-vnan",
    "repo": "demo/repo",
    "repo_language": "python",
    "base_commit": "b" * 40,
    "problem_statement": "The `loadUserInfo` helper drops the pending flag.",
    "requirements": "It must attach the pending flag.",
    "interface": "Type: Method\n\nName: db.mget\n\nPath: src/user/emails.py\n",
    "fail_to_pass": ["tests/t.py | attaches pending"],
    "pass_to_pass": [],
    "selected_test_files_to_run": ["tests/t.py"],
    "before_repo_set_cmd": "git checkout abc123 -- tests/t.py",
    "dockerhub_tag": "demo.repo-demo__repo-abc123",
    "image_name": "jefzda/sweap-images:demo.repo-demo__repo-abc123",
    "repo_workdir": "/app",
}

PATCH = ("```diff\ndiff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n"
         "@@ -1 +1 @@\n-old\n+new\n```")

SOURCE = "PENDING_FLAG = 'email:pending'\n"


class FakeEnv:
    """Container stand-in that also stands in for the repository."""

    instances = []
    sources = {"src/user/emails.py": SOURCE}

    def __init__(self, problem, timeout=None, scripts=None):
        self.problem = problem
        self.started = True
        self.setup_error = ""
        self.calls = []
        self.events = []
        self.last_ratio = 0.25
        self.last_report = {"resolved": False, "required": 4, "missing": 3}
        FakeEnv.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read_source(self, paths, budget=None, per_file=None, max_files=None):
        self.events.append("read_source")
        read = [p for p in paths if p in self.sources]
        return ([f"--- FILE: {p} ---\n{self.sources[p]}\n" for p in read],
                read, [p for p in paths if p not in self.sources])

    def grep_paths(self, terms, limit=40):
        self.events.append("grep")
        return []

    def score(self, patch):
        self.events.append("score")
        self.calls.append(patch)
        return False, _evidence()


def _evidence():
    """A typed capture, so the evidence gate has something real to read."""
    items = [types.SimpleNamespace(id=f"t{i}", failure_class="AssertionError")
             for i in range(5)]
    graph = types.SimpleNamespace(items=items, aggregate={"failing": 5})
    run = types.SimpleNamespace(evidence_graph=lambda: graph, profile="pytest/v1")

    class E(str):
        pass
    e = E("FAILED tests/t.py::test_a")
    e.run = run
    e.reason = ""
    e.failure_class = ""
    e.digest = "DIGEST"
    return e


@pytest.fixture
def wired(monkeypatch):
    FakeEnv.instances = []
    calls = []

    def fake_dispatch(model_id, prompt, max_tokens=2560, thinking_level=None,
                      problem=None):
        calls.append({"model": model_id, "prompt": prompt,
                      "max_tokens": max_tokens})
        return PATCH, {"as_run_usd": 0.05, "output": 10, "total_tokens": 20,
                       "input_raw": 10, "prompt_tokens": 10}, 0.1

    monkeypatch.setattr(sbp, "SWEBenchProEnv", FakeEnv)
    monkeypatch.setattr(sbp, "dispatch_model", fake_dispatch)
    monkeypatch.setattr(sbp, "_treat_error",
                        lambda err, t, problem=None: ("DIGEST", {
                            "as_run_usd": 0.0, "output": 0, "total_tokens": 0}, 0.0))
    monkeypatch.setattr("src.architectures.sj.require", lambda: None)
    return calls


# ==============================================================================
# --- THE PLANNER SEES THE REPOSITORY ---
# ==============================================================================

def test_the_architect_prompt_carries_the_quoted_source(wired):
    """The whole point. `PLANNER_ROLE` asks for real paths; before this it was
    asking a model that had never seen one."""
    sbp.run_sbp_plan_exec(dict(PROBLEM))
    plan_call = wired[0]
    assert "IMPLEMENTATION PLAN" in plan_call["prompt"]
    assert "--- FILE: src/user/emails.py ---" in plan_call["prompt"]
    assert SOURCE.strip() in plan_call["prompt"]


def test_the_locator_prompt_carries_the_quoted_source(wired):
    sbp.run_sbp_grounded_contract(dict(PROBLEM))
    contract_call = wired[0]
    assert "IMPLEMENTATION CONTRACT" in contract_call["prompt"]
    assert "--- FILE: src/user/emails.py ---" in contract_call["prompt"]


def test_the_planner_runs_after_the_tree_is_read_and_before_any_scoring(wired):
    """Ordering is the fix. Reading the tree afterwards would be no better than
    not reading it, and scoring first would spend an oracle call on a patch the
    plan had no part in."""
    sbp.run_sbp_plan_exec(dict(PROBLEM))
    events = FakeEnv.instances[0].events
    assert events[0] == "read_source"
    assert events.index("score") > 0
    # The plan was the first model call of the whole task.
    assert wired[0]["model"] == OPUS_5_ID


def test_the_planner_is_still_the_first_model_call(wired):
    """H2's shape: the frontier model is bought BEFORE the first oracle call,
    not after a failure."""
    sbp.run_sbp_plan_exec(dict(PROBLEM))
    assert wired[0]["model"] == OPUS_5_ID
    assert all(c["model"] == GEMINI_37_FLASH_ID for c in wired[1:])


def test_the_locator_is_a_cheap_model_and_a_small_budget(wired):
    """The contract is <150 words by design; a large cap would let it drift
    into writing the diff, which is the executor's job."""
    sbp.run_sbp_grounded_contract(dict(PROBLEM))
    assert wired[0]["model"] == SONNET_ID
    assert wired[0]["max_tokens"] == 512


def test_the_architects_budget_is_the_one_its_role_asks_for(wired):
    sbp.run_sbp_plan_exec(dict(PROBLEM))
    assert wired[0]["max_tokens"] == 2048


# ==============================================================================
# --- THE PLAN REACHES THE EXECUTOR ---
# ==============================================================================

def test_the_plan_is_handed_to_every_executor_rung(wired):
    """An arm whose plan does not reach the executor is a flash single wearing
    an expensive name."""
    sbp.run_sbp_plan_exec(dict(PROBLEM))
    for call in wired[1:]:
        assert "Architect's implementation plan" in call["prompt"]


def test_the_executor_rungs_also_see_the_source(wired):
    """Both halves are grounded, or the comparison to S0b/S1/S2 is not like
    for like."""
    sbp.run_sbp_plan_exec(dict(PROBLEM))
    assert "--- FILE: src/user/emails.py ---" in wired[1]["prompt"]


# ==============================================================================
# --- BUDGETS AND LEDGERS ---
# ==============================================================================

def test_the_planner_costs_land_in_the_arms_ledger(wired):
    """It used to be accumulated outside `_ladder`; moving it inside must not
    make the frontier model free."""
    out = sbp.run_sbp_plan_exec(dict(PROBLEM))
    assert out["as_run_usd"] == pytest.approx(0.05 * len(wired))
    assert len(wired) == sbp.MAX_ORACLE_CALLS + 1, "one plan plus every rung"


def test_the_plan_is_an_llm_call_not_an_oracle_call(wired):
    """H2 holds container test runs constant; the plan is paid for in dollars,
    not in oracle budget."""
    sbp.run_sbp_plan_exec(dict(PROBLEM))
    assert len(FakeEnv.instances[0].calls) == sbp.MAX_ORACLE_CALLS


def test_the_record_states_whether_the_planner_was_grounded(wired):
    """A run with `SBP_GROUNDING_CHARS=0` and one without produce different
    arms; the row has to say which it was rather than leaving it to be
    inferred from the pass rate."""
    out = sbp.run_sbp_plan_exec(dict(PROBLEM))
    assert out["grounding"]["planner_grounded"] is True
    assert out["grounding"]["read"] == ["src/user/emails.py"]


def test_a_blind_planner_is_recorded_as_blind(wired, monkeypatch):
    """The A/B leg: `SBP_GROUNDING_CHARS=0` reproduces the old behaviour, and
    the record says so instead of looking identical to the grounded run."""
    monkeypatch.setattr("src.swebench_pro.SBP_GROUNDING_CHARS", 0)
    out = sbp.run_sbp_plan_exec(dict(PROBLEM))
    assert out["grounding"]["planner_grounded"] is False
    assert "--- FILE:" not in wired[0]["prompt"]


# ==============================================================================
# --- THE ARMS THAT PASS NO PLANNER ARE UNCHANGED ---
# ==============================================================================
#
# This is what keeps report 30's cached S0b/S1/S2 records valid: the branch
# added to `_ladder` is skipped entirely when no planner is supplied, so those
# three arms take the same path they took when their records were written.

@pytest.mark.parametrize("arm,first_model", [
    (lambda p: sbp.run_sbp_single(p, model_id=SONNET_ID), SONNET_ID),
    (sbp.run_sbp_cascade, GEMINI_37_FLASH_ID),
    (sbp.run_sbp_evidence_gate, GEMINI_37_FLASH_ID),
])
def test_an_arm_with_no_planner_makes_no_extra_model_call(arm, first_model, wired):
    arm(dict(PROBLEM))
    assert len(wired) == sbp.MAX_ORACLE_CALLS, "no pre-oracle call was added"
    assert wired[0]["model"] == first_model
    assert wired[0]["max_tokens"] == sbp.MAX_PATCH_TOKENS


@pytest.mark.parametrize("arm", [
    lambda p: sbp.run_sbp_single(p, model_id=SONNET_ID),
    sbp.run_sbp_cascade,
    sbp.run_sbp_evidence_gate,
])
def test_an_arm_with_no_planner_records_no_planner_flag(arm, wired):
    """`planner_grounded` is absent, not False: the key exists only for arms
    that actually have a planner, so a report cannot read S0b as 'a blind
    planner arm'."""
    out = arm(dict(PROBLEM))
    assert "planner_grounded" not in (out["grounding"] or {})


@pytest.mark.parametrize("arm", [
    lambda p: sbp.run_sbp_single(p, model_id=SONNET_ID),
    sbp.run_sbp_cascade,
    sbp.run_sbp_evidence_gate,
])
def test_an_arm_with_no_planner_spends_only_the_oracle_budget(arm, wired):
    arm(dict(PROBLEM))
    assert len(FakeEnv.instances[0].calls) == sbp.MAX_ORACLE_CALLS


def test_the_first_solve_prompt_of_a_plannerless_arm_uses_the_solver_role(wired):
    """`EXECUTOR_ROLE` is for arms working to a plan. A plannerless arm reaching
    for it would be a silent prompt change to three cached arms."""
    sbp.run_sbp_cascade(dict(PROBLEM))
    assert "Read the issue, the requirements and the interfaces" in wired[0]["prompt"]
    assert "Architect's implementation plan" not in wired[0]["prompt"]
