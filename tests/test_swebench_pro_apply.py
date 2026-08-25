# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Contract tests for patch application and the evidence it produces.

Why this exists
---------------
On FeatureBench, whose per-task records are committed, `patch did not apply`
is the *final* error on 75-90% of tasks across every arm:

    F0a flash    38/48    F0b sonnet   42/48    F1 cascade   36/48
    F2 evi-gate  40/48    F3 plan-exec 43/48

SWE-bench Pro runs the same code path against larger, multi-file patches. Two
things follow, and both are pinned here.

**The apply ladder.** `git apply` is all-or-nothing on context, and the defect
it most often rejects — a miscounted `@@` header — carries no information about
whether the edit was right. The relaxations added here each target one such
artefact and none of them relax *grading*: the candidate still has to make the
repository's own suite pass afterwards.

**The evidence.** `_try_apply` was collecting `git apply --verbose` output —
the failing file, the hunk, the context block it searched for — into
`self.apply_log`, and `SWEBenchProEnv.score` was throwing it away in favour of
a fixed sentence. The repair turn received 31 tokens of constant, which makes
the second rung an independent re-roll rather than a repair, and makes the
straitjacket digest under test a no-op on the dataset's dominant failure.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import evaluator                                            # noqa: E402
from src.evaluator import (SWEBenchProEnv, _APPLY_STRATEGIES, _guard_evidence,
                           guard_reason, missing_patch_error, extract_patch,
                           GUARD_REASONS, ENVIRONMENT_REASONS)        # noqa: E402


PROBLEM = {
    "instance_id": "instance_demo__repo-abc123-vnan",
    "repo": "demo/repo",
    "repo_language": "python",
    "base_commit": "0" * 40,
    "problem_statement": "s", "requirements": "r", "interface": "i",
    "fail_to_pass": ["tests/t.py | a"], "pass_to_pass": [],
    "selected_test_files_to_run": ["tests/t.py"],
    "before_repo_set_cmd": "git checkout abc123 -- tests/t.py",
    "dockerhub_tag": "demo.repo-demo__repo-abc123",
    "image_name": "jefzda/sweap-images:demo.repo-demo__repo-abc123",
    "repo_workdir": "/app",
}

PATCH = ("diff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n"
         "@@ -1 +1 @@\n-old\n+new\n")


# ==============================================================================
# --- THE APPLY LADDER ---
# ==============================================================================

def test_the_strictest_strategy_is_tried_first():
    """A patch that applies cleanly must never be graded through a fuzzy path."""
    assert _APPLY_STRATEGIES[0] == ["git", "apply", "--verbose"]


def test_recount_is_available_because_miscounted_hunk_headers_carry_no_meaning():
    """`@@ -a,b +c,d @@` line counts are the most common defect in a
    model-authored diff and the one that says least about the edit."""
    assert any("--recount" in s for s in _APPLY_STRATEGIES)


def test_a_three_way_merge_is_available_because_the_tree_is_at_the_base_commit():
    assert any("--3way" in s for s in _APPLY_STRATEGIES)


def test_patch_is_forbidden_from_reversing_a_diff_it_thinks_is_applied():
    """Without `--forward`, `patch --batch` exits 0 having *deleted* the change
    the next step needs. Silence in exactly the place a wrong number comes from."""
    fallback = [s for s in _APPLY_STRATEGIES if s[0] == "patch"]
    assert fallback and all("--forward" in s for s in fallback)


def test_every_strategy_ends_in_the_patch_path_slot():
    """`_try_apply` appends the path, so a strategy taking a `-i` flag must have
    it last or the file becomes an argument to something else."""
    for argv in _APPLY_STRATEGIES:
        assert argv[0] in ("git", "patch")
        if argv[0] == "patch":
            assert argv[-1] == "-i"


class _Applier:
    """A `_DockerRepoEnv` stub: `succeed_at` is the strategy index that works."""

    def __init__(self, succeed_at=None):
        self.succeed_at = succeed_at
        self.cmds = []
        self.resets = 0
        self.apply_log = []

    _sh = None

    def sh(self, script, timeout=None, check=True):
        self.cmds.append(script)
        idx = len(self.cmds) - 1
        ok = self.succeed_at is not None and idx == self.succeed_at
        return types.SimpleNamespace(
            returncode=0 if ok else 1,
            stdout="", stderr="" if ok else
            f"error: patch failed: w.py:1\nerror: while searching for:\nold\n")

    def reset(self):
        self.resets += 1


def _bind(applier):
    applier._sh = applier.sh
    applier._try_apply = types.MethodType(SWEBenchProEnv._try_apply, applier)
    applier.apply_evidence = types.MethodType(SWEBenchProEnv.apply_evidence, applier)
    return applier


def test_apply_stops_at_the_first_strategy_that_works():
    a = _bind(_Applier(succeed_at=0))
    assert a._try_apply("/workspace/patch.diff") is True
    assert len(a.cmds) == 1


def test_a_later_strategy_can_rescue_a_diff_the_strict_one_rejected():
    a = _bind(_Applier(succeed_at=1))
    assert a._try_apply("/workspace/patch.diff") is True
    assert len(a.cmds) == 2
    assert "--recount" in a.cmds[1]


def test_every_strategy_is_tried_before_giving_up():
    a = _bind(_Applier(succeed_at=None))
    assert a._try_apply("/workspace/patch.diff") is False
    assert len(a.cmds) == len(_APPLY_STRATEGIES)


def test_the_worktree_is_reset_between_failed_strategies_but_not_before_the_first():
    """`git apply --3way` writes conflict markers and *then* exits non-zero.
    Without a reset the next strategy applies on top of that debris, and any
    success it reports is a success against a tree nobody chose."""
    a = _bind(_Applier(succeed_at=None))
    a._try_apply("/workspace/patch.diff", reset=a.reset)
    assert a.resets == len(_APPLY_STRATEGIES) - 1


def test_no_reset_happens_when_the_caller_supplies_none():
    """FeatureBench applies a test patch during setup, where a reset would
    discard the very thing being staged."""
    a = _bind(_Applier(succeed_at=None))
    a._try_apply("/workspace/patch.diff")
    assert a.resets == 0


def test_the_apply_log_records_the_command_and_what_it_said():
    a = _bind(_Applier(succeed_at=None))
    a._try_apply("/workspace/patch.diff")
    log = a.apply_evidence()
    assert "git apply --verbose" in log
    assert "while searching for" in log


def test_the_apply_log_is_bounded():
    a = _bind(_Applier(succeed_at=None))
    a._try_apply("/workspace/patch.diff")
    assert len(a.apply_evidence(limit=200)) <= 200


# ==============================================================================
# --- WHAT THE REPAIR TURN ACTUALLY RECEIVES ---
# ==============================================================================

@pytest.fixture
def env(monkeypatch):
    e = SWEBenchProEnv(PROBLEM)
    e.started = True
    e.sandbox = "/tmp/sandbox"
    e.scripts = {"run_script": "run_all_tests", "parser": "#", "env_exports": ""}
    e.apply_ok = False
    e.calls = []

    monkeypatch.setattr(e, "_sh", lambda s, timeout=None, check=True: (
        e.calls.append(s) or types.SimpleNamespace(returncode=0, stdout="", stderr="")))
    monkeypatch.setattr(e, "_write", lambda p, t: None)

    def fake_apply(path, reset=None):
        e.apply_log = ["$ git apply --verbose /workspace/patch.diff\n"
                       "[exit 1] error: patch failed: src/user/emails.py:120\n"
                       "error: while searching for:\n    if user.email:\n"]
        return e.apply_ok
    monkeypatch.setattr(e, "_try_apply", fake_apply)
    return e


def test_an_apply_failure_hands_the_model_the_apply_log(env):
    """The whole point. Before this the repair turn received a fixed sentence,
    so the second rung had the same information as the first and 'repair' was
    an independent re-roll."""
    passed, evidence = env.score(PATCH)
    assert passed is False
    assert "src/user/emails.py:120" in str(evidence)
    assert "while searching for" in str(evidence)
    assert "if user.email:" in str(evidence)


def test_an_apply_failure_says_no_test_was_run_rather_than_that_tests_failed(env):
    passed, evidence = env.score(PATCH)
    assert "no test was run" in str(evidence)
    assert "did NOT apply" in str(evidence)


def test_an_apply_failure_is_typed_so_the_router_can_read_it(env):
    passed, evidence = env.score(PATCH)
    assert evidence.reason == "apply_failed"
    assert evidence.failure_class == "PatchApplyError"


def test_the_digest_of_a_guard_failure_is_the_guard_message_itself(env):
    """There is no captured stream to contain, so containment must not invent
    one — and must not leave the digest empty either."""
    _, evidence = env.score(PATCH)
    assert evidence.digest == str(evidence)
    assert evidence.contained is False


def test_apply_is_given_the_reset_callback(env, monkeypatch):
    seen = {}
    monkeypatch.setattr(env, "_try_apply",
                        lambda path, reset=None: seen.setdefault("reset", reset) and False)
    env.score(PATCH)
    assert seen["reset"] == env.reset


# ==============================================================================
# --- THE GUARD REASON VOCABULARY ---
# ==============================================================================

def test_every_guard_reason_has_a_failure_class():
    assert all(GUARD_REASONS.values())


def test_environment_reasons_are_exactly_the_ones_no_model_can_fix():
    assert ENVIRONMENT_REASONS == {"container_unavailable", "restore_failed",
                                   "row_no_test_files", "execution_error",
                                   "harness_error"}


@pytest.mark.parametrize("response,reason", [
    ("I would change the widget module.", "no_patch"),
    ("```diff\n@@ -1 +1 @@\n-a\n+b\n```", "not_a_diff"),
    ("```diff\ndiff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n```", "no_hunk"),
])
def test_malformed_responses_are_named_not_lumped_together(response, reason):
    """`no_patch` and `no_hunk` are different defects with different fixes, and
    a report that shows only their sum cannot tell which one to act on."""
    guard = missing_patch_error(extract_patch(response))
    assert guard is not None
    assert guard.reason == reason
    assert guard.failure_class == "MalformedPatch"


def test_a_well_formed_patch_passes_the_birth_gate():
    assert missing_patch_error(PATCH) is None


def test_a_container_that_never_started_is_an_environment_failure():
    e = SWEBenchProEnv(PROBLEM)
    e.started, e.setup_error = False, "docker run failed: manifest unknown"
    passed, evidence = e.score(PATCH)
    assert passed is False
    assert evidence.reason == "container_unavailable"
    assert evidence.reason in ENVIRONMENT_REASONS


def test_a_failed_test_restore_is_an_environment_failure(env):
    """Grading against the repository's original tests would measure nothing, so
    this must never be filed as a model failure."""
    env.apply_ok = True
    env._sh = lambda s, timeout=None, check=True: types.SimpleNamespace(
        returncode=1 if s.startswith("git checkout abc123 --") else 0,
        stdout="", stderr="pathspec")
    passed, evidence = env.score(PATCH)
    assert passed is False and evidence.reason == "restore_failed"


def test_guard_reason_is_empty_when_the_suite_actually_ran():
    """The one case where a pass/fail is a statement about the model."""
    run = types.SimpleNamespace(digest="D", native_payload=lambda: "raw",
                                metrics=lambda: {})
    assert guard_reason(evaluator._from_run(run)) == ""


def test_guard_reason_of_a_plain_string_is_empty_rather_than_raising():
    assert guard_reason("some error text") == ""
    assert guard_reason(None) == ""
