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
    fb.run_fb_diff_contract,
    fb.run_fb_diff_aware_gate,
    fb.run_fb_spec_deconstruct,
])
def test_every_arm_makes_at_most_three_oracle_calls(wired, arm):
    arm() if not callable(getattr(arm, "__wrapped__", None)) else arm(PROBLEM)
    env = FakeEnv.instances[-1]
    assert len(env.calls) == fb.MAX_ORACLE_CALLS, (
        "arms must be matched on oracle calls -- that is the scarce resource "
        "H2 makes a claim about")


# -- who holds each rung ---------------------------------------------------

def test_cascade_escalates_one_rung_per_failure_ending_at_the_frontier(wired):
    """Cheap rung, dear rung, then the frontier once the ladder is exhausted.

    At the old two-call budget this arm stopped at sonnet, and it stopped there
    for an arithmetic reason rather than a routing one: one gate evaluation,
    at `attempt == 1`, which `gate_after_ladder` answers with "cheap rungs
    remain". `docs/featurebench-n48-lessons.md` §2 is the audit."""
    out = fb.run_fb_cascade(PROBLEM)
    assert out["routing"]["rungs"] == [
        f"{fb.GEMINI_37_FLASH_ID}/low", f"{fb.SONNET_ID}/off",
        f"{fb.FRONTIER}/off"]
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


def test_diff_aware_gate_escalates_on_broad_test_failure(wired):
    out = fb.run_fb_diff_aware_gate(PROBLEM)
    rungs = out["routing"]["rungs"]
    assert rungs[1] == f"{fb.FRONTIER}/off"
    assert out["routing"]["frontier_used"] is True


def test_diff_aware_gate_escalates_on_consecutive_patch_apply_failures(wired, monkeypatch):
    """The arm's stated purpose, which it did not do before.

    Its old gate substring-matched the evidence prose for
    `"patch did not apply"`, so it depended on the exact wording of a message
    in another module; and with a two-call budget the branch could not be
    reached at all. Both are gone: the guard is typed, so a repeat arrives as
    `stalled`, and the budget leaves a second gate evaluation to fire on."""
    from src.evaluator import _guard_evidence

    def patch_fail_score(self, patch):
        self.calls.append(patch)
        return False, _guard_evidence("did NOT apply", "apply_failed")

    monkeypatch.setattr(FakeEnv, "score", patch_fail_score)
    out = fb.run_fb_diff_aware_gate(PROBLEM)
    rungs = out["routing"]["rungs"]
    assert rungs[0] == f"{fb.GEMINI_37_FLASH_ID}/low"
    assert rungs[1] == f"{fb.SONNET_ID}/off"
    assert rungs[2] == f"{fb.FRONTIER}/off"
    assert out["routing"]["frontier_used"] is True


def test_diff_aware_gate_never_escalates_on_an_environment_failure(wired, monkeypatch):
    """A container that would not start is the worst possible reason to buy the
    study's most expensive tier."""
    from src.evaluator import _guard_evidence

    def env_fail_score(self, patch):
        self.calls.append(patch)
        return False, _guard_evidence("no such image", "container_unavailable")

    monkeypatch.setattr(FakeEnv, "score", env_fail_score)
    out = fb.run_fb_diff_aware_gate(PROBLEM)
    assert out["routing"]["frontier_used"] is False
    assert all("environment failure" in d["why"]
               for d in out["routing"]["decisions"])


def test_the_diff_aware_gate_is_reachable_from_module_scope():
    """It was a closure inside the arm, so the registry invariant could not
    evaluate it and no test could reach it directly."""
    assert fb.gate_diff_aware.requires_typed_evidence is True


def test_diff_contract_uses_contracted_roles(wired):
    out = fb.run_fb_diff_contract(PROBLEM)
    assert "CRITICAL UNIFIED DIFF REQUIREMENTS" in wired[0]["prompt"]
    assert "principal software engineer repairing" in wired[1]["prompt"]
    assert out["routing"]["rungs"][:2] == [
        f"{fb.GEMINI_37_FLASH_ID}/low", f"{fb.SONNET_ID}/off"]


def test_spec_deconstruct_extracts_manifest_first(wired):
    out = fb.run_fb_spec_deconstruct(PROBLEM)
    assert wired[0]["model"] == fb.GEMINI_37_FLASH_ID, "the manifest extractor runs first"
    assert fb.MANIFEST_ROLE[:30] in wired[0]["prompt"]
    assert [c["model"] for c in wired[1:]] == [
        fb.GEMINI_37_FLASH_ID, fb.SONNET_ID, fb.SONNET_ID]
    assert "FILE AND INTERFACE MANIFEST" in wired[0]["prompt"]


RANK = {f"{fb.GEMINI_37_FLASH_ID}/low": 0, f"{fb.SONNET_ID}/off": 1,
        f"{fb.FRONTIER}/off": 2}


def test_the_ladder_never_de_escalates(wired):
    """The strongest finding in the repo, enforced rather than hoped for.

    N=100 measured a repair turn that de-escalates rescuing 16% of failures
    against 41% for one that escalates (z = +3.55, p = 0.0004). Escalation here
    is a one-way ratchet: once an arm is at a rung it never drops below it, and
    a spare oracle call is spent re-running the rung it holds.
    """
    for arm in (fb.run_fb_cascade, fb.run_fb_evidence_gate, fb.run_fb_diff_aware_gate):
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
    # ...and never again: the executor and one repair are the cheap model.
    assert ([c["model"] for c in wired[1:]]
            == [fb.GEMINI_37_FLASH_ID] * fb.MAX_ORACLE_CALLS)
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


# -- image coverage planning -----------------------------------------------

def _preflight():
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "fb_preflight", os.path.join(root, "tools", "featurebench_preflight.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic(counts):
    return {f"{img}__case{k}": {"image_name": img}
            for img, n in counts for k in range(n)}


def test_top_images_ranks_by_instance_coverage():
    """These images are ~10 GB each, so which ones you pull decides how much of
    the split is runnable. The ranking is what makes that choice cheap."""
    pf = _preflight()
    problems = _synthetic([("a", 5), ("b", 21), ("c", 18), ("d", 1)])
    ranked, by_image = pf.top_images(problems, 3)
    assert [img for img, _ in ranked] == ["b", "c", "a"]
    assert [len(iids) for _, iids in ranked] == [21, 18, 5]
    assert len(by_image) == 4


def test_top_images_is_deterministic_on_ties():
    pf = _preflight()
    ranked, _ = pf.top_images(_synthetic([("z", 4), ("a", 4), ("m", 4)]), 2)
    assert [img for img, _ in ranked] == ["a", "m"], "ties break by name, not hash order"


def test_top_images_selection_covers_exactly_those_images(tmp_path):
    pf = _preflight()
    pf._hub_size = lambda img: None
    pf._image_present = lambda img: False
    problems = _synthetic([("a", 5), ("b", 21), ("c", 18), ("d", 1)])
    out = tmp_path / "ids.txt"
    covered = pf.report_top(problems, 2, out_path=str(out))
    assert len(covered) == 21 + 18
    assert {i.split("__")[0] for i in covered} == {"b", "c"}
    assert out.read_text().strip().splitlines() == covered


def test_preflight_and_runner_share_one_tasks_grammar(tmp_path):
    """A divergence here would send gold verification and the sweep to
    different task sets, which is the exact failure the preflight exists to
    prevent."""
    import importlib.util
    import os
    pf = _preflight()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "fb_runner", os.path.join(root, "run_benchmark.py"))
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)

    f = tmp_path / "ids.txt"
    f.write_text("x1\n# comment\n\nx2\nx1\n")
    for spec_str in ("a, b ,a,c", f"@{f}"):
        assert pf._requested_tasks(spec_str) == rb._requested_tasks(spec_str)
    assert pf._requested_tasks(f"@{f}") == ["x1", "x2"]


def test_registry_excludes_the_frontier_single_from_the_default_group():
    from src.architectures import get_configurations
    ids = {c["id"] for c in get_configurations("featurebench", "featurebench")}
    assert "fb_evidence_gate" in ids
    assert "fb_single_opus" not in ids, (
        "opus is priced far above the rest; including it would silently reprice "
        "every routine sweep, as ClassEval's baseline does not")


# ==============================================================================
# --- REGISTRY INVARIANTS ---
# ==============================================================================
#
# Rule 2 of `docs/featurebench-n48-lessons.md` §8 — "pick one oracle budget and
# assert it" — as an executable check rather than a note. The audit it comes
# from found that report 20's arms did not share a budget, that three rows are
# labelled as architectures they did not run, and that at two oracle calls over
# a two-rung ladder the frontier tier is unreachable code.

def test_no_arm_advertises_a_frontier_rung_it_cannot_reach():
    assert fb.unreachable_frontier_arms() == []


def test_the_check_catches_the_budget_that_shipped_report_20():
    assert "fb_cascade" in fb.unreachable_frontier_arms(2)


def test_the_registry_refuses_to_finalise_at_an_unrunnable_budget(monkeypatch):
    monkeypatch.setattr(fb, "MAX_ORACLE_CALLS", 2)
    with pytest.raises(RuntimeError, match="cannot be reached"):
        fb._finalise_registry()


def test_arm_labels_state_the_rung_count_that_will_actually_be_spent():
    """Report 20 shipped rows reading "(2 rungs)" beside arms that had spent
    three, because the count was typed into the label by hand."""
    for cfg in fb.FEATUREBENCH_VARIANTS.values():
        assert "{rungs}" not in cfg["name"] and "{rungs}" not in cfg["models"]
        assert " x2" not in cfg["models"], "a stale rung count in the models column"


def test_the_oracle_budget_exceeds_the_rung_count():
    """The invariant behind every unreachable-frontier bug in this repository:
    K oracle calls give K-1 gate evaluations, and every gate compares `attempt`
    against `len(TIERS)`."""
    assert fb.MAX_ORACLE_CALLS > len(fb.TIERS)


# ==========================================================================
# --- THE APPLIER CONTRACT ---
# ==========================================================================
#
# These pin the two defects that made the N=48 sweep unmeasurable, both of
# them in the harness rather than in any arm. See
# docs/featurebench-n48-lessons.md.


def test_extracted_patch_always_ends_in_a_newline():
    """`git apply` exits 128 `corrupt patch` on a diff with no final newline.

    Measured on a scratch repository: a byte-perfect diff applies under all
    five entries of `_APPLY_STRATEGIES` with the trailing newline and under
    *none* of them without it -- the strict applier never even reads the
    worktree. `extract_patch` used to end every return path with `.strip("\n")`,
    so widening the strategy ladder bought nothing at all.
    """
    fenced = extract_patch(GOOD_PATCH)
    unfenced = extract_patch("diff --git a/w.py b/w.py\n--- a/w.py\n"
                             "+++ b/w.py\n@@ -1 +1 @@\n-old\n+new")
    for patch in (fenced, unfenced):
        assert patch.endswith("\n")
        assert not patch.endswith("\n\n")


def test_extract_patch_keeps_every_fenced_block():
    """A multi-file feature is routinely answered one fence per file.

    Returning only the first block scores a fraction of the candidate, which
    reads as a model failure and is not one.
    """
    two = (GOOD_PATCH + "\nand the second file:\n```diff\n"
           "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
           "@@ -1 +1 @@\n-a\n+b\n```")
    got = extract_patch(two)
    assert [l for l in got.splitlines() if l.startswith("diff --git")] == [
        "diff --git a/w.py b/w.py", "diff --git a/x.py b/x.py"]


def test_extract_patch_still_returns_empty_for_prose():
    assert extract_patch("I cannot help with that.") == ""
    assert extract_patch("") == ""


# ==========================================================================
# --- REPOSITORY GROUNDING ---
# ==========================================================================


class GroundedEnv(FakeEnv):
    """A container whose files can be read, so grounding has something to quote."""

    FILES = {"src/widget.py": "class Widget:\n    pass\n",
             "src/registry.py": "REGISTRY = {}\n"}

    def _sh(self, script, timeout=None, check=True):
        if script.startswith("git grep"):
            return types.SimpleNamespace(stdout="src/registry.py\n", stderr="",
                                         returncode=0)
        out = []
        for path, body in self.FILES.items():
            if f'"{path}"' in script:
                out.append(f"@@FB_FILE@@ {path}\n{body}")
        return types.SimpleNamespace(stdout="".join(out), stderr="", returncode=0)


GROUNDING_PROBLEM = dict(PROBLEM,
                         problem_statement="Add a registry to src/widget.py and "
                                           "src/registry.py. Call `Widget.register`.")


def test_grounded_arm_quotes_the_repository_into_the_solve_prompt(wired, monkeypatch):
    """The defect this arm exists to test: the model was never shown the files.

    A unified diff for an existing file is a claim about bytes already on disk,
    so a model that has not read the file is guessing its context lines. 94% of
    the N=48 sweep's failures were `patch did not apply`.
    """
    monkeypatch.setattr(fb, "FeatureBenchEnv", GroundedEnv)
    out = fb.run_fb_grounded(GROUNDING_PROBLEM)

    assert "class Widget:" in wired[0]["prompt"]
    assert "--- BEGIN src/widget.py ---" in wired[0]["prompt"]
    # ...and the repair turn keeps it, or the second rung re-guesses too.
    assert "class Widget:" in wired[1]["prompt"]
    assert out["grounding"]["read"] == ["src/widget.py", "src/registry.py"]
    assert out["grounding"]["chars"] > 0


def test_grounding_never_quotes_the_graded_tests_by_default():
    """Quoting the graded tests changes what the benchmark measures.

    Opt in with `FB_GROUND_TESTS=1` and say so in the report; do not let it
    happen because a test file was mentioned in the problem statement.
    """
    problem = dict(PROBLEM,
                   problem_statement="Fix tests/test_widget.py and src/widget.py.")
    assert fb._candidate_paths(problem) == ["src/widget.py"]


def test_a_container_that_cannot_be_read_falls_back_to_the_blind_prompt(wired,
                                                                       monkeypatch):
    """Grounding is an enrichment, never a precondition.

    A row whose files cannot be read is still a valid blind run, recorded as
    `chars: 0` rather than lost.
    """
    class Unreadable(FakeEnv):
        def _sh(self, script, timeout=None, check=True):
            raise RuntimeError("docker exec failed")

    monkeypatch.setattr(fb, "FeatureBenchEnv", Unreadable)
    out = fb.run_fb_grounded(GROUNDING_PROBLEM)
    assert out["grounding"]["chars"] == 0
    assert "--- BEGIN" not in wired[0]["prompt"]
    assert out["repair_loops"] >= 1          # the arm still ran


def test_ungrounded_arms_are_byte_identical_to_before(wired, monkeypatch):
    """Grounding is opt-in per arm, so the existing rows stay reproducible."""
    monkeypatch.setattr(fb, "FeatureBenchEnv", GroundedEnv)
    fb.run_fb_cascade(PROBLEM)
    assert "--- BEGIN" not in wired[0]["prompt"]
    assert "@@FB_FILE@@" not in wired[0]["prompt"]


def test_grounded_gate_can_actually_reach_the_frontier():
    """Rule 4 of the lessons doc, asserted rather than remembered."""
    assert "fb_grounded_gate" not in fb.unreachable_frontier_arms()


# ==========================================================================
# --- WHAT THE N=2 RUN EXPOSED ---
# ==========================================================================


def test_blank_lines_inside_a_hunk_get_their_marker_back():
    """The N=2 failure signature: `corrupt patch at line N` on all 5 strategies.

    A blank source line must be written with a column-0 marker -- a lone space
    for context, `+` for an addition. Models emit a genuinely empty line, which
    `git apply` cannot parse *at all*: measured on a scratch repository, an
    unmarked blank inside one new-file hunk applies under 0 of 5 strategies,
    and `--recount` / `--ignore-whitespace` / `--3way` cannot rescue it because
    the patch never parses.
    """
    from src.evaluator import normalise_hunk_markers

    # New file: the blank belongs to the added side.
    new_file = ("diff --git a/n.py b/n.py\nnew file mode 100644\n"
                "--- /dev/null\n+++ b/n.py\n@@ -0,0 +1,3 @@\n+import os\n\n+x = 1\n")
    assert normalise_hunk_markers(new_file).splitlines()[6] == "+"

    # Existing file: the blank is context.
    edit = ("diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n"
            "@@ -1,3 +1,4 @@\n import os\n\n+y = 2\n")
    assert normalise_hunk_markers(edit).splitlines()[5] == " "


def test_a_blank_between_two_files_is_not_hunk_content():
    """Models separate per-file diffs with a blank line. It is not a context line."""
    from src.evaluator import normalise_hunk_markers

    two = ("diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1,2 @@\n"
           " import os\n+y = 2\n\ndiff --git a/n.py b/n.py\n--- a/n.py\n+++ b/n.py\n"
           "@@ -1 +1,2 @@\n import os\n+z = 3\n")
    got = normalise_hunk_markers(two).splitlines()
    # The blank is dropped, not turned into a context line, so the last line of
    # the first file's hunk sits directly against the next file's header.
    assert got[got.index("diff --git a/n.py b/n.py") - 1] == "+y = 2"


def test_container_absolute_paths_resolve_to_repository_paths():
    """`/testbed/src/pkg/x.py` is `src/pkg/x.py` inside the container.

    The N=2 run asked for `testbed/src/packaging/metadata.py` -- the leading
    slash lost, the prefix kept -- so the one file it most needed was recorded
    as skipped, and grounding quoted `docs/conf.py` instead.
    """
    assert fb._normalise_repo_path("/testbed/src/packaging/metadata.py") == \
        "src/packaging/metadata.py"
    assert fb._normalise_repo_path("src/packaging/metadata.py") == \
        "src/packaging/metadata.py"


def test_grounding_spends_its_budget_on_source_before_documentation():
    problem = dict(PROBLEM, problem_statement=(
        "Update docs/conf.py and /testbed/src/pkg/core.py for the new option."))
    assert fb._candidate_paths(problem) == ["src/pkg/core.py", "docs/conf.py"]


def test_a_failed_row_keeps_the_candidate_patch_for_diagnosis(wired):
    """`corrupt patch at line 8` is unreadable without line 8.

    The N=2 run recorded that error under all five strategies and the patch was
    nowhere on disk, so the cause had to be reproduced from scratch instead of
    looked up.
    """
    out = fb.run_fb_cascade(PROBLEM)
    assert not out["passed"]
    assert "diff --git a/w.py b/w.py" in out["candidate_patch"]


def test_a_passing_row_does_not_store_its_patch(wired, monkeypatch):
    class Solves(FakeEnv):
        def score(self, patch):
            self.calls.append(patch)
            return True, FakeEvidence("2 passed", typed=True)

    monkeypatch.setattr(fb, "FeatureBenchEnv", Solves)
    assert "candidate_patch" not in fb.run_fb_cascade(PROBLEM)
