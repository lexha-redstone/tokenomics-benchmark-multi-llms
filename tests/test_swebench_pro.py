# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Contract tests for the SWE-bench Pro harness and arms.

Nothing here touches Docker or an API. What they pin is the set of things that
would silently produce a *wrong number* rather than an error:

* the list columns are Python literals, not JSON -- `json.loads` drops the rows
  whose test names contain an apostrophe, which shrinks the grading set;
* `before_repo_set_cmd` contributes only its last line, because the first three
  lines would throw the candidate patch away;
* the graded test files are restored **after** the patch is applied, which is
  the only thing stopping a patch from passing by editing the tests;
* a stale `output.json` is deleted before each run, or an attempt whose parser
  crashed is graded against the previous rung's results;
* resolution is `(fail_to_pass | pass_to_pass) <= passed`, so breaking a
  pass_to_pass test is a failure even when the issue was fixed;
* every arm spends exactly three oracle calls, which is the comparison H2
  rests on.

The one test that needs Docker is opt-in (`SBP_INTEGRATION=1`) and runs a row's
own gold patch: the only end-to-end check that this harness agrees with the
benchmark it claims to implement.
"""

import json
import os
import subprocess
import sys
import tempfile
import types

import pytest

from src import swebench_pro as sbp
from src import datasets as ds
from src.evaluator import (SWEBenchProEnv, extract_patch, missing_patch_error,
                           sbp_resolution, sbp_test_script, docker_available)


PROBLEM = {
    "instance_id": "instance_demo__repo-abc123-vnan",
    "repo": "demo/repo",
    "repo_language": "python",
    "base_commit": "0" * 40,
    "problem_statement": "Email validation status is not handled correctly.",
    "requirements": "The loadUserInfo function should attach pending flags.",
    "interface": "Type: Method\nName: db.mget",
    "fail_to_pass": ["tests/t.py | Key methods should return null"],
    "pass_to_pass": ["tests/t.py | Key methods should set a key"],
    "selected_test_files_to_run": ["tests/t.py", "tests/u.py"],
    "before_repo_set_cmd": ("git reset --hard 0000\ngit clean -fd \n"
                            "git checkout 0000 \n"
                            "git checkout abc123 -- tests/t.py tests/u.py"),
    "dockerhub_tag": "demo.repo-demo__repo-abc123",
    "image_name": "jefzda/sweap-images:demo.repo-demo__repo-abc123",
    "repo_workdir": "/app",
}

GOOD_PATCH = ("```diff\n"
              "diff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n"
              "@@ -1 +1 @@\n-old\n+new\n```")


def _output(*named):
    return {"tests": [{"name": n, "status": s} for n, s in named]}


# ==============================================================================
# --- DATASET COLUMNS ---
# ==============================================================================

def test_list_columns_are_python_literals_not_json():
    """The upstream column mixes quote styles; `json.loads` fails on exactly it."""
    raw = """["a | it should work", 'b | it should not error if key doesn\\'t exist']"""
    with pytest.raises(Exception):
        json.loads(raw)
    assert ds._sbp_list(raw) == ["a | it should work",
                                 "b | it should not error if key doesn't exist"]


def test_list_columns_accept_json_and_lists_and_empty():
    assert ds._sbp_list('["x", "y"]') == ["x", "y"]
    assert ds._sbp_list(["x"]) == ["x"]
    assert ds._sbp_list("") == []
    assert ds._sbp_list(None) == []


def test_unparseable_list_column_becomes_one_entry_not_nothing():
    """A schema change must surface as a failing row, not as a smaller test set."""
    assert ds._sbp_list("tests/t.py | only one") == ["tests/t.py | only one"]


def test_image_comes_from_the_dataset_tag_not_a_derivation():
    assert ds.swebench_pro_image(PROBLEM) == \
        "jefzda/sweap-images:demo.repo-demo__repo-abc123"
    assert ds.swebench_pro_image(PROBLEM, username="me") == \
        "me/sweap-images:demo.repo-demo__repo-abc123"
    assert ds.swebench_pro_image({"instance_id": "x"}) == ""


def test_only_the_last_line_of_before_repo_set_cmd_is_used():
    """The first three lines reset the tree; running them would discard the patch."""
    cmd = ds.sbp_restore_tests_cmd(PROBLEM)
    assert cmd == "git checkout abc123 -- tests/t.py tests/u.py"
    assert "reset --hard" not in cmd
    assert ds.sbp_restore_tests_cmd({}) == ""


def test_required_tests_are_fail_to_pass_first_and_deduplicated():
    p = dict(PROBLEM, fail_to_pass=["a", "b"], pass_to_pass=["b", "c"])
    assert ds.sbp_required_tests(p) == ["a", "b", "c"]


def test_loader_parses_columns_and_filters_by_language(tmp_path, monkeypatch):
    rows = [
        dict(PROBLEM, instance_id="py-row", repo_language="python",
             fail_to_pass='["a"]', pass_to_pass='["b"]',
             selected_test_files_to_run='["tests/t.py"]'),
        dict(PROBLEM, instance_id="js-row", repo_language="js",
             fail_to_pass='["a"]', pass_to_pass='["b"]',
             selected_test_files_to_run='["test/t.js"]'),
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(ds, "ensure_swebench_pro_dataset", lambda split=None: str(path))
    monkeypatch.setattr(ds, "load_swebench_pro_quarantine", lambda split=None: {})

    both = ds.load_swebench_pro_problems()
    assert set(both) == {"py-row", "js-row"}
    row = both["py-row"]
    assert row["fail_to_pass"] == ["a"] and row["pass_to_pass"] == ["b"]
    assert row["selected_test_files_to_run"] == ["tests/t.py"]
    assert row["image_name"].startswith("jefzda/sweap-images:")
    assert row["repo_workdir"] == "/app" and row["dataset_type"] == "swebench_pro"

    assert set(ds.load_swebench_pro_problems(languages=["python"])) == {"py-row"}


def test_loader_honours_the_quarantine_file(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(dict(PROBLEM, instance_id="bad-row",
                                    fail_to_pass='["a"]', pass_to_pass='[]',
                                    selected_test_files_to_run='["t.py"]')),
                    encoding="utf-8")
    monkeypatch.setattr(ds, "ensure_swebench_pro_dataset", lambda split=None: str(path))
    monkeypatch.setattr(ds, "load_swebench_pro_quarantine",
                        lambda split=None: {"bad-row": {"reason": "missing_image"}})
    assert ds.load_swebench_pro_problems() == {}
    assert set(ds.load_swebench_pro_problems(apply_quarantine=False)) == {"bad-row"}


# ==============================================================================
# --- THE GRADING RULE ---
# ==============================================================================

def test_resolved_needs_every_required_test_including_pass_to_pass():
    out = _output(("tests/t.py | Key methods should return null", "PASSED"),
                  ("tests/t.py | Key methods should set a key", "PASSED"))
    r = sbp_resolution(out, PROBLEM)
    assert r["resolved"] is True and r["test_pass_ratio"] == 1.0
    assert r["required"] == 2 and r["missing"] == 0


def test_a_regression_in_pass_to_pass_fails_the_row():
    """Fixing the issue while breaking something else is not a resolve."""
    out = _output(("tests/t.py | Key methods should return null", "PASSED"),
                  ("tests/t.py | Key methods should set a key", "FAILED"))
    r = sbp_resolution(out, PROBLEM)
    assert r["resolved"] is False
    assert r["test_pass_ratio"] == 0.5
    assert r["missing_names"] == ["tests/t.py | Key methods should set a key"]


def test_a_test_that_was_never_reported_counts_as_missing():
    """A suite that crashed before collection reports nothing, which is not a pass."""
    r = sbp_resolution({"tests": []}, PROBLEM)
    assert r["resolved"] is False and r["test_pass_ratio"] == 0.0
    assert r["reported"] == 0


def test_absent_output_json_is_a_failure_not_an_exception():
    r = sbp_resolution(None, PROBLEM)
    assert r["resolved"] is False and r["missing"] == 2


def test_a_row_with_no_required_tests_never_resolves():
    """Upstream's `set() <= passed` is True; a row that passes everything is worse
    than a row that is skipped."""
    r = sbp_resolution(_output(("x", "PASSED")),
                       dict(PROBLEM, fail_to_pass=[], pass_to_pass=[]))
    assert r["resolved"] is False and r["test_pass_ratio"] is None


def test_only_passed_counts_skipped_is_not_passed():
    out = _output(("tests/t.py | Key methods should return null", "SKIPPED"),
                  ("tests/t.py | Key methods should set a key", "PASSED"))
    assert sbp_resolution(out, PROBLEM)["resolved"] is False


def test_missing_names_are_bounded():
    p = dict(PROBLEM, fail_to_pass=[f"t{i}" for i in range(40)], pass_to_pass=[])
    r = sbp_resolution({"tests": []}, p)
    assert r["missing"] == 40 and len(r["missing_names"]) == 10


# ==============================================================================
# --- THE ATTEMPT SCRIPT ---
# ==============================================================================

def test_test_script_deletes_the_previous_verdict_before_running():
    """Without this, a crashed parser grades the attempt against the last rung."""
    script = sbp_test_script(["a.py", "b.py"])
    body = script.splitlines()
    rm = next(i for i, l in enumerate(body) if "rm -f /workspace/output.json" in l)
    run = next(i for i, l in enumerate(body) if "run_script.sh" in l)
    assert rm < run


def test_test_script_uses_upstreams_script_and_parser_with_comma_joined_files():
    script = sbp_test_script(["a.py", "b.py"], workdir="/srv/app",
                             env_exports="export FOO=1")
    assert "bash /workspace/run_script.sh 'a.py,b.py'" in script
    assert "/workspace/parser.py /workspace/stdout.log /workspace/stderr.log" in script
    assert "cd /srv/app" in script
    assert "export FOO=1" in script


def test_test_script_echoes_the_logs_so_the_harness_captures_what_the_parser_read():
    script = sbp_test_script(["a.py"])
    assert "cat /workspace/stdout.log" in script
    assert "cat /workspace/stderr.log >&2" in script


# ==============================================================================
# --- THE EXECUTOR, WITHOUT DOCKER ---
# ==============================================================================

class FakeRun:
    """Stands in for a straitjacket capture."""

    digest = "DIGEST: 1 failing test"
    exit_code = 1

    def native_payload(self):
        return "FAILED tests/t.py::test_a"

    def metrics(self):
        return {"raw_tokens_est": 100, "digest_tokens_est": 10}


@pytest.fixture
def env(monkeypatch):
    """A SWEBenchProEnv with every container call recorded instead of run."""
    e = SWEBenchProEnv(PROBLEM)
    e.started = True
    e.sandbox = "/tmp/sandbox"
    e.scripts = {"run_script": "run_all_tests", "parser": "#", "env_exports": ""}
    e.calls = []
    e.output = _output(("tests/t.py | Key methods should return null", "PASSED"),
                       ("tests/t.py | Key methods should set a key", "PASSED"))
    e.apply_ok = True

    def fake_sh(script, timeout=None, check=True):
        e.calls.append(("sh", script))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_write(path, text):
        e.calls.append(("write", path))

    def fake_apply(path):
        e.calls.append(("apply", path))
        return e.apply_ok

    monkeypatch.setattr(e, "_sh", fake_sh)
    monkeypatch.setattr(e, "_write", fake_write)
    monkeypatch.setattr(e, "_try_apply", fake_apply)
    monkeypatch.setattr(e, "_read_output", lambda: e.output)
    monkeypatch.setattr("src.evaluator.sj.contained_run",
                        lambda *a, **k: e.calls.append(("run", a[0])) or FakeRun())
    return e


def test_score_resolves_when_every_required_test_passes(env):
    passed, evidence = env.score(extract_patch(GOOD_PATCH))
    assert passed is True
    assert env.last_ratio == 1.0 and env.last_report["required"] == 2
    assert evidence.digest == "DIGEST: 1 failing test"


def test_graded_tests_are_restored_after_the_patch_is_applied(env):
    """The anti-cheat. Restoring first would let a patch edit the tests."""
    env.score(extract_patch(GOOD_PATCH))
    kinds = [c for c in env.calls]
    apply_at = next(i for i, (k, v) in enumerate(kinds) if k == "apply")
    restore_at = next(i for i, (k, v) in enumerate(kinds)
                      if k == "sh" and v.startswith("git checkout abc123 --"))
    run_at = next(i for i, (k, v) in enumerate(kinds) if k == "run")
    assert apply_at < restore_at < run_at


def test_a_patch_that_does_not_apply_never_runs_the_tests(env):
    env.apply_ok = False
    passed, evidence = env.score(extract_patch(GOOD_PATCH))
    assert passed is False
    assert "did not apply" in str(evidence)
    assert not [c for c in env.calls if c[0] == "run"]


def test_a_failed_test_restore_is_an_error_not_a_silent_grade(monkeypatch, env):
    """Grading against the repository's original tests would measure nothing."""
    def bad_sh(script, timeout=None, check=True):
        env.calls.append(("sh", script))
        rc = 1 if script.startswith("git checkout abc123 --") else 0
        return types.SimpleNamespace(returncode=rc, stdout="", stderr="pathspec")
    monkeypatch.setattr(env, "_sh", bad_sh)
    passed, evidence = env.score(extract_patch(GOOD_PATCH))
    assert passed is False
    assert "restore the graded test files" in str(evidence)
    assert not [c for c in env.calls if c[0] == "run"]


def test_prose_is_rejected_before_the_container_is_touched(env):
    passed, evidence = env.score("I would change the widget module.")
    assert passed is False and env.calls == []
    assert missing_patch_error("I would change it") is not None


def test_a_container_that_never_started_says_why(monkeypatch):
    e = SWEBenchProEnv(PROBLEM)
    e.started, e.setup_error = False, "docker run failed: manifest unknown"
    passed, evidence = e.score(extract_patch(GOOD_PATCH))
    assert passed is False
    assert "manifest unknown" in str(evidence)


def test_env_reads_the_image_and_workdir_from_the_row():
    e = SWEBenchProEnv(PROBLEM)
    assert e.image == "jefzda/sweap-images:demo.repo-demo__repo-abc123"
    assert e.workdir == "/app"
    assert e.name.startswith("tokenomics-sbp-")


# ==============================================================================
# --- THE ARMS ---
# ==============================================================================

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
        return types.SimpleNamespace(evidence_graph=lambda: graph, profile="pytest/v1")


class FakeEnv:
    """Records every scoring call; never resolves, so the ladder runs to its cap."""

    instances = []

    def __init__(self, problem, timeout=None, scripts=None):
        self.problem = problem
        self.started = True
        self.setup_error = ""
        self.calls = []
        self.last_ratio = 0.25
        self.last_report = {"resolved": False, "required": 4, "missing": 3}
        FakeEnv.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def score(self, patch):
        self.calls.append(patch)
        return False, FakeEvidence("FAILED tests/t.py::test_a", typed=True)


@pytest.fixture
def wired(monkeypatch):
    """Replace the container and the API with recorders."""
    FakeEnv.instances = []
    calls = []

    def fake_dispatch(model_id, prompt, max_tokens=2560, thinking_level=None,
                      problem=None):
        calls.append({"model": model_id, "thinking": thinking_level, "prompt": prompt})
        return GOOD_PATCH, {"as_run_usd": 0.01, "output": 10, "total_tokens": 20,
                            "input_raw": 10, "prompt_tokens": 10}, 0.1

    monkeypatch.setattr(sbp, "SWEBenchProEnv", FakeEnv)
    monkeypatch.setattr(sbp, "dispatch_model", fake_dispatch)
    monkeypatch.setattr(sbp, "_treat_error",
                        lambda err, t, problem=None: ("DIGEST",
                                                      {"as_run_usd": 0.0, "output": 0,
                                                       "total_tokens": 0}, 0.0))
    # The arms are `sj_required`; the harness is not installed in CI.
    monkeypatch.setattr("src.architectures.sj.require", lambda: None)
    return calls


@pytest.mark.parametrize("arm", [
    lambda p: sbp.run_sbp_single(p),
    sbp.run_sbp_cascade,
    sbp.run_sbp_evidence_gate,
    sbp.run_sbp_plan_exec,
])
def test_every_arm_spends_exactly_three_oracle_calls(arm, wired):
    """The budget H2's comparison rests on: container runs, not LLM calls."""
    out = arm(dict(PROBLEM))
    assert len(FakeEnv.instances) == 1
    assert len(FakeEnv.instances[0].calls) == sbp.MAX_ORACLE_CALLS == 3
    assert out["repair_loops"] == 2 and out["passed"] is False


def test_cascade_climbs_one_rung_per_failure(wired):
    sbp.run_sbp_cascade(dict(PROBLEM))
    assert [c["model"] for c in wired] == [sbp.TIERS[0][0], sbp.TIERS[1][0],
                                           sbp.FRONTIER]


def test_evidence_gate_jumps_to_the_frontier_on_a_hard_digest(wired):
    out = sbp.run_sbp_evidence_gate(dict(PROBLEM))
    models = [c["model"] for c in wired]
    assert models[0] == sbp.TIERS[0][0]
    assert sbp.FRONTIER in models
    assert out["routing"]["frontier_used"] is True


def test_evidence_gate_never_escalates_twice(wired):
    """One frontier rung is the ceiling; a second would double the arm's price."""
    sbp.run_sbp_evidence_gate(dict(PROBLEM))
    assert [c["model"] for c in wired].count(sbp.FRONTIER) <= 2


def test_plan_exec_buys_the_frontier_model_before_the_first_oracle_call(wired):
    sbp.run_sbp_plan_exec(dict(PROBLEM))
    assert wired[0]["model"] == sbp.FRONTIER
    assert "IMPLEMENTATION PLAN" in wired[0]["prompt"]
    # ...and never again: the executor rungs are all the cheap model.
    assert [c["model"] for c in wired[1:]] == [sbp.TIERS[0][0]] * 3
    assert "Architect's implementation plan" in wired[1]["prompt"]


def test_the_partial_credit_metric_survives_into_the_result(wired):
    out = sbp.run_sbp_cascade(dict(PROBLEM))
    assert out["test_pass_ratio"] == 0.25
    assert out["sbp"]["required"] == 4


def test_the_prompt_carries_upstreams_three_blocks(wired):
    """Resolve rates for this dataset were measured with all three present."""
    sbp.run_sbp_cascade(dict(PROBLEM))
    prompt = wired[0]["prompt"]
    assert PROBLEM["problem_statement"] in prompt
    assert PROBLEM["requirements"] in prompt
    assert PROBLEM["interface"] in prompt
    assert "tests/t.py, tests/u.py" in prompt
    assert "Do NOT modify test files" in prompt


def test_the_repair_prompt_carries_the_digest_and_the_failing_patch(wired):
    sbp.run_sbp_cascade(dict(PROBLEM))
    repair = wired[1]["prompt"]
    assert "DIGEST" in repair and "diff --git" in repair


def test_oversized_support_blocks_are_clipped_but_the_statement_is_not(wired):
    huge = "x" * (sbp.MAX_SUPPORT_CHARS + 5000)
    statement = "y" * (sbp.MAX_SUPPORT_CHARS + 5000)
    sbp.run_sbp_cascade(dict(PROBLEM, requirements=huge, interface=huge,
                             problem_statement=statement))
    prompt = wired[0]["prompt"]
    assert statement in prompt
    assert huge not in prompt and "chars omitted" in prompt


def test_every_variant_is_wired_and_uniquely_named():
    for key, cfg in sbp.SWEBENCH_PRO_VARIANTS.items():
        assert cfg["id"] == key
        assert callable(cfg["fn"])
        assert set(cfg) >= {"id", "name", "category", "models", "triage_mode", "fn"}
    names = [c["name"] for c in sbp.SWEBENCH_PRO_VARIANTS.values()]
    assert len(set(names)) == len(names)


def test_the_group_selector_finds_the_arms_and_leaves_the_opus_baseline_out():
    from src.architectures import get_configurations
    ids = {c["id"] for c in get_configurations(dataset="swebench_pro", group="sbp")}
    assert "sbp_cascade" in ids and "sbp_evidence_gate" in ids
    # The frontier single is priced far above the rest and is opt-in.
    assert "sbp_single_opus" not in ids
    picked = get_configurations(dataset="swebench_pro", variant_keys=["sbp_single_opus"])
    assert [c["id"] for c in picked] == ["sbp_single_opus"]


# ==============================================================================
# --- END TO END (opt-in: needs Docker and a multi-GB image pull) ---
# ==============================================================================

_docker_ok, _docker_why = docker_available()


@pytest.mark.skipif(os.environ.get("SBP_INTEGRATION") != "1",
                    reason="set SBP_INTEGRATION=1 to run the real container check")
@pytest.mark.skipif(not _docker_ok, reason=f"docker unavailable: {_docker_why}")
def test_gold_patch_resolves_a_real_instance():
    """The only check that this harness agrees with the benchmark it implements.

    Pick the row with SBP_INSTANCE, or let it take the first Python row of the
    split. A failure here means the harness is wrong, not the model.
    """
    problems = ds.load_swebench_pro_problems(
        max_tasks=None, apply_quarantine=False,
        languages=[os.environ.get("SBP_LANGUAGE", "python")])
    iid = os.environ.get("SBP_INSTANCE") or next(iter(problems))
    problem = problems[iid]
    with SWEBenchProEnv(problem) as e:
        passed, evidence = e.score(problem["patch"])
        assert e.started, e.setup_error
        assert passed, (f"gold did not resolve {iid}: {e.last_report}\n"
                        f"{str(evidence)[:2000]}")


# ==============================================================================
# --- THE ATTEMPT SCRIPT, ACTUALLY EXECUTED ---
# ==============================================================================
#
# The script is the one part of this harness that is shell rather than Python,
# and the one part a unit test would normally only assert *strings* about. It
# is short enough to run for real against a stub run script and a stub parser,
# which is how the stale-verdict guard below is checked rather than asserted.

def _stub_workspace(tmp_path, stdout="ok", stderr="warn", rc=0, parser="write"):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "run_script.sh").write_text(
        f'#!/bin/bash\necho "files=$1"\necho "{stdout}"\necho "{stderr}" >&2\n'
        f"exit {rc}\n", encoding="utf-8")
    if parser == "write":
        body = ("import json,sys\n"
                "open(sys.argv[3],'w').write(json.dumps("
                "{'tests':[{'name':'t','status':'PASSED'}]}))\n")
    else:                                   # a parser that dies on this output
        body = "import sys\nsys.exit(3)\n"
    (ws / "parser.py").write_text(body, encoding="utf-8")
    return ws


def _run(script, cwd):
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          cwd=str(cwd), timeout=60)


def test_the_generated_script_parses_the_run_and_forwards_its_exit_code(tmp_path):
    ws = _stub_workspace(tmp_path, rc=1)
    script = sbp_test_script(["a.py", "b.py"], workdir=str(tmp_path),
                             workspace=str(ws))
    p = _run(script, tmp_path)
    assert p.returncode == 1                       # the suite's own verdict
    assert "files=a.py,b.py" in p.stdout           # comma-joined, as upstream
    assert "warn" in p.stderr and "ok" in p.stdout  # logs echoed for capture
    out = json.loads((ws / "output.json").read_text(encoding="utf-8"))
    assert out["tests"][0]["status"] == "PASSED"


def test_a_crashed_parser_cannot_be_graded_against_the_previous_attempt(tmp_path):
    """The repair loop reuses one container, so a stale output.json is the
    previous *rung's* verdict. Deleting it first is what makes that impossible."""
    ws = _stub_workspace(tmp_path, parser="crash")
    (ws / "output.json").write_text(json.dumps(
        _output(("tests/t.py | Key methods should return null", "PASSED"),
                ("tests/t.py | Key methods should set a key", "PASSED"))),
        encoding="utf-8")
    script = sbp_test_script(["a.py"], workdir=str(tmp_path), workspace=str(ws))
    p = _run(script, tmp_path)
    assert "PARSER FAILED" in p.stderr
    assert not (ws / "output.json").exists()
    # Which is what the executor would then read, and it does not resolve.
    assert sbp_resolution({}, PROBLEM)["resolved"] is False


def test_the_script_fails_loudly_when_the_repository_is_not_where_it_should_be(tmp_path):
    ws = _stub_workspace(tmp_path)
    p = _run(sbp_test_script(["a.py"], workdir=str(tmp_path / "nope"),
                             workspace=str(ws)), tmp_path)
    assert p.returncode == 90


# ==============================================================================
# --- UPSTREAM'S PARSER AGREES WITH UPSTREAM'S TEST NAMES ---
# ==============================================================================

_NODEBB = "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan"
_NODEBB_SCRIPTS = ds.swebench_pro_scripts_dir(_NODEBB)


@pytest.mark.skipif(not os.path.exists(os.path.join(_NODEBB_SCRIPTS, "parser.py")),
                    reason="run scripts not cached; fetch with "
                           "`python3 tools/swebench_pro_preflight.py --tasks "
                           f"{_NODEBB} --gold 1`")
def test_the_parsers_test_names_match_the_datasets_required_names():
    """The assumption everything else rests on, checked rather than assumed.

    `resolved` is a set comparison between names upstream's per-instance parser
    invents from the test output and names the dataset stored in
    `fail_to_pass`. If those two naming conventions disagree by so much as a
    prefix, every row grades as unresolved and the sweep reports a uniform zero
    that looks like a hard dataset.

    The input below is what mocha's JSON reporter emits *after* the run script's
    sed prefixes every `describe` title with `<file>::` -- which is why the
    stored names carry a `::` in the middle and the parser's greedy `(\\S+)::`
    lands on the outer file, not the inner one.
    """
    mocha = {
        "stats": {"suites": 1, "tests": 2, "passes": 2, "pending": 0, "failures": 0},
        "tests": [], "pending": [], "failures": [],
        "passes": [
            {"title": "should return multiple keys and null if key doesn't exist",
             "fullTitle": "test/database.js::Test database "
                          "test/database/keys.js::Key methods should return "
                          "multiple keys and null if key doesn't exist",
             "file": "/app/test/database.js"},
        ],
    }
    with open(os.path.join(_NODEBB_SCRIPTS, "parser.py"), encoding="utf-8") as f:
        parser_src = f.read()

    with tempfile.TemporaryDirectory() as d:
        for name, body in (("stdout.log", json.dumps(mocha, indent=2)),
                           ("stderr.log", ""), ("parser.py", parser_src)):
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(body)
        out_path = os.path.join(d, "output.json")
        r = subprocess.run([sys.executable, os.path.join(d, "parser.py"),
                    os.path.join(d, "stdout.log"), os.path.join(d, "stderr.log"),
                    out_path], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        with open(out_path, encoding="utf-8") as f:
            output = json.load(f)

    produced = {t["name"] for t in output["tests"]}
    expected = ("test/database.js | Test database test/database/keys.js::Key "
                "methods should return multiple keys and null if key doesn't exist")
    assert produced == {expected}

    # And that name is graded as a pass by this repository's resolution rule.
    row = {"fail_to_pass": [expected], "pass_to_pass": []}
    assert sbp_resolution(output, row)["resolved"] is True
