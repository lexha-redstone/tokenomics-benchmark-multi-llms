# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Contract tests for the straitjacket integration.

These pin the properties that make a "Straitjacket ($0.00)" benchmark row
mean something:

  * the digest comes from the upstream ``ctx`` profile registry, not from a
    local re-implementation;
  * capture happens at the birth gate, so the full output is stored and the
    contained payload is orders of magnitude smaller than the raw one;
  * what the digest omits stays retrievable at an exact address;
  * a straitjacket arm refuses to run when the harness is missing, instead of
    silently degrading to something that only looks like a digest.

Run with:  pytest tests/test_straitjacket_integration.py -q
"""

import os
import pathlib
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import straitjacket as sj  # noqa: E402

pytestmark = pytest.mark.skipif(
    not sj.available(),
    reason="ctx-harness not installed (pip install ctx-harness)",
)

NOISY_FAILING_PROGRAM = (
    "def task_func(x):\n"
    "    return x + 5\n"
    "\n"
    "import unittest\n"
    "class TestCases(unittest.TestCase):\n"
    "    def test_quiet_needle(self):\n"
    "        for i in range(3000):\n"
    "            print('routine progress line %d' % i)\n"
    "        self.assertEqual(task_func(1), 2)\n"
    "\n"
    "import unittest as _ut, sys as _sys\n"
    "_res = _ut.TextTestRunner(verbosity=0).run("
    "_ut.TestLoader().loadTestsFromTestCase(TestCases))\n"
    "_sys.exit(0 if _res.wasSuccessful() else 1)\n"
)


@pytest.fixture(scope="module")
def noisy_run():
    d = sj.new_sandbox("test")
    try:
        (d / "prog.py").write_text(NOISY_FAILING_PROGRAM, encoding="utf-8")
        yield sj.contained_run([sys.executable, "prog.py"], cwd=d,
                               record_argv=["python3", "prog.py"])
    finally:
        sj.drop_sandbox(d)


# ---------------------------------------------------------------- provenance

def test_digest_comes_from_upstream_profile_registry(noisy_run):
    """The header names an upstream profile version, not a local invention.

    The CLI may print a reflex notice above the header, so it is located
    rather than assumed to be line 1.
    """
    assert f"[ctx {noisy_run.handle} profile=unittest/v1]" in noisy_run.digest
    assert noisy_run.profile == "unittest/v1"


def test_status_names_the_harness_that_ran():
    st = sj.status()
    assert st["available"] is True
    assert st["backend"] in ("library", "cli")
    assert st["ctx_version"]


# ---------------------------------------------------------------- containment

def test_full_output_is_captured_not_truncated(noisy_run):
    """Birth-gate capture: the whole stream is stored, all of it parsed."""
    assert noisy_run.stdout_lines == 3000
    total = noisy_run.stdout_lines + noisy_run.stderr_lines
    m = re.search(r"parsed: ([\d,]+)/([\d,]+) lines", noisy_run.digest)
    assert m, noisy_run.digest
    parsed, of = (int(g.replace(",", "")) for g in m.groups())
    assert parsed == of == total


def test_digest_is_orders_of_magnitude_smaller_than_raw(noisy_run):
    m = noisy_run.metrics()
    assert m["raw_tokens_est"] > 5000
    assert m["digest_tokens_est"] < 400
    assert m["containment_ratio"] > 10
    assert m["tokens_kept_out"] == m["raw_tokens_est"] - m["digest_tokens_est"]


def test_digest_carries_a_coverage_receipt_and_addresses(noisy_run):
    """Omission without amnesia: the digest says what it dropped and how to
    get it back. This is exactly what a keyword filter cannot provide."""
    assert "coverage:" in noisy_run.digest
    assert "omitted:" in noisy_run.digest
    assert "next:" in noisy_run.digest
    assert noisy_run.handle in noisy_run.digest


def test_digest_keeps_the_failing_identity_and_the_innermost_frame(noisy_run):
    assert "test_quiet_needle" in noisy_run.digest
    assert "innermost frame" in noisy_run.digest
    assert "AssertionError: 6 != 2" in noisy_run.digest


@pytest.mark.skipif(sj.status()["backend"] != "library",
                    reason="needs the in-process manifest (library backend)")
def test_digest_is_deterministic(noisy_run):
    """Same bytes, same manifest → same digest. A benchmark cannot compare
    arms whose evidence rendering wobbles between runs."""
    from ctx.digest import render_run_digest
    again, _ = render_run_digest(sj._store, sj._ws, dict(noisy_run._manifest))
    assert again == noisy_run.digest


def test_innermost_frame_keeps_its_line_number(noisy_run):
    """The frame row exists to name the file AND line a fix lands on.

    Upstream clips that row at 160 chars. A long sandbox path eats the budget
    and silently truncates `", line 42, in test_x"` off the end, leaving a
    frame that points at a file and nothing else. Regression: the in-repo
    default sandbox (83-char checkout path + `.straitjacket/workspace/
    sandbox/<uuid>/`) did exactly that.
    """
    frame = [ln for ln in noisy_run.digest.splitlines() if "innermost frame" in ln]
    assert frame, noisy_run.digest
    assert re.search(r', line \d+, in test_quiet_needle', frame[0]), frame[0]


def test_sandbox_path_fits_the_frame_budget():
    fb = sj.frame_budget()
    assert fb["frame_fits"], fb


# A failure heavy enough that the runner's elapsed time actually varies. The
# first version of this test used a single trivial assertion, which always
# rounded to `Ran 1 test in 0.001s` — so it passed while the bug it was
# supposed to catch was live. A determinism test that only holds for fast
# fixtures is worse than no test.
_SLOW_FAILING_PROGRAM = (
    "def task_func(n):\n    return [str(i) for i in range(n)]\n\n"
    "import unittest\n"
    "class TestCases(unittest.TestCase):\n"
    + "".join(
        f"    def test_{i:02d}(self):\n"
        f"        self.assertEqual(task_func(60), list(range(60)))\n"
        for i in range(12)
    )
    + sj.DETERMINISTIC_UNITTEST_TAIL
)


def _capture(program, prefix="repro"):
    d = sj.new_sandbox(prefix)
    try:
        (d / "prog.py").write_text(program, encoding="utf-8")
        return sj.contained_run([sys.executable, "prog.py"], cwd=d,
                                record_argv=["python3", "prog.py"],
                                env_extra=sj.CAPTURE_ENV)
    finally:
        sj.drop_sandbox(d)


def test_same_failure_renders_the_same_digest():
    """Byte-identical digests — header included — for the same failure.

    Two separate causes have broken this. A unique sandbox name per attempt
    leaked into the manifest cwd and into every traceback frame. And the
    runner's own `Ran 12 tests in 0.114s` line varied between processes, which
    left the digest *body* identical but minted a new artifact and a new run
    handle each time.
    """
    runs = [_capture(_SLOW_FAILING_PROGRAM) for _ in range(3)]
    assert runs[0].digest == runs[1].digest == runs[2].digest
    assert runs[0].handle == runs[1].handle == runs[2].handle


def test_runner_elapsed_time_is_pinned():
    """The specific noise source, asserted directly, so the regression is
    named rather than inferred from a hash mismatch."""
    run = _capture(_SLOW_FAILING_PROGRAM)
    timing = [ln for ln in run.raw_stderr.splitlines() if ln.startswith("Ran ")]
    assert timing == ["Ran 12 tests in 0.000s"]
    # ...and the upstream profile still recognises the shape it detects on.
    assert run.profile == "unittest/v1"


def test_capture_environment_is_pinned():
    """Hash randomisation reorders any set a candidate prints, and a GUI
    matplotlib backend behaves differently per host. Both would show up as
    run-to-run byte churn in the artifact."""
    assert sj.CAPTURE_ENV["PYTHONHASHSEED"] == "0"
    assert sj.CAPTURE_ENV["MPLBACKEND"] == "Agg"


def test_harness_home_is_outside_the_repository():
    """The sandbox path is model-visible evidence, not private bookkeeping —
    it must not inherit the checkout's arbitrary depth."""
    repo = pathlib.Path(__file__).resolve().parent.parent
    assert not sj.workspace_root().resolve().is_relative_to(repo)


# ------------------------------------------------- the uncontained baseline

def test_native_baseline_reads_the_failing_stream(noisy_run):
    """The untreated arm must get the baseline this benchmark always defined.

    Regression: concatenating stdout+stderr and tail-truncating the pair spent
    the native arm's whole budget on stdout chatter it never used to forward.
    That inflates the baseline's token cost and degrades the evidence it
    repairs from — biasing every comparison toward straitjacket.
    """
    assert noisy_run.stdout_lines == 3000          # the chatter is on stdout
    assert noisy_run.native_stream == "stderr"     # the failure is on stderr

    payload = noisy_run.native_payload()
    assert payload.startswith("=" * 20)
    assert "FAIL: test_quiet_needle" in payload
    assert "routine progress line" not in payload  # no stdout chatter


@pytest.mark.skipif(sj.status()["backend"] != "library",
                    reason="contain_text is library-backend only")
def test_native_baseline_falls_back_to_stdout():
    """Runners that report on stdout (the SWE-bench pytest shape) still get a
    baseline rather than an empty string."""
    run = sj.contain_text(
        "============================= test session starts ====================\n"
        "FAILED tests/test_x.py::test_y - AssertionError: boom\n"
        "=========================== 1 failed, 3 passed =======================\n",
        argv=["pytest", "-q"], exit_code=1, stream="stdout")
    assert run.native_stream == "stdout"
    assert "FAILED tests/test_x.py::test_y" in run.native_payload()


def test_metrics_do_not_conflate_containment_with_the_ab_delta(noisy_run):
    """`raw - digest` (captured but never resident) is a bigger number than
    `native - digest` (what the treatment actually bought). Reporting the
    first as the second overstates the mechanism."""
    m = noisy_run.metrics()
    assert m["tokens_kept_out"] == m["raw_tokens_est"] - m["digest_tokens_est"]
    assert m["delta_vs_native_tokens"] == (
        m["native_sent_tokens_est"] - m["digest_tokens_est"])
    # This fixture floods stdout, which the baseline never forwarded, so the
    # honest A/B is far smaller than the containment figure.
    assert m["tokens_kept_out"] > m["delta_vs_native_tokens"]


def test_native_arm_measures_zero_advantage_over_itself():
    """The self-consistency check for the whole receipt.

    An arm whose treatment IS the baseline must show a delta of exactly zero.
    Two separate bugs each broke this: a second tail-truncation inside the arm
    (measured baseline 4,000 chars vs 2,500 actually sent), and summing the
    baseline over captures while summing the payload over treatments (a repair
    loop treats N failures but captures N+1 runs).
    """
    import src.architectures as arch
    from src.evaluator import begin_containment, containment_report

    problem = {
        "entry_point": "task_func",
        "test": ("import unittest\n"
                 "class TestCases(unittest.TestCase):\n"
                 "    def test_a(self):\n"
                 "        for i in range(400):\n"
                 "            print('noise %d' % i)\n"
                 "        self.assertEqual(task_func(1), 2)\n"),
    }
    from src.evaluator import run_bigcodebench
    begin_containment()
    _, err = run_bigcodebench(problem, "def task_func(x):\n    return x + 9\n")

    arch._treat_error(err, "native", problem=problem)
    c = containment_report()
    assert c["treatment_events"] == 1
    assert c["delta_vs_native_tokens"] == 0
    assert c["evidence_sent_tokens_est"] == c["native_baseline_tokens_est"]

    # ...and the contained treatment on the same failure does beat it, or is
    # honestly reported as not beating it. Either way both sides are counted
    # over the same event.
    begin_containment()
    arch._treat_error(err, "straitjacket", problem=problem)
    c = containment_report()
    assert c["treatment_events"] == 1
    assert c["delta_vs_native_tokens"] == (
        c["native_baseline_tokens_est"] - c["evidence_sent_tokens_est"])


def test_one_native_truncation_knob():
    """`SJ_RAW_CAP` is the only place the uncontained payload is clipped."""
    import src.architectures as arch

    src_text = (pathlib.Path(__file__).resolve().parent.parent
                / "src" / "architectures.py").read_text()
    assert "native_tail" not in src_text
    payload, _, _ = arch._treat_error("x" * 10_000, "native")
    assert len(payload) == sj.raw_cap()


def test_per_task_path_never_materialises_the_stream():
    """The adapter must not undo the disk spooling it depends on.

    ``run_capture`` streams the child's output to disk so a flood never sits
    in memory. The first version of this bridge then read both streams back in
    full on every capture: a 40.9 MB stdout pushed the harness process to a
    150.8 MB peak heap in order to produce a 663-char digest and a 506-char
    native payload. Everything on the per-task path is now a bounded tail read.
    """
    import tracemalloc

    big = (
        "def task_func(x):\n    return x + 5\n\n"
        "import unittest\n"
        "class TestCases(unittest.TestCase):\n"
        "    def test_a(self):\n"
        "        for i in range(120000): print('progress record %d' % i)\n"
        "        self.assertEqual(task_func(1), 2)\n"
        + sj.DETERMINISTIC_UNITTEST_TAIL
    )
    run = _capture(big, prefix="mem")
    assert run.stdout_bytes > 2_000_000        # a real flood was captured

    tracemalloc.start()
    try:
        payload = run.native_payload()
        run.metrics()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert len(payload) <= sj.raw_cap()
    assert peak < 1_000_000, f"per-task path allocated {peak:,} bytes"


def test_bounded_tail_read_returns_the_end_of_the_stream():
    run = _capture(_SLOW_FAILING_PROGRAM)
    assert run.raw_tail("stderr", 200) == run.raw_stderr[-200:]
    assert run.native_payload().endswith("FAILED (failures=12)")


# ------------------------------------------------------------------ retrieval

def test_bounded_retrieval_returns_the_exact_region(noisy_run):
    out = noisy_run.get("stderr", (1, 8))
    assert "selector: --lines 1:8" in out
    assert "FAIL: test_quiet_needle" in out


def test_search_finds_evidence_left_out_of_the_digest(noisy_run):
    out = noisy_run.search(["routine progress line 1500"], context=0)
    assert "L1501" in out or "routine progress line 1500" in out


def test_retrieval_cannot_become_the_second_flood(noisy_run):
    """A request for the whole 3,000-line stream is clamped."""
    from src.architectures import RETRIEVAL_MAX_LINES, serve_retrieval
    from src.evaluator import Evidence

    ev = Evidence("", run=noisy_run, digest=noisy_run.digest, contained=True)
    served, note = serve_retrieval(ev, "ctx get run:deadbeef#stdout --lines 1:3000")
    assert served is not None
    assert note.endswith(f"--lines 1:{RETRIEVAL_MAX_LINES}")


def test_retrieval_is_served_against_this_runs_artifact_only(noisy_run):
    """A handle typed by the model is ignored; the served command names the
    real handle, so a hallucinated id cannot address another run."""
    from src.architectures import serve_retrieval
    from src.evaluator import Evidence

    ev = Evidence("", run=noisy_run, digest=noisy_run.digest, contained=True)
    _, note = serve_retrieval(ev, "ctx get run:0000000000ff#stderr --lines 1:5")
    assert noisy_run.handle in note
    assert "0000000000ff" not in note


# -------------------------------------------------------------------- evaluator

def test_evaluator_returns_contained_evidence():
    from src.evaluator import Evidence, run_bigcodebench, triage_error_straitjacket

    problem = {
        "entry_point": "task_func",
        "test": ("import unittest\n"
                 "class TestCases(unittest.TestCase):\n"
                 "    def test_a(self):\n"
                 "        for i in range(1200):\n"
                 "            print('noise %d' % i)\n"
                 "        self.assertEqual(task_func(1), 2)\n"),
    }
    passed, err = run_bigcodebench(problem, "def task_func(x):\n    return x + 9\n")
    assert passed is False
    assert isinstance(err, Evidence)
    assert err.contained is True
    assert err.run is not None

    digest, usage, seconds = triage_error_straitjacket(err)
    assert digest == err.digest
    assert usage["as_run_usd"] == 0.0
    assert seconds == 0.0
    # Bounded against what was captured. Whether it also beats the untreated
    # baseline depends on which stream carried the flood — this fixture floods
    # stdout, which the baseline never forwarded. See
    # test_metrics_do_not_conflate_containment_with_the_ab_delta.
    assert len(digest.encode("utf-8")) < err.run.raw_bytes


def test_no_keyword_filter_remains_in_the_evaluator():
    """The specific anti-pattern this integration replaced: selecting lines
    because they contain a marker word. If it comes back, this fails."""
    src = pathlib.Path(__file__).resolve().parent.parent / "src" / "evaluator.py"
    text = src.read_text()
    body = text.split('"""', 2)[-1]  # skip the module docstring, which names it
    assert 'any(k in line for k in' not in body
    assert 'any(k in l for k in' not in body


# ------------------------------------------------- every registered arm

_FAILING_PROBLEM = {
    "task_id": "instrumentation/1",
    "entry_point": "task_func",
    "complete_prompt": "def task_func(x):\n    return x + 1\n",
    "test": ("import unittest\n"
             "class TestCases(unittest.TestCase):\n"
             "    def test_a(self):\n"
             "        for i in range(300):\n"
             "            print('noise %d' % i)\n"
             "        self.assertEqual(task_func(1), 2)\n"),
}


@pytest.fixture
def stub_models(monkeypatch):
    """Every model call returns a solution that keeps failing, so each arm is
    forced all the way through its repair path."""
    import src.architectures as arch
    import src.client as client
    import src.evaluator as ev

    def fake(model_id, prompt, max_tokens=2048, thinking_level=None, problem=None):
        return ("```python\ndef task_func(x):\n    return x + 99\n```",
                {"as_run_usd": 0.001, "input": 10, "output": 10, "total_tokens": 20}, 0.1)

    for mod in (client, ev, arch):
        monkeypatch.setattr(mod, "dispatch_model", fake, raising=False)
    return fake


def _registered_arms():
    from src.architectures import VARIANT_REGISTRY
    return sorted(VARIANT_REGISTRY.items())


@pytest.mark.parametrize("variant_id", [k for k, _ in _registered_arms()])
def test_every_arm_records_the_treatment_it_applied(variant_id, stub_models):
    """The receipt must not be blank for an arm that actually repaired.

    This is the test that was missing. Six ``*_straitjacket`` arms called
    ``triage_error_straitjacket`` directly rather than through
    ``_treat_error``, so ``record_evidence_sent`` never fired for them. The
    containment mechanism worked — real ``unittest/v1`` digests, real stored
    artifacts — but an N=100 sweep published "Treatment: n/a, Sent: 0, delta:
    +0" for every headline straitjacket row. Covering only the arms already
    converted is what let that through, so this parametrises over the whole
    registry.
    """
    from src.architectures import VARIANT_REGISTRY

    result = VARIANT_REGISTRY[variant_id]["fn"](dict(_FAILING_PROBLEM))
    c = result["containment"]

    assert result["repair_loops"] > 0, "arm never repaired; fixture is not exercising it"
    assert c["captures"] > 0
    assert result.get("containment_instrumentation") != "MISSING"
    assert c["treatment_events"] > 0, f"{variant_id} recorded no evidence treatment"
    assert c["treatments"], f"{variant_id} reported no treatment name"
    assert c["evidence_sent_tokens_est"] > 0
    assert c["native_baseline_tokens_est"] > 0


@pytest.mark.parametrize(
    "variant_id",
    [k for k, v in _registered_arms()
     if "straitjacket" in v["triage_mode"].lower()])
def test_straitjacket_arms_report_the_straitjacket_treatment(variant_id, stub_models):
    """A row labelled with containment must have applied containment."""
    from src.architectures import VARIANT_REGISTRY

    c = VARIANT_REGISTRY[variant_id]["fn"](dict(_FAILING_PROBLEM))["containment"]
    assert c["treatments"] == ["straitjacket"], (variant_id, c["treatments"])
    assert "unittest/v1" in c["profiles"]


# ------------------------------------------------------------------- refusal

def test_straitjacket_arm_refuses_without_the_harness(monkeypatch):
    """Better no row than a row that credits a mechanism which never ran."""
    monkeypatch.setattr(sj, "_state", {"resolved": True, "backend": "off",
                                       "version": None, "reason": "test"})
    with pytest.raises(sj.SJUnavailable):
        sj.require()
