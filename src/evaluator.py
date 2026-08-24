# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Unified Evaluator and Straitjacket Context-Containment Bridge.

Handles:
  1. Python function code extraction and sandboxed unit test execution (BigCodeBench & WebDev).
  2. Containerised repository evaluation for FeatureBench: apply a candidate
     diff inside the task's own Docker image and run its pytest suite.
  3. Comparison between LLM-based triage ($) and Straitjacket local containment ($0.00).

Containment contract
--------------------
Candidate solutions are executed **through** the straitjacket harness
(``src/straitjacket.py`` → ``ctx.execution.run_capture``), so test output is
captured at the birth gate: it is stored whole and never returned to the
caller as an unbounded blob. Every evaluation therefore yields an
:class:`Evidence` value that carries three distinct things:

* ``str(evidence)``  — the *uncontained* payload an ordinary (native) arm
  sends to the model, after the native path's own tail truncation;
* ``evidence.digest`` — the *contained* payload a straitjacket arm sends: the
  real upstream digest, with its coverage receipt and retrieval addresses;
* ``evidence.run``    — the addressable handle, so an arm can retrieve an
  exact region instead of re-flooding.

``Evidence`` subclasses ``str`` so existing call sites that interpolate or
serialise the error keep working unchanged, while straitjacket-aware call
sites can reach the digest and the handle.

What is deliberately NOT here
-----------------------------
There is no keyword/substring filter producing a "digest". Selecting lines
because they contain ``"FAIL:"`` or ``"AssertionError"`` is the anti-pattern
straitjacket exists to replace: it has no coverage receipt, no address for
what it dropped, and it silently loses the quiet needle. If the harness is
unavailable, the straitjacket arms refuse to run rather than fabricate one.
"""

import os
import sys
import re
import subprocess
import tempfile
import shutil
import threading

from .config import GEMINI_35_FLASH_LITE_ID, TRIAGE_ROLE
from .client import dispatch_model
from . import straitjacket as sj

# run_capture inherits the parent environment, so the capture-determinism
# settings have to be present before the first child is spawned.
for _k, _v in sj.CAPTURE_ENV.items():
    os.environ.setdefault(_k, _v)


# ==============================================================================
# --- EVIDENCE: ONE CAPTURE, TWO TREATMENTS ---
# ==============================================================================

class Evidence(str):
    """A captured failure. Behaves as the uncontained error string.

    ``run`` is the :class:`~src.straitjacket.ContainedRun` when the failure
    came from a real execution through the harness; ``None`` for failures
    detected before execution (e.g. a response with no function definition),
    which carry no unbounded output to contain.
    """

    run = None
    digest = ""
    contained = False

    def __new__(cls, native_text, *, run=None, digest="", contained=False):
        obj = super().__new__(cls, native_text or "")
        obj.run = run
        obj.digest = digest or ""
        obj.contained = bool(contained)
        return obj

    @property
    def metrics(self):
        return self.run.metrics() if self.run is not None else {}


def _guard_evidence(message):
    """A pre-execution guard failure: short, bounded, already its own digest."""
    return Evidence(message, run=None, digest=message, contained=False)


def _from_run(run):
    """Wrap a harness capture: native payload as the string value, real
    straitjacket digest alongside it."""
    _record_capture(run)
    return Evidence(run.native_payload(), run=run, digest=run.digest, contained=True)


# ==============================================================================
# --- CONTAINMENT LEDGER (per task attempt) ---
# ==============================================================================
#
# straitjacket's own claim is "task success at lower context residency", so a
# benchmark that only reports pass rate and dollars measures half of it. The
# ledger records what every capture in one task attempt actually kept out of
# context, straight from the harness's own accounting.

_ledger = threading.local()


def begin_containment():
    """Start a fresh ledger for one task attempt."""
    _ledger.items = []
    _ledger.sent = []


def _record_capture(run):
    items = getattr(_ledger, "items", None)
    if items is None:
        items = _ledger.items = []
    try:
        items.append(run.metrics())
    except Exception:
        pass


def record_evidence_sent(evidence, payload, treatment):
    """Record what an arm put in the repair prompt, and its counterfactual.

    Capture happens for every arm — the harness runs the tests either way —
    so "raw vs digest" alone does not separate the arms. What separates them
    is which payload crossed the wire, measured against what the untreated
    path would have sent *for that same failure*.

    Both numbers are recorded here, at the same moment, for the same event.
    Sourcing the baseline from captures instead meant summing over a different
    number of events (a repair loop treats N failures but captures N+1 runs),
    which made the native arm score a 33% improvement over itself.
    ``str(evidence)`` is by definition the uncontained payload.
    """
    sent = getattr(_ledger, "sent", None)
    if sent is None:
        sent = _ledger.sent = []
    sent.append((
        treatment,
        sj.estimate_tokens(len(str(payload).encode("utf-8"))),
        sj.estimate_tokens(len(str(evidence).encode("utf-8"))),
    ))


def containment_report():
    """Aggregate the current task attempt's captures."""
    items = list(getattr(_ledger, "items", []) or [])
    sent = list(getattr(_ledger, "sent", []) or [])
    raw = sum(i.get("raw_tokens_est", 0) for i in items)
    dig = sum(i.get("digest_tokens_est", 0) for i in items)
    sent_tok = sum(t for _, t, _ in sent)
    base_tok = sum(b for _, _, b in sent)
    return {
        # -- capture side: what the harness held back from the transcript
        "captures": len(items),
        "raw_tokens_est": raw,
        "digest_tokens_est": dig,
        "tokens_kept_out": max(0, raw - dig),
        "containment_ratio": round(raw / dig, 2) if dig else None,
        "profiles": sorted({i.get("sj_profile") for i in items if i.get("sj_profile")}),
        "backend": next((i.get("sj_backend") for i in items if i.get("sj_backend")), None),
        "handles": [i.get("sj_handle") for i in items if i.get("sj_handle")],
        "raw_exact": all(i.get("raw_exact", False) for i in items) if items else None,
        # -- treatment side: the A/B, event for event
        "treatment_events": len(sent),
        "evidence_sent_tokens_est": sent_tok,
        "native_baseline_tokens_est": base_tok,
        "delta_vs_native_tokens": base_tok - sent_tok,
        "treatments": sorted({t for t, _, _ in sent}),
    }


# ==============================================================================
# --- PYTHON FUNCTION EVALUATION (BigCodeBench-Hard & WebDev) ---
# ==============================================================================

def extract_code(text):
    """Extract Python code block from LLM model response."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()

# The runner tail appended to every candidate program. It lives in
# src/straitjacket.py because its shape is dictated by capture determinism:
# identical failing code must produce identical bytes, or the same failure
# mints a new artifact on every attempt.
_UNITTEST_RUNNER = sj.DETERMINISTIC_UNITTEST_TAIL

def run_bigcodebench(problem, solution_code):
    """Execute the BigCodeBench unittest suite for one candidate solution.

    Returns ``(passed, evidence)``. When the harness is available the program
    runs under ``ctx run``: stdout/stderr are content-addressed into the
    artifact store and the failure comes back as a bounded digest plus a
    retrieval handle. Otherwise it falls back to a plain subprocess with the
    native tail truncation, and the evidence is marked ``contained=False``.
    """
    program = solution_code + "\n\n" + problem["test"] + _UNITTEST_RUNNER

    if sj.available():
        return _run_bigcodebench_contained(program)
    return _run_bigcodebench_native(program)


def _run_bigcodebench_contained(program):
    workdir = sj.new_sandbox("bcb")
    try:
        (workdir / "prog.py").write_text(program, encoding="utf-8")
        try:
            run = sj.contained_run(
                [sys.executable, "prog.py"],
                cwd=workdir,
                timeout=120.0,
                # Keep the host's absolute interpreter path out of the
                # manifest and the model-visible digest.
                record_argv=["python3", "prog.py"],
                env_extra=sj.CAPTURE_ENV,
            )
        except sj.SJUnavailable as e:
            return False, _guard_evidence(f"harness_error: {e}")

        if run.timed_out:
            return False, _from_run(run)
        if run.exit_code == 0:
            _record_capture(run)
            return True, Evidence("", run=run, digest=run.digest, contained=True)
        return False, _from_run(run)
    finally:
        sj.drop_sandbox(workdir)


def _run_bigcodebench_native(program):
    """Uncontained execution path (harness disabled). Kept so the benchmark
    can still run the non-straitjacket arms without ctx-harness installed."""
    workdir = tempfile.mkdtemp(prefix="bcb_")
    path = os.path.join(workdir, "prog.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(program)
    try:
        env = {**os.environ, **sj.CAPTURE_ENV}
        r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=120, cwd=workdir, env=env)
        if r.returncode == 0:
            return True, Evidence("")
        return False, Evidence(sj.tail_to_cap(r.stderr.strip() or "test failed"))
    except subprocess.TimeoutExpired:
        return False, Evidence("timeout: execution exceeded 120s")
    except Exception as e:
        return False, Evidence(f"execution_error: {e}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def missing_code_error(code, entry_point):
    """Check if the extracted code defines the requested function entry point.

    This is a birth-gate rejection: the candidate is refused before execution,
    so there is no unbounded output to contain. The message is returned as
    bounded :class:`Evidence` so downstream treatment is uniform.
    """
    if f"def {entry_point}" in code:
        return None
    return _guard_evidence(
        f"model response contains no `def {entry_point}` code block")

# ==============================================================================
# --- CLASSEVAL CLASS / METHOD EVALUATION ---
# ==============================================================================
#
# BigCodeBench scores one function with one verdict. ClassEval scores a class,
# and the whole reason it is here is that the verdict can be taken per METHOD:
# each method owns a test class, so a pass can be attributed to whichever model
# wrote that method. That attribution is what turns "the routed arm won" into
# "the cheap model was routed 60 standalone methods and got 47 of them".
#
# The candidate program is assembled the same way for both granularities --
# candidate class + the row's complete test source -- and only the runner tail
# differs, naming which test classes to load. Keeping the program text
# identical between a per-method run and a class-level run means the two
# numbers are comparable; selecting the tests by slicing the source instead
# would quietly change what the candidate is compiled against.


def run_classeval_tests(problem, class_code, test_classes):
    """Run exactly ``test_classes`` against a candidate class. Returns
    ``(passed, evidence)``.

    ``test_classes`` may be one method's test class, several, or every class in
    the row -- the caller decides the granularity.
    """
    names = [t for t in (test_classes or []) if t]
    if not names:
        return False, _guard_evidence("no test class named for this ClassEval task")

    program = (class_code + "\n\n" + problem.get("test", "")
               + sj.unittest_tail(names))

    if sj.available():
        return _run_bigcodebench_contained(program)
    return _run_bigcodebench_native(program)


def run_classeval_method(problem, class_code, subtask):
    """Score one method of a candidate class against that method's own tests."""
    return run_classeval_tests(problem, class_code, [subtask.get("test_class")])


def run_classeval_class(problem, class_code):
    """Score a candidate class against every test class in the row.

    This is ClassEval's own class-level metric and it is strictly harder than
    "every method passed its own tests": 89 of the 100 rows carry an extra
    integration test class that belongs to no single method, and it is the one
    that fails when methods are individually correct but do not compose.
    """
    return run_classeval_tests(problem, class_code, problem.get("test_classes") or [])


def missing_class_error(code, class_name):
    """Birth gate: refuse a candidate that never defined the required class.

    Same contract as :func:`missing_code_error` -- rejected before execution,
    returned as bounded Evidence so the repair turn is fed the same shape of
    payload whichever gate stopped it.
    """
    if re.search(rf"^\s*class\s+{re.escape(str(class_name))}\b", code or "", re.M):
        return None
    return _guard_evidence(
        f"model response contains no `class {class_name}` definition")


def classeval_subtask_summary(subtask_records):
    """Aggregate per-method records into per-tier and per-model tallies.

    The per-tier split is the measurement the routing hypothesis lives or dies
    on: an arm that routed `standalone` methods to a cheap model has to show
    that the cheap model actually passed them.
    """
    by_tier, by_model = {}, {}
    for r in subtask_records or []:
        for key, bucket in ((r.get("tier", "?"), by_tier), (r.get("model_id", "?"), by_model)):
            b = bucket.setdefault(key, {"n": 0, "passed": 0, "usd": 0.0})
            b["n"] += 1
            b["passed"] += 1 if r.get("passed") else 0
            b["usd"] = round(b["usd"] + float(r.get("as_run_usd", 0.0) or 0.0), 6)
    for bucket in (by_tier, by_model):
        for b in bucket.values():
            b["pass_rate"] = round(b["passed"] / b["n"], 3) if b["n"] else 0.0
    return {"by_tier": by_tier, "by_model": by_model,
            "n_subtasks": len(subtask_records or []),
            "passed_subtasks": sum(1 for r in (subtask_records or []) if r.get("passed"))}


# ==============================================================================
# --- FEATUREBENCH: CONTAINERISED REPOSITORY EVALUATION ---
# ==============================================================================
#
# The first dataset in this repository whose oracle is *expensive*. BCB-Hard and
# ClassEval run a sandboxed unittest in well under a second for $0; a
# FeatureBench attempt applies a patch inside a repository container and runs
# pytest, which the upstream paper measures at 57.2 s/instance on gold patches.
# That is the whole point of adopting it -- it is the only P4 axis the repo has
# (docs/pattern-dataset-selection.md section 7).
#
# Two consequences shape this code:
#
#   1. **One container per task, not per attempt.** A repair ladder runs up to
#      three attempts against the same repository; paying container start-up
#      three times would triple the dominant cost and make the arms measure
#      Docker rather than the models. `FeatureBenchEnv` starts the container
#      once, resets the worktree between attempts, and tears it down after.
#   2. **`docker exec` goes through the harness unchanged.** `sj.contained_run`
#      takes an argv, so the pytest output is captured at the birth gate by the
#      same code path BCB-Hard uses. Nothing about containment is
#      re-implemented for this dataset; the `pytest/v*` profile does the
#      extraction, which is also what makes the evidence gate work here.

FB_CONTAINER_PREFIX = "tokenomics-fb"

# Two apply strategies, strict first. A model-authored diff that needs fuzz is
# still a legitimate solve -- SWE-bench's own harness allows the same latitude
# -- but a patch that fails both is a real failure and is fed back as evidence
# rather than silently scored zero.
_APPLY_STRATEGIES = (
    ["git", "apply", "--verbose"],
    ["patch", "--batch", "--fuzz=5", "-p1", "-i"],
)


def extract_patch(text):
    """Pull a unified diff out of a model response.

    Re-introduced for FeatureBench. The identically-named helper that served
    the deleted SWE-bench Pro path fed a scorer that never ran the repository's
    tests; this one feeds a real pytest run inside the repository's own
    container, so the diff is executed rather than string-matched.
    """
    m = re.search(r"```(?:diff|patch)\s*\n(.*?)```", text or "", re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    m = re.search(r"```\s*\n(diff --git .*?)```", text or "", re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    # An unfenced diff is common enough to accept, but only from the first
    # marker onward -- prose before it would break `git apply`.
    idx = (text or "").find("diff --git ")
    if idx == -1:
        idx = (text or "").find("--- ")
    return (text or "")[idx:].strip("\n") if idx != -1 else ""


def missing_patch_error(patch_str):
    """Birth gate: refuse a response that carries no applicable diff.

    Same contract as :func:`missing_code_error` -- rejected before the
    container is touched, returned as bounded Evidence so the repair turn is
    fed the same shape of payload whichever gate stopped it.
    """
    text = (patch_str or "").strip()
    if not text:
        return _guard_evidence("model response contains no patch")
    if "--- " not in text or "+++ " not in text:
        return _guard_evidence(
            "model response is not a unified diff: no `---`/`+++` file headers")
    if "@@" not in text:
        return _guard_evidence(
            "model response has diff headers but no `@@` hunk -- nothing to apply")
    return None


def docker_available():
    """(ok, reason). Checked before a sweep rather than failing task by task."""
    if shutil.which("docker") is None:
        return False, "`docker` is not on PATH"
    try:
        p = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True, timeout=30)
    except Exception as e:                                   # noqa: BLE001
        return False, f"could not run `docker info`: {e}"
    if p.returncode != 0:
        return False, f"`docker info` failed: {(p.stderr or '').strip()[:200]}"
    return True, f"docker server {(p.stdout or '').strip()}"


class FeatureBenchEnv:
    """One repository container, reused across a task's repair attempts.

    Use as a context manager. `score(patch)` applies a candidate diff, runs the
    task's tests through the harness, then resets the worktree so the next
    attempt starts from the same base.
    """

    def __init__(self, problem, timeout=900.0):
        self.problem = problem
        self.timeout = float(timeout)
        self.image = problem.get("image_name") or ""
        self.workdir = problem.get("repo_workdir") or "/workspace"
        self.name = f"{FB_CONTAINER_PREFIX}-{os.getpid()}-{abs(hash(problem.get('instance_id', ''))) % 10 ** 8}"
        self.sandbox = None
        self.started = False
        self.setup_error = ""

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self):
        self.sandbox = sj.new_sandbox("fb")
        try:
            self._start()
            self.started = True
        except Exception as e:                               # noqa: BLE001
            self.setup_error = str(e)
        return self

    def __exit__(self, *exc):
        subprocess.run(["docker", "rm", "-f", self.name],
                       capture_output=True, text=True, timeout=120)
        return False

    def _sh(self, script, timeout=None, check=True):
        """Run a shell snippet inside the container, outside the harness.

        Setup plumbing (checkout, patch application, worktree reset) is not
        evidence: containing it would put `git apply` chatter into the
        containment ledger and dilute the receipt the test runs produce.
        """
        p = subprocess.run(
            ["docker", "exec", "-w", self.workdir, self.name, "bash", "-lc", script],
            capture_output=True, text=True, timeout=timeout or self.timeout)
        if check and p.returncode != 0:
            raise RuntimeError(
                f"{script.splitlines()[0][:60]}: exit {p.returncode}: "
                f"{((p.stderr or '') + (p.stdout or '')).strip()[-400:]}")
        return p

    def _start(self):
        if not self.image:
            raise RuntimeError("row carries no `image_name`")
        subprocess.run(["docker", "rm", "-f", self.name],
                       capture_output=True, text=True, timeout=120)
        p = subprocess.run(
            ["docker", "run", "-d", "--name", self.name,
             "--network", "none",              # the task is offline by construction
             "-w", self.workdir, self.image, "sleep", "infinity"],
            capture_output=True, text=True, timeout=self.timeout)
        if p.returncode != 0:
            raise RuntimeError(
                f"docker run failed for {self.image}: {(p.stderr or '').strip()[-300:]}")

        base = self.problem.get("base_commit") or ""
        if base:
            self._sh(f"git checkout -f {base} 2>/dev/null || true", check=False)
        # Stage the tests, then commit them, so a per-attempt `git checkout -- .`
        # resets the candidate's edits without also reverting the test files the
        # grade depends on.
        test_patch = self.problem.get("test_patch") or ""
        if test_patch.strip():
            self._write("/tmp/fb_test.patch", test_patch)
            if not self._try_apply("/tmp/fb_test.patch"):
                raise RuntimeError("the dataset's own test_patch did not apply")
        self._sh("git config user.email fb@local && git config user.name fb && "
                 "git add -A && git commit -q -m fb-tests --allow-empty")

    def _write(self, path, text):
        """Materialise a file inside the container without a shell quoting hazard."""
        p = subprocess.run(
            ["docker", "exec", "-i", self.name, "bash", "-lc", f"cat > {path}"],
            input=text, capture_output=True, text=True, timeout=self.timeout)
        if p.returncode != 0:
            raise RuntimeError(f"could not write {path}: {(p.stderr or '').strip()[-200:]}")

    def _try_apply(self, path):
        for argv in _APPLY_STRATEGIES:
            cmd = " ".join(argv) + (f" {path}" if argv[0] == "git" else f" {path}")
            if self._sh(cmd, check=False).returncode == 0:
                return True
        return False

    def reset(self):
        self._sh("git checkout -- . && git clean -fdq", check=False)

    # -- scoring -----------------------------------------------------------
    def score(self, patch):
        """Apply `patch`, run the task's tests, reset. Returns (resolved, evidence).

        `resolved` mirrors FeatureBench's own Resolved Rate: pytest exits 0 over
        the fail-to-pass and pass-to-pass files together.
        """
        if not self.started:
            return False, _guard_evidence(
                f"FeatureBench container unavailable: {self.setup_error}")
        guard = missing_patch_error(patch)
        if guard:
            return False, guard

        try:
            self._write("/tmp/fb_cand.patch", patch)
            if not self._try_apply("/tmp/fb_cand.patch"):
                self.reset()
                return False, _guard_evidence(
                    "patch did not apply (tried `git apply` then `patch --fuzz=5`). "
                    "Re-emit the diff against the files as they exist at this commit.")
            return self._pytest()
        except Exception as e:                               # noqa: BLE001
            return False, _guard_evidence(f"FeatureBench execution error: {e}")
        finally:
            try:
                self.reset()
            except Exception:                                # noqa: BLE001
                pass

    def _pytest(self):
        files = featurebench_test_files(self.problem)
        if not files:
            return False, _guard_evidence("row names no FAIL_TO_PASS test file")
        argv = ["docker", "exec", "-w", self.workdir, self.name,
                "python", "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
                *files]
        run = sj.contained_run(argv, cwd=self.sandbox, timeout=self.timeout,
                               # Keep the container name (which carries a pid)
                               # out of the recorded argv, or every attempt
                               # digests as a different command.
                               record_argv=["python", "-m", "pytest", "-q", *files])
        return run.exit_code == 0, _from_run(run)


def featurebench_test_files(problem):
    """The files pytest is pointed at: fail-to-pass first, then pass-to-pass."""
    out = []
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        v = problem.get(key) or []
        if isinstance(v, str):
            v = [v]
        out.extend(str(x) for x in v if str(x).strip())
    seen, uniq = set(), []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


_PYTEST_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors)")


def featurebench_test_ratio(evidence):
    """Fraction of executed test cases that passed, from pytest's own summary.

    FeatureBench reports a *Passed Rate* beside Resolved Rate precisely because
    a binary verdict throws away the signal a cheap model still produces on a
    task it cannot finish. This is the same idea read off the captured summary
    line; it is named `test_pass_ratio` rather than `passed_rate` because the
    upstream denominator (fail-to-pass tests only) is not something this code
    can verify from the output alone.

    Returns None when nothing countable was captured -- a crash before
    collection, or a patch that never applied.
    """
    counts = {}
    for n, kind in _PYTEST_COUNT_RE.findall(str(evidence or "")):
        key = "failed" if kind.startswith("error") else kind
        counts[key] = counts.get(key, 0) + int(n)
    total = counts.get("passed", 0) + counts.get("failed", 0)
    if not total:
        return None
    return round(counts.get("passed", 0) / total, 4)


# ==============================================================================
# --- TRIAGE MECHANISMS: LLM TRIAGE VS STRAITJACKET CONTAINMENT ---
# ==============================================================================

def triage_error(raw_err, model_id=GEMINI_35_FLASH_LITE_ID):
    """
    Standard LLM triage ($ costs input + output tokens and API latency).
    Uses a cheap model to compress verbose error logs into a short digest.

    This is the *uncontained* comparison arm: it pays to move the raw log
    across the wire. ``raw_err`` is used as a plain string, which for an
    :class:`Evidence` value is exactly the native payload.
    """
    prompt = TRIAGE_ROLE + "```\n" + str(raw_err) + "\n```"
    text, usage, dt = dispatch_model(model_id, prompt, max_tokens=768)
    digest = text.strip() or str(raw_err)[-1200:]
    return digest[:1200], usage, dt


def triage_error_straitjacket(raw_err, problem=None, cwd=None):
    """
    Straitjacket containment ($0.000000, no API call).

    Returns the digest the upstream harness already produced for this
    execution — profile-detected, coverage-attested, and carrying retrieval
    addresses for everything it left out. Nothing is re-summarised here and
    no line is selected by keyword.

    ``raw_err`` is normally the :class:`Evidence` returned by the evaluator.
    A plain string reaching this function means the failure was captured
    outside the harness; it is contained now (birth gate missed, entry gate
    still enforced) rather than keyword-filtered.
    """
    usage = {"as_run_usd": 0.0, "input": 0, "output": 0, "total_tokens": 0}

    if isinstance(raw_err, Evidence) and raw_err.digest:
        return raw_err.digest, usage, 0.0

    text = str(raw_err or "")
    if not text.strip():
        return "", usage, 0.0

    if sj.available():
        try:
            run = sj.contain_text(
                text,
                argv=["python3", "-m", "unittest", "prog.py"],
                exit_code=1,
                stream="stderr",
                cwd=str((problem or {}).get("repo", ".")) if problem else ".",
            )
            _record_capture(run)
            return run.digest, usage, 0.0
        except sj.SJUnavailable:
            pass

    # The harness is genuinely unavailable. Do not invent a digest: say so and
    # hand back the evidence unchanged, so the arm's numbers are never
    # attributed to a containment mechanism that did not run.
    sj.require()  # raises SJUnavailable with install instructions
    return text, usage, 0.0  # pragma: no cover - require() always raises here


def aggregate_containment(results):
    """Sum per-task containment ledgers into one row-level receipt.

    One definition, used by every runner: a benchmark row that reports pass
    rate and dollars but not context residency is reporting half of what the
    harness is for.
    """
    items = [r.get("containment") or {} for r in results]
    raw = sum(i.get("raw_tokens_est", 0) for i in items)
    dig = sum(i.get("digest_tokens_est", 0) for i in items)
    sent = sum(i.get("evidence_sent_tokens_est", 0) for i in items)
    base = sum(i.get("native_baseline_tokens_est", 0) for i in items)
    return {
        "captures": sum(i.get("captures", 0) for i in items),
        "treatment_events": sum(i.get("treatment_events", 0) for i in items),
        "raw_tokens_est": raw,
        "digest_tokens_est": dig,
        "evidence_sent_tokens_est": sent,
        "native_baseline_tokens_est": base,
        "delta_vs_native_tokens": base - sent,
        "tokens_kept_out": max(0, raw - dig),
        "containment_ratio": round(raw / dig, 2) if dig else None,
        "profiles": sorted({p for i in items for p in (i.get("profiles") or [])}),
        "treatments": sorted({t for i in items for t in (i.get("treatments") or [])}),
    }


def straitjacket_status():
    """Harness provenance for result files and reports."""
    return sj.status()
