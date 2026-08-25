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
import json
import platform
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
    # Why the attempt failed, when it failed *before* the suite ran. Empty
    # string means "the suite ran and the harness captured it", which is the
    # only case where `run` carries a typed evidence graph.
    reason = ""
    # The name a router should reason over for a pre-execution failure. The
    # counterpart of a profile's `failure_class` for evidence that has no
    # profile, so `src/routing.py` has one thing to read instead of two.
    failure_class = ""

    def __new__(cls, native_text, *, run=None, digest="", contained=False,
                reason="", failure_class=""):
        obj = super().__new__(cls, native_text or "")
        obj.run = run
        obj.digest = digest or ""
        obj.contained = bool(contained)
        obj.reason = reason or ""
        obj.failure_class = failure_class or ""
        return obj

    @property
    def metrics(self):
        return self.run.metrics() if self.run is not None else {}


# Every way an attempt can die *before* the repository's suite runs, named once.
# The sweep records the name per task and the reporter tabulates the
# distribution, because "0% pass rate" and "89% of attempts never reached the
# tests" are different findings and only the second one is actionable.
GUARD_REASONS = {
    # the model's response was not a usable patch
    "no_patch": "MalformedPatch",
    "not_a_diff": "MalformedPatch",
    "no_hunk": "MalformedPatch",
    "truncated_output": "MalformedPatch",
    # the patch was well-formed but did not land on this tree
    "apply_failed": "PatchApplyError",
    # nothing the model did could have changed the outcome
    "container_unavailable": "EnvironmentError",
    "restore_failed": "EnvironmentError",
    "row_no_test_files": "EnvironmentError",
    "execution_error": "EnvironmentError",
    "harness_error": "EnvironmentError",
}

# Reasons that say the *environment* failed, not the model. Escalating to a
# frontier model on one of these buys nothing; see `src/routing.py`.
ENVIRONMENT_REASONS = frozenset(
    r for r, cls in GUARD_REASONS.items() if cls == "EnvironmentError")


def _guard_evidence(message, reason="", failure_class=""):
    """A pre-execution guard failure: short, bounded, already its own digest.

    ``reason`` is one of :data:`GUARD_REASONS`. It is what makes a guard
    failure legible to both the router and the report: without it every
    pre-execution death arrives as an untyped string and classifies as
    `shallow`, which is how an evidence gate ends up never firing on the most
    common failure in the sweep.
    """
    return Evidence(message, run=None, digest=message, contained=False,
                    reason=reason,
                    failure_class=failure_class or GUARD_REASONS.get(reason, ""))


def guard_reason(evidence):
    """The reason slug for one attempt: `''` when the suite actually ran."""
    return getattr(evidence, "reason", "") or ""


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
            return False, _guard_evidence(f"harness_error: {e}", "harness_error")

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
# Strict first, then progressively looser. Every relaxation here targets a
# failure mode that is an artefact of *how a model writes a diff*, never one
# that would let a wrong patch score:
#
#   --recount            the `@@ -a,b +c,d @@` line counts are recomputed from
#                        the hunk body. Miscounted hunk headers are the single
#                        most common defect in a model-authored diff, and they
#                        are the one defect that carries no information about
#                        whether the *edit* was right.
#   --ignore-whitespace  indentation drift in the context lines.
#   -C1                  match on one line of context instead of three.
#   --3way               fall back to a real merge using the blobs the
#                        repository already has. Only possible because the tree
#                        is at `base_commit`, and it fails loudly on conflict
#                        rather than guessing.
#   patch --forward      refuses to *reverse* a diff it thinks is already
#                        applied. Without it `patch` exits 0 having deleted the
#                        change the next step needs -- silence in exactly the
#                        place a wrong number comes from.
#
# The candidate still has to make the repository's own tests pass afterwards,
# so a looser apply widens what gets *graded*, not what counts as resolved.
_APPLY_STRATEGIES = (
    ["git", "apply", "--verbose"],
    ["git", "apply", "--verbose", "--recount"],
    ["git", "apply", "--verbose", "--recount", "--ignore-whitespace", "-C1"],
    ["git", "apply", "--verbose", "--recount", "--3way"],
    ["patch", "--batch", "--forward", "--fuzz=5", "-p1", "-i"],
)


def _terminate_patch(patch):
    """Guarantee the one byte `git apply` refuses to work without.

    A unified diff whose final line has no newline is not a diff `git apply`
    will read: it exits **128** with `corrupt patch at line N` before it looks
    at the worktree at all. Every strategy in `_APPLY_STRATEGIES` inherits that
    -- widening the ladder buys nothing, because none of its entries ever gets
    to run. Measured on a scratch repository: a byte-perfect diff applies under
    all five strategies with the newline and under none of them without it, and
    a diff with drifted context is rescued by `--recount --ignore-whitespace`
    with the newline and by nothing without it.

    This was the harness's own bug, not the models'. `extract_patch` used to end
    every return path with `.strip("\n")`, which deletes exactly this byte from
    every candidate the sweep ever scored -- so `git apply` never once ran to
    completion on FeatureBench, and the loose `patch --fuzz` fallback was
    silently the only applier in the pipeline. See
    docs/featurebench-n48-lessons.md.
    """
    body = (patch or "").strip("\n")
    return body + "\n" if body else ""


def extract_patch(text):
    """Pull a unified diff out of a model response.

    Re-introduced for FeatureBench. The identically-named helper that served
    the deleted SWE-bench Pro path fed a scorer that never ran the repository's
    tests; this one feeds a real pytest run inside the repository's own
    container, so the diff is executed rather than string-matched.

    **Every** fenced diff block is taken, not just the first. A multi-file
    feature is routinely answered with one fence per file, and returning only
    the first silently scored a fraction of the candidate -- which reads as a
    model failure and is not one. Blocks are joined with the newline that
    :func:`_terminate_patch` guarantees, so a concatenation is still a diff.
    """
    blocks = re.findall(r"```(?:diff|patch)\s*\n(.*?)```", text or "", re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n(diff --git .*?)```", text or "", re.DOTALL)
    if blocks:
        joined = "\n".join(b.strip("\n") for b in blocks if b.strip())
        return _terminate_patch(joined)
    # An unfenced diff is common enough to accept, but only from the first
    # marker onward -- prose before it would break `git apply`.
    idx = (text or "").find("diff --git ")
    if idx == -1:
        idx = (text or "").find("--- ")
    return _terminate_patch((text or "")[idx:]) if idx != -1 else ""


def missing_patch_error(patch_str):
    """Birth gate: refuse a response that carries no applicable diff.

    Same contract as :func:`missing_code_error` -- rejected before the
    container is touched, returned as bounded Evidence so the repair turn is
    fed the same shape of payload whichever gate stopped it.
    """
    text = (patch_str or "").strip()
    if not text:
        return _guard_evidence("model response contains no patch", "no_patch")
    if "--- " not in text or "+++ " not in text:
        return _guard_evidence(
            "model response is not a unified diff: no `---`/`+++` file headers",
            "not_a_diff")
    if "@@" not in text:
        return _guard_evidence(
            "model response has diff headers but no `@@` hunk -- nothing to apply",
            "no_hunk")
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


class _DockerRepoEnv:
    """Container plumbing shared by the two executors that grade a patch inside
    a repository's own image.

    Only the three operations that are *identical* live here -- run a snippet,
    write a file, apply a diff. Container start-up, what "reset" means and how
    a row is scored differ per dataset and stay in the subclasses, because
    those are where the benchmarks disagree.

    ``SHELL_FLAGS`` is not decoration. FeatureBench's images want a login shell
    (`-lc`) to pick up the interpreter their `test_cmd` assumes; SWE-bench Pro's
    do not, and must not get one -- upstream runs `bash <script>` directly, and
    sourcing `/etc/profile` can reorder a PATH the image set deliberately.
    """

    SHELL_FLAGS = "-lc"

    def _sh(self, script, timeout=None, check=True):
        """Run a shell snippet inside the container, outside the harness.

        Setup plumbing (checkout, patch application, worktree reset) is not
        evidence: containing it would put `git apply` chatter into the
        containment ledger and dilute the receipt the test runs produce.
        """
        p = subprocess.run(
            ["docker", "exec", "-w", self.workdir, self.name,
             "bash", self.SHELL_FLAGS, script],
            capture_output=True, text=True, timeout=timeout or self.timeout)
        if check and p.returncode != 0:
            raise RuntimeError(
                f"{script.splitlines()[0][:60]}: exit {p.returncode}: "
                f"{((p.stderr or '') + (p.stdout or '')).strip()[-400:]}")
        return p

    def _write(self, path, text):
        """Materialise a file inside the container without a shell quoting hazard."""
        p = subprocess.run(
            ["docker", "exec", "-i", self.name, "bash", self.SHELL_FLAGS,
             f"mkdir -p $(dirname {path}) && cat > {path}"],
            input=text, capture_output=True, text=True, timeout=self.timeout)
        if p.returncode != 0:
            raise RuntimeError(f"could not write {path}: {(p.stderr or '').strip()[-200:]}")

    def _try_apply(self, path, reset=None):
        """Apply a diff, strict strategy first. Records what each attempt said.

        `patch --batch` will happily *reverse* a diff it thinks was already
        applied and exit 0, which silently deletes a file the next step needs.
        So the log is kept on the instance rather than discarded, and callers
        that care check the outcome rather than the return code alone.

        ``reset`` is called between failed strategies when supplied. It has to
        be: `git apply --3way` writes conflict markers into the worktree on a
        partial merge and *then* exits non-zero, so without a reset the next
        strategy would be applied on top of that debris and any success it
        reported would be a success against a tree nobody chose.
        """
        self.apply_log = []
        for i, argv in enumerate(_APPLY_STRATEGIES):
            if i and reset is not None:
                reset()
            cmd = " ".join(argv) + f" {path}"
            r = self._sh(cmd, check=False)
            out = ((r.stdout or "") + (r.stderr or "")).strip()
            self.apply_log.append(f"$ {cmd}\n[exit {r.returncode}] {out[:400]}")
            if r.returncode == 0:
                return True
        return False

    def apply_evidence(self, limit=1800):
        """What the apply attempts actually said, as repair-turn evidence.

        `git apply --verbose` names the file, the hunk number and prints the
        `error: while searching for:` block that did not match. That is the
        only actionable signal a patch-apply failure produces, and it was
        being collected and then dropped -- so the repair turn received a
        fixed 31-token sentence and the second rung was an independent
        re-roll rather than a repair.
        """
        log = "\n".join(getattr(self, "apply_log", []) or [])
        return log[-limit:] if len(log) > limit else log


class FeatureBenchEnv(_DockerRepoEnv):
    """One repository container, reused across a task's repair attempts.

    Use as a context manager. `score(patch)` applies a candidate diff, runs the
    task's tests through the harness, then resets the worktree so the next
    attempt starts from the same base.
    """

    def __init__(self, problem, timeout=None):
        from .datasets import fb_timeout
        self.problem = problem
        self.timeout = float(timeout) if timeout else fb_timeout(problem, "timeout_run")
        self.image = problem.get("image_name") or ""
        # Only a fallback. `_resolve_workdir` replaces it from the image itself
        # once the container is up -- `repo_settings` carries no path key, so a
        # guess here would decide what every arm is scored on.
        self.workdir = problem.get("repo_workdir") or "/workspace"
        self.name = f"{FB_CONTAINER_PREFIX}-{os.getpid()}-{abs(hash(problem.get('instance_id', ''))) % 10 ** 8}"
        self.sandbox = None
        self.started = False
        self.setup_error = ""
        self.workdir_unverified = False
        self._staged = []
        self.apply_log = []

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

    def _start(self):
        if not self.image:
            raise RuntimeError("row carries no `image_name`")
        subprocess.run(["docker", "rm", "-f", self.name],
                       capture_output=True, text=True, timeout=120)
        p = subprocess.run(
            ["docker", "run", "-d", "--name", self.name,
             # Hermetic would be nicer, but `--network none` also blocks any
             # dependency the image expects to resolve at run time, and turns a
             # fixable setup problem into an unexplained import error. Set
             # FB_NETWORK=none to force isolation once a split is known good.
             "--network", os.environ.get("FB_NETWORK", "bridge"),
             "-w", self.workdir, self.image, "sleep", "infinity"],
            capture_output=True, text=True, timeout=self.timeout)
        if p.returncode != 0:
            raise RuntimeError(
                f"docker run failed for {self.image}: {(p.stderr or '').strip()[-300:]}")

        self._resolve_workdir()
        self._sh("git config user.email fb@local && git config user.name fb",
                 check=False)

        base = self.problem.get("base_commit") or ""
        if base:
            head = self._sh("git rev-parse HEAD", check=False).stdout.strip()
            # These are per-instance images, so HEAD is usually already right.
            # Checking out unconditionally would move a correctly prepared tree.
            if not head.startswith(base[:12]):
                self._sh(f"git checkout -f {base} 2>/dev/null || true", check=False)

        # Materialise the graded test files, then take the tree back to base.
        #
        # Order matters and it is not the obvious one. FeatureBench applies the
        # *solution* patch to the base tree and only then restores the
        # fail-to-pass test file. Applying `test_patch` first -- the obvious
        # reading -- makes every gold patch that also touches a test file
        # conflict, because the context it was generated against is already
        # gone. So the test files are staged aside here and copied back over
        # each candidate in `score()`.
        files = featurebench_test_files(self.problem)

        # Materialise graded test files from git HEAD first. FeatureBench
        # images ship with fail-to-pass test files deleted from the worktree
        # (or skip-worktree flagged), while git HEAD carries the committed
        # versions. In FeatureBench, `test_patch` is often a deletion diff
        # (masking the test file for inference), so applying it unconditionally
        # deletes the test file from the worktree. Restore from HEAD first, and
        # only try applying `test_patch` if files remain missing and the patch
        # is not a deletion diff.
        missing = self._restore_from_head(files) if files else []
        test_patch = self.problem.get("test_patch") or ""
        if missing and test_patch.strip() and "deleted file mode" not in test_patch:
            self._write("/tmp/fb_test.patch", test_patch)
            self._try_apply("/tmp/fb_test.patch")

        still_missing = [f for f in files
                         if self._sh(f"test -f {f}", check=False).returncode != 0]
        if still_missing:
            raise RuntimeError(
                f"{len(still_missing)} graded test file(s) are absent ({still_missing[0]}). "
                f"Apply log:\n" + "\n".join(getattr(self, "apply_log", []) or ["(none)"]))
        self._staged = []
        if files:
            # Staged one file at a time, with `cp --parents` avoided (it is a
            # GNU extension, absent on busybox) and NO error suppression. An
            # earlier version wrote `cp --parents ... 2>/dev/null || true`,
            # which silently staged nothing: the restore then became a no-op,
            # every arm ran the repository's ORIGINAL tests, and gold "failed"
            # because the old tests describe the old behaviour. Silence is the
            # bug here, so this verifies rather than hopes.
            script = "rm -rf /tmp/fb_tests && " + " && ".join(
                f"mkdir -p /tmp/fb_tests/$(dirname {f}) && cp {f} /tmp/fb_tests/{f}"
                for f in files)
            r = self._sh(script, check=False)
            if r.returncode != 0:
                raise RuntimeError(
                    "could not stage the graded test files: "
                    f"{((r.stderr or '') + (r.stdout or '')).strip()[-300:]}")
            got = self._sh("find /tmp/fb_tests -type f | wc -l", check=False)
            n_staged = int((got.stdout or "0").strip() or 0)
            if n_staged != len(files):
                raise RuntimeError(
                    f"staged {n_staged} of {len(files)} graded test files; the "
                    "restore would run the repository's original tests instead")
            self._staged = list(files)

        if files:
            self._sh("git checkout -- " + " ".join(files), check=False)

    def _restore_from_head(self, files):
        """Materialise the graded test files from the commit, whatever git thinks.

        `git checkout -- <path>` is the obvious way and it does not work here:
        the images hide the fail-to-pass file with an index flag (skip-worktree
        / sparse-checkout), so the checkout is a no-op and the file stays gone.
        `git show HEAD:<path>` reads the blob directly and is immune to that.

        Files genuinely absent from the commit are fine -- `test_patch` may be
        creating them -- so only a write that leaves nothing behind is an error.
        """
        missing = []
        for f in files:
            self._sh(f"mkdir -p $(dirname {f}) && "
                     f"git show HEAD:{f} > {f} 2>/dev/null || true", check=False)
            if self._sh(f"test -f {f}", check=False).returncode != 0:
                missing.append(f)
        return missing

    def _restore_tests(self):
        """Copy the graded test files back over whatever the candidate did.

        This is what makes the run measure the *feature* rather than the
        repository's pre-existing behaviour, and what stops a candidate from
        passing by editing the tests. Failure here is fatal to the row's
        meaning, so it is checked.
        """
        if not self._staged:
            return
        script = " && ".join(
            f"mkdir -p $(dirname {f}) && cp /tmp/fb_tests/{f} {f}"
            for f in self._staged)
        r = self._sh(script, check=False)
        if r.returncode != 0:
            raise RuntimeError(
                "could not restore the graded test files: "
                f"{((r.stderr or '') + (r.stdout or '')).strip()[-300:]}")

    def _resolve_workdir(self):
        """Find the repository inside the running container.

        Authoritative sources first: the image's own `WORKDIR`, then the git
        root reachable from it. `repo_settings` has no path key -- checked
        across all 100 rows of the fast split -- so the constructor's
        `/workspace/<name>` is a last resort, not the plan.
        """
        p = subprocess.run(
            ["docker", "inspect", "-f", "{{.Config.WorkingDir}}", self.image],
            capture_output=True, text=True, timeout=60)
        candidate = (p.stdout or "").strip()
        for probe in (candidate, self.workdir):
            if not probe or probe == "/":
                continue
            q = subprocess.run(
                ["docker", "exec", "-w", probe, self.name, "bash", "-lc",
                 "git rev-parse --show-toplevel"],
                capture_output=True, text=True, timeout=60)
            root = (q.stdout or "").strip()
            if q.returncode == 0 and root:
                self.workdir = root
                return
        # Neither worked: leave the fallback in place and let gold fail loudly
        # in the preflight rather than scoring arms against the wrong tree.
        self.workdir_unverified = True

    def reset(self):
        r = self._sh("git status --porcelain", check=False)
        lines = [line.strip().split(maxsplit=1) for line in (r.stdout or "").splitlines() if line.strip()]
        dirty = [parts[1] for parts in lines if len(parts) == 2]
        if dirty:
            self._sh("git checkout -- " + " ".join(dirty) + " && git clean -fdq", check=False)

    # -- scoring -----------------------------------------------------------
    def score(self, patch, allow_empty=False):
        """Apply `patch`, run the task's tests, reset. Returns (resolved, evidence).

        `resolved` mirrors FeatureBench's own Resolved Rate: pytest exits 0 over
        the fail-to-pass and pass-to-pass files together.
        """
        if not self.started:
            return False, _guard_evidence(
                f"FeatureBench container unavailable: {self.setup_error}",
                "container_unavailable")
        if not (allow_empty and not (patch or "").strip()):
            guard = missing_patch_error(patch)
            if guard:
                return False, guard

        try:
            if patch and patch.strip():
                self._write("/tmp/fb_cand.patch", patch)
                if not self._try_apply("/tmp/fb_cand.patch", reset=self.reset):
                    log = self.apply_evidence()
                    self.reset()
                    # Same reasoning as `SWEBenchProEnv.score`: the apply log
                    # names the file, the hunk and the context block that did
                    # not match, and it was being collected and discarded. On
                    # this dataset 75-90% of every arm's final failures are
                    # this branch, so a fixed sentence here is a repair turn
                    # with no information in it.
                    return False, _guard_evidence(
                        "The patch did NOT apply to the repository -- no test was run.\n"
                        f"{len(_APPLY_STRATEGIES)} strategies were tried, strictest first.\n"
                        "Apply log:\n" + (log or "(no output)") +
                        "\n\nRe-emit the COMPLETE diff against the files as they exist "
                        "at this commit.",
                        "apply_failed")
            self._restore_tests()
            return self._pytest()
        except Exception as e:                               # noqa: BLE001
            return False, _guard_evidence(
                f"FeatureBench execution error: {e}", "execution_error")
        finally:
            try:
                self.reset()
            except Exception:                                # noqa: BLE001
                pass

    def _pytest(self):
        from .datasets import fb_test_command
        files = featurebench_test_files(self.problem)
        if not files:
            return False, _guard_evidence(
                "row names no FAIL_TO_PASS test file", "row_no_test_files")
        # Every row ships its own `test_cmd`; using a hardcoded pytest line
        # would score the arms on a command the benchmark never specified.
        cmd = fb_test_command(self.problem)
        shell = f"{cmd} {' '.join(files)}"
        argv = ["docker", "exec", "-w", self.workdir, self.name, "bash", "-lc", shell]
        if sj.available():
            run = sj.contained_run(argv, cwd=self.sandbox, timeout=self.timeout,
                                   # Keep the container name (which carries a pid)
                                   # out of the recorded argv, or every attempt
                                   # digests as a different command.
                                   record_argv=cmd.split() + files)
            return run.exit_code == 0, _from_run(run)
        # Fallback when straitjacket harness is off / unavailable
        p = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        output = ((p.stdout or "") + (p.stderr or "")).strip()
        return p.returncode == 0, Evidence(sj.tail_to_cap(output or "test failed"))


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
# --- SWE-BENCH PRO: THE BENCHMARK'S OWN GRADING, RUN LOCALLY ---
# ==============================================================================
#
# The contract, in one place, because every part of it is load-bearing:
#
#   1. The image (`jefzda/sweap-images:<dockerhub_tag>`) has the repository at
#      /app with dependencies installed, and `ENTRYPOINT ["/bin/bash"]` -- so a
#      container has to be started with `--entrypoint` overridden, or `docker
#      run <image> sleep infinity` becomes `bash sleep infinity` and dies.
#   2. Reset to `base_commit`, apply the candidate diff, THEN run the last line
#      of `before_repo_set_cmd`. That order is upstream's and it is the
#      anti-cheat: the graded test files are checked out from the solution
#      commit *over* whatever the candidate did to them.
#   3. Run upstream's own `run_script.sh <files>` and `parser.py`, never a
#      reimplementation. `parser.py` is per-instance because "what does a
#      passing test look like" is per-repository (mocha JSON here, pytest
#      there, `go test -json` elsewhere).
#   4. Resolved == every name in fail_to_pass + pass_to_pass reported PASSED.
#      Upstream computes `(f2p | p2p) <= passed`; so does `sbp_resolution`.
#
# Network stays ON by default. Several run scripts install dependencies at test
# time (NodeBB's runs `npm install`), so `--network none` does not harden the
# run, it fails it. SBP_NETWORK=none once a split is known to be self-contained.

SBP_CONTAINER_PREFIX = "tokenomics-sbp"

# Some repositories' suites are slow, and the run scripts already carry their
# own inner `timeout`. This is the outer bound on one attempt.
SBP_DEFAULT_TIMEOUT = float(os.environ.get("SBP_TIMEOUT", "1800"))


SBP_WORKSPACE = "/workspace"

# How many characters of repository source may be read out of the container and
# placed in the solver prompt. `0` reproduces the original blind setting, which
# is what makes the grounding an A/B rather than a one-way change.
#
# Why this exists at all: the row hands the model an issue, a requirements
# block and an interface block, and asks for a complete unified diff with real
# line numbers. Measured over the published split, only 8.8% of rows name
# *every* file their reference patch touches anywhere in those three blocks,
# and the median reference patch is 9 hunks across 4 files. `git apply` needs
# every context line to match a file the model was never shown, so the blind
# setting has a localisation ceiling near 9% before a single hunk is written.
SBP_GROUNDING_CHARS = int(os.environ.get("SBP_GROUNDING_CHARS", "60000"))

# Per-file cap, so one 400 kB vendored bundle cannot consume the whole budget.
SBP_GROUNDING_FILE_CHARS = int(os.environ.get("SBP_GROUNDING_FILE_CHARS", "16000"))

# Upper bound on how many files are quoted, whatever the character budget says.
SBP_GROUNDING_MAX_FILES = int(os.environ.get("SBP_GROUNDING_MAX_FILES", "12"))


def sbp_test_script(test_files, workdir="/app", env_exports="", workspace=SBP_WORKSPACE):
    """The script one attempt runs inside the container.

    Two details are the difference between a number and a wrong number:

    * `rm -f /workspace/output.json` first. The parser writes that file only
      when it runs; without the delete, an attempt whose parser crashed would
      be graded against the *previous* attempt's results -- which, in a repair
      loop that reuses one container, is the previous rung's results.
    * the logs are written to files (upstream's parser takes file paths) and
      then echoed, so the harness capture wrapping this command sees exactly
      the bytes the parser saw. Nothing is summarised twice.
    """
    csv = ",".join(str(f) for f in test_files)
    w = workspace.rstrip("/")
    return "\n".join([
        "set -u",
        env_exports or "",
        f"cd {workdir} || exit 90",
        f"rm -f {w}/output.json {w}/stdout.log {w}/stderr.log",
        "PY=python; command -v python >/dev/null 2>&1 || PY=python3",
        f"bash {w}/run_script.sh '{csv}' > {w}/stdout.log 2> {w}/stderr.log",
        "rc=$?",
        f'"$PY" {w}/parser.py {w}/stdout.log {w}/stderr.log {w}/output.json >&2 '
        '|| echo "PARSER FAILED" >&2',
        f"cat {w}/stderr.log >&2",
        f"cat {w}/stdout.log",
        "exit $rc",
    ])


def sbp_resolution(output, problem):
    """Score one attempt from the parser's JSON. Upstream's rule, verbatim.

    Returns a dict rather than a bool because the binary verdict throws away
    what a cheap rung achieved: on a dataset where frontier agents resolve
    ~20-40%, `test_pass_ratio` is the only thing separating "wrote nothing
    useful" from "one assertion short".
    """
    from .datasets import sbp_required_tests
    required = sbp_required_tests(problem)
    tests = (output or {}).get("tests") or []
    passed = {str(t.get("name")) for t in tests
              if str(t.get("status", "")).upper() == "PASSED"}
    missing = [n for n in required if n not in passed]
    # An empty requirement set is a broken row, not a free pass. Upstream's
    # `(f2p | p2p) <= passed` returns True for it; here it does not, because a
    # row that grades every patch as resolved is worse than a skipped row.
    resolved = bool(required) and not missing
    return {
        "resolved": resolved,
        "test_pass_ratio": (round((len(required) - len(missing)) / len(required), 4)
                            if required else None),
        "required": len(required),
        "missing": len(missing),
        "reported": len(tests),
        # Enough names to see the shape of the failure in a results file,
        # bounded so a row with 900 pass_to_pass entries cannot inflate it.
        "missing_names": missing[:10],
    }


class SWEBenchProEnv(_DockerRepoEnv):
    """One SWE-bench Pro container, reused across a task's repair attempts.

    Use as a context manager. `score(patch)` resets to the base commit, applies
    the candidate diff, restores the graded test files, runs the instance's own
    test script through the straitjacket harness, and reads the verdict from
    the instance's own parser.

    Reusing one container across the repair loop is what makes an expensive
    oracle affordable enough to study: the image pull and the dependency
    install happen once per task rather than once per attempt.
    """

    SHELL_FLAGS = "-c"

    def __init__(self, problem, timeout=None, scripts=None):
        from .datasets import swebench_pro_image
        self.problem = problem
        self.timeout = float(timeout or SBP_DEFAULT_TIMEOUT)
        self.image = problem.get("image_name") or swebench_pro_image(problem)
        self.workdir = problem.get("repo_workdir") or "/app"
        self.name = (f"{SBP_CONTAINER_PREFIX}-{os.getpid()}-"
                     f"{abs(hash(problem.get('instance_id', ''))) % 10 ** 8}")
        self.sandbox = None
        self.started = False
        self.setup_error = ""
        self.apply_log = []
        self.scripts = scripts
        # Last attempt's verdict, for the arm's soft metric and the results row.
        self.last_report = {}
        self.last_ratio = None

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self):
        self.sandbox = sj.new_sandbox("sbp")
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

    def _start(self):
        from .datasets import ensure_swebench_pro_scripts
        if not self.image:
            raise RuntimeError("row carries no `dockerhub_tag`")
        iid = self.problem.get("instance_id") or ""
        # Fetched before the container exists: a missing run script is a setup
        # error worth one clear message, not a container that starts, pulls a
        # multi-gigabyte image and then has nothing to run.
        self.scripts = self.scripts or ensure_swebench_pro_scripts(iid)

        subprocess.run(["docker", "rm", "-f", self.name],
                       capture_output=True, text=True, timeout=120)
        argv = ["docker", "run", "-d", "--name", self.name,
                # The images set ENTRYPOINT ["/bin/bash"]; without this
                # override the command below is read as a script *filename*.
                "--entrypoint", "/bin/bash",
                "--network", os.environ.get("SBP_NETWORK", "bridge")]
        platform = os.environ.get("SBP_PLATFORM") or _sbp_default_platform()
        if platform:
            argv += ["--platform", platform]
        argv += ["-w", self.workdir, self.image, "-c", "sleep infinity"]
        p = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        if p.returncode != 0:
            raise RuntimeError(
                f"docker run failed for {self.image}: {(p.stderr or '').strip()[-300:]}")

        root = self._sh("git rev-parse --show-toplevel", check=False).stdout.strip()
        if not root:
            raise RuntimeError(
                f"no git repository at {self.workdir} in {self.image} -- the "
                "image is not the one this row names, or it failed to build")
        self.workdir = root

        base = self.problem.get("base_commit") or ""
        if base:
            r = self._sh(f"git cat-file -e {base}^{{commit}}", check=False)
            if r.returncode != 0:
                raise RuntimeError(
                    f"base commit {base[:12]} is not in the image's clone; "
                    "grading would run against a different tree")

        self._write(f"{SBP_WORKSPACE}/run_script.sh", self.scripts["run_script"])
        self._write(f"{SBP_WORKSPACE}/parser.py", self.scripts["parser"])

    def reset(self):
        """Back to the base commit, discarding everything the last attempt did."""
        base = self.problem.get("base_commit") or ""
        if base:
            self._sh(f"git reset --hard {base}", check=False)
            self._sh(f"git checkout -f {base}", check=False)
        self._sh("git clean -fdq", check=False)

    # -- scoring -----------------------------------------------------------
    def score(self, patch):
        """Apply `patch`, run the instance's tests, reset. Returns (resolved, evidence)."""
        from .datasets import sbp_restore_tests_cmd, sbp_test_files
        self.last_report, self.last_ratio = {}, None
        if not self.started:
            return False, _guard_evidence(
                f"SWE-bench Pro container unavailable: {self.setup_error}",
                "container_unavailable")
        guard = missing_patch_error(patch)
        if guard:
            return False, guard

        try:
            self.reset()
            self._write(f"{SBP_WORKSPACE}/patch.diff", patch)
            if not self._try_apply(f"{SBP_WORKSPACE}/patch.diff", reset=self.reset):
                # The apply log goes to the model. `git apply --verbose` prints
                # the failing file, the hunk number and the context block it
                # searched for, which is the difference between "try again" and
                # "this hunk expected these three lines and the file has these".
                return False, _guard_evidence(
                    "The patch did NOT apply to the repository -- no test was run.\n"
                    f"{len(_APPLY_STRATEGIES)} strategies were tried, strictest first.\n"
                    "Apply log:\n" + (self.apply_evidence() or "(no output)") +
                    "\n\nRe-emit the COMPLETE diff against the files as they exist at "
                    "this commit. Prefer fewer, larger hunks with exact context lines.",
                    "apply_failed")

            # The anti-cheat, and the step whose silent failure would make every
            # arm's number meaningless: the graded tests are restored from the
            # solution commit *after* the candidate patch, so a patch that
            # edited them gains nothing.
            restore = sbp_restore_tests_cmd(self.problem)
            if restore:
                r = self._sh(restore, check=False)
                if r.returncode != 0:
                    return False, _guard_evidence(
                        "could not restore the graded test files "
                        f"(`{restore[:80]}`): "
                        f"{((r.stderr or '') + (r.stdout or '')).strip()[-200:]}",
                        "restore_failed")

            files = sbp_test_files(self.problem)
            if not files:
                return False, _guard_evidence(
                    "row names no `selected_test_files_to_run`", "row_no_test_files")
            script = sbp_test_script(files, workdir=self.workdir,
                                     env_exports=(self.scripts or {}).get("env_exports", ""))
            self._write(f"{SBP_WORKSPACE}/testscript.sh", script)
            run = sj.contained_run(
                ["docker", "exec", "-w", self.workdir, self.name,
                 "bash", f"{SBP_WORKSPACE}/testscript.sh"],
                cwd=self.sandbox, timeout=self.timeout,
                # The container name carries a pid; recording it would mint a
                # different digest handle for every identical failure.
                record_argv=["bash", "run_script.sh"] + list(files))
            evidence = _from_run(run)

            report = sbp_resolution(self._read_output(), self.problem)
            self.last_report = report
            self.last_ratio = report["test_pass_ratio"]
            return report["resolved"], evidence
        except Exception as e:                               # noqa: BLE001
            return False, _guard_evidence(
                f"SWE-bench Pro execution error: {e}", "execution_error")
        finally:
            try:
                self.reset()
            except Exception:                                # noqa: BLE001
                pass

    # -- repository grounding ---------------------------------------------
    #
    # Read-only views of the tree at `base_commit`. They exist so the solver
    # prompt can quote the files it is being asked to patch. Everything goes
    # through `git show <base>:<path>` rather than the worktree, so a previous
    # attempt's patch can never leak into the next attempt's prompt.

    def list_paths(self, limit=4000):
        """Every tracked path at the base commit."""
        base = self.problem.get("base_commit") or "HEAD"
        r = self._sh(f"git ls-tree -r --name-only {base}", check=False)
        if r.returncode != 0:
            return []
        return [p for p in (r.stdout or "").splitlines() if p][:limit]

    def grep_paths(self, terms, limit=40):
        """Paths at the base commit whose contents mention any of `terms`.

        Fixed-string, case-insensitive, filenames only. This is the cheap
        stand-in for the search an agent would do: it turns a symbol named in
        the issue text into the file that defines it.
        """
        base = self.problem.get("base_commit") or "HEAD"
        out, seen = [], set()
        for term in terms:
            term = str(term or "").strip()
            if len(term) < 3:
                continue
            quoted = term.replace("'", "'\\''")
            r = self._sh(f"git grep -l -I -i -F -e '{quoted}' {base} -- "
                         f"| head -{limit}", check=False)
            if r.returncode != 0:
                continue
            for line in (r.stdout or "").splitlines():
                # `git grep <rev>` prefixes every hit with `<rev>:`.
                path = line.split(":", 1)[1] if ":" in line else line
                if path and path not in seen:
                    seen.add(path)
                    out.append(path)
                if len(out) >= limit:
                    return out
        return out

    def read_source(self, paths, budget=None, per_file=None, max_files=None):
        """Quote files from the base commit, within a character budget.

        Returns ``(blocks, read, skipped)``: the rendered text per path, the
        paths that fit, and the paths that did not. The skipped list is
        returned rather than dropped so the prompt can say what it left out --
        an excerpt the model does not know is an excerpt is worse than none.
        """
        budget = SBP_GROUNDING_CHARS if budget is None else budget
        per_file = SBP_GROUNDING_FILE_CHARS if per_file is None else per_file
        max_files = SBP_GROUNDING_MAX_FILES if max_files is None else max_files
        base = self.problem.get("base_commit") or "HEAD"
        blocks, read, skipped, spent = [], [], [], 0
        for path in paths:
            if len(read) >= max_files or spent >= budget:
                skipped.append(path)
                continue
            quoted = str(path).replace("'", "'\\''")
            r = self._sh(f"git show '{base}:{quoted}'", check=False)
            if r.returncode != 0:
                skipped.append(path)
                continue
            text = r.stdout or ""
            room = min(per_file, budget - spent)
            clipped = len(text) > room
            body = text[:room]
            blocks.append(
                f"--- FILE: {path}"
                + (f"  [first {room} of {len(text)} chars]" if clipped else "")
                + f" ---\n{body}\n")
            read.append(path)
            spent += len(body)
        return blocks, read, skipped

    def _read_output(self):
        """The parser's verdict file. Absent means the parser never ran."""
        p = subprocess.run(
            ["docker", "exec", self.name, "cat", f"{SBP_WORKSPACE}/output.json"],
            capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            return {}
        try:
            return json.loads(p.stdout or "{}")
        except json.JSONDecodeError:
            return {}


def _sbp_default_platform():
    """`linux/amd64` on Apple Silicon: upstream publishes amd64 images only.

    Without this, Docker Desktop picks the host architecture, fails to find a
    matching manifest and reports it as a missing image.
    """
    try:
        return "linux/amd64" if platform.machine().lower() in ("arm64", "aarch64") else ""
    except Exception:                                        # noqa: BLE001
        return ""


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
