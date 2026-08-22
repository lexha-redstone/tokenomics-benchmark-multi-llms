# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Comprehensive Benchmark Architectures & Variant Registry for Multi-LLM Tokenomics.
Provides unified implementations of single-model baselines, multi-model cascades,
advisor-executor splits, and zero-cost Straitjacket context containment variants.
"""

from .config import (
    GEMINI_37_FLASH_ID, GEMINI_35_FLASH_LITE_ID,
    GEMINI_FLASH_ID, GEMINI_FLASH_LITE_ID,
    SONNET_ID, OPUS_5_ID, OPUS_48_ID, OPUS_ID,
    SOLVER_ROLE, ADVISOR_ROLE, EXECUTOR_ROLE, REPAIR_ROLE,
    SWEBENCH_SOLVER_ROLE, SWEBENCH_ADVISOR_ROLE, SWEBENCH_EXECUTOR_ROLE, SWEBENCH_REPAIR_ROLE,
    WEBDEV_SOLVER_ROLE, WEBDEV_ADVISOR_ROLE,
    RETRIEVAL_PROTOCOL_ROLE
)
from .client import dispatch_model
from .evaluator import (
    extract_code, run_bigcodebench, missing_code_error,
    extract_patch, run_swebench_pro_task, missing_patch_error,
    triage_error, triage_error_straitjacket,
    begin_containment, containment_report, record_evidence_sent,
    straitjacket_status
)
from . import straitjacket as sj

import functools
import re
import sys

# ==============================================================================
# --- ERROR TREATMENT: THE BENCHMARK'S INDEPENDENT VARIABLE ---
# ==============================================================================
#
# Every architecture below eventually has to put a failed test run in front of
# a model. HOW it does that is the thing this suite measures, so it is named
# rather than hard-coded:
#
#   "native"       raw test output, tail-truncated by the caller. $0, and the
#                  bytes stay resident in the transcript for every later turn.
#   "llm"          a cheap model rewrites the log into a short digest. Costs
#                  input + output tokens and a round trip.
#   "straitjacket" the harness already captured the run at its birth gate;
#                  the bounded, coverage-attested digest it produced is used
#                  as-is. $0, no round trip, and the omitted regions stay
#                  retrievable by address.
#
# An arm's registry label must match the treatment it actually applies.

TREATMENTS = ("native", "llm", "straitjacket")

_ZERO_USAGE = {"as_run_usd": 0.0, "input": 0, "output": 0, "total_tokens": 0}

# Prompt slot headings. The straitjacket wording is kept verbatim from the
# original arms so switching treatment is the only thing that changes.
_EVIDENCE_LABEL = {
    "native": "Raw Test Output",
    "llm": "LLM Triaged Error Digest",
    "straitjacket": "Straitjacket Triaged Error Digest",
}


def _treat_error(err, treatment="straitjacket", problem=None,
                 triage_model=GEMINI_35_FLASH_LITE_ID, is_swe=False):
    """Apply one error treatment. Returns ``(payload, usage, seconds)``.

    The native payload is NOT re-truncated here. ``Evidence`` already carries
    the uncontained path's tail-truncated payload (``SJ_RAW_CAP``); clipping it
    a second time with a different budget is what made the native arm measure
    as 37% better than its own baseline.
    """
    if treatment == "native":
        # Idempotent for Evidence (already capped at capture); enforces the
        # invariant for a plain string that never went through the harness.
        out = (sj.tail_to_cap(err), dict(_ZERO_USAGE), 0.0)
    elif treatment == "llm":
        out = triage_error(err, model_id=triage_model, is_swe=is_swe)
    elif treatment == "straitjacket":
        out = triage_error_straitjacket(err, problem=problem)
    else:
        raise ValueError(
            f"unknown error treatment {treatment!r}; expected one of {TREATMENTS}")
    record_evidence_sent(err, out[0], treatment)
    return out


def _arm(sj_required=False):
    """Wrap an architecture: per-task containment ledger + harness preflight.

    ``sj_required`` arms refuse to start when the harness is missing. A row
    labelled "Straitjacket ($0.00)" that was actually produced without the
    harness is worse than no row at all.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(problem, *args, **kwargs):
            if sj_required or kwargs.get("error_treatment") == "straitjacket":
                sj.require()
            begin_containment()
            out = fn(problem, *args, **kwargs)
            if isinstance(out, dict):
                c = containment_report()
                out["containment"] = c
                _check_instrumented(fn.__name__, out, c)
            return out
        wrapper.sj_required = sj_required
        return wrapper
    return deco


def _check_instrumented(name, out, containment):
    """An arm that repaired something must have recorded the treatment.

    Measured failure this catches: the six ``*_straitjacket`` arms called
    ``triage_error_straitjacket`` directly instead of going through
    :func:`_treat_error`, so ``record_evidence_sent`` never fired. The digests
    were real — the harness ran, ``unittest/v1`` was detected, the artifacts
    were stored — but the containment receipt was empty for exactly the arms
    the report is named after, and an N=100 sweep published a table reading
    "Treatment: n/a · Sent: 0 · Δ: +0" for all four of them.

    A missing measurement is not a cosmetic problem when the measurement IS
    the deliverable, so it fails loudly here instead of reaching a report.
    """
    if out.get("repair_loops", 0) > 0 and containment.get("treatment_events", 0) == 0:
        _warn_once(
            f"{name}: performed {out['repair_loops']} repair loop(s) but recorded no "
            "evidence treatment. The arm is bypassing _treat_error(), so its "
            "containment receipt will be blank. Route the repair turn through "
            '_treat_error(err, "straitjacket", problem=problem).'
        )
        out["containment_instrumentation"] = "MISSING"


_warned_arms: set = set()


def _warn_once(message):
    if message in _warned_arms:
        return
    _warned_arms.add(message)
    print(message, file=sys.stderr)


# ==============================================================================
# --- PROBLEM EVALUATION DISPATCHER HELPER ---
# ==============================================================================

def _is_swebench_problem(problem):
    return problem.get("dataset_type") == "swebench" or "instance_id" in problem or "base_commit" in problem

def _build_initial_prompt(problem, role_type="solver"):
    is_swe = _is_swebench_problem(problem)
    if is_swe:
        if role_type == "advisor":
            return (
                SWEBENCH_ADVISOR_ROLE +
                f"Repository: {problem.get('repo', 'enterprise/repo')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Code Context:\n{problem.get('code_context', '')}"
            )
        else:
            return (
                SWEBENCH_SOLVER_ROLE +
                f"Repository: {problem.get('repo', 'enterprise/repo')}\n"
                f"Base Commit: {problem.get('base_commit', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Code Context:\n{problem.get('code_context', '')}\n\n"
                "Generate the COMPLETE unified git patch/diff."
            )
    else:
        # Python function completion (BCB / WebDev)
        prompt_text = problem.get("complete_prompt", "")
        if role_type == "advisor":
            return ADVISOR_ROLE + f"Problem:\n```python\n{prompt_text}\n```"
        else:
            return SOLVER_ROLE + f"Problem:\n```python\n{prompt_text}\n```\n\nWrite the complete solution."

def _eval_solution(problem, text):
    is_swe = _is_swebench_problem(problem)
    if is_swe:
        patch = extract_patch(text)
        passed, err = run_swebench_pro_task(problem, patch)
        return passed, err, patch
    else:
        code = extract_code(text)
        guard = missing_code_error(code, problem.get("entry_point", "task_func"))
        if guard:
            return False, guard, code
        passed, err = run_bigcodebench(problem, code)
        return passed, err, code

# ==============================================================================
# --- CATEGORY 1: SINGLE MODEL BASELINES ---
# ==============================================================================

@_arm()
def run_single(problem, model_id=GEMINI_37_FLASH_ID, thinking_level=None, max_loops=1,
               error_treatment="straitjacket"):
    """Direct single-shot code/patch generation with optional multi-turn self-repair.

    ``error_treatment`` defaults to ``"straitjacket"``: the repair turn is fed
    the contained digest of the failing run. Set it to ``"native"`` for a
    genuinely untreated single-model baseline.
    """
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, dt = dispatch_model(model_id, prompt, thinking_level=thinking_level, problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    tot_dt = dt
    loop = 0
    triage_usd = 0.0

    while not passed and loop < max_loops:
        loop += 1
        is_swe = _is_swebench_problem(problem)
        digest, tr_usage, _ = _treat_error(err, error_treatment, problem=problem, is_swe=is_swe)
        tot_usd += tr_usage["as_run_usd"]
        triage_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]
        label = _EVIDENCE_LABEL[error_treatment]

        if is_swe:
            repair_prompt = (
                SWEBENCH_REPAIR_ROLE +
                f"Repository: {problem.get('repo', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Current candidate patch:\n```diff\n{sol}\n```\n\n"
                f"{label}:\n```\n{digest}\n```\n\n"
                "Output COMPLETE corrected unified git patch/diff."
            )
        else:
            repair_prompt = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
                f"Current solution:\n```python\n{sol}\n```\n\n"
                f"{label}:\n```\n{digest}\n```\n\n"
                "Write the complete corrected solution."
            )

        r_text, r_usage, r_dt = dispatch_model(model_id, repair_prompt, thinking_level=thinking_level, problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        tot_dt += r_dt
        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(tot_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": round(triage_usd, 6),
        "patch": sol,
    }


# ==============================================================================
# --- BOUNDED RETRIEVAL: RETURNING TO THE MODEL AT AN UNCERTAINTY BOUNDARY ---
# ==============================================================================

_CTX_REQUEST_RE = re.compile(r"^\s*CTX:\s*(?P<cmd>ctx\s+(?:get|search)\s+.+?)\s*$",
                             re.MULTILINE | re.IGNORECASE)
_CTX_VERB_RE = re.compile(r"\bctx\s+(get|search)\b", re.IGNORECASE)
_CTX_LINES_RE = re.compile(r"--lines\s+(\d+)\s*:\s*(\d+)")
_CTX_STREAM_RE = re.compile(r"#(stdout|stderr)\b", re.IGNORECASE)
_CTX_PATTERN_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")
_CTX_CONTEXT_RE = re.compile(r"--context\s+(\d+)")

# One lookup, and the lookup itself is bounded. Retrieval must not become the
# second flood -- that is the failure mode the harness exists to prevent, and
# an arm that re-pastes the whole stream has simply moved the flood one turn
# later.
RETRIEVAL_MAX_LINES = 80
RETRIEVAL_MAX_CONTEXT = 6


def parse_retrieval_request(text):
    """Extract a single ``CTX: ctx <verb> ...`` line from a model reply.

    Returns the command string, or ``None`` when the reply is an ordinary
    answer. Only ``get`` and ``search`` are recognised; anything else is
    treated as "no request", never executed.
    """
    m = _CTX_REQUEST_RE.search(text or "")
    if not m:
        return None
    # A reply that also contains a code block is an answer, not a request.
    if "```" in (text or ""):
        return None
    return m.group("cmd").strip()


def serve_retrieval(evidence, command):
    """Serve one bounded retrieval against the frozen artifact.

    Runs locally against the stored blob: no model call, no re-execution, and
    the span is clamped so a request for "everything" cannot re-flood. The
    handle the model typed is deliberately ignored — retrieval is always
    served against *this* failure's artifact, so a hallucinated or copied
    handle can never address someone else's run.

    Returns ``(rendered_text, served_command)`` or ``(None, reason)``.
    """
    run = getattr(evidence, "run", None)
    if run is None:
        return None, "no addressable artifact for this failure"

    verb_m = _CTX_VERB_RE.search(command or "")
    if not verb_m:
        return None, f"unsupported retrieval verb: {str(command)[:80]}"
    verb = verb_m.group(1).lower()

    if verb == "get":
        span = _CTX_LINES_RE.search(command)
        if not span:
            return None, "ctx get needs a --lines A:B span"
        a, b = int(span.group(1)), int(span.group(2))
        if b < a:
            a, b = b, a
        a = max(1, a)
        b = min(b, a + RETRIEVAL_MAX_LINES - 1)
        sm = _CTX_STREAM_RE.search(command)
        stream = sm.group(1).lower() if sm else "stderr"
        try:
            return run.get(stream, (a, b)), f"ctx get {run.handle}#{stream} --lines {a}:{b}"
        except Exception as e:
            return None, f"retrieval refused: {type(e).__name__}: {e}"

    pm = _CTX_PATTERN_RE.search(command)
    if pm:
        pattern = pm.group(1) or pm.group(2)
    else:
        # Bare token after the handle, e.g. `ctx search run:abc Traceback`
        tail = command[verb_m.end():].split()
        tail = [t for t in tail if not t.startswith("run:") and not t.startswith("--")]
        pattern = tail[0] if tail else ""
    if not pattern:
        return None, "ctx search needs a pattern"
    cm = _CTX_CONTEXT_RE.search(command)
    ctx_n = min(int(cm.group(1)) if cm else 0, RETRIEVAL_MAX_CONTEXT)
    try:
        return (run.search([pattern], context=ctx_n),
                f"ctx search {run.handle} '{pattern}' --context {ctx_n}")
    except Exception as e:
        return None, f"retrieval refused: {type(e).__name__}: {e}"


@_arm(sj_required=True)
def run_contained_retrieval_cascade(problem, gen_model=GEMINI_35_FLASH_LITE_ID,
                                    esc_model=GEMINI_37_FLASH_ID, max_repairs=2,
                                    escalate_after=1, allow_retrieval=True):
    """Cascade that exercises BOTH halves of context containment.

    Each repair turn starts from the bounded digest. The model may spend one
    zero-cost, bounded lookup against the frozen artifact when the digest is
    genuinely not enough — the "return to the model at uncertainty boundaries"
    rule, rather than pasting the whole log back in on the chance it helps.

    Compare against ``run_cascade`` (native raw output) and
    ``run_cascade_straitjacket`` (digest only, no retrieval) on the same
    models: the three differ only in how failure evidence is delivered.
    """
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, dt = dispatch_model(gen_model, prompt, problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    tot_dt = dt
    loop = 0
    retrievals = []

    while not passed and loop < max_repairs:
        loop += 1
        escalated = loop >= escalate_after
        target_model = esc_model if escalated else gen_model
        think_level = "low" if (escalated and target_model in (GEMINI_37_FLASH_ID, GEMINI_FLASH_ID)) else None

        digest, tr_usage, _ = _treat_error(err, "straitjacket", problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        statement = problem.get("problem_statement", problem.get("complete_prompt", ""))
        base = (
            role
            + (RETRIEVAL_PROTOCOL_ROLE if allow_retrieval else "")
            + f"Problem:\n```\n{statement}\n```\n\n"
            + f"Current solution:\n```\n{sol}\n```\n\n"
            + f"Straitjacket Contained Digest:\n```\n{digest}\n```\n\n"
        )

        r_text, r_usage, r_dt = dispatch_model(target_model, base, thinking_level=think_level, problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        tot_dt += r_dt

        request = parse_retrieval_request(r_text) if allow_retrieval else None
        if request:
            served, note = serve_retrieval(err, request)
            retrievals.append({
                "loop": loop,
                "requested": request,
                "served": note,
                "ok": served is not None,
                "response_tokens_est": sj.estimate_tokens(len(served.encode("utf-8"))) if served else 0,
            })
            follow = base + (
                f"Retrieved region ({note}):\n```\n{served}\n```\n\n"
                if served else
                f"Retrieval unavailable ({note}). Work from the digest.\n\n"
            ) + "Now write the complete corrected solution."
            r_text, r2_usage, r2_dt = dispatch_model(target_model, follow,
                                                     thinking_level=think_level, problem=problem)
            tot_usd += r2_usage["as_run_usd"]
            tot_out += r2_usage["output"]
            tot_tok += r2_usage["total_tokens"]
            tot_dt += r2_dt

        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(tot_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "retrievals": retrievals,
        "patch": sol,
    }

# ==============================================================================
# --- CATEGORY 2: ADVISOR-EXECUTOR & MULTI-TIER CASCADES ---
# ==============================================================================

@_arm()
def run_read_write(problem, planner_model=GEMINI_37_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID,
                   error_treatment="straitjacket"):
    """Read-Heavy Advisor (Planner) + Write-Heavy Executor Split.

    The repair turn is fed the contained digest by default — this arm has
    always used straitjacket containment, and its registry label now says so.
    """
    adv_prompt = _build_initial_prompt(problem, role_type="advisor")
    guidance, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024, problem=problem)

    is_swe = _is_swebench_problem(problem)
    if is_swe:
        exec_prompt = (
            SWEBENCH_EXECUTOR_ROLE +
            f"Repository: {problem.get('repo', '')}\n"
            f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
            f"Code Context:\n{problem.get('code_context', '')}\n\n"
            f"Software Architect Contract Guidance:\n{guidance}\n\n"
            "Generate the COMPLETE unified git patch/diff."
        )
    else:
        exec_prompt = (
            EXECUTOR_ROLE +
            f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
            f"Advisor guidance:\n{guidance}\n\nWrite the complete solution."
        )

    sol_text, exec_usage, exec_dt = dispatch_model(executor_model, exec_prompt, max_tokens=2560, problem=problem)
    passed, err, sol = _eval_solution(problem, sol_text)

    tot_usd = round(adv_usage["as_run_usd"] + exec_usage["as_run_usd"], 6)
    tot_out = adv_usage["output"] + exec_usage["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage["total_tokens"]
    tot_dt = adv_dt + exec_dt
    loop = 0
    triage_usd = 0.0

    if not passed:
        loop = 1
        digest, tr_usage, _ = _treat_error(err, error_treatment, problem=problem, is_swe=is_swe)
        tot_usd += tr_usage["as_run_usd"]
        triage_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        if is_swe:
            repair_prompt = (
                SWEBENCH_REPAIR_ROLE +
                f"Repository: {problem.get('repo', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Contract:\n{guidance}\n\n"
                f"Current candidate patch:\n```diff\n{sol}\n```\n\n"
                f"{_EVIDENCE_LABEL[error_treatment]}:\n```\n{digest}\n```\n\n"
                "Output COMPLETE corrected unified git patch/diff."
            )
        else:
            repair_prompt = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
                f"Advisor Guidance:\n{guidance}\n\n"
                f"Current solution:\n```python\n{sol}\n```\n\n"
                f"Unit test error:\n```\n{digest}\n```\n\n"
                "Write the complete corrected solution."
            )

        r_text, r_usage, r_dt = dispatch_model(planner_model, repair_prompt, thinking_level="low", problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        tot_dt += r_dt
        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(tot_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": round(triage_usd, 6),
        "patch": sol,
    }

@_arm()
def run_cascade(problem, gen_model=GEMINI_35_FLASH_LITE_ID, esc_model=GEMINI_37_FLASH_ID,
                max_repairs=2, escalate_after=1, error_treatment="native"):
    """Two-tier generation cascade. Uncontained baseline: the raw test output
    goes straight into the repair prompt, tail-truncated at 2,500 chars."""
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, dt = dispatch_model(gen_model, prompt, problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    tot_dt = dt
    loop = 0
    triage_usd = 0.0

    while not passed and loop < max_repairs:
        loop += 1
        escalated = loop >= escalate_after
        target_model = esc_model if escalated else gen_model
        think_level = "low" if (escalated and target_model in (GEMINI_37_FLASH_ID, GEMINI_FLASH_ID)) else None

        is_swe = _is_swebench_problem(problem)
        payload, tr_usage, _ = _treat_error(err, error_treatment, problem=problem,
                                            is_swe=is_swe)
        tot_usd += tr_usage["as_run_usd"]
        triage_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        if is_swe:
            repair_prompt = (
                SWEBENCH_REPAIR_ROLE +
                f"Repository: {problem.get('repo', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Current candidate patch:\n```diff\n{sol}\n```\n\n"
                f"{_EVIDENCE_LABEL[error_treatment]}:\n```\n{payload}\n```\n\n"
                "Output COMPLETE corrected unified git patch/diff."
            )
        else:
            repair_prompt = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
                f"Current solution:\n```python\n{sol}\n```\n\n"
                f"Unit test error:\n```\n{payload}\n```\n\n"
                "Write the complete corrected solution."
            )

        r_text, r_usage, r_dt = dispatch_model(target_model, repair_prompt, thinking_level=think_level, problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        tot_dt += r_dt
        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(tot_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": round(triage_usd, 6),
        "patch": sol,
    }

@_arm()
def run_hybrid(problem, planner_model=GEMINI_37_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID,
               escalate_model=GEMINI_37_FLASH_ID, triage_model=GEMINI_35_FLASH_LITE_ID,
               error_treatment="llm"):
    """Sweet Spot Hybrid with standard LLM Triage (the paid comparison arm)."""
    adv_prompt = _build_initial_prompt(problem, role_type="advisor")
    guidance, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024, problem=problem)

    is_swe = _is_swebench_problem(problem)
    if is_swe:
        exec_prompt = (
            SWEBENCH_EXECUTOR_ROLE +
            f"Repository: {problem.get('repo', '')}\n"
            f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
            f"Contract Guidance:\n{guidance}\n\nGenerate COMPLETE unified git patch/diff."
        )
    else:
        exec_prompt = (
            EXECUTOR_ROLE +
            f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
            f"Advisor guidance:\n{guidance}\n\nWrite the complete solution."
        )

    sol_text, exec_usage, exec_dt = dispatch_model(executor_model, exec_prompt, max_tokens=2560, problem=problem)
    passed, err, sol = _eval_solution(problem, sol_text)

    tot_usd = adv_usage["as_run_usd"] + exec_usage["as_run_usd"]
    tot_out = adv_usage["output"] + exec_usage["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage["total_tokens"]
    triage_usd = 0.0
    loop = 0

    if not passed:
        loop = 1
        digest, tr_usage, _ = _treat_error(err, error_treatment, problem=problem,
                                           triage_model=triage_model, is_swe=is_swe)
        tot_usd += tr_usage["as_run_usd"]
        triage_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        if is_swe:
            repair_prompt = (
                SWEBENCH_REPAIR_ROLE +
                f"Repository: {problem.get('repo', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Contract:\n{guidance}\n\n"
                f"Current candidate patch:\n```diff\n{sol}\n```\n\n"
                f"{_EVIDENCE_LABEL[error_treatment]}:\n```\n{digest}\n```\n\n"
                "Output COMPLETE corrected unified git patch/diff."
            )
        else:
            repair_prompt = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
                f"Current solution:\n```python\n{sol}\n```\n\n"
                f"{_EVIDENCE_LABEL[error_treatment]}:\n```\n{digest}\n```\n\n"
                "Write the complete corrected solution."
            )

        esc_text, esc_usage, _ = dispatch_model(escalate_model, repair_prompt, thinking_level="low", problem=problem)
        tot_usd += esc_usage["as_run_usd"]
        tot_out += esc_usage["output"]
        tot_tok += esc_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, esc_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(adv_dt + exec_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": round(triage_usd, 6),
        "patch": sol,
    }

# ==============================================================================
# --- CATEGORY 3: STRAITJACKET ZERO-COST LOCAL TRIAGE VARIANTS ---
# ==============================================================================

@_arm(sj_required=True)
def run_hybrid_straitjacket(problem, planner_model=GEMINI_37_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID,
                            escalate_model=GEMINI_37_FLASH_ID):
    """Architecture E-SJ: Sweet Spot Hybrid fed the straitjacket contained digest ($0.00)."""
    adv_prompt = _build_initial_prompt(problem, role_type="advisor")
    guidance, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024, problem=problem)

    is_swe = _is_swebench_problem(problem)
    if is_swe:
        exec_prompt = (
            SWEBENCH_EXECUTOR_ROLE +
            f"Repository: {problem.get('repo', '')}\n"
            f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
            f"Contract Guidance:\n{guidance}\n\nGenerate COMPLETE unified git patch/diff."
        )
    else:
        exec_prompt = (
            EXECUTOR_ROLE +
            f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
            f"Advisor guidance:\n{guidance}\n\nWrite the complete solution."
        )

    sol_text, exec_usage, exec_dt = dispatch_model(executor_model, exec_prompt, max_tokens=2560, problem=problem)
    passed, err, sol = _eval_solution(problem, sol_text)

    tot_usd = adv_usage["as_run_usd"] + exec_usage["as_run_usd"]
    tot_out = adv_usage["output"] + exec_usage["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage["total_tokens"]
    loop = 0

    if not passed:
        loop = 1
        digest, tr_usage, _ = _treat_error(err, "straitjacket", problem=problem)
        tot_usd += tr_usage["as_run_usd"]  # $0.000000
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        if is_swe:
            repair_prompt = (
                SWEBENCH_REPAIR_ROLE +
                f"Repository: {problem.get('repo', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Contract:\n{guidance}\n\n"
                f"Current candidate patch:\n```diff\n{sol}\n```\n\n"
                f"Straitjacket Zero-Cost Triaged Digest:\n```\n{digest}\n```\n\n"
                "Output COMPLETE corrected unified git patch/diff."
            )
        else:
            repair_prompt = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
                f"Current solution:\n```python\n{sol}\n```\n\n"
                f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
                "Write the complete corrected solution."
            )

        esc_text, esc_usage, _ = dispatch_model(escalate_model, repair_prompt, thinking_level="low", problem=problem)
        tot_usd += esc_usage["as_run_usd"]
        tot_out += esc_usage["output"]
        tot_tok += esc_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, esc_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(adv_dt + exec_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

@_arm(sj_required=True)
def run_cascade_straitjacket(problem, gen_model=GEMINI_35_FLASH_LITE_ID, esc_model=GEMINI_37_FLASH_ID, max_repairs=2, escalate_after=1):
    """Architecture C-SJ: Generation Cascade with Straitjacket Local Triage & Cache Warming."""
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, dt = dispatch_model(gen_model, prompt, problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    tot_dt = dt
    loop = 0

    while not passed and loop < max_repairs:
        loop += 1
        escalated = loop >= escalate_after
        target_model = esc_model if escalated else gen_model
        think_level = "low" if (escalated and target_model in (GEMINI_37_FLASH_ID, GEMINI_FLASH_ID)) else None

        digest, tr_usage, _ = _treat_error(err, "straitjacket", problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        if is_swe:
            repair_prompt = (
                SWEBENCH_REPAIR_ROLE +
                f"Repository: {problem.get('repo', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Current candidate patch:\n```diff\n{sol}\n```\n\n"
                f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
                "Output COMPLETE corrected unified git patch/diff."
            )
        else:
            repair_prompt = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
                f"Current solution:\n```python\n{sol}\n```\n\n"
                f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
                "Write the complete corrected solution."
            )

        r_text, r_usage, r_dt = dispatch_model(target_model, repair_prompt, thinking_level=think_level, problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        tot_dt += r_dt
        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(tot_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

@_arm(sj_required=True)
def run_escalation_shield_straitjacket(problem, lite_model=GEMINI_35_FLASH_LITE_ID, flash_model=GEMINI_37_FLASH_ID, sonnet_model=SONNET_ID):
    """
    Straitjacket Escalation Shield (Gemini 3.5-Lite -> Gemini 3.6-Flash -> Claude Sonnet-5).
    Repair turns are fed the harness's contained digest ($0.00) and reuse the warm prompt prefix.
    """
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, _ = dispatch_model(lite_model, prompt, problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    loop = 0

    if not passed:
        loop = 1
        digest, tr_usage, _ = _treat_error(err, "straitjacket", problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        r1_prompt = (
            role + f"Problem:\n```\n{problem.get('problem_statement', problem.get('complete_prompt', ''))}\n```\n\n"
            f"Current candidate:\n```\n{sol}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
            "Write the complete corrected solution."
        )
        r1_text, r1_usage, _ = dispatch_model(flash_model, r1_prompt, thinking_level="low", problem=problem)
        tot_usd += r1_usage["as_run_usd"]
        tot_out += r1_usage["output"]
        tot_tok += r1_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, r1_text)

    if not passed:
        loop = 2
        digest, tr_usage, _ = _treat_error(err, "straitjacket", problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        r2_prompt = (
            role + f"Problem:\n```\n{problem.get('problem_statement', problem.get('complete_prompt', ''))}\n```\n\n"
            f"Current candidate:\n```\n{sol}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
            "Write the complete corrected solution."
        )
        r2_text, r2_usage, _ = dispatch_model(sonnet_model, r2_prompt, problem=problem)
        tot_usd += r2_usage["as_run_usd"]
        tot_out += r2_usage["output"]
        tot_tok += r2_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, r2_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": 5.0,
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

@_arm(sj_required=True)
def run_smart_repair_straitjacket(problem, flash_model=GEMINI_37_FLASH_ID, lite_model=GEMINI_35_FLASH_LITE_ID):
    """
    Straitjacket Smart Repair (Pure Gemini: 3.6-Flash -> 3.5-Lite -> 3.6-Flash Medium).
    Contained digest ($0.00) in a native Google Cloud stack.
    """
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, _ = dispatch_model(flash_model, prompt, thinking_level="low", problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    loop = 0

    if not passed:
        loop = 1
        digest, tr_usage, _ = _treat_error(err, "straitjacket", problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        r1_prompt = (
            role + f"Problem:\n```\n{problem.get('problem_statement', problem.get('complete_prompt', ''))}\n```\n\n"
            f"Current candidate:\n```\n{sol}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
            "Write the complete corrected solution."
        )
        r1_text, r1_usage, _ = dispatch_model(lite_model, r1_prompt, problem=problem)
        tot_usd += r1_usage["as_run_usd"]
        tot_out += r1_usage["output"]
        tot_tok += r1_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, r1_text)

    if not passed:
        loop = 2
        digest, tr_usage, _ = _treat_error(err, "straitjacket", problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        r2_prompt = (
            role + f"Problem:\n```\n{problem.get('problem_statement', problem.get('complete_prompt', ''))}\n```\n\n"
            f"Current candidate:\n```\n{sol}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
            "Write the complete corrected solution."
        )
        r2_text, r2_usage, _ = dispatch_model(flash_model, r2_prompt, thinking_level="medium", problem=problem)
        tot_usd += r2_usage["as_run_usd"]
        tot_out += r2_usage["output"]
        tot_tok += r2_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, r2_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": 5.0,
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

@_arm(sj_required=True)
def run_ultra_sweet_straitjacket(problem, sonnet_model=SONNET_ID, lite_model=GEMINI_35_FLASH_LITE_ID, opus_model=OPUS_ID):
    """
    Straitjacket Ultra-Sweet Hybrid (Claude Sonnet-5 Advisor -> Gemini Lite Executor -> Claude Opus-5 Repair).
    Contract-guided cross-vendor synthesis with zero-cost local triage.
    """
    adv_prompt = _build_initial_prompt(problem, role_type="advisor")
    guidance, adv_usage, _ = dispatch_model(sonnet_model, adv_prompt, max_tokens=1024, problem=problem)
    tot_usd = adv_usage["as_run_usd"]
    tot_out = adv_usage["output"]
    tot_tok = adv_usage["total_tokens"]

    is_swe = _is_swebench_problem(problem)
    if is_swe:
        exec_prompt = (
            SWEBENCH_EXECUTOR_ROLE +
            f"Repository: {problem.get('repo', '')}\n"
            f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
            f"Contract Specification:\n{guidance}\n\nGenerate COMPLETE unified git patch/diff."
        )
    else:
        exec_prompt = (
            EXECUTOR_ROLE +
            f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
            f"Contract Specification:\n{guidance}\n\nWrite complete solution."
        )

    sol_text, exec_usage, _ = dispatch_model(lite_model, exec_prompt, problem=problem)
    tot_usd += exec_usage["as_run_usd"]
    tot_out += exec_usage["output"]
    tot_tok += exec_usage["total_tokens"]
    passed, err, sol = _eval_solution(problem, sol_text)

    loop = 0
    if not passed:
        loop = 1
        digest, tr_usage, _ = _treat_error(err, "straitjacket", problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        repair_prompt = (
            role + f"Problem:\n```\n{problem.get('problem_statement', problem.get('complete_prompt', ''))}\n```\n\n"
            f"Contract:\n{guidance}\n\n"
            f"Current candidate:\n```\n{sol}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
            "Write the complete corrected solution."
        )
        r_text, r_usage, _ = dispatch_model(opus_model, repair_prompt, problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": 6.0,
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

# ==============================================================================
# --- CATEGORY 4: NEXT-GEN MULTI-PROVIDER ARCHITECTURES + STRAITJACKET ---
# ==============================================================================

@_arm(sj_required=True)
def run_dual_verifier_cascade_straitjacket(problem, lite_model=GEMINI_35_FLASH_LITE_ID, flash_model=GEMINI_37_FLASH_ID,
                                          sonnet_model=SONNET_ID, opus_model=OPUS_ID):
    """4-Tier Multi-Provider Dual-Verifier Cascade fed the straitjacket contained digest ($0.00)."""
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, _ = dispatch_model(lite_model, prompt, problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    loop = 0

    tiers = [
        (flash_model, "low"),
        (sonnet_model, None),
        (opus_model, None)
    ]

    for target_model, think_level in tiers:
        if passed:
            break
        loop += 1
        digest, tr_usage, _ = _treat_error(err, "straitjacket", problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        r_prompt = (
            role + f"Problem:\n```\n{problem.get('problem_statement', problem.get('complete_prompt', ''))}\n```\n\n"
            f"Current candidate:\n```\n{sol}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
            "Write the complete corrected solution."
        )
        r_text, r_usage, _ = dispatch_model(target_model, r_prompt, thinking_level=think_level, problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": 7.0,
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

# ==============================================================================
# --- VARIANT REGISTRY ---
# ==============================================================================

VARIANT_REGISTRY = {
    # --- SINGLE MODELS ---
    "single_flash_lite": {
        "id": "single_flash_lite",
        "name": "1. Single: gemini-3.5-flash-lite",
        "category": "1. Single models",
        "triage_mode": "Straitjacket digest ($0.00, repair turn)",
        "models": "Gemini 3.5 Flash-Lite",
        "fn": lambda p: run_single(p, model_id=GEMINI_35_FLASH_LITE_ID),
    },
    "single_flash37": {
        "id": "single_flash37",
        "name": "2. Single: gemini-3.7-flash",
        "category": "1. Single models",
        "triage_mode": "Straitjacket digest ($0.00, repair turn)",
        "models": "Gemini 3.7 Flash",
        "fn": lambda p: run_single(p, model_id=GEMINI_37_FLASH_ID, thinking_level="low"),
    },
    "single_sonnet5": {
        "id": "single_sonnet5",
        "name": "3. Single: claude-sonnet-5",
        "category": "1. Single models",
        "triage_mode": "Straitjacket digest ($0.00, repair turn)",
        "models": "Claude Sonnet-5",
        "fn": lambda p: run_single(p, model_id=SONNET_ID),
    },
    "single_opus5": {
        "id": "single_opus5",
        "name": "4. Single: claude-opus-5",
        "category": "1. Single models",
        "triage_mode": "Straitjacket digest ($0.00, repair turn)",
        "models": "Claude Opus-5",
        "fn": lambda p: run_single(p, model_id=OPUS_5_ID),
    },
    
    # --- COMBINATIONS (NO STRAITJACKET) ---
    "combo_read_write": {
        "id": "combo_read_write",
        "name": "5. Read/Write: 3.7-Flash Plan + 3.5-Lite Exec",
        "category": "2. Combination of models",
        "triage_mode": "Straitjacket digest ($0.00, repair turn)",
        "models": "Gemini 3.7 Flash Plan + 3.5 Lite Exec",
        "fn": lambda p: run_read_write(p, planner_model=GEMINI_37_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID),
    },
    "combo_cascade_llm": {
        "id": "combo_cascade_llm",
        "name": "6. Cascade Baseline (Gemini 3-Tier Raw Stderr)",
        "category": "2. Combination of models",
        "triage_mode": "Native raw output (uncontained, $0.00)",
        "models": "Gemini 3.5 Lite -> 3.7 Flash",
        "fn": lambda p: run_cascade(p, gen_model=GEMINI_35_FLASH_LITE_ID, esc_model=GEMINI_37_FLASH_ID),
    },
    "combo_hybrid_llm": {
        "id": "combo_hybrid_llm",
        "name": "7. Escalation Shield LLM Triage (Gemini -> Claude)",
        "category": "2. Combination of models",
        "triage_mode": "LLM triage ($ per repair)",
        "models": "Gemini Lite -> Flash -> Claude Sonnet-5",
        "fn": lambda p: run_hybrid(p, planner_model=GEMINI_37_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID, escalate_model=SONNET_ID),
    },
    
    # --- COMBINATIONS + STRAITJACKET ZERO-COST TRIAGE ---
    "sj_cascade": {
        "id": "sj_cascade",
        "name": "8. Straitjacket Cascade (3.5-Lite -> 3.7-Flash)",
        "category": "3. Combination of models + straitjacket",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "models": "Gemini 3.5 Lite -> 3.7 Flash",
        "fn": lambda p: run_cascade_straitjacket(p, gen_model=GEMINI_35_FLASH_LITE_ID, esc_model=GEMINI_37_FLASH_ID),
    },
    "sj_hybrid": {
        "id": "sj_hybrid",
        "name": "9. Straitjacket Hybrid (Flash Plan + Lite Exec + Flash Repair)",
        "category": "3. Combination of models + straitjacket",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "models": "Gemini 3.7 Flash + 3.5 Lite",
        "fn": lambda p: run_hybrid_straitjacket(p, planner_model=GEMINI_37_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID, escalate_model=GEMINI_37_FLASH_ID),
    },
    "sj_escalation_shield": {
        "id": "sj_escalation_shield",
        "name": "10. Straitjacket Escalation Shield (Gemini -> Claude)",
        "category": "3. Combination of models + straitjacket",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "models": "Gemini Lite -> Flash -> Claude Sonnet-5",
        "fn": lambda p: run_escalation_shield_straitjacket(p, lite_model=GEMINI_35_FLASH_LITE_ID, flash_model=GEMINI_37_FLASH_ID, sonnet_model=SONNET_ID),
    },
    "sj_smart_repair": {
        "id": "sj_smart_repair",
        "name": "11. Straitjacket Smart Repair (Pure Gemini)",
        "category": "3. Combination of models + straitjacket",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "models": "Gemini 3.7 Flash -> 3.5 Lite -> Flash (Med)",
        "fn": lambda p: run_smart_repair_straitjacket(p, flash_model=GEMINI_37_FLASH_ID, lite_model=GEMINI_35_FLASH_LITE_ID),
    },
    "sj_ultra_sweet": {
        "id": "sj_ultra_sweet",
        "name": "12. Straitjacket Ultra-Sweet Hybrid (Claude + Gemini)",
        "category": "3. Combination of models + straitjacket",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "models": "Claude Sonnet-5 -> Gemini Lite -> Claude Opus-5",
        "fn": lambda p: run_ultra_sweet_straitjacket(p, sonnet_model=SONNET_ID, lite_model=GEMINI_35_FLASH_LITE_ID, opus_model=OPUS_5_ID),
    },
    # --- EVIDENCE-TREATMENT ABLATION -------------------------------------
    # Identical model pipeline, identical prompts, identical evaluation.
    # The ONLY difference is how the failing run's output reaches the model.
    # This triplet is what licenses any claim about containment.
    "ablate_cascade_native": {
        "id": "ablate_cascade_native",
        "name": "A1. Ablation: Cascade + native raw output",
        "category": "5. Evidence-treatment ablation",
        "triage_mode": "Native raw output (uncontained, $0.00)",
        "models": "Gemini 3.5 Lite -> 3.7 Flash",
        "fn": lambda p: run_cascade(p, gen_model=GEMINI_35_FLASH_LITE_ID,
                                    esc_model=GEMINI_37_FLASH_ID, error_treatment="native"),
    },
    "ablate_cascade_llm_triage": {
        "id": "ablate_cascade_llm_triage",
        "name": "A2. Ablation: Cascade + LLM triage",
        "category": "5. Evidence-treatment ablation",
        "triage_mode": "LLM triage ($ per repair)",
        "models": "Gemini 3.5 Lite -> 3.7 Flash (+ Lite triage)",
        "fn": lambda p: run_cascade(p, gen_model=GEMINI_35_FLASH_LITE_ID,
                                    esc_model=GEMINI_37_FLASH_ID, error_treatment="llm"),
    },
    "ablate_cascade_straitjacket": {
        "id": "ablate_cascade_straitjacket",
        "name": "A3. Ablation: Cascade + straitjacket digest",
        "category": "5. Evidence-treatment ablation",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "models": "Gemini 3.5 Lite -> 3.7 Flash",
        "fn": lambda p: run_cascade(p, gen_model=GEMINI_35_FLASH_LITE_ID,
                                    esc_model=GEMINI_37_FLASH_ID, error_treatment="straitjacket"),
    },
    "sj_contained_retrieval": {
        "id": "sj_contained_retrieval",
        "name": "A4. Straitjacket digest + one bounded retrieval",
        "category": "5. Evidence-treatment ablation",
        "triage_mode": "Straitjacket digest + ctx get/search ($0.00)",
        "models": "Gemini 3.5 Lite -> 3.7 Flash",
        "fn": lambda p: run_contained_retrieval_cascade(
            p, gen_model=GEMINI_35_FLASH_LITE_ID, esc_model=GEMINI_37_FLASH_ID),
    },

    "sj_dual_verifier": {
        "id": "sj_dual_verifier",
        "name": "13. Straitjacket Dual-Verifier Cascade (4-Tier Synergy)",
        "category": "4. Next-Gen Multi-Provider + straitjacket",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "models": "Gemini Lite -> Flash -> Sonnet-5 -> Opus-5",
        "fn": lambda p: run_dual_verifier_cascade_straitjacket(p, lite_model=GEMINI_35_FLASH_LITE_ID, flash_model=GEMINI_37_FLASH_ID, sonnet_model=SONNET_ID, opus_model=OPUS_5_ID),
    },
}

def get_configurations(dataset="swebench", group="all", variant_keys=None):
    """
    Retrieve list of benchmark configurations matching dataset, group filter, or specific variant keys.
    """
    if variant_keys:
        keys = [k.strip() for k in variant_keys if k.strip()]
        selected = []
        for k in keys:
            if k in VARIANT_REGISTRY:
                selected.append(VARIANT_REGISTRY[k])
            else:
                # Find matching by partial ID or name
                matches = [v for v in VARIANT_REGISTRY.values() if k.lower() in v["id"].lower() or k.lower() in v["name"].lower()]
                if matches:
                    selected.extend(matches)
        return selected

    all_configs = list(VARIANT_REGISTRY.values())
    if group == "single":
        return [c for c in all_configs if "1. Single" in c["category"]]
    elif group == "combo":
        return [c for c in all_configs if "2. Combination" in c["category"]]
    elif group in ("straitjacket", "sj"):
        return [c for c in all_configs if "straitjacket" in c["category"].lower()]
    elif group == "nextgen":
        return [c for c in all_configs if "4. Next-Gen" in c["category"]]
    elif group in ("ablation", "ablate"):
        return [c for c in all_configs if "5. Evidence-treatment" in c["category"]]
    
    return all_configs
