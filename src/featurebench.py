# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
FeatureBench architecture arms: does escalation still win when the oracle costs?

Why this module exists
----------------------
Every result in this repository so far sits on the cheap side of one
conditional. BigCodeBench-Hard runs ~6 unit tests in a sandbox for $0;
ClassEval runs a per-method test class for $0. Both said the same thing --
escalate on a test failure rather than plan in advance -- and both said it in
exactly the regime where escalation is most favoured, because *fail -> escalate*
is a free routing signal when failing is free.

FeatureBench breaks that. An attempt applies a diff inside the repository's own
Docker image and runs pytest, which upstream measures at 57.2 s on gold
patches. That is H2 from docs/pattern-dataset-selection.md:

    H2. As the oracle gets more expensive or more partial, the cascade's
        advantage shrinks, because *fail -> escalate* stops being a free
        routing signal. Front-loaded planning is then paying for information
        the tests can no longer hand over for nothing.

Reading the arms
----------------
    fb_single_flash    gemini-3.7-flash (low) x3        -- the cheap single
    fb_single_sonnet   claude-sonnet-5 x3               -- best $/solved at N=148
    fb_single_opus     claude-opus-5 x3                 -- the ceiling; opt-in
    fb_cascade         flash -> sonnet -> opus, escalate when the rung fails
    fb_evidence_gate   same tiers, escalate when the digest says the failure is
                       hard. THE RECOMMENDED SHAPE, carried over from N=148
                       where it reached 96% of frontier accuracy for 74% of
                       frontier spend.
    fb_plan_exec       THE H2 CHALLENGER: opus-5 writes the plan *before* any
                       test runs, then gemini-3.7-flash implements and repairs.

Why the arm set looks like this
-------------------------------
The models are the ones that earned their place in the two sweeps that ran at
size, and the ones that lost are absent on purpose:

* `gemini-3.5-flash-lite` is **not** a rung here. On BCB-Hard it was a
  reasonable first attempt because a wasted attempt cost a fraction of a cent
  and a millisecond. Here a wasted attempt costs a container run, and a
  multi-file feature is far outside what Lite solved even on single functions.
  Spending the expensive resource on a rung that cannot plausibly succeed is
  precisely the mistake H2 is about.
* **No `medium`/`high` thinking ladder.** N=148 measured `gemini-3.7-flash` at
  `medium` costing 33% more than `claude-opus-5` while solving 14 fewer tasks,
  because it emits 5.3x the output tokens. Escalate the model, not the budget.
* `fb_plan_exec` uses **the same frontier model as `fb_cascade`**, spent before
  the first oracle call instead of after the last. That is the contrast H2
  makes a claim about: not "is opus good", but "when should you buy it".

Budget matching
---------------
Every arm makes **exactly 3 oracle calls** -- three container test runs. That is
the resource H2 says is scarce, so it is the one held constant; `fb_plan_exec`
buys one extra *LLM* call for its plan, which shows up in dollars where it
belongs. Matching on attempts rather than reconstructing a fair comparison
afterwards is the lesson ClassEval's control arm taught (see src/classeval.py).

Two consequences of holding that constant, both deliberate:

* **Escalation is a one-way ratchet.** An arm never drops to a cheaper rung --
  N=100 measured a de-escalating repair turn rescuing 16% of failures against
  41% for an escalating one (z = +3.55, p = 0.0004), so building the losing
  direction into an arm would be designing in a known defect.
* **A spare attempt is spent on the rung already held.** `fb_evidence_gate` can
  jump straight to the frontier on its first repair and then have budget left
  with nothing above it. It re-runs the frontier rather than handing the
  attempt back: an arm that quietly returns a container run reads as cheaper
  for a reason that has nothing to do with its routing policy. (The N=148
  routing study let rung counts float instead, because a spare attempt there
  cost a fraction of a cent rather than the resource under study.)
"""

from .config import GEMINI_37_FLASH_ID, SONNET_ID, OPUS_5_ID
from .client import dispatch_model
from .evaluator import (FeatureBenchEnv, extract_patch, featurebench_test_files,
                        featurebench_test_ratio)
from .architectures import _arm, _treat_error
from .routing import GATES, EscalationTrace, classify

# The ladder, cheapest first. One place, so "one rung up" is defined once.
TIERS = [
    (GEMINI_37_FLASH_ID, "low"),
    (SONNET_ID, None),
]
FRONTIER = OPUS_5_ID

MAX_ORACLE_CALLS = 3

SOLVER_ROLE = (
    "You are a senior engineer implementing a complete feature in an existing "
    "repository. Read the feature request and produce the COMPLETE unified git "
    "diff that implements it, spanning every file that needs to change. The "
    "diff is applied with `git apply` at the repository root, so use `a/` and "
    "`b/` prefixes and real line numbers. Output ONLY one ```diff code block.\n\n"
)
PLANNER_ROLE = (
    "You are a senior software architect. Read the feature request below and "
    "write an IMPLEMENTATION PLAN for an engineer who will write the patch: "
    "which files to create or modify, the functions and classes each needs, the "
    "data flow between them, and the edge cases the tests will probe. Be "
    "concrete and name real paths. Do NOT write the diff. Under 500 words.\n\n"
)
EXECUTOR_ROLE = (
    "You are a senior engineer implementing a complete feature in an existing "
    "repository, working to an architect's plan. Produce the COMPLETE unified "
    "git diff implementing the feature, following the plan. The diff is applied "
    "with `git apply` at the repository root. Output ONLY one ```diff code block.\n\n"
)
REPAIR_ROLE = (
    "You are a senior engineer. Your patch was applied to the repository and its "
    "test suite FAILED. Read the feature request, your patch, and the contained "
    "test digest below. Fix the root cause and output the COMPLETE corrected "
    "unified git diff -- the whole patch against the original tree, not an "
    "increment on top of it. Output ONLY one ```diff code block.\n\n"
)


def _context(problem):
    """The task as the model sees it: statement, repo, and what will be run."""
    files = featurebench_test_files(problem)
    return (
        f"Repository: {problem.get('repo', '')}\n"
        f"Base commit: {problem.get('base_commit', '')}\n"
        f"Tests that will be run: {', '.join(files) if files else '(unknown)'}\n\n"
        f"Feature request:\n{problem.get('problem_statement', '')}\n"
    )


def _solve_prompt(problem, role=SOLVER_ROLE, plan=""):
    plan_block = f"\nArchitect's implementation plan:\n{plan}\n" if plan else ""
    return role + _context(problem) + plan_block


def _repair_prompt(problem, patch, digest, plan=""):
    plan_block = f"\nArchitect's implementation plan:\n{plan}\n" if plan else ""
    return (
        REPAIR_ROLE + _context(problem) + plan_block
        + f"\nYour current patch:\n```diff\n{patch}\n```\n\n"
        + f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n"
    )


def _spend(acc, usage):
    acc["usd"] += usage["as_run_usd"]
    acc["out"] += usage["output"]
    acc["tok"] += usage["total_tokens"]
    return usage["as_run_usd"]


def _result(passed, evidence, acc, loops, trace=None, ratio=None):
    out = {
        "passed": bool(passed),
        "as_run_usd": round(acc["usd"], 6),
        "output_tokens": acc["out"],
        "total_tokens": acc["tok"],
        "repair_loops": loops,
        "triage_usd": 0.0,
        "error": "" if passed else str(evidence)[:500],
        # FeatureBench reports a soft partial-credit metric beside its binary
        # Resolved Rate, because a binary verdict throws away everything a
        # cheap rung still achieved on a task it could not finish. Kept for the
        # same reason: on a dataset where frontier models resolve 20-47%, the
        # cheap arms would otherwise all read as an undifferentiated zero.
        "test_pass_ratio": ratio,
    }
    if trace is not None:
        out["routing"] = trace.as_dict()
    return out


def _ladder(problem, tiers, gate="after_ladder", frontier=FRONTIER,
            plan="", planner_usd=0.0, acc=None):
    """One escalating repair loop against the containerised oracle.

    The gate, the difficulty classifier and the degradation warning are the
    *same* ones the BigCodeBench-Hard routing study used (`src/routing.py`), so
    an `fb_evidence_gate` row and an `r9_opus_on_evidence` row mean the same
    thing on two datasets with very different oracle costs. Only the prompts
    and the execution backend differ.
    """
    gate_fn = GATES[gate] if isinstance(gate, str) else gate
    trace = EscalationTrace()
    acc = acc if acc is not None else {"usd": planner_usd, "out": 0, "tok": 0}

    with FeatureBenchEnv(problem) as env:
        model, think = tiers[0]
        text, usage, _ = dispatch_model(
            model, _solve_prompt(problem, EXECUTOR_ROLE if plan else SOLVER_ROLE, plan),
            max_tokens=8192, thinking_level=think, problem=problem)
        _spend(acc, usage)
        trace.rungs.append(f"{model}/{think or 'off'}")
        held = (model, think)
        patch = extract_patch(text)
        passed, evidence = env.score(patch)
        ratio = featurebench_test_ratio(evidence)
        oracle_calls = 1
        loops = 0
        difficulty = None
        if passed:
            trace.solved_at = trace.rungs[-1]

        while not passed and oracle_calls < MAX_ORACLE_CALLS:
            difficulty = classify(evidence, previous=difficulty)
            if (getattr(gate_fn, "requires_typed_evidence", False)
                    and difficulty is not None and not difficulty.typed
                    and not trace.degraded):
                trace.degraded = True
            escalate, why = gate_fn(difficulty, loops + 1, len(tiers))
            if escalate and trace.frontier_used:
                escalate, why = False, "already at the frontier rung"
            trace.record(loops + 1, difficulty, escalate, why)

            if escalate:
                target, think = frontier, None
            elif loops + 1 < len(tiers):
                target, think = tiers[loops + 1]
            else:
                # Oracle budget remains but the ladder is out of rungs -- an
                # evidence gate that jumped straight to the frontier gets here
                # with a spare attempt. Re-run the highest rung reached rather
                # than returning the budget unused: these arms are matched on
                # oracle calls, and an arm that quietly hands one back looks
                # cheaper for a reason that has nothing to do with its routing
                # policy. (The N=148 study let rung counts float because a
                # spare attempt there cost a fraction of a cent; here it costs
                # a container run, which is the resource under study.)
                target, think = held
            held = (target, think)

            digest, tr_usage, _ = _treat_error(evidence, "straitjacket", problem=problem)
            _spend(acc, tr_usage)
            r_text, r_usage, _ = dispatch_model(
                target, _repair_prompt(problem, patch, digest, plan),
                max_tokens=8192, thinking_level=think, problem=problem)
            _spend(acc, r_usage)
            trace.rungs.append(f"{target}/{think or 'off'}")
            if escalate:
                trace.frontier_used = True
                trace.frontier_rung = len(trace.rungs)

            patch = extract_patch(r_text)
            passed, evidence = env.score(patch)
            oracle_calls += 1
            loops += 1
            r = featurebench_test_ratio(evidence)
            if r is not None:
                ratio = r
            if passed:
                trace.solved_at = trace.rungs[-1]

    return _result(passed, evidence, acc, loops, trace, ratio)


# ==============================================================================
# --- ARMS ---
# ==============================================================================

@_arm(sj_required=True)
def run_fb_single(problem, model_id=GEMINI_37_FLASH_ID, thinking_level=None):
    """One model writes the feature and repairs it twice. Three oracle calls."""
    return _ladder(problem, [(model_id, thinking_level)] * MAX_ORACLE_CALLS,
                   gate="never")


@_arm(sj_required=True)
def run_fb_cascade(problem):
    """Attempt-count ladder: escalate one rung every time the tests fail.

    The `r6_opus_after_ladder` shape. On BCB-Hard at N=148 this tied the plain
    frontier baseline exactly, at 99% of its cost per solved task -- the ladder
    bought nothing there. Whether an expensive oracle changes that is the point.
    """
    return _ladder(problem, TIERS, gate="after_ladder")


@_arm(sj_required=True)
def run_fb_evidence_gate(problem):
    """Escalate to the frontier model when the digest says the failure is hard.

    The `r9_opus_on_evidence` shape, and the recommended default carried over
    from N=148. Needs the library backend: without a typed fact tier the gate
    has nothing to read, `routing.degraded` is set, and the row must not be
    quoted as an evidence-gate result.
    """
    return _ladder(problem, TIERS, gate="evidence")


@_arm(sj_required=True)
def run_fb_plan_exec(problem, planner_model=OPUS_5_ID,
                     executor_model=GEMINI_37_FLASH_ID):
    """H2 challenger: buy the frontier model BEFORE the first oracle call.

    Same frontier model as `fb_cascade`, same three oracle calls, opposite
    timing. If H2 holds, this is where front-loaded planning finally pays --
    it lost on BigCodeBench-Hard and on ClassEval, both of which had a free
    oracle.
    """
    acc = {"usd": 0.0, "out": 0, "tok": 0}
    plan, usage, _ = dispatch_model(planner_model, _solve_prompt(problem, PLANNER_ROLE),
                                    max_tokens=2048, problem=problem)
    _spend(acc, usage)
    return _ladder(problem, [(executor_model, "low")] * MAX_ORACLE_CALLS,
                   gate="never", plan=plan, acc=acc)


# ==============================================================================
# --- VARIANT REGISTRY ---
# ==============================================================================

CATEGORY = "8. FeatureBench expensive-oracle study"

# Same reasoning as ClassEval's opus baseline: the frontier single is priced far
# above the rest, so it sits in its own category and is opt-in rather than
# silently repricing every `--group featurebench` sweep.
OPUS_CATEGORY = "8b. FeatureBench frontier baseline"

FEATUREBENCH_VARIANTS = {
    "fb_single_flash": {
        "id": "fb_single_flash", "category": CATEGORY,
        "name": "F0a. Single: gemini-3.7-flash low (3 rungs)",
        "models": "Gemini 3.7 Flash (low) x3",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": lambda p: run_fb_single(p, model_id=GEMINI_37_FLASH_ID, thinking_level="low"),
    },
    "fb_single_sonnet": {
        "id": "fb_single_sonnet", "category": CATEGORY,
        "name": "F0b. Single: claude-sonnet-5 (3 rungs)",
        "models": "Claude Sonnet-5 x3",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": lambda p: run_fb_single(p, model_id=SONNET_ID),
    },
    "fb_cascade": {
        "id": "fb_cascade", "category": CATEGORY,
        "name": "F1. Cascade: 3.7-flash -> sonnet-5 -> opus-5 (attempt-count gate)",
        "models": "Gemini 3.7 Flash -> Claude Sonnet-5 -> Claude Opus-5",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": run_fb_cascade,
    },
    "fb_evidence_gate": {
        "id": "fb_evidence_gate", "category": CATEGORY,
        "name": "F2. Evidence gate: same tiers, escalate when the digest says hard",
        "models": "Gemini 3.7 Flash -> Claude Sonnet-5 -> Claude Opus-5 (evidence gate)",
        "triage_mode": "Straitjacket digest + evidence-gated escalation ($0.00)",
        "fn": run_fb_evidence_gate,
    },
    "fb_plan_exec": {
        "id": "fb_plan_exec", "category": CATEGORY,
        "name": "F3. H2: opus-5 plans first, 3.7-flash implements and repairs",
        "models": "Claude Opus-5 plan + Gemini 3.7 Flash exec x3",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": run_fb_plan_exec,
    },
    "fb_single_opus": {
        "id": "fb_single_opus", "category": OPUS_CATEGORY,
        "name": "F0c. Single: claude-opus-5 (3 rungs)",
        "models": "Claude Opus-5 x3",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": lambda p: run_fb_single(p, model_id=OPUS_5_ID),
    },
}
