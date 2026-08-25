# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
SWE-bench Pro architecture arms: the H2 study, on rows that are actually gradable.

Why this module exists
----------------------
`src/featurebench.py` asks H2 -- does front-loaded planning beat fail->escalate
once the oracle stops being free? -- and the arms there are sound. The dataset
under them is not, on this environment: a FeatureBench row is only scorable
when the repository's own `test_patch` applies to the image it ships with, and
the harness has to rebuild the graded tree itself before anything runs. Rows
that fail that step fail it for *every* arm, so they are not a hard task, they
are a missing measurement.

SWE-bench Pro removes that step rather than working around it. Upstream ships,
per instance, the image (repo at /app, dependencies installed), the git command
that puts the graded tests in place, the script that runs that repository's
suite, and the parser that reads its output. `src/evaluator.py` runs those
four things; nothing about the grading is reconstructed here. What stays
identical to FeatureBench is everything the study is about: the tiers, the
gates, the digest on every repair turn, and the oracle-call budget.

    H2. As the oracle gets more expensive or more partial, the cascade's
        advantage shrinks, because *fail -> escalate* stops being a free
        routing signal. Front-loaded planning is then paying for information
        the tests can no longer hand over for nothing.

SWE-bench Pro is a *stronger* test of H2 than FeatureBench was. An attempt here
runs a real repository's suite -- `npm install` and all -- and the tasks are
long-horizon enough that frontier agents resolve roughly 20-40% of them. If the
free-oracle result (escalate on failure, do not plan ahead) survives at this
oracle price, it survives most prices worth paying.

Reading the arms
----------------
    sbp_single_flash   gemini-3.7-flash (low) x3        -- the cheap single
    sbp_single_sonnet  claude-sonnet-5 x3               -- best $/solved at N=148
    sbp_single_opus    claude-opus-5 x3                 -- the ceiling; opt-in
    sbp_cascade        flash -> sonnet -> opus, escalate when the rung fails
    sbp_evidence_gate  same tiers, escalate when the digest says the failure is
                       hard. THE RECOMMENDED SHAPE from N=148.
    sbp_plan_exec      THE H2 CHALLENGER: opus-5 plans before any test runs,
                       then gemini-3.7-flash implements and repairs.

Every arm makes exactly three oracle calls -- three container test runs -- for
the same reason as FeatureBench: that is the resource H2 says is scarce, so it
is the one held constant. `sbp_plan_exec` buys one extra *LLM* call for its
plan, which shows up in dollars where it belongs. See `src/featurebench.py` for
why escalation is a one-way ratchet and why a spare attempt is spent on the
rung already held; both hold here unchanged.

One thing that is genuinely different
-------------------------------------
The straitjacket digest's typed fact tier is profile-detected from the test
output, and SWE-bench Pro spans four languages. A Python row's pytest output
digests as a typed profile the evidence gate can read; a mocha row's JSON
reporter blob does not, and `routing.degraded` is set for it. That is not a
defect to paper over, but it does mean a mixed-language `sbp_evidence_gate`
sweep is two arms wearing one name. Run it per language --
`SBP_LANGUAGES=python`, or `load_swebench_pro_problems(languages=["python"])`
-- and say which in the report.
"""

import os
from .config import GEMINI_37_FLASH_ID, SONNET_ID, OPUS_5_ID
from .client import dispatch_model
from .evaluator import SWEBenchProEnv, extract_patch
from .architectures import _arm, _treat_error
from .routing import GATES, EscalationTrace, classify

# The ladder, cheapest first. One place, so "one rung up" is defined once.
TIERS = [
    (GEMINI_37_FLASH_ID, "low"),
    (SONNET_ID, None),
]
FRONTIER = OPUS_5_ID

MAX_ORACLE_CALLS = int(os.environ.get("SBP_MAX_ORACLE_CALLS", "2"))

# SWE-bench Pro's gold patches run to ~12k chars on a median row and far past
# that on the large ones. The same ceiling FeatureBench needed, for the same
# reason: an 8k cap truncates a correct answer into an inapplicable one.
MAX_PATCH_TOKENS = 32768

# The problem statement, requirements and interface block together exceed 30k
# chars on some rows. Cutting the *statement* would remove the task; this cuts
# the two supporting blocks, tail first, and says so in the prompt so the model
# knows it is reading an excerpt.
MAX_SUPPORT_CHARS = 12000

SOLVER_ROLE = (
    "You are a senior engineer resolving an issue in an existing repository. "
    "Read the issue, the requirements and the interfaces below, then produce "
    "the COMPLETE unified git diff that resolves it, spanning every file that "
    "needs to change. The diff is applied with `git apply` at the repository "
    "root, so use `a/` and `b/` prefixes and real line numbers. Do NOT modify "
    "test files: the graded tests are restored from the reference commit after "
    "your patch is applied, so edits to them are discarded. Output ONLY one "
    "```diff code block.\n\n"
)
CONTRACT_LOCATOR_ROLE = (
    "You are a principal software engineer and codebase navigator. Read the issue, "
    "requirements and new interfaces below. Produce a CONCISE IMPLEMENTATION CONTRACT (<150 words) "
    "for the engineer who will generate the unified git diff. Specify:\n"
    "1. TARGET_FILES: Exact repository file paths that must be modified or created (use standard relative paths).\n"
    "2. MODIFICATIONS: The exact function, method, class, or logic block to modify in each file.\n"
    "3. TEST_INVARIANTS: What key behaviors must hold to pass the test suite.\n"
    "Do NOT write diffs or full code. Output ONLY the concise contract.\n\n"
)
PLANNER_ROLE = (
    "You are a senior software architect. Read the issue, requirements and "
    "interfaces below and write an IMPLEMENTATION PLAN for an engineer who "
    "will write the patch: which files to create or modify, the functions and "
    "classes each needs, the data flow between them, and the edge cases the "
    "tests will probe. Be concrete and name real paths. Do NOT write the diff. "
    "Under 500 words.\n\n"
)
EXECUTOR_ROLE = (
    "You are a senior engineer resolving an issue in an existing repository, "
    "working to an architect's plan. Produce the COMPLETE unified git diff "
    "resolving the issue, following the plan. The diff is applied with "
    "`git apply` at the repository root. Do NOT modify test files. Output ONLY "
    "one ```diff code block.\n\n"
)
REPAIR_ROLE = (
    "You are a senior engineer. Your patch was applied to the repository and "
    "its test suite FAILED. Read the issue, your patch, and the contained test "
    "digest below. Fix the root cause and output the COMPLETE corrected "
    "unified git diff -- the whole patch against the original tree, not an "
    "increment on top of it. Output ONLY one ```diff code block.\n\n"
)


def _clip(text, limit=MAX_SUPPORT_CHARS):
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} chars omitted ...]"


def _context(problem):
    """The task as the model sees it.

    The statement/requirements/interface triple is upstream's own prompt shape
    (`helper_code/create_problem_statement.py`). Reproducing it matters: the
    published resolve rates for this dataset were measured with those three
    blocks present, so dropping the last two would make every number here
    quietly incomparable to the leaderboard.
    """
    files = problem.get("selected_test_files_to_run") or []
    return (
        f"Repository: {problem.get('repo', '')} ({problem.get('repo_language', '?')})\n"
        f"Base commit: {problem.get('base_commit', '')}\n"
        f"Test files that will be run: {', '.join(files) if files else '(unknown)'}\n\n"
        f"Issue:\n{problem.get('problem_statement', '')}\n\n"
        f"Requirements:\n{_clip(problem.get('requirements'))}\n\n"
        f"New interfaces introduced:\n{_clip(problem.get('interface'))}\n"
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


def _result(passed, evidence, acc, loops, trace=None, ratio=None, report=None):
    out = {
        "passed": bool(passed),
        "as_run_usd": round(acc["usd"], 6),
        "output_tokens": acc["out"],
        "total_tokens": acc["tok"],
        "repair_loops": loops,
        "triage_usd": 0.0,
        "error": "" if passed else str(evidence)[:500],
        # Partial credit over the required test names, from the benchmark's own
        # parser rather than scraped from a summary line. On a dataset where
        # frontier agents resolve 20-40%, a binary verdict makes every cheap
        # arm read as an undifferentiated zero.
        "test_pass_ratio": ratio,
    }
    if report:
        out["sbp"] = report
    if trace is not None:
        out["routing"] = trace.as_dict()
    return out


def _ladder(problem, tiers, gate="after_ladder", frontier=FRONTIER,
            plan="", planner_usd=0.0, acc=None):
    """One escalating repair loop against the containerised oracle.

    The gate, the difficulty classifier and the degradation warning are the
    *same* ones the BigCodeBench-Hard routing study used (`src/routing.py`), so
    an `sbp_evidence_gate` row and an `r9_opus_on_evidence` row mean the same
    thing on two datasets with very different oracle costs.
    """
    gate_fn = GATES[gate] if isinstance(gate, str) else gate
    trace = EscalationTrace()
    acc = acc if acc is not None else {"usd": planner_usd, "out": 0, "tok": 0}

    with SWEBenchProEnv(problem) as env:
        model, think = tiers[0]
        text, usage, _ = dispatch_model(
            model, _solve_prompt(problem, EXECUTOR_ROLE if plan else SOLVER_ROLE, plan),
            max_tokens=MAX_PATCH_TOKENS, thinking_level=think, problem=problem)
        _spend(acc, usage)
        trace.rungs.append(f"{model}/{think or 'off'}")
        held = (model, think)
        patch = extract_patch(text)
        passed, evidence = env.score(patch)
        ratio, report = env.last_ratio, env.last_report
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
                # Oracle budget left, ladder out of rungs: re-run the highest
                # rung reached rather than handing an attempt back. See
                # src/featurebench.py for why an arm that quietly returns a
                # container run reads as cheaper for the wrong reason.
                target, think = held
            held = (target, think)

            digest, tr_usage, _ = _treat_error(evidence, "straitjacket", problem=problem)
            _spend(acc, tr_usage)
            r_text, r_usage, _ = dispatch_model(
                target, _repair_prompt(problem, patch, digest, plan),
                max_tokens=MAX_PATCH_TOKENS, thinking_level=think, problem=problem)
            _spend(acc, r_usage)
            trace.rungs.append(f"{target}/{think or 'off'}")
            if escalate:
                trace.frontier_used = True
                trace.frontier_rung = len(trace.rungs)

            patch = extract_patch(r_text)
            passed, evidence = env.score(patch)
            oracle_calls += 1
            loops += 1
            if env.last_ratio is not None:
                ratio = env.last_ratio
            if env.last_report:
                report = env.last_report
            if passed:
                trace.solved_at = trace.rungs[-1]

    return _result(passed, evidence, acc, loops, trace, ratio, report)


# ==============================================================================
# --- ARMS ---
# ==============================================================================

@_arm(sj_required=True)
def run_sbp_single(problem, model_id=GEMINI_37_FLASH_ID, thinking_level=None):
    """One model writes the patch and repairs it twice. Three oracle calls."""
    return _ladder(problem, [(model_id, thinking_level)] * MAX_ORACLE_CALLS,
                   gate="never")


@_arm(sj_required=True)
def run_sbp_cascade(problem):
    """Attempt-count ladder: escalate one rung every time the tests fail.

    The `r6_opus_after_ladder` shape, which on BCB-Hard at N=148 tied the plain
    frontier baseline exactly, at 99% of its cost per solved task.
    """
    return _ladder(problem, TIERS, gate="after_ladder")


@_arm(sj_required=True)
def run_sbp_evidence_gate(problem):
    """Escalate to the frontier model when the digest says the failure is hard.

    The `r9_opus_on_evidence` shape and the recommended default from N=148.
    Needs the library backend *and* a language whose test output the harness can
    type: without a typed fact tier the gate has nothing to read,
    `routing.degraded` is set, and the row must not be quoted as an
    evidence-gate result.
    """
    return _ladder(problem, TIERS, gate="evidence")


@_arm(sj_required=True)
def run_sbp_plan_exec(problem, planner_model=OPUS_5_ID,
                      executor_model=GEMINI_37_FLASH_ID):
    """H2 challenger: buy the frontier model BEFORE the first oracle call.

    Same frontier model as `sbp_cascade`, same three oracle calls, opposite
    timing. It lost on BigCodeBench-Hard and on ClassEval, both of which had a
    free oracle; this is the expensive-oracle rerun of that comparison.
    """
    acc = {"usd": 0.0, "out": 0, "tok": 0}
    plan, usage, _ = dispatch_model(planner_model, _solve_prompt(problem, PLANNER_ROLE),
                                    max_tokens=2048, problem=problem)
    _spend(acc, usage)
    return _ladder(problem, [(executor_model, "low")] * MAX_ORACLE_CALLS,
                   gate="never", plan=plan, acc=acc)


def gate_patch_health(difficulty, attempt, total_rungs):
    """Patch-health aware escalation gate for SWE-bench Pro."""
    if difficulty is None:
        return False, "no difficulty signal"
    if difficulty.is_hard:
        return True, f"patch health & evidence says {difficulty.level}: {'; '.join(difficulty.reasons)}"
    if attempt >= total_rungs:
        return True, f"cheap rungs exhausted ({attempt}/{total_rungs})"
    return False, f"failure is {difficulty.level}; standard repair"


@_arm(sj_required=True)
def run_sbp_grounded_contract(problem, locator_model=SONNET_ID,
                              executor_model=GEMINI_37_FLASH_ID,
                              frontier=OPUS_5_ID):
    """Candidate 1: Grounded Micro-Contract Localization Cascade.
    
    Sonnet-5 generates a strict, compact file & interface contract (<150 words)
    at minimal cost (~$0.0015). Gemini 3.7 Flash executes the grounded diff.
    Escalates to Opus-5 on broad/stalled test failure evidence.
    """
    acc = {"usd": 0.0, "out": 0, "tok": 0}
    contract, usage, _ = dispatch_model(
        locator_model, _solve_prompt(problem, CONTRACT_LOCATOR_ROLE),
        max_tokens=512, problem=problem)
    _spend(acc, usage)
    return _ladder(problem, [(executor_model, "low"), (SONNET_ID, None)],
                   gate="evidence", frontier=frontier, plan=contract, acc=acc)


@_arm(sj_required=True)
def run_sbp_patch_health_router(problem, initial_model=GEMINI_37_FLASH_ID,
                                mid_model=SONNET_ID, frontier=OPUS_5_ID):
    """Candidate 2: Patch-Health & Semantic Error-Class Aware Router.
    
    Gemini 3.7 Flash writes the initial diff. The router discriminates between
    patch-applicability/syntax issues (stays cheap/Sonnet) vs broad semantic regressions
    (instantly escalates to Opus-5).
    """
    return _ladder(problem, [(initial_model, "low"), (mid_model, None)],
                   gate=gate_patch_health, frontier=frontier)


@_arm(sj_required=True)
def run_sbp_sonnet_opus_sweetspot(problem, initial_model=SONNET_ID,
                                  frontier=OPUS_5_ID):
    """Candidate 3: Cross-Provider Pareto Sweet-Spot (Sonnet Drafter + Opus Escalator).
    
    Claude Sonnet-5 writes the initial diff with high first-shot syntax fidelity (~$0.005).
    Deterministic Straitjacket triage escalates to Claude Opus-5 on broad/stalled regressions,
    providing maximum reliability in the expensive Docker oracle regime.
    """
    return _ladder(problem, [(initial_model, None), (initial_model, None)],
                   gate="evidence", frontier=frontier)


# ==============================================================================
# --- VARIANT REGISTRY ---
# ==============================================================================

CATEGORY = "9. SWE-bench Pro expensive-oracle study"

# Same reasoning as ClassEval's and FeatureBench's opus baselines: the frontier
# single is priced far above the rest, so it sits in its own category and is
# opt-in rather than silently repricing every `--group swebench-pro` sweep.
OPUS_CATEGORY = "9b. SWE-bench Pro frontier baseline"
CANDIDATES_CATEGORY = "9c. SWE-bench Pro candidate architectures"

SWEBENCH_PRO_VARIANTS = {
    "sbp_single_flash": {
        "id": "sbp_single_flash", "category": CATEGORY,
        "name": "S0a. Single: gemini-3.7-flash low (2 rungs)",
        "models": "Gemini 3.7 Flash (low) x2",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": lambda p: run_sbp_single(p, model_id=GEMINI_37_FLASH_ID,
                                        thinking_level="low"),
    },
    "sbp_single_sonnet": {
        "id": "sbp_single_sonnet", "category": CATEGORY,
        "name": "S0b. Single: claude-sonnet-5 (2 rungs)",
        "models": "Claude Sonnet-5 x2",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": lambda p: run_sbp_single(p, model_id=SONNET_ID),
    },
    "sbp_cascade": {
        "id": "sbp_cascade", "category": CATEGORY,
        "name": "S1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate, 2 rungs)",
        "models": "Gemini 3.7 Flash -> Claude Sonnet-5",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": run_sbp_cascade,
    },
    "sbp_evidence_gate": {
        "id": "sbp_evidence_gate", "category": CATEGORY,
        "name": "S2. Evidence gate: flash -> sonnet/opus (evidence gate, 2 rungs)",
        "models": "Gemini 3.7 Flash -> Claude Sonnet-5 / Claude Opus-5 (evidence gate)",
        "triage_mode": "Straitjacket digest + evidence-gated escalation ($0.00)",
        "fn": run_sbp_evidence_gate,
    },
    "sbp_plan_exec": {
        "id": "sbp_plan_exec", "category": CATEGORY,
        "name": "S3. H2: opus-5 plans first, 3.7-flash implements and repairs (2 rungs)",
        "models": "Claude Opus-5 plan + Gemini 3.7 Flash exec x2",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": run_sbp_plan_exec,
    },
    "sbp_single_opus": {
        "id": "sbp_single_opus", "category": OPUS_CATEGORY,
        "name": "S0c. Single: claude-opus-5 (2 rungs)",
        "models": "Claude Opus-5 x2",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": lambda p: run_sbp_single(p, model_id=OPUS_5_ID),
    },
    # -- 3 Optimized Candidate Architectures --
    "sbp_grounded_contract": {
        "id": "sbp_grounded_contract", "category": CANDIDATES_CATEGORY,
        "name": "S4. Grounded Contract: Sonnet locator -> Flash exec -> Opus escalation (2 rungs)",
        "models": "Claude Sonnet-5 contract + Gemini 3.7 Flash exec -> Claude Opus-5",
        "triage_mode": "Straitjacket digest + evidence-gated escalation ($0.00)",
        "fn": run_sbp_grounded_contract,
    },
    "sbp_patch_health_router": {
        "id": "sbp_patch_health_router", "category": CANDIDATES_CATEGORY,
        "name": "S5. Patch-Health Router: Flash -> Sonnet / Opus (health-aware gate, 2 rungs)",
        "models": "Gemini 3.7 Flash -> Claude Sonnet-5 / Claude Opus-5 (health gate)",
        "triage_mode": "Straitjacket digest + patch-health router ($0.00)",
        "fn": run_sbp_patch_health_router,
    },
    "sbp_sonnet_opus_sweetspot": {
        "id": "sbp_sonnet_opus_sweetspot", "category": CANDIDATES_CATEGORY,
        "name": "S6. Sweetspot: Sonnet-5 draft -> Evidence gate -> Opus-5 repair (2 rungs)",
        "models": "Claude Sonnet-5 draft -> Claude Sonnet-5 / Claude Opus-5 (evidence gate)",
        "triage_mode": "Straitjacket digest + evidence-gated escalation ($0.00)",
        "fn": run_sbp_sonnet_opus_sweetspot,
    },
}
