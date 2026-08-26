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
import re

from .config import GEMINI_37_FLASH_ID, SONNET_ID, OPUS_5_ID
from .client import dispatch_model
from .evaluator import (SWEBenchProEnv, extract_patch, guard_reason,
                        _guard_evidence, SBP_GROUNDING_CHARS)
from .architectures import _arm, _treat_error
from .routing import GATES, EscalationTrace, classify, frontier_is_reachable

# The ladder, cheapest first. One place, so "one rung up" is defined once.
TIERS = [
    (GEMINI_37_FLASH_ID, "low"),
    (SONNET_ID, None),
]
FRONTIER = OPUS_5_ID

# Three, not two, and the difference is structural rather than a budget
# preference. `_ladder` evaluates its gate once per repair turn, so a budget of
# K oracle calls produces K-1 gate evaluations at attempts 1..K-1. Every gate
# here compares `attempt` against `len(tiers)`, which is 2. At K=2 the only
# evaluation is `attempt == 1`, no gate can answer "escalate", and the frontier
# rung is unreachable code for *every* arm that has one. See
# `routing.frontier_is_reachable`, which is asserted over the registry below.
MAX_ORACLE_CALLS = int(os.environ.get("SBP_MAX_ORACLE_CALLS", "3"))

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
# A repair turn that opens with "your patch was applied and the tests failed"
# is a false premise when the patch never applied, and the model spends the
# turn debugging a test run that did not happen. The dominant failure on this
# dataset is exactly that one, so it gets its own instruction.
APPLY_REPAIR_ROLE = (
    "You are a senior engineer. Your diff was REJECTED by `git apply` -- it "
    "never reached the repository and NO test was run. This is a diff-format "
    "problem, not a logic problem. Read the apply log below: it names the file "
    "and hunk that failed and prints the context lines it searched for and did "
    "not find. Re-emit the COMPLETE unified git diff so that every hunk's "
    "context lines match the file exactly as quoted above, and every `@@` "
    "header's line counts match the hunk body. Output ONLY one ```diff code "
    "block.\n\n"
)
# Nothing the model writes changes a container that never started or a graded
# test file that could not be restored. Saying so stops the turn from being
# spent inventing a code-level explanation for a Docker failure.
ENVIRONMENT_REPAIR_ROLE = (
    "You are a senior engineer. The attempt failed for an ENVIRONMENT reason "
    "reported below, not because of your code -- the harness could not run the "
    "suite. Re-emit your best COMPLETE unified git diff for the issue, "
    "unchanged in approach unless you see an actual defect in it. Output ONLY "
    "one ```diff code block.\n\n"
)

_REPAIR_ROLES = {
    "apply_failed": APPLY_REPAIR_ROLE,
    "container_unavailable": ENVIRONMENT_REPAIR_ROLE,
    "restore_failed": ENVIRONMENT_REPAIR_ROLE,
    "row_no_test_files": ENVIRONMENT_REPAIR_ROLE,
    "execution_error": ENVIRONMENT_REPAIR_ROLE,
    "harness_error": ENVIRONMENT_REPAIR_ROLE,
}


def repair_role(evidence):
    """The repair instruction that matches how the attempt actually died."""
    return _REPAIR_ROLES.get(guard_reason(evidence), REPAIR_ROLE)


# ==============================================================================
# --- REPOSITORY GROUNDING ---
# ==============================================================================
#
# The row hands the model an issue, a requirements block and an interface
# block, and asks for a complete unified diff with real line numbers against a
# tree it has never seen. Measured over the published split:
#
#     reference-patch files named anywhere in those three blocks : 19.8%
#     rows where EVERY changed file is named                     :  8.8%
#     median reference patch                                     :  9 hunks
#                                                                   4 files
#     patches that only create new files (no context needed)     :  1.8%
#
# So the blind setting has a localisation ceiling near 9% before a single hunk
# is written, and `git apply` needs every context line of every hunk to match.
# The container is already running and already holds the tree at `base_commit`,
# so the files can simply be quoted. `SBP_GROUNDING_CHARS=0` restores the blind
# setting, which is what makes this an A/B rather than a one-way change.

_PATH_RE = re.compile(
    r"\b[\w./-]*[\w-]+\.(?:py|go|js|jsx|ts|tsx|rb|java|json|yml|yaml|toml|cfg|md)\b")
_BACKTICK_RE = re.compile(r"`([^`\n]{3,80})`")
_INTERFACE_PATH_RE = re.compile(r"^\s*Path:\s*(.+)$", re.M)
_INTERFACE_NAME_RE = re.compile(r"^\s*Name:\s*(.+)$", re.M)


def _dedup(items):
    seen, out = set(), []
    for i in items:
        i = str(i or "").strip().strip("`,;")
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def candidate_paths(problem):
    """Paths the row's own text points at, best signal first.

    Explicit `Path:` lines in the interface block are the strongest — they are
    upstream's own statement of where the new surface lives. Then the graded
    test files, which describe the behaviour being demanded. Then anything
    path-shaped anywhere in the prose.
    """
    interface = str(problem.get("interface") or "")
    explicit = []
    for line in _INTERFACE_PATH_RE.findall(interface):
        explicit += [p.strip() for p in line.split(",")]
    prose = (str(problem.get("problem_statement") or "") + "\n"
             + str(problem.get("requirements") or "") + "\n" + interface)
    return _dedup(explicit
                  + list(problem.get("selected_test_files_to_run") or [])
                  + _PATH_RE.findall(prose))


def search_terms(problem, limit=8):
    """Identifiers worth grepping for when the text names no usable path."""
    interface = str(problem.get("interface") or "")
    names = []
    for raw in _INTERFACE_NAME_RE.findall(interface):
        raw = raw.strip()
        names.append(raw)
        if "." in raw:
            names.append(raw.rsplit(".", 1)[-1])
    ticked = _BACKTICK_RE.findall(
        str(problem.get("problem_statement") or "") + "\n"
        + str(problem.get("requirements") or ""))
    # A backticked token is only a useful grep term when it looks like an
    # identifier; whole sentences in backticks match everything or nothing.
    ticked = [t for t in ticked if re.fullmatch(r"[\w.$-]{3,60}", t or "")]
    return _dedup(names + ticked)[:limit]


def collect_grounding(env, problem, budget=None):
    """Quote the repository files this row is most likely to be about.

    Returns ``(text, meta)``. ``text`` is empty when grounding is disabled or
    nothing could be read, in which case the prompt is byte-identical to the
    blind one. ``meta`` records what was read, what was skipped and how the
    paths were found, so a report can say whether an arm was grounded rather
    than leaving it to be inferred from the pass rate.
    """
    budget = SBP_GROUNDING_CHARS if budget is None else budget
    meta = {"enabled": bool(budget), "read": [], "skipped": [],
            "searched": [], "chars": 0}
    if not budget:
        return "", meta

    wanted = candidate_paths(problem)
    terms = search_terms(problem)
    try:
        blocks, read, skipped = env.read_source(wanted, budget=budget)
        # Only pay for a search when the row's own text did not locate enough.
        # 91% of rows do not name every file they need, so this is the common
        # path rather than the exception.
        if len(read) < 3 and terms:
            meta["searched"] = terms
            found = [p for p in env.grep_paths(terms) if p not in read]
            more, read2, skipped2 = env.read_source(
                found, budget=max(0, budget - sum(len(b) for b in blocks)))
            blocks, read, skipped = blocks + more, read + read2, skipped + skipped2
    except Exception as e:                                   # noqa: BLE001
        # Grounding is an enrichment. A container that cannot be read still
        # runs the blind prompt rather than failing the task.
        meta["error"] = str(e)[:200]
        return "", meta

    if not blocks:
        return "", meta
    meta.update(read=read, skipped=skipped, chars=sum(len(b) for b in blocks))
    header = (f"Repository files at commit {str(problem.get('base_commit', ''))[:12]} "
              f"({len(read)} file(s) quoted below). Your diff's context lines "
              "MUST match these exactly.\n")
    tail = (f"\n[{len(skipped)} further candidate file(s) not quoted: "
            f"{', '.join(skipped[:8])}]\n" if skipped else "")
    return header + "\n" + "\n".join(blocks) + tail, meta


def _clip(text, limit=MAX_SUPPORT_CHARS):
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} chars omitted ...]"


def _context(problem, grounding=""):
    """The task as the model sees it.

    The statement/requirements/interface triple is upstream's own prompt shape
    (`helper_code/create_problem_statement.py`). Reproducing it matters: the
    published resolve rates for this dataset were measured with those three
    blocks present, so dropping the last two would make every number here
    quietly incomparable to the leaderboard.

    `grounding` is quoted repository source, appended rather than substituted,
    so the three upstream blocks are still present verbatim and the comparison
    to the leaderboard prompt survives.
    """
    files = problem.get("selected_test_files_to_run") or []
    text = (
        f"Repository: {problem.get('repo', '')} ({problem.get('repo_language', '?')})\n"
        f"Base commit: {problem.get('base_commit', '')}\n"
        f"Test files that will be run: {', '.join(files) if files else '(unknown)'}\n\n"
        f"Issue:\n{problem.get('problem_statement', '')}\n\n"
        f"Requirements:\n{_clip(problem.get('requirements'))}\n\n"
        f"New interfaces introduced:\n{_clip(problem.get('interface'))}\n"
    )
    if grounding:
        text += f"\n{grounding}\n"
    return text


def _solve_prompt(problem, role=SOLVER_ROLE, plan="", grounding=""):
    plan_block = f"\nArchitect's implementation plan:\n{plan}\n" if plan else ""
    return role + _context(problem, grounding) + plan_block


def _repair_prompt(problem, patch, digest, plan="", grounding="", role=REPAIR_ROLE):
    plan_block = f"\nArchitect's implementation plan:\n{plan}\n" if plan else ""
    return (
        role + _context(problem, grounding) + plan_block
        + f"\nYour current patch:\n```diff\n{patch}\n```\n\n"
        + f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n"
    )


# Guard reasons a truncated response is the more likely explanation for. A
# diff cut off at the output cap still carries `---`, `+++` and `@@`, so it
# passes the birth gate and dies at `git apply` -- filed, before this, as the
# same "patch did not apply" as a diff that was simply wrong.
_TRUNCATION_MASKS = frozenset(
    {"no_patch", "not_a_diff", "no_hunk", "apply_failed"})


def _relabel_truncated(evidence, usage):
    """Re-attribute a failure to the output cap when that is what caused it.

    Diagnosis, not leniency: the attempt still failed and still counts as a
    failure. What changes is that the sweep records `truncated_output` instead
    of blaming the model's diff, which is the difference between "raise
    MAX_PATCH_TOKENS" and "the models cannot write patches".
    """
    if not (usage or {}).get("truncated"):
        return evidence
    if guard_reason(evidence) not in _TRUNCATION_MASKS:
        return evidence
    return _guard_evidence(
        "The response hit the output token cap and the diff is incomplete.\n"
        f"Downstream failure was: {str(evidence)[:400]}\n"
        "Emit a smaller, complete diff rather than a larger truncated one.",
        "truncated_output")


def _pre_oracle_call(model_id, role, max_tokens):
    """A model bought *before* the first test run, reading the same tree the
    executor will.

    Returned as a callable rather than executed inline because the repository
    is only readable inside `_ladder`'s container context. Buying the plan
    outside it -- which is what these arms used to do -- meant the H2
    challenger's architect and the grounded contract's locator were the only
    models in the study working blind, while every rung they were briefing
    could see the files. Both roles ask in so many words for real paths.
    """
    def call(problem, grounding):
        text, usage, _ = dispatch_model(
            model_id, _solve_prompt(problem, role, grounding=grounding),
            max_tokens=max_tokens, problem=problem)
        return text, usage
    call.model_id = model_id
    return call


def _spend(acc, usage):
    acc["usd"] += usage["as_run_usd"]
    acc["out"] += usage["output"]
    acc["tok"] += usage["total_tokens"]
    return usage["as_run_usd"]


def _result(passed, evidence, acc, loops, trace=None, ratio=None, report=None,
            grounding=None, guard_reasons=None):
    out = {
        "passed": bool(passed),
        "as_run_usd": round(acc["usd"], 6),
        "output_tokens": acc["out"],
        "total_tokens": acc["tok"],
        "repair_loops": loops,
        "triage_usd": 0.0,
        "error": "" if passed else str(evidence)[:500],
        # Why the LAST attempt died, and why every attempt died. `""` means the
        # repository's suite actually ran, which is the only case where the
        # pass/fail is a statement about the model. A sweep that reports a pass
        # rate without this cannot tell "the models are bad at this dataset"
        # apart from "89% of attempts never reached a test".
        "guard_reason": guard_reason(evidence),
        "guard_reasons": list(guard_reasons or []),
        "suite_reached": sum(1 for g in (guard_reasons or []) if not g),
        "attempts": len(guard_reasons or []),
        "grounding": grounding,
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
            plan="", planner_usd=0.0, acc=None, planner=None):
    """One escalating repair loop against the containerised oracle.

    The gate, the difficulty classifier and the degradation warning are the
    *same* ones the BigCodeBench-Hard routing study used (`src/routing.py`), so
    an `sbp_evidence_gate` row and an `r9_opus_on_evidence` row mean the same
    thing on two datasets with very different oracle costs.

    ``planner`` is a callable ``(problem, grounding) -> (text, usage)`` run
    once, after the repository has been read and before the first solve. It
    exists because the arms that buy a model *before* the oracle -- the H2
    challenger and the grounded-contract locator -- used to call it outside
    this function, which is outside the container, which meant the one model
    whose whole job is "name the real files" was the only one that never saw
    them. `PLANNER_ROLE` says *"Be concrete and name real paths"* to a model
    holding an issue and nothing else.

    Arms that pass no planner take a byte-identical path: the branch below is
    skipped, and `tests/test_swebench_pro_planner.py` pins that.
    """
    gate_fn = GATES[gate] if isinstance(gate, str) else gate
    trace = EscalationTrace()
    acc = acc if acc is not None else {"usd": planner_usd, "out": 0, "tok": 0}

    with SWEBenchProEnv(problem) as env:
        # Read the tree before the first token is spent. The container is up
        # either way, so this costs a handful of `git show` calls and removes
        # the localisation ceiling the blind prompt has.
        grounding, ground_meta = collect_grounding(env, problem)
        if planner is not None:
            plan, p_usage = planner(problem, grounding)
            _spend(acc, p_usage)
            ground_meta = dict(ground_meta or {}, planner_grounded=bool(grounding))
        model, think = tiers[0]
        text, usage, _ = dispatch_model(
            model, _solve_prompt(problem, EXECUTOR_ROLE if plan else SOLVER_ROLE,
                                 plan, grounding),
            max_tokens=MAX_PATCH_TOKENS, thinking_level=think, problem=problem)
        _spend(acc, usage)
        trace.rungs.append(f"{model}/{think or 'off'}")
        held = (model, think)
        patch = extract_patch(text)
        passed, evidence = env.score(patch)
        evidence = _relabel_truncated(evidence, usage)
        ratio, report = env.last_ratio, env.last_report
        guard_reasons = [guard_reason(evidence)]
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
                target, _repair_prompt(problem, patch, digest, plan, grounding,
                                       role=repair_role(evidence)),
                max_tokens=MAX_PATCH_TOKENS, thinking_level=think, problem=problem)
            _spend(acc, r_usage)
            trace.rungs.append(f"{target}/{think or 'off'}")
            if escalate:
                trace.frontier_used = True
                trace.frontier_rung = len(trace.rungs)

            patch = extract_patch(r_text)
            passed, evidence = env.score(patch)
            evidence = _relabel_truncated(evidence, r_usage)
            guard_reasons.append(guard_reason(evidence))
            oracle_calls += 1
            loops += 1
            if env.last_ratio is not None:
                ratio = env.last_ratio
            if env.last_report:
                report = env.last_report
            if passed:
                trace.solved_at = trace.rungs[-1]

    return _result(passed, evidence, acc, loops, trace, ratio, report,
                   grounding=ground_meta, guard_reasons=guard_reasons)


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

    The architect reads the repository, like every rung it briefs -- see
    `_pre_oracle_call`. Grounded, its call prices at ~$0.13/task against
    ~$0.06 blind; the difference buys the one thing `PLANNER_ROLE` asks for and
    the issue text cannot supply, which is which files actually exist.
    """
    return _ladder(problem, [(executor_model, "low")] * MAX_ORACLE_CALLS,
                   gate="never",
                   planner=_pre_oracle_call(planner_model, PLANNER_ROLE, 2048))


def gate_patch_health(difficulty, attempt, total_rungs):
    """Patch-health aware escalation gate for SWE-bench Pro."""
    if difficulty is None:
        return False, "no difficulty signal"
    if difficulty.is_environment:
        return False, f"environment failure ({difficulty.guard}); no model fixes this"
    if difficulty.is_hard:
        return True, f"patch health & evidence says {difficulty.level}: {'; '.join(difficulty.reasons)}"
    if attempt >= total_rungs:
        return True, f"cheap rungs exhausted ({attempt}/{total_rungs})"
    return False, f"failure is {difficulty.level}; standard repair"


# Declared for the same reason `gate_on_evidence` declares it: this gate reads
# `difficulty.level`, so on a row whose evidence carries no fact tier it
# degenerates into a counter gate. Without the flag `_ladder` never sets
# `trace.degraded` and the row reads as an evidence-routed result when it was
# not one.
gate_patch_health.requires_typed_evidence = True


@_arm(sj_required=True)
def run_sbp_grounded_contract(problem, locator_model=SONNET_ID,
                              executor_model=GEMINI_37_FLASH_ID,
                              frontier=OPUS_5_ID):
    """Candidate 1: Grounded Micro-Contract Localization Cascade.

    Sonnet-5 generates a strict, compact file & interface contract (<150 words)
    from the repository as it stands, then Gemini 3.7 Flash executes the diff.
    Escalates to Opus-5 on broad/stalled test failure evidence.

    "Grounded" was aspirational until the locator was moved inside the
    container: it was being asked for `TARGET_FILES: Exact repository file
    paths` while holding only the issue text. Reading the tree prices the
    locator at ~$0.035/task rather than ~$0.008.
    """
    return _ladder(problem, [(executor_model, "low"), (SONNET_ID, None)],
                   gate="evidence", frontier=frontier,
                   planner=_pre_oracle_call(locator_model,
                                            CONTRACT_LOCATOR_ROLE, 512))


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
        "name": "S0a. Single: gemini-3.7-flash low ({rungs} rungs)",
        "models": "Gemini 3.7 Flash (low) x{rungs}",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": lambda p: run_sbp_single(p, model_id=GEMINI_37_FLASH_ID,
                                        thinking_level="low"),
    },
    "sbp_single_sonnet": {
        "id": "sbp_single_sonnet", "category": CATEGORY,
        "name": "S0b. Single: claude-sonnet-5 ({rungs} rungs)",
        "models": "Claude Sonnet-5 x{rungs}",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": lambda p: run_sbp_single(p, model_id=SONNET_ID),
    },
    "sbp_cascade": {
        "id": "sbp_cascade", "category": CATEGORY,
        "name": "S1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate, {rungs} rungs)",
        "models": "Gemini 3.7 Flash -> Claude Sonnet-5",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": run_sbp_cascade,
    },
    "sbp_evidence_gate": {
        "id": "sbp_evidence_gate", "category": CATEGORY,
        "name": "S2. Evidence gate: flash -> sonnet/opus (evidence gate, {rungs} rungs)",
        "models": "Gemini 3.7 Flash -> Claude Sonnet-5 / Claude Opus-5 (evidence gate)",
        "triage_mode": "Straitjacket digest + evidence-gated escalation ($0.00)",
        "fn": run_sbp_evidence_gate,
    },
    "sbp_plan_exec": {
        "id": "sbp_plan_exec", "category": CATEGORY,
        "name": "S3. H2: opus-5 plans first, 3.7-flash implements and repairs ({rungs} rungs)",
        "models": "Claude Opus-5 plan + Gemini 3.7 Flash exec x{rungs}",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": run_sbp_plan_exec,
    },
    "sbp_single_opus": {
        "id": "sbp_single_opus", "category": OPUS_CATEGORY,
        "name": "S0c. Single: claude-opus-5 ({rungs} rungs)",
        "models": "Claude Opus-5 x{rungs}",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": lambda p: run_sbp_single(p, model_id=OPUS_5_ID),
    },
    # -- 3 Optimized Candidate Architectures --
    "sbp_grounded_contract": {
        "id": "sbp_grounded_contract", "category": CANDIDATES_CATEGORY,
        "name": "S4. Grounded Contract: Sonnet locator -> Flash exec -> Opus escalation ({rungs} rungs)",
        "models": "Claude Sonnet-5 contract + Gemini 3.7 Flash exec -> Claude Opus-5",
        "triage_mode": "Straitjacket digest + evidence-gated escalation ($0.00)",
        "fn": run_sbp_grounded_contract,
    },
    "sbp_patch_health_router": {
        "id": "sbp_patch_health_router", "category": CANDIDATES_CATEGORY,
        "name": "S5. Patch-Health Router: Flash -> Sonnet / Opus (health-aware gate, {rungs} rungs)",
        "models": "Gemini 3.7 Flash -> Claude Sonnet-5 / Claude Opus-5 (health gate)",
        "triage_mode": "Straitjacket digest + patch-health router ($0.00)",
        "fn": run_sbp_patch_health_router,
    },
    "sbp_sonnet_opus_sweetspot": {
        "id": "sbp_sonnet_opus_sweetspot", "category": CANDIDATES_CATEGORY,
        "name": "S6. Sweetspot: Sonnet-5 draft -> Evidence gate -> Opus-5 repair ({rungs} rungs)",
        "models": "Claude Sonnet-5 draft -> Claude Sonnet-5 / Claude Opus-5 (evidence gate)",
        "triage_mode": "Straitjacket digest + evidence-gated escalation ($0.00)",
        "fn": run_sbp_sonnet_opus_sweetspot,
    },
}


# ==============================================================================
# --- REGISTRY INVARIANTS ---
# ==============================================================================
#
# Two things about an arm are asserted here rather than left to a reader: what
# its name claims about the rung count, and whether the frontier rung it
# advertises can be reached at all.

# Which ladder and gate each frontier-claiming arm actually runs. Kept beside
# the registry so a new arm cannot quietly opt out of the check by not being
# listed -- `test_swebench_pro.py` asserts this table covers every variant
# whose `models` string names the frontier model.
_ARM_SHAPES = {
    "sbp_cascade": (len(TIERS), GATES["after_ladder"]),
    "sbp_evidence_gate": (len(TIERS), GATES["evidence"]),
    "sbp_grounded_contract": (2, GATES["evidence"]),
    "sbp_patch_health_router": (2, gate_patch_health),
    "sbp_sonnet_opus_sweetspot": (2, GATES["evidence"]),
}


def unreachable_frontier_arms(max_oracle_calls=None):
    """Arms whose advertised frontier rung cannot be called at this budget.

    This is the check that was missing when reports 21 and 23 were produced.
    Both ran with `MAX_ORACLE_CALLS = 2` over two-rung ladders, which makes
    `attempt == 1` the only gate evaluation of the whole run; no gate escalates
    on that, so Opus-5 was never invoked once. Five arms named after an Opus
    escalation shipped as plain flash->sonnet ladders, the three "candidate
    architectures" were the same arm three times, and their identical 0% was
    read as a finding about the models.
    """
    budget = MAX_ORACLE_CALLS if max_oracle_calls is None else max_oracle_calls
    return sorted(
        arm for arm, (n_tiers, gate) in _ARM_SHAPES.items()
        if not frontier_is_reachable(gate, n_tiers, budget))


def _finalise_registry():
    """Render `{rungs}` in the variant names and refuse a dishonest registry.

    The names carry the rung count because a report quotes them verbatim; a
    name that says "2 rungs" beside a run that made three is the same class of
    error as an unreachable frontier, just quieter.
    """
    for cfg in SWEBENCH_PRO_VARIANTS.values():
        cfg["name"] = cfg["name"].format(rungs=MAX_ORACLE_CALLS)
        cfg["models"] = cfg["models"].format(rungs=MAX_ORACLE_CALLS)
    broken = unreachable_frontier_arms()
    if broken:
        raise RuntimeError(
            "SWE-bench Pro registry is not runnable: "
            f"{', '.join(broken)} advertise a frontier rung that cannot be "
            f"reached with SBP_MAX_ORACLE_CALLS={MAX_ORACLE_CALLS} over a "
            f"{len(TIERS)}-rung ladder. Raise the oracle budget above the rung "
            "count, or drop the frontier model from those arms' names.")


_finalise_registry()
