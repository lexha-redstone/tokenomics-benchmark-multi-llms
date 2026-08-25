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

import os
import re

from .config import GEMINI_37_FLASH_ID, SONNET_ID, OPUS_5_ID
from .client import dispatch_model
from .evaluator import (FeatureBenchEnv, extract_patch, featurebench_test_files,
                        featurebench_test_ratio)
from .architectures import _arm, _treat_error
from .routing import (GATES, EscalationTrace, classify,
                      frontier_is_reachable)

# The ladder, cheapest first. One place, so "one rung up" is defined once.
TIERS = [
    (GEMINI_37_FLASH_ID, "low"),
    (SONNET_ID, None),
]
FRONTIER = OPUS_5_ID

# Three, not two. `_ladder` evaluates its gate once per repair turn, so a budget
# of K oracle calls yields K-1 evaluations at attempts 1..K-1, and every gate
# compares `attempt` against `len(tiers)` -- which is 2. At K=2 the only
# evaluation is `attempt == 1`, no gate can answer "escalate", and the frontier
# rung is unreachable code for every arm that has one.
#
# `docs/featurebench-n48-lessons.md` §2 is the audit of what that produced:
# report 20's arms did not share a budget, three rows are labelled as
# architectures they did not run, and the H2 challenger was compared against
# rivals that had an extra oracle call. `unreachable_frontier_arms` below is
# rule 2 of that document's §8, asserted rather than remembered.
MAX_ORACLE_CALLS = int(os.environ.get("FB_MAX_ORACLE_CALLS", "3"))

# The sample row's gold patch is 51k chars (~13k tokens) and the schema tops out
# at 227k, so the 8192 this started at truncated a correct answer into an
# inapplicable one. 32k is a practical ceiling rather than a comfortable one:
# some rows' gold exceeds any output budget a model can hit in one turn, which
# is part of why frontier models resolve only 20-47% here.
MAX_PATCH_TOKENS = 32768

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

# Contracted roles for unified diff precision: enforces minimal context hunks,
# zero-context new file creation, and relative paths (stripping /testbed/).
DIFF_CONTRACT_SOLVER_ROLE = (
    "You are a principal software engineer implementing a feature in an existing repository.\n"
    "CRITICAL UNIFIED DIFF REQUIREMENTS:\n"
    "1. File Paths: Always use relative paths starting with `a/` and `b/` (strip any `/testbed/` prefix).\n"
    "2. New Files: If creating a new file specified in the prompt, use the standard header:\n"
    "   --- /dev/null\n"
    "   +++ b/<path>\n"
    "   @@ -0,0 +1,<total_lines> @@\n"
    "   followed by each line prefixed with '+'.\n"
    "3. Existing Files: Keep hunk context lines minimal (1-2 lines) so `git apply` / `patch -p1` applies cleanly.\n"
    "4. Output format: Output ONLY ONE ```diff code block spanning all modified/created files.\n\n"
)

DIFF_CONTRACT_REPAIR_ROLE = (
    "You are a principal software engineer repairing a failed feature patch in an existing repository.\n"
    "If the previous attempt failed with `patch did not apply`: Ensure all paths are relative without `/testbed/`, "
    "new files use `--- /dev/null` and `+++ b/<path>`, and existing files use minimal context lines.\n"
    "If unit tests failed: Analyze the Straitjacket test error digest below, fix the root cause, and output "
    "the COMPLETE corrected unified git diff against the base commit.\n"
    "Output ONLY ONE ```diff code block.\n\n"
)

GROUNDED_SOLVER_ROLE = (
    "You are a senior engineer implementing a complete feature in an existing "
    "repository. The current contents of the relevant files are quoted below.\n"
    "RULES FOR THE DIFF:\n"
    "1. Every context line and every `-` line you write for an EXISTING file must "
    "be copied character-for-character from the quoted text, including indentation.\n"
    "2. Include at least 3 lines of real context above and below each change, so "
    "the hunk can be located even if the line numbers are off.\n"
    "3. For a file that is NOT quoted above, treat it as absent: create it with "
    "`--- /dev/null` and `+++ b/<path>`. Never invent the contents of a file you "
    "were not shown.\n"
    "4. Use `a/` and `b/` prefixes and relative paths (no `/testbed/`).\n"
    "5. Output ONLY ONE ```diff code block, spanning every file you change.\n\n"
)

GROUNDED_REPAIR_ROLE = (
    "You are a senior engineer repairing a failed patch. The current contents of "
    "the relevant files are quoted below, and the previous attempt's failure "
    "follows them.\n"
    "If the failure says the patch did NOT apply, the applier prints the exact "
    "text it searched for and could not find. Compare that block against the "
    "quoted file and copy the real bytes -- do not re-guess them.\n"
    "If the failure is a test error, fix the cause and re-emit the whole patch.\n"
    "The repository is reset to the base commit before your patch is applied, so "
    "always emit the COMPLETE diff, never an increment.\n"
    "Output ONLY ONE ```diff code block.\n\n"
)

MANIFEST_ROLE = (
    "You are a senior software architect. Read the feature request below and extract a structured "
    "FILE AND INTERFACE MANIFEST.\n"
    "List each target file (relative path from repository root, stripping any `/testbed/` prefix) and every class, "
    "function, and exception that belongs to that file according to the Interface Descriptions.\n"
    "Be concise, concrete, and output ONLY the manifest. Under 300 words.\n\n"
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


# ==============================================================================
# --- REPOSITORY GROUNDING ---
# ==============================================================================
#
# The deepest defect the N=48 sweep exposed was not in the routing policy, the
# budget or the labels. It was in `_context`: the model was given the repository
# *name*, the base commit, the test filenames and the feature request, and then
# asked for a unified diff. A unified diff for an existing file is a claim about
# bytes that are already on disk -- `git apply` matches the context lines
# literally -- so a model that has never seen the file is guessing them. 331 of
# the 353 recorded failures (94%) were `patch did not apply`, and the two arms
# built to fix it (F4's diff contract, F6's manifest) both worked on diff
# *syntax*, which was never the thing that was wrong.
#
# Grounding closes that. Before the first attempt it quotes the files the row is
# about, read out of the row's own container at the row's own base commit, so
# the context lines the model writes can be copied rather than invented.
#
# Two deliberate limits:
#
#   * **The graded test files are not quoted by default.** They are the API
#     contract the row is scored on, and quoting them would change what the
#     benchmark measures rather than how well the harness measures it. Set
#     `FB_GROUND_TESTS=1` to include them, and say so in the report if you do.
#   * **Grounding is an enrichment, never a precondition.** A container that
#     cannot be read yields an empty block and the arm runs the blind prompt, so
#     a grounding failure shows up as a `grounding` receipt with `chars: 0`
#     rather than as a task failure.

FB_GROUNDING_CHARS = int(os.environ.get("FB_GROUNDING_CHARS", "48000"))
FB_GROUNDING_PER_FILE = int(os.environ.get("FB_GROUNDING_PER_FILE", "16000"))
FB_GROUND_TESTS = os.environ.get("FB_GROUND_TESTS", "") not in ("", "0", "false")

# A path is only read when it looks like a path. Anything else is a shell
# hazard, and these strings come out of a model-readable problem statement.
_SAFE_PATH_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,190}")
_PATH_IN_TEXT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|pyi|cfg|toml|ini)")
_TICKED_RE = re.compile(r"`([^`\n]{3,60})`")
_FILE_MARKER = "@@FB_FILE@@"


# These images root the repository at `/testbed`, and the problem statements
# quote absolute paths inside it. `_PATH_IN_TEXT_RE` cannot capture the leading
# slash without also matching prose, so the prefix is stripped here instead --
# the N=2 run read `docs/conf.py` and skipped
# `testbed/src/packaging/metadata.py`, which is this bug: the file it most
# needed was requested under a path that does not exist.
_CONTAINER_ROOTS = ("testbed/", "workspace/", "repo/", "app/", "src/app/")

# Quoted budget is finite, so spend it on code. A statement that mentions
# `docs/conf.py` is almost never asking for a change to `docs/conf.py`.
_LOW_VALUE_DIRS = ("docs/", "doc/", "examples/", "example/", "benchmarks/",
                   "benchmark/", "scripts/", ".github/")


def _normalise_repo_path(path):
    """Make a path from the problem statement resolvable inside the container."""
    p = (path or "").strip().lstrip("/")
    for root in _CONTAINER_ROOTS:
        if p.startswith(root):
            return p[len(root):]
    return p


def _rank_paths(paths):
    """Source before documentation, shallow before deep, order otherwise kept."""
    return sorted(paths, key=lambda p: (p.startswith(_LOW_VALUE_DIRS),
                                        p.count("/")))


def _dedup(items):
    seen, out = set(), []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _candidate_paths(problem, limit=24):
    """Repository paths this row is most likely to be about, best guess first.

    The statement names them far more often than not: FeatureBench rows are
    generated from real PRs and carry their file list in prose. What the
    statement does not give, :func:`_grep_paths` looks for.
    """
    graded = set(featurebench_test_files(problem))
    found = _PATH_IN_TEXT_RE.findall(str(problem.get("problem_statement") or ""))
    paths = _dedup(_normalise_repo_path(p) for p in found)
    paths = [p for p in paths if p and _SAFE_PATH_RE.fullmatch(p)]
    if not FB_GROUND_TESTS:
        paths = [p for p in paths if p not in graded]
    return _rank_paths(paths)[:limit]


def _search_terms(problem, limit=8):
    """Backticked identifiers from the statement, usable as `git grep` terms."""
    ticked = _TICKED_RE.findall(str(problem.get("problem_statement") or ""))
    return _dedup(t.strip() for t in ticked
                  if re.fullmatch(r"[\w.]{3,60}", (t or "").strip()))[:limit]


def _grep_paths(env, terms, limit=12):
    """Files that mention any of `terms`, via the repository's own index."""
    hits = []
    for term in terms:
        if not re.fullmatch(r"[\w.]{3,60}", term):
            continue
        r = env._sh(f"git grep -l --fixed-strings -- '{term}' | head -n 20",
                    check=False)
        hits.extend(l.strip() for l in (r.stdout or "").splitlines() if l.strip())
        if len(_dedup(hits)) >= limit:
            break
    py = [p for p in _dedup(hits) if p.endswith((".py", ".pyi"))]
    return _rank_paths(py)[:limit]


def _resolve_by_basename(env, missing, limit=8):
    """Find a named file that is not where the statement said it was.

    Statements quote paths against several roots (`/testbed/...`, the package
    directory, the sdist layout). When the literal path is not present, the
    basename usually still is and is usually unique.
    """
    found = []
    for path in missing[:limit]:
        base = path.rsplit("/", 1)[-1]
        if not re.fullmatch(r"[\w.-]{3,80}", base):
            continue
        r = env._sh(f"git ls-files '*/{base}' '{base}' | head -n 4", check=False)
        found.extend(l.strip() for l in (r.stdout or "").splitlines() if l.strip())
    return _rank_paths(_dedup(found))


def _read_repo_files(env, paths, budget, per_file=None):
    """Quote `paths` from inside the container. Returns (blocks, read, skipped).

    One `docker exec` for the whole batch: a per-file exec would add a process
    round-trip per candidate to a dataset whose cost is already dominated by
    container time.
    """
    per_file = FB_GROUNDING_PER_FILE if per_file is None else per_file
    safe = [p for p in paths if _SAFE_PATH_RE.fullmatch(p or "")]
    if not safe or budget <= 0:
        return [], [], list(paths)

    script = "; ".join(
        f'if [ -f "{p}" ]; then echo "{_FILE_MARKER} {p}"; '
        f'head -c {per_file} "{p}"; echo; fi'
        for p in safe)
    r = env._sh(script, check=False)

    blocks, read, skipped, spent = [], [], [], 0
    chunks = (r.stdout or "").split(_FILE_MARKER + " ")
    present = set()
    for chunk in chunks[1:]:
        head, _, body = chunk.partition("\n")
        path = head.strip()
        present.add(path)
        block = f"--- BEGIN {path} ---\n{body.rstrip()}\n--- END {path} ---\n"
        if spent + len(block) > budget:
            skipped.append(path)
            continue
        blocks.append(block)
        read.append(path)
        spent += len(block)
    skipped.extend(p for p in safe if p not in present and p not in skipped)
    return blocks, read, skipped


def collect_repo_context(env, problem, budget=None):
    """Quote the repository files this row is about. Returns (text, meta).

    `text` is empty whenever grounding is disabled or nothing could be read, in
    which case the prompt is byte-identical to the blind one -- so an arm that
    fails to ground is still a valid run of the blind arm rather than a lost row.
    """
    budget = FB_GROUNDING_CHARS if budget is None else budget
    meta = {"enabled": bool(budget), "read": [], "skipped": [], "searched": [],
            "relocated": [], "chars": 0, "tests_quoted": bool(FB_GROUND_TESTS)}
    if not budget:
        return "", meta

    try:
        wanted = _candidate_paths(problem)
        blocks, read, skipped = _read_repo_files(env, wanted, budget)

        # A path the statement named but the container does not have is usually
        # the right file under a different root, not a file that is absent.
        if skipped:
            spent = sum(len(b) for b in blocks)
            relocated = [p for p in _resolve_by_basename(env, skipped)
                         if p not in read]
            if relocated:
                meta["relocated"] = relocated
                more, read2, _ = _read_repo_files(env, relocated, budget - spent)
                blocks, read = blocks + more, read + read2
                skipped = [p for p in skipped
                           if p.rsplit("/", 1)[-1] not in
                           {q.rsplit("/", 1)[-1] for q in read2}]

        # The statement does not always name every file. Only pay for a search
        # when it did not name enough of them.
        if len(read) < 3:
            terms = _search_terms(problem)
            if terms:
                meta["searched"] = terms
                extra = [p for p in _grep_paths(env, terms) if p not in read]
                more, read2, skipped2 = _read_repo_files(
                    env, extra, budget - sum(len(b) for b in blocks))
                blocks, read = blocks + more, read + read2
                skipped = skipped + skipped2
    except Exception as e:                                   # noqa: BLE001
        meta["error"] = str(e)[:200]
        return "", meta

    if not blocks:
        return "", meta
    meta.update(read=read, skipped=_dedup(skipped), chars=sum(len(b) for b in blocks))
    header = (
        f"Repository files at commit {str(problem.get('base_commit', ''))[:12]}, "
        f"quoted verbatim ({len(read)} file(s)). Your diff's context lines MUST "
        "match these bytes exactly -- copy them, do not retype them. A file that "
        "is not quoted here you have NOT seen: create it with `--- /dev/null` "
        "rather than guessing its current contents.\n\n")
    tail = ("\n[not quoted: " + ", ".join(meta["skipped"][:8]) + "]\n"
            if meta["skipped"] else "")
    return header + "\n".join(blocks) + tail, meta


def _solve_prompt(problem, role=SOLVER_ROLE, plan="", repo=""):
    plan_block = f"\nArchitect's implementation plan:\n{plan}\n" if plan else ""
    repo_block = f"\n{repo}\n" if repo else ""
    return role + _context(problem) + repo_block + plan_block


def _repair_prompt(problem, patch, digest, plan="", role=REPAIR_ROLE, repo=""):
    plan_block = f"\nArchitect's implementation plan:\n{plan}\n" if plan else ""
    repo_block = f"\n{repo}\n" if repo else ""
    return (
        role + _context(problem) + repo_block + plan_block
        + f"\nYour current patch:\n```diff\n{patch}\n```\n\n"
        + f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n"
    )


def _spend(acc, usage):
    acc["usd"] += usage["as_run_usd"]
    acc["out"] += usage["output"]
    acc["tok"] += usage["total_tokens"]
    return usage["as_run_usd"]


# Enough of the candidate to read the line `git apply` complained about, and
# far short of a 51k-char gold patch. The N=2 run recorded
# `corrupt patch at line 8` on all five strategies and the patch itself was
# nowhere on disk, so the diagnosis needed a scratch-repository reproduction
# instead of a lookup.
FB_PATCH_EXCERPT_CHARS = int(os.environ.get("FB_PATCH_EXCERPT_CHARS", "4000"))


def _patch_excerpt(patch, limit=None):
    limit = FB_PATCH_EXCERPT_CHARS if limit is None else limit
    text = patch or ""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]}\n[... {len(text) - limit} chars elided ...]\n{text[-(limit - head):]}"


def _result(passed, evidence, acc, loops, trace=None, ratio=None, grounding=None,
            patch=None):
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
    if not passed and patch:
        # Only on a failure, and only the last candidate: a passing row's patch
        # is not a diagnostic and every stored byte is paid for on every read.
        out["candidate_patch"] = _patch_excerpt(patch)
    if grounding is not None:
        # What the arm was actually shown, so "grounded" is a receipt rather
        # than an inference from the pass rate. An arm whose container could
        # not be read records `chars: 0` and is a blind run, not a grounded one.
        out["grounding"] = grounding
    return out


def _ladder(problem, tiers, gate="after_ladder", frontier=FRONTIER,
            plan="", planner_usd=0.0, acc=None,
            solver_role=None, repair_role=REPAIR_ROLE, ground=False):
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
    init_role = solver_role or (EXECUTOR_ROLE if plan else SOLVER_ROLE)

    with FeatureBenchEnv(problem) as env:
        # Read once, before the first attempt. The worktree is reset to base
        # between attempts, so the files do not change and re-reading them
        # would buy nothing but container time.
        repo, grounding = collect_repo_context(env, problem) if ground else ("", None)

        model, think = tiers[0]
        text, usage, _ = dispatch_model(
            model, _solve_prompt(problem, init_role, plan, repo=repo),
            max_tokens=MAX_PATCH_TOKENS, thinking_level=think, problem=problem)
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
            try:
                escalate, why = gate_fn(difficulty, loops + 1, len(tiers), last_err=evidence)
            except TypeError:
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
                target,
                _repair_prompt(problem, patch, digest, plan,
                               role=repair_role, repo=repo),
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
            r = featurebench_test_ratio(evidence)
            if r is not None:
                ratio = r
            if passed:
                trace.solved_at = trace.rungs[-1]

    return _result(passed, evidence, acc, loops, trace, ratio, grounding,
                   patch=patch)


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
# --- NEW CANDIDATE ARCHITECTURES (H2 PARETO-OPTIMAL EXTENSIONS) ---
# ==============================================================================

@_arm(sj_required=True)
def run_fb_diff_contract(problem, model_id=GEMINI_37_FLASH_ID, repair_model_id=SONNET_ID):
    """Candidate 1: Diff-Contracted Multi-File Solver.

    Directly addresses the dominant failure mode on FeatureBench (80%+ of errors
    are `patch did not apply`). Enforces strict unified diff path/hunk standards:
    relative paths (no /testbed/), zero-context new file headers (--- /dev/null),
    and minimal context lines to ensure git apply and patch --fuzz=5 cleanly succeed.
    """
    return _ladder(
        problem,
        [(model_id, "low"), (repair_model_id, None if repair_model_id == SONNET_ID else "low")],
        gate="never",
        solver_role=DIFF_CONTRACT_SOLVER_ROLE,
        repair_role=DIFF_CONTRACT_REPAIR_ROLE,
    )


def gate_diff_aware(difficulty, loop, n_tiers, last_err=""):
    """Escalate on a typed hard failure, or on a repeated patch-apply failure.

    Module-level rather than a closure inside the arm, for two reasons: the
    registry invariant at the bottom of this file has to be able to evaluate
    it, and a gate nobody can reach is a gate nobody can test.

    It reads `difficulty` -- which now types pre-execution failures, so
    `apply_failed` twice running arrives as `stalled` -- rather than
    substring-matching the evidence prose. The original spelled
    `"patch did not apply" in str(last_err)`, making this gate's behaviour
    depend on the exact wording of a message in another module. It broke
    silently the moment that message changed, and a silently-off gate is
    precisely the defect this arm was added to fix.
    """
    if difficulty is not None and difficulty.is_environment:
        return False, f"environment failure ({difficulty.guard}); no model fixes this"
    if difficulty is not None and difficulty.level in ("broad", "stalled"):
        return True, f"typed failure classified as {difficulty.level}"
    if loop >= n_tiers:
        return True, "reached end of tier ladder"
    return False, "stay on standard tier"


gate_diff_aware.requires_typed_evidence = True


@_arm(sj_required=True)
def run_fb_diff_aware_gate(problem, tiers=TIERS, frontier=FRONTIER):
    """Candidate 2: Diff-Aware Evidence Escalation Gate.

    Fixes the blind spot in standard evidence gate where `patch did not apply` was
    classified as shallow and never escalated to Opus-5. Escalate to Frontier (Opus-5)
    when:
    (1) Unit tests show `broad` or `stalled` failure (>=3 failing identities), OR
    (2) Patch application fails consecutively across attempts (stalled format).
    """
    return _ladder(
        problem,
        tiers,
        gate=gate_diff_aware,
        frontier=frontier,
        solver_role=DIFF_CONTRACT_SOLVER_ROLE,
        repair_role=DIFF_CONTRACT_REPAIR_ROLE,
    )


@_arm(sj_required=True)
def run_fb_spec_deconstruct(problem, architect_model=GEMINI_37_FLASH_ID,
                            solver_model=GEMINI_37_FLASH_ID, repair_model=SONNET_ID):
    """Candidate 3: Manifest Deconstruction & Component Synthesis.

    Deconstructs 40k+ character problem statements into a structured File-to-Interface
    Manifest before synthesizing unified diffs. Solves multi-file interface omission
    and enables component-targeted repair without vague natural language overhead.
    """
    acc = {"usd": 0.0, "out": 0, "tok": 0}
    manifest_prompt = MANIFEST_ROLE + _context(problem)
    manifest, usage, _ = dispatch_model(architect_model, manifest_prompt,
                                        max_tokens=1024, problem=problem)
    _spend(acc, usage)

    return _ladder(
        problem,
        [(solver_model, "low"), (repair_model, None if repair_model == SONNET_ID else "low")],
        gate="never",
        plan=f"Target File & Interface Manifest:\n{manifest}",
        acc=acc,
        solver_role=DIFF_CONTRACT_SOLVER_ROLE,
        repair_role=DIFF_CONTRACT_REPAIR_ROLE,
    )


@_arm(sj_required=True)
def run_fb_grounded(problem, tiers=TIERS):
    """Same ladder as `fb_cascade`, with the repository actually shown.

    The control for the only defect the N=48 sweep left unexplained. F4 and F6
    attacked the 94% `patch did not apply` rate through diff *syntax* -- a
    stricter contract, a file manifest -- and neither moved it, because the
    models were not writing malformed diffs. They were writing well-formed
    diffs about files they had never read. This arm changes one thing: the
    files are quoted from the row's own container at its own base commit.

    Held constant against `fb_cascade`: tiers, gate, oracle budget, output
    budget. The extra input tokens are the cost of the treatment and show up in
    `as_run_usd` where they belong.
    """
    return _ladder(problem, tiers, gate="after_ladder", ground=True,
                   solver_role=GROUNDED_SOLVER_ROLE,
                   repair_role=GROUNDED_REPAIR_ROLE)


@_arm(sj_required=True)
def run_fb_grounded_gate(problem, tiers=TIERS, frontier=FRONTIER):
    """Grounded ladder with the N=148 evidence gate on top.

    `fb_grounded` is to `fb_cascade` what this is to `fb_evidence_gate`, so the
    pair answers the routing question and the grounding question separately
    rather than confounding them the way report 22's arms did.
    """
    return _ladder(problem, tiers, gate="evidence", frontier=frontier,
                   ground=True, solver_role=GROUNDED_SOLVER_ROLE,
                   repair_role=GROUNDED_REPAIR_ROLE)


# ==============================================================================
# --- VARIANT REGISTRY ---
# ==============================================================================

CATEGORY = "8. FeatureBench expensive-oracle study"

# Same reasoning as ClassEval's opus baseline: the frontier single is priced far
# above the rest, so it sits in its own category and is opt-in rather than
# silently repricing every `--group featurebench` sweep.
OPUS_CATEGORY = "8b. FeatureBench frontier baseline"

# Same reasoning as the frontier baseline above, for the opposite reason: the
# grounded arms buy their advantage with input tokens (~30k more per attempt),
# so folding them into `--group featurebench` would silently reprice every
# default sweep. They are the A/B partner of `fb_cascade` and
# `fb_evidence_gate`, and the honest way to run them is against those two by
# name:
#
#   python3 run_benchmark.py --dataset featurebench --no-cache \
#       --variants fb_cascade,fb_grounded,fb_evidence_gate,fb_grounded_gate
GROUNDED_CATEGORY = "8g. FeatureBench repository-grounded"

FEATUREBENCH_VARIANTS = {
    "fb_single_flash": {
        "id": "fb_single_flash", "category": CATEGORY,
        "name": "F0a. Single: gemini-3.7-flash low ({rungs} rungs)",
        "models": "Gemini 3.7 Flash (low) x{rungs}",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": lambda p: run_fb_single(p, model_id=GEMINI_37_FLASH_ID, thinking_level="low"),
    },
    "fb_single_sonnet": {
        "id": "fb_single_sonnet", "category": CATEGORY,
        "name": "F0b. Single: claude-sonnet-5 ({rungs} rungs)",
        "models": "Claude Sonnet-5 x{rungs}",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": lambda p: run_fb_single(p, model_id=SONNET_ID),
    },
    "fb_cascade": {
        "id": "fb_cascade", "category": CATEGORY,
        "name": "F1. Cascade: 3.7-flash -> sonnet-5 (attempt-count gate)",
        "models": "Gemini 3.7 Flash -> Claude Sonnet-5",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": run_fb_cascade,
    },
    "fb_evidence_gate": {
        "id": "fb_evidence_gate", "category": CATEGORY,
        "name": "F2. Evidence gate: same tiers, escalate when the digest says hard",
        "models": "Gemini 3.7 Flash -> Claude Sonnet-5 / Opus-5 (evidence gate)",
        "triage_mode": "Straitjacket digest + evidence-gated escalation ($0.00)",
        "fn": run_fb_evidence_gate,
    },
    "fb_plan_exec": {
        "id": "fb_plan_exec", "category": CATEGORY,
        "name": "F3. H2: opus-5 plans first, 3.7-flash implements and repairs",
        "models": "Claude Opus-5 plan + Gemini 3.7 Flash exec x{rungs}",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": run_fb_plan_exec,
    },
    "fb_diff_contract": {
        "id": "fb_diff_contract", "category": CATEGORY,
        "name": "F4. Diff-Contract: Flash low -> Sonnet-5 (Strict unified diff anchoring)",
        "models": "Gemini 3.7 Flash (low) -> Claude Sonnet-5 (contracted diffs)",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": run_fb_diff_contract,
    },
    "fb_diff_aware_gate": {
        "id": "fb_diff_aware_gate", "category": CATEGORY,
        "name": "F5. Diff-Aware Evidence Gate: Flash low -> Sonnet-5 / Opus-5 on hard/stalled",
        "models": "Gemini 3.7 Flash -> Claude Sonnet-5 / Opus-5 (diff-aware gate)",
        "triage_mode": "Straitjacket digest + diff-aware escalation ($0.00)",
        "fn": run_fb_diff_aware_gate,
    },
    "fb_spec_deconstruct": {
        "id": "fb_spec_deconstruct", "category": CATEGORY,
        "name": "F6. Spec Deconstruct: Manifest extraction + Flash low synthesis & repair",
        "models": "Gemini 3.7 Flash manifest + Flash/Sonnet diff synthesis",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": run_fb_spec_deconstruct,
    },
    "fb_grounded": {
        "id": "fb_grounded", "category": GROUNDED_CATEGORY,
        "name": "F7. Grounded cascade: repository quoted, Flash low -> Sonnet-5 ({rungs} rungs)",
        "models": "Gemini 3.7 Flash (low) -> Claude Sonnet-5, source-grounded",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": run_fb_grounded,
    },
    "fb_grounded_gate": {
        "id": "fb_grounded_gate", "category": GROUNDED_CATEGORY,
        "name": "F8. Grounded evidence gate: repository quoted, escalate on hard evidence ({rungs} rungs)",
        "models": "Gemini 3.7 Flash -> Claude Sonnet-5 / Opus-5, source-grounded",
        "triage_mode": "Straitjacket digest + evidence-gated escalation ($0.00)",
        "fn": run_fb_grounded_gate,
    },
    "fb_single_opus": {
        "id": "fb_single_opus", "category": OPUS_CATEGORY,
        "name": "F0c. Single: claude-opus-5 ({rungs} rungs)",
        "models": "Claude Opus-5 x{rungs}",
        "triage_mode": "Straitjacket contained digest ($0.00)",
        "fn": lambda p: run_fb_single(p, model_id=OPUS_5_ID),
    },
}



# ==============================================================================
# --- REGISTRY INVARIANTS ---
# ==============================================================================
#
# Rule 2 of `docs/featurebench-n48-lessons.md` §8 -- "pick one oracle budget and
# assert it" -- made executable. The audit it comes from found that report 20's
# arms did not share a budget, that three rows are labelled as architectures
# they did not run, and that at two oracle calls over a two-rung ladder the
# frontier tier is unreachable for every arm that advertises one.

_ARM_SHAPES = {
    "fb_cascade": (len(TIERS), GATES["after_ladder"]),
    "fb_evidence_gate": (len(TIERS), GATES["evidence"]),
    "fb_diff_aware_gate": (len(TIERS), gate_diff_aware),
    "fb_grounded_gate": (len(TIERS), GATES["evidence"]),
}


def unreachable_frontier_arms(max_oracle_calls=None):
    """Arms whose advertised frontier rung cannot be called at this budget."""
    budget = MAX_ORACLE_CALLS if max_oracle_calls is None else max_oracle_calls
    return sorted(
        arm for arm, (n_tiers, gate) in _ARM_SHAPES.items()
        if not frontier_is_reachable(gate, n_tiers, budget))


def _finalise_registry():
    """Render `{rungs}` in the variant names and refuse a dishonest registry.

    A report quotes these names verbatim, and report 20 shipped rows reading
    "(2 rungs)" beside arms that had spent three. The count is derived from the
    constant that decides it rather than typed out beside it.
    """
    for cfg in FEATUREBENCH_VARIANTS.values():
        cfg["name"] = cfg["name"].format(rungs=MAX_ORACLE_CALLS)
        cfg["models"] = cfg["models"].format(rungs=MAX_ORACLE_CALLS)
    broken = unreachable_frontier_arms()
    if broken:
        raise RuntimeError(
            "FeatureBench registry is not runnable: "
            f"{', '.join(broken)} advertise a frontier rung that cannot be "
            f"reached with FB_MAX_ORACLE_CALLS={MAX_ORACLE_CALLS} over a "
            f"{len(TIERS)}-rung ladder. Raise the oracle budget above the rung "
            "count, or drop the frontier model from those arms' names.")


_finalise_registry()
