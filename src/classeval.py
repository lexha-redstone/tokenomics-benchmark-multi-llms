# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
ClassEval architecture arms: does routing sub-tasks by difficulty beat a cascade?

Why this module exists
----------------------
On BigCodeBench-Hard a task is one function with one verdict, so the only thing
an architecture can do with a failure is escalate the *whole* task. The N=100
sweep found that what the next turn escalates **to** is the only thing that
reliably moved the number (README section 1).

ClassEval breaks that symmetry. A task is a class of ~4 methods, the dataset
labels each method's dependency structure, and each method ships its own test
class. So a sub-task can be routed to a model by its difficulty, and -- this is
the part that makes the experiment worth running -- the result can be
*attributed*: every arm here records, per method, which model wrote it, what
tier it was, whether it passed, and what it cost.

The hypothesis under test (H1 in docs/pattern-dataset-selection.md): when one
task contains sub-tasks of unequal difficulty, difficulty routing should beat a
cascade at matched cost, because the cascade re-solves the easy methods at
frontier prices every time it escalates.

Reading the arms
----------------
    ce_single_*        one model, whole class, one self-repair turn
    ce_cascade         lite -> flash -> flash on the WHOLE class (the shape to beat)
    ce_plan_exec       flash plans per-method contracts, lite implements
    ce_route_by_tier   THE HYPOTHESIS: each method to a model chosen by its tier
    ce_route_flat      THE CONTROL: same per-method loop, one model for every
                       method. Without it, a win for ce_route_by_tier cannot be
                       told apart from "generating method-by-method is better",
                       which is a different claim entirely.

`ce_route_flat` is not optional. The BCB analysis had to reconstruct a
budget-matched comparison after the fact because the cascade arms got three
attempts to sj_hybrid's two; designing the control in from the start is what
keeps this sweep from needing the same rescue.
"""

import re
import textwrap

from .config import GEMINI_35_FLASH_LITE_ID, GEMINI_37_FLASH_ID, SONNET_ID, OPUS_5_ID
from .client import dispatch_model
from .evaluator import (extract_code, missing_class_error, run_classeval_class,
                        run_classeval_method, classeval_subtask_summary)

# Imported for their side-effect-free helpers; the decorator and the error
# treatment must be the same ones every other arm in the repo uses, or the
# containment ledger and the $0.00 triage claim stop meaning the same thing.
from .architectures import _arm, _treat_error

# The ladder a difficulty router draws from, cheapest first. Kept as one list so
# "one tier up" is defined in a single place.
LADDER = [GEMINI_35_FLASH_LITE_ID, GEMINI_37_FLASH_ID, SONNET_ID, OPUS_5_ID]

# Which rung each difficulty tier is routed to. This is the arm's policy and the
# thing the experiment is actually testing, so it lives here in the open rather
# than being buried in the routing function.
TIER_ROUTE = {
    "standalone": (GEMINI_35_FLASH_LITE_ID, None),
    "lib_dep":    (GEMINI_35_FLASH_LITE_ID, None),
    "field_dep":  (GEMINI_35_FLASH_LITE_ID, None),
    "field_lib":  (GEMINI_37_FLASH_ID, "low"),
    "method_dep": (GEMINI_37_FLASH_ID, "medium"),
}

SOLVER_ROLE = (
    "You are an expert Python programmer. Implement every method of the class "
    "below. Keep the class name, method names, decorators and signatures exactly "
    "as given. Output ONLY one ```python code block containing the complete "
    "module, imports included.\n\n"
)
METHOD_ROLE = (
    "You are an expert Python programmer. Implement EXACTLY ONE method of the "
    "class below. Keep its name, decorators and signature exactly as given. "
    "Assume the other methods already exist and behave as documented. Output "
    "ONLY one ```python code block containing that single method definition and "
    "nothing else -- no class statement, no imports.\n\n"
)
PLANNER_ROLE = (
    "You are a senior software architect. For the class below, write a short "
    "implementation contract for EACH method: the algorithm in one or two lines, "
    "the edge cases that its tests will probe, and any method it must call. Do "
    "not write code. Be terse.\n\n"
)
REPAIR_ROLE = (
    "You are an expert Python programmer. The class below FAILED its unit tests. "
    "Fix it. Output ONLY one ```python code block containing the complete "
    "corrected module.\n\n"
)
METHOD_REPAIR_ROLE = (
    "You are an expert Python programmer. One method of the class below FAILED "
    "its unit tests. Rewrite THAT METHOD ONLY. Output ONLY one ```python code "
    "block containing the single corrected method definition.\n\n"
)


# ==============================================================================
# --- CANDIDATE ASSEMBLY ---
# ==============================================================================

def normalise_method(src, decorators=()):
    """Re-indent one method definition so it sits inside a class body.

    Two shapes arrive here and both have to come out identical:

      * the dataset's, where only the FIRST line was dedented to column 0 and
        the rest still carry their in-class indentation (`def` at 0, body at 8);
      * a model's, which is normally a self-consistent block (`def` at 0, body
        at 4).

    Distinguishing them by the remainder's minimum indent handles both, and
    handles the decorated case where the first line is `@staticmethod` and the
    `def` beneath it is already at 4.

    ``decorators`` are prepended when the source does not already carry them.
    The dataset drops them inconsistently -- see `_classeval_decorators` in
    src/datasets.py -- and a lost `@staticmethod` fails as an argument-count
    error that looks exactly like a model mistake.
    """
    lines = (src or "").strip("\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""

    first, rest = lines[0], lines[1:]
    indents = [len(l) - len(l.lstrip()) for l in rest if l.strip()]
    if indents:
        m = min(indents)
        # A decorator first line means the `def` below it should land at 0;
        # otherwise the body should land at 4.
        shift = m if first.lstrip().startswith("@") else max(m - 4, 0)
        rest = [l[shift:] if len(l) - len(l.lstrip()) >= shift else l.lstrip()
                for l in rest]

    body = "\n".join([first.lstrip()] + rest)
    have = {d.strip() for d in re.findall(r"^\s*(@[\w.]+)", body, re.M)}
    missing = [d for d in (decorators or []) if d.strip() not in have]
    if missing:
        body = "\n".join(list(missing) + [body])
    return textwrap.indent(body, "    ")


def assemble_class(problem, method_sources):
    """Build a runnable module from per-method sources.

    ``method_sources`` maps method name -> source. A method with no entry is
    emitted as a `pass` stub so the module still imports; its own tests then
    fail honestly instead of taking the whole class down with a SyntaxError and
    charging the failure to every other method.
    """
    parts = [problem.get("import_block", ""), "",
             problem.get("class_constructor", "").rstrip(), ""]
    for sub in problem.get("subtasks", []):
        src = method_sources.get(sub["name"])
        if src:
            parts.append(normalise_method(src, sub.get("decorators")))
        else:
            parts.append(normalise_method(
                f"def {sub['name']}(self, *args, **kwargs):\n"
                f"    raise NotImplementedError({sub['name']!r})",
                sub.get("decorators")))
        parts.append("")
    return "\n".join(parts)


def extract_method(text, method_name):
    """Pull a single method definition out of a model response.

    Falls back to the whole code block: a model that wrapped the method in its
    class is better re-indented than rejected, and `normalise_method` copes.
    """
    code = extract_code(text)
    m = re.search(rf"((?:^[ \t]*@[\w.]+[ \t]*\n)*^[ \t]*def[ \t]+{re.escape(method_name)}\b.*?)"
                  rf"(?=^[ \t]*(?:@[\w.]+|def|class)\b|\Z)",
                  code, re.M | re.S)
    return (m.group(1) if m else code).rstrip()


def _class_prompt(problem, role=SOLVER_ROLE, extra=""):
    return (role + f"```python\n{problem.get('skeleton', '')}\n```\n" + extra)


def _method_prompt(problem, sub, guidance=""):
    note = f"\nContract:\n{guidance}\n" if guidance else ""
    return (METHOD_ROLE + f"```python\n{problem.get('skeleton', '')}\n```\n"
            f"{note}\nImplement ONLY: `{sub['name']}`\n"
            f"{sub.get('description', '')}\n")


# ==============================================================================
# --- SCORING ---
# ==============================================================================

def _score(problem, class_code, owner, first_owner=None):
    """Run the class-level suite and every method's own suite.

    ``owner`` maps method name -> the model that wrote the version being
    scored; ``first_owner`` maps it to whoever wrote the FIRST version. Both
    are recorded because a repair turn changes the writer, and a rollup that
    only knew the final writer would credit the escalation rung with work the
    cheap rung had already done. Returns ``(class_passed, evidence, records)``.
    """
    first_owner = first_owner or owner
    guard = missing_class_error(class_code, problem.get("class_name", ""))
    if guard:
        return False, guard, [{"name": s["name"], "tier": s["tier"],
                               "rank": s["rank"],
                               "model_id": owner.get(s["name"], ""),
                               "initial_model_id": first_owner.get(s["name"], ""),
                               "passed": False, "reason": "no class definition"}
                              for s in problem.get("subtasks", [])]

    records = []
    for sub in problem.get("subtasks", []):
        ok, _ev = run_classeval_method(problem, class_code, sub)
        records.append({"name": sub["name"], "tier": sub["tier"],
                        "rank": sub["rank"], "model_id": owner.get(sub["name"], ""),
                        "initial_model_id": first_owner.get(sub["name"], ""),
                        "repaired": owner.get(sub["name"]) != first_owner.get(sub["name"]),
                        "passed": bool(ok)})
    class_passed, evidence = run_classeval_class(problem, class_code)
    return class_passed, evidence, records


def _result(problem, passed, evidence, records, usd, out_tok, tot_tok, loops,
            usd_by_method=None):
    # A whole-class arm cannot attribute spend to a method -- one call wrote
    # them all. Recording 0.00 there would make the by-tier rollup read as if
    # those methods were free, and the cost comparison that H1 turns on would
    # be silently wrong in the cascade's favour. An even split is the honest
    # statement of "this arm bought them as a bundle".
    if usd_by_method is None and records:
        share = usd / len(records)
        usd_by_method = {r["name"]: share for r in records}
        for r in records:
            r["cost_basis"] = "class-level spend, split evenly"
    for r in records:
        r["as_run_usd"] = round(float((usd_by_method or {}).get(r["name"], 0.0)), 6)
    return {
        "passed": bool(passed),
        "as_run_usd": round(usd, 6),
        "output_tokens": out_tok,
        "total_tokens": tot_tok,
        "repair_loops": loops,
        "triage_usd": 0.0,
        "error": "" if passed else str(evidence)[:500],
        "subtasks": records,
        "subtask_summary": classeval_subtask_summary(records),
    }


def _spend(acc, usage):
    acc["usd"] += usage["as_run_usd"]
    acc["out"] += usage["output"]
    acc["tok"] += usage["total_tokens"]
    return usage["as_run_usd"]


# ==============================================================================
# --- ARMS ---
# ==============================================================================

@_arm(sj_required=True)
def run_ce_single(problem, model_id=GEMINI_37_FLASH_ID, thinking_level=None,
                  max_repairs=1):
    """One model writes the whole class, then repairs its own failures."""
    acc = {"usd": 0.0, "out": 0, "tok": 0}
    text, usage, _ = dispatch_model(model_id, _class_prompt(problem),
                                    max_tokens=3072, thinking_level=thinking_level,
                                    problem=problem)
    _spend(acc, usage)
    code = extract_code(text)
    owner = {s["name"]: model_id for s in problem["subtasks"]}
    passed, evidence, records = _score(problem, code, owner)

    loops = 0
    while not passed and loops < max_repairs:
        loops += 1
        digest, tr_usage, _ = _treat_error(evidence, "straitjacket", problem=problem)
        _spend(acc, tr_usage)
        text, usage, _ = dispatch_model(
            model_id,
            REPAIR_ROLE + f"```python\n{code}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n",
            max_tokens=3072, thinking_level=thinking_level, problem=problem)
        _spend(acc, usage)
        code = extract_code(text)
        passed, evidence, records = _score(problem, code, owner)

    return _result(problem, passed, evidence, records, acc["usd"], acc["out"],
                   acc["tok"], loops)


@_arm(sj_required=True)
def run_ce_cascade(problem, rungs=None):
    """Whole-class escalation: each failure hands the ENTIRE class up a rung.

    This is the BCB-winning shape transplanted unchanged. It cannot spend
    selectively -- when one hard method fails, the expensive rung re-solves the
    easy ones too, which is exactly the waste H1 predicts.
    """
    rungs = rungs or [(GEMINI_35_FLASH_LITE_ID, None),
                      (GEMINI_37_FLASH_ID, "low"),
                      (GEMINI_37_FLASH_ID, "medium")]
    acc = {"usd": 0.0, "out": 0, "tok": 0}
    model, think = rungs[0]
    text, usage, _ = dispatch_model(model, _class_prompt(problem), max_tokens=3072,
                                    thinking_level=think, problem=problem)
    _spend(acc, usage)
    code = extract_code(text)
    owner = {s["name"]: model for s in problem["subtasks"]}
    passed, evidence, records = _score(problem, code, owner)

    loops = 0
    for model, think in rungs[1:]:
        if passed:
            break
        loops += 1
        digest, tr_usage, _ = _treat_error(evidence, "straitjacket", problem=problem)
        _spend(acc, tr_usage)
        text, usage, _ = dispatch_model(
            model,
            REPAIR_ROLE + f"```python\n{code}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n",
            max_tokens=3072, thinking_level=think, problem=problem)
        _spend(acc, usage)
        code = extract_code(text)
        owner = {s["name"]: model for s in problem["subtasks"]}
        passed, evidence, records = _score(problem, code, owner)

    return _result(problem, passed, evidence, records, acc["usd"], acc["out"],
                   acc["tok"], loops)


@_arm(sj_required=True)
def run_ce_plan_exec(problem, planner_model=GEMINI_37_FLASH_ID,
                     executor_model=GEMINI_35_FLASH_LITE_ID,
                     repair_model=GEMINI_37_FLASH_ID):
    """Planner writes per-method contracts; a cheap executor writes the class.

    The planner call is unconditional -- it is paid on every task, before any
    test has run. That is the property the BCB sweep found expensive; here the
    plan has genuine work to do, because the methods call each other and the
    order they are written in matters.
    """
    acc = {"usd": 0.0, "out": 0, "tok": 0}
    plan, p_usage, _ = dispatch_model(planner_model, _class_prompt(problem, PLANNER_ROLE),
                                      max_tokens=1024, problem=problem)
    _spend(acc, p_usage)

    text, e_usage, _ = dispatch_model(
        executor_model,
        _class_prompt(problem, SOLVER_ROLE, extra=f"\nContracts:\n{plan}\n"),
        max_tokens=3072, problem=problem)
    _spend(acc, e_usage)
    code = extract_code(text)
    owner = {s["name"]: executor_model for s in problem["subtasks"]}
    passed, evidence, records = _score(problem, code, owner)

    loops = 0
    if not passed:
        loops = 1
        digest, tr_usage, _ = _treat_error(evidence, "straitjacket", problem=problem)
        _spend(acc, tr_usage)
        text, r_usage, _ = dispatch_model(
            repair_model,
            REPAIR_ROLE + f"```python\n{code}\n```\n\n"
            f"Contracts:\n{plan}\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n",
            max_tokens=3072, thinking_level="low", problem=problem)
        _spend(acc, r_usage)
        code = extract_code(text)
        owner = {s["name"]: repair_model for s in problem["subtasks"]}
        passed, evidence, records = _score(problem, code, owner)

    return _result(problem, passed, evidence, records, acc["usd"], acc["out"],
                   acc["tok"], loops)


def _per_method_arm(problem, pick, max_repairs=1, plan=None, planner_usd=0.0,
                    acc=None):
    """Shared body of the two per-method arms.

    ``pick(subtask)`` returns ``(model_id, thinking_level)``. The only
    difference between the hypothesis arm and its control is that function,
    which is what makes them a matched pair: identical call counts, identical
    prompts, identical repair policy.
    """
    acc = acc if acc is not None else {"usd": 0.0, "out": 0, "tok": 0}
    sources, owner, spent = {}, {}, {}

    for sub in problem["subtasks"]:
        model, think = pick(sub)
        text, usage, _ = dispatch_model(model, _method_prompt(problem, sub, plan or ""),
                                        max_tokens=1536, thinking_level=think,
                                        problem=problem)
        cost = _spend(acc, usage)
        sources[sub["name"]] = extract_method(text, sub["name"])
        owner[sub["name"]] = model
        spent[sub["name"]] = cost

    first_owner = dict(owner)
    code = assemble_class(problem, sources)
    passed, evidence, records = _score(problem, code, owner, first_owner)

    # Repair only what failed, one rung up from whoever wrote it. This is the
    # whole point: a cascade cannot express "re-solve this method only".
    loops = 0
    while not passed and loops < max_repairs:
        loops += 1
        failed = [r for r in records if not r["passed"]]
        if not failed:
            break                      # methods pass individually, integration does not
        digest, tr_usage, _ = _treat_error(evidence, "straitjacket", problem=problem)
        _spend(acc, tr_usage)
        for rec in failed:
            sub = next(s for s in problem["subtasks"] if s["name"] == rec["name"])
            wrote = owner[sub["name"]]
            idx = LADDER.index(wrote) if wrote in LADDER else 0
            up = LADDER[min(idx + 1, len(LADDER) - 1)]
            text, usage, _ = dispatch_model(
                up,
                METHOD_REPAIR_ROLE + f"```python\n{code}\n```\n\n"
                f"Failing method: `{sub['name']}`\n"
                f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n",
                max_tokens=1536, thinking_level="low", problem=problem)
            spent[sub["name"]] = spent.get(sub["name"], 0.0) + _spend(acc, usage)
            sources[sub["name"]] = extract_method(text, sub["name"])
            owner[sub["name"]] = up
        code = assemble_class(problem, sources)
        passed, evidence, records = _score(problem, code, owner, first_owner)

    if planner_usd and problem["subtasks"]:
        share = planner_usd / len(problem["subtasks"])
        for name in spent:
            spent[name] += share

    return _result(problem, passed, evidence, records, acc["usd"], acc["out"],
                   acc["tok"], loops, usd_by_method=spent)


@_arm(sj_required=True)
def run_ce_route_by_tier(problem, route=None, max_repairs=1):
    """H1: route each method to a model chosen by its labelled difficulty tier.

    The tier comes from the dataset's own `dependencies` annotation, never from
    a guess made here, so a win is reported against a label this repository did
    not choose.
    """
    table = route or TIER_ROUTE

    def pick(sub):
        return table.get(sub["tier"], (GEMINI_37_FLASH_ID, "low"))

    return _per_method_arm(problem, pick, max_repairs=max_repairs)


@_arm(sj_required=True)
def run_ce_route_flat(problem, model_id=GEMINI_35_FLASH_LITE_ID,
                      thinking_level=None, max_repairs=1):
    """Control for `run_ce_route_by_tier`: same loop, one model for every method.

    If this matches the routed arm, then any advantage belongs to writing the
    class method-by-method, not to routing by difficulty, and H1 is unsupported
    however good the headline looks.
    """
    def pick(_sub):
        return (model_id, thinking_level)

    return _per_method_arm(problem, pick, max_repairs=max_repairs)


@_arm(sj_required=True)
def run_ce_plan_route(problem, planner_model=GEMINI_37_FLASH_ID, max_repairs=1):
    """Planner contracts + difficulty routing -- both halves of the hypothesis."""
    acc = {"usd": 0.0, "out": 0, "tok": 0}
    plan, usage, _ = dispatch_model(planner_model, _class_prompt(problem, PLANNER_ROLE),
                                    max_tokens=1024, problem=problem)
    planner_usd = _spend(acc, usage)

    def pick(sub):
        return TIER_ROUTE.get(sub["tier"], (GEMINI_37_FLASH_ID, "low"))

    return _per_method_arm(problem, pick, max_repairs=max_repairs, plan=plan,
                           planner_usd=planner_usd, acc=acc)


# ==============================================================================
# --- VARIANT REGISTRY ---
# ==============================================================================

CATEGORY = "7. ClassEval sub-task routing"

CLASSEVAL_VARIANTS = {
    "ce_single_lite": {
        "id": "ce_single_lite", "category": CATEGORY,
        "name": "C0a. Single: gemini-3.5-flash-lite (whole class)",
        "models": "Gemini 3.5 Flash-Lite", "triage_mode": "Straitjacket ($0.00)",
        "fn": lambda p: run_ce_single(p, model_id=GEMINI_35_FLASH_LITE_ID),
    },
    "ce_single_flash": {
        "id": "ce_single_flash", "category": CATEGORY,
        "name": "C0b. Single: gemini-3.7-flash low (whole class)",
        "models": "Gemini 3.7 Flash", "triage_mode": "Straitjacket ($0.00)",
        "fn": lambda p: run_ce_single(p, model_id=GEMINI_37_FLASH_ID,
                                      thinking_level="low"),
    },
    "ce_single_sonnet": {
        "id": "ce_single_sonnet", "category": CATEGORY,
        "name": "C0c. Single: claude-sonnet-5 (whole class)",
        "models": "Claude Sonnet-5", "triage_mode": "Straitjacket ($0.00)",
        "fn": lambda p: run_ce_single(p, model_id=SONNET_ID),
    },
    "ce_cascade": {
        "id": "ce_cascade", "category": CATEGORY,
        "name": "C1. Cascade: whole class, Lite -> 3.7 low -> 3.7 medium",
        "models": "Gemini 3.5 Lite -> 3.7 Flash", "triage_mode": "Straitjacket ($0.00)",
        "fn": lambda p: run_ce_cascade(p),
    },
    "ce_plan_exec": {
        "id": "ce_plan_exec", "category": CATEGORY,
        "name": "C2. Plan & execute: 3.7 contracts -> Lite writes class",
        "models": "Gemini 3.7 Flash plan + 3.5 Lite exec",
        "triage_mode": "Straitjacket ($0.00)",
        "fn": lambda p: run_ce_plan_exec(p),
    },
    "ce_route_flat": {
        "id": "ce_route_flat", "category": CATEGORY,
        "name": "C3. CONTROL: per-method, every method to Lite",
        "models": "Gemini 3.5 Lite per method", "triage_mode": "Straitjacket ($0.00)",
        "fn": lambda p: run_ce_route_flat(p),
    },
    "ce_route_by_tier": {
        "id": "ce_route_by_tier", "category": CATEGORY,
        "name": "C4. H1: per-method, routed by labelled difficulty tier",
        "models": "Lite (easy) / 3.7 Flash (hard) per method",
        "triage_mode": "Straitjacket ($0.00)",
        "fn": lambda p: run_ce_route_by_tier(p),
    },
    "ce_plan_route": {
        "id": "ce_plan_route", "category": CATEGORY,
        "name": "C5. H1+plan: 3.7 contracts, then routed per-method execution",
        "models": "3.7 plan + Lite/3.7 routed per method",
        "triage_mode": "Straitjacket ($0.00)",
        "fn": lambda p: run_ce_plan_route(p),
    },
}
