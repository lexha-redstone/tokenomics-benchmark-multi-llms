# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Evidence-gated escalation: deciding when a task is hard enough for a frontier model.

Why this exists
---------------
A frontier model is worth its price on the tasks a cheap model cannot solve and
nowhere else. On the BigCodeBench-Hard N=100 sweep
(`bigCodeBench-hard/results/archive/bcb_n100_instrumented_20260822T2129.json`):

    tasks solved by gemini-3.7-flash but not claude-opus-5 :  3
    tasks solved by claude-opus-5 but not gemini-3.7-flash : 19
    solved by both                                          : 57
    solved by neither                                       : 21
    ------------------------------------------------------------
    perfect-router ceiling (flash OR opus)                  : 79

So a router that spends Opus only where Flash fails can reach ~79% while paying
Opus on ~40 tasks instead of 100. The whole design question is *how early* you
can tell which 40.

Two gate families
-----------------
**Attempt-count gates** are the obvious baseline: escalate after K failed
rungs. The same sweep shows why they work — in the cascade arm, tasks resolved
at repair loop 0 or 1 passed 100% of the time, while tasks still failing at
loop 2 passed only 15%. "Still failing after the cheap ladder" is a strong,
if late, signal.

**Evidence gates** read the harness's own typed extraction instead of a
counter, so they can escalate on turn one when the failure already looks hard,
and refuse to escalate on a one-line syntax slip that any model fixes. They
cost nothing: the evidence graph is a by-product of a capture that already
happened.

Nothing here re-derives evidence. :meth:`ContainedRun.evidence_graph` returns
the profile's own ``extract()`` output — typed failing identities, failure
classes and ``file:line`` loci. This module only reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A failure with at least this many distinct failing identities is treated as
# broad rather than local. Chosen from the N=100 distribution, where
# single-identity failures were overwhelmingly fixed by the next cheap rung.
BROAD_FAILURE_ITEMS = 3

# Failure classes that indicate the candidate never really ran: the model
# produced something that does not import or parse. These are cheap to fix and
# escalating on them wastes frontier budget.
_SHALLOW_CLASSES = {
    "SyntaxError", "IndentationError", "TabError",
    "ImportError", "ModuleNotFoundError", "NameError",
}

LEVELS = ("shallow", "local", "broad", "stalled", "environment")

# Guard reasons whose failure says nothing about the model. A row whose
# container never started, whose graded tests could not be restored or whose
# execution raised is a missing measurement; buying a frontier model for it
# spends the study's most expensive resource on a Docker problem.
_ENVIRONMENT_CLASSES = {"EnvironmentError"}

# Pre-execution guard failures that a cheap rung fixes as readily as an
# expensive one: the model wrote prose, or fenced something that is not a diff,
# or was cut off mid-diff by the output cap.
_MALFORMED_CLASSES = {"MalformedPatch"}


@dataclass
class Difficulty:
    """What the contained evidence says about how hard this failure is."""

    level: str                          # one of LEVELS
    reasons: tuple = ()
    failing: int = 0
    failure_classes: tuple = ()
    profile: str = ""
    identities: frozenset = frozenset()
    typed: bool = True                  # False when no fact tier was available
    # The guard slug (`src.evaluator.GUARD_REASONS`) when this failure happened
    # before the suite ran; `""` when the suite ran and produced real evidence.
    guard: str = ""

    @property
    def is_hard(self) -> bool:
        """Worth a frontier model, if the policy allows one."""
        return self.level in ("broad", "stalled")

    @property
    def is_environment(self) -> bool:
        """Nothing a different model would have changed."""
        return self.level == "environment"

    def as_dict(self):
        return {
            "level": self.level,
            "reasons": list(self.reasons),
            "failing": self.failing,
            "failure_classes": list(self.failure_classes),
            "profile": self.profile,
            "typed": self.typed,
            "guard": self.guard,
        }


@dataclass
class EscalationTrace:
    """Per-task record of what the router decided and why.

    Written into the result so a sweep can be audited after the fact: which
    rung solved it, whether the frontier tier was invoked, and what the gate
    saw when it decided.
    """

    rungs: list = field(default_factory=list)     # model ids actually called
    decisions: list = field(default_factory=list)  # one entry per gate evaluation
    frontier_used: bool = False
    # True when the gate needed typed evidence and the backend could not
    # supply it, so the arm did not test what its name says.
    degraded: bool = False
    frontier_rung: int | None = None
    solved_at: str | None = None

    def record(self, attempt, difficulty, escalate, why):
        self.decisions.append({
            "attempt": attempt,
            "difficulty": difficulty.as_dict() if difficulty else None,
            "escalate": bool(escalate),
            "why": why,
        })

    def as_dict(self):
        return {
            "degraded": self.degraded,
            "rungs": list(self.rungs),
            "decisions": list(self.decisions),
            "frontier_used": self.frontier_used,
            "frontier_rung": self.frontier_rung,
            "solved_at": self.solved_at,
        }


def _classify_guard(evidence, guard, previous):
    """Type a failure that happened *before* the repository's suite ran.

    Why this is not just "shallow": on a dataset where the model has to emit a
    complete unified diff, `apply_failed` is the single most common outcome,
    and it is neither shallow nor readable from a fact tier — the suite never
    ran, so there is no profile and no census. Before this existed every such
    failure landed in the untyped branch below, classified as `shallow`, and
    an evidence gate consequently never fired on the *dominant* failure in the
    sweep. The router was reading a constant.
    """
    failure_class = getattr(evidence, "failure_class", "") or ""
    common = dict(profile="guard/v1", typed=True, guard=guard,
                  failure_classes=(failure_class,) if failure_class else ())

    if failure_class in _ENVIRONMENT_CLASSES:
        return Difficulty(
            level="environment",
            reasons=(f"{guard}: the environment failed, not the model",),
            **common)

    if failure_class in _MALFORMED_CLASSES:
        return Difficulty(
            level="shallow",
            reasons=(f"{guard}: response was not a usable diff",),
            **common)

    # `apply_failed`. One occurrence is a local defect — a miscounted hunk, a
    # context line off by a word — and the next cheap rung fixes those. The
    # same thing twice running means the model cannot see the tree it is
    # patching, which is exactly the "hand this over" signal a stall is.
    repeated = previous is not None and getattr(previous, "guard", "") == guard
    if repeated:
        return Difficulty(
            level="stalled",
            reasons=(f"{guard} survived the last repair turn",),
            **common)
    return Difficulty(
        level="local",
        reasons=(f"{guard}: patch did not land on this tree",),
        **common)


def classify(evidence, previous=None):
    """Read the typed evidence graph for one failure.

    ``previous`` is the :class:`Difficulty` from the prior attempt, used to
    detect a stall: the same failing identities surviving a repair turn means
    the model is not converging, which is the clearest "hand this over" signal
    there is.
    """
    run = getattr(evidence, "run", None)
    graph = run.evidence_graph() if run is not None else None
    profile = getattr(run, "profile", "") or ""

    guard = getattr(evidence, "reason", "") or ""
    if guard:
        return _classify_guard(evidence, guard, previous)

    if graph is None:
        # No fact tier. `text/v1` means nothing recognised the output as a test
        # run at all — usually a crash before the suite started. Treat it as
        # shallow: cheap models fix these, and there is no census to reason
        # over anyway.
        return Difficulty(
            level="shallow",
            reasons=("no typed evidence (profile has no fact tier)",),
            profile=profile,
            typed=False,
        )

    items = tuple(graph.items)
    identities = frozenset(i.id for i in items)
    classes = tuple(sorted({i.failure_class for i in items if i.failure_class}))
    failing = int(graph.aggregate.get("failing", len(items)))

    reasons = []
    level = "local"

    if items and all((c in _SHALLOW_CLASSES) for c in classes):
        level = "shallow"
        reasons.append(f"failure classes are all shallow: {', '.join(classes)}")
    elif failing >= BROAD_FAILURE_ITEMS:
        level = "broad"
        reasons.append(f"{failing} distinct failing identities (>= {BROAD_FAILURE_ITEMS})")
    else:
        reasons.append(f"{failing} failing identity/identities")

    if previous is not None and identities and identities == previous.identities:
        level = "stalled"
        reasons.append("identical failing identities survived the last repair turn")

    return Difficulty(
        level=level,
        reasons=tuple(reasons),
        failing=failing,
        failure_classes=classes,
        profile=profile,
        identities=identities,
        typed=True,
    )


# ==============================================================================
# --- GATES ---
# ==============================================================================
#
# A gate answers one question: given this failure, at this attempt, may the
# frontier model be called now? Each returns (escalate: bool, why: str).


def _plain(fn):
    fn.requires_typed_evidence = False
    return fn


@_plain
def gate_never(difficulty, attempt, total_rungs):
    """Control arm: the frontier model is never invoked."""
    return False, "frontier disabled"


@_plain
def gate_after_ladder(difficulty, attempt, total_rungs):
    """Escalate only once every cheap rung has failed. The conservative
    baseline, and the shape `sj_escalation_shield` already uses."""
    if difficulty is not None and difficulty.is_environment:
        return False, f"environment failure ({difficulty.guard}); no model fixes this"
    if attempt >= total_rungs:
        return True, f"all {total_rungs} cheap rungs exhausted"
    return False, f"cheap rungs remain ({attempt}/{total_rungs})"


def gate_after_attempts(k):
    """Escalate after `k` failed attempts, regardless of what failed."""
    def gate(difficulty, attempt, total_rungs):
        if attempt >= k:
            return True, f"{attempt} failed attempts (>= {k})"
        return False, f"{attempt} failed attempts (< {k})"
    gate.__name__ = f"gate_after_attempts_{k}"
    gate.requires_typed_evidence = False
    return gate


def gate_on_evidence(min_attempt=1):
    """Escalate as soon as the evidence looks hard, but never before
    `min_attempt` failures — one cheap attempt is worth making regardless,
    because it is nearly free and often enough."""
    def gate(difficulty, attempt, total_rungs):
        if attempt < min_attempt:
            return False, f"attempt {attempt} < min_attempt {min_attempt}"
        if difficulty is None:
            return False, "no difficulty signal"
        # Checked before `is_hard` and before the exhaustion fallback: a
        # container that never started does not become worth $15/Mtok because
        # the cheap rungs ran out.
        if difficulty.is_environment:
            return False, f"environment failure ({difficulty.guard}); no model fixes this"
        if difficulty.is_hard:
            return True, f"evidence says {difficulty.level}: {'; '.join(difficulty.reasons)}"
        if attempt >= total_rungs:
            return True, f"cheap rungs exhausted ({attempt}/{total_rungs})"
        return False, f"evidence says {difficulty.level}; keep it cheap"
    gate.__name__ = f"gate_on_evidence_{min_attempt}"
    # Declared so the router can refuse to present an evidence-gated arm that
    # silently ran as a counter gate. Without the fact tier every failure
    # classifies as `shallow`, the gate never fires early, and the row looks
    # like a result rather than a no-op.
    gate.requires_typed_evidence = True
    return gate


def frontier_is_reachable(gate_fn, n_tiers, max_oracle_calls):
    """Can this (gate, ladder, budget) triple *ever* call the frontier model?

    The measured failure this exists to stop: a sweep ran with
    `MAX_ORACLE_CALLS = 2` over a two-rung ladder, so the only gate evaluation
    that ever happened was `attempt == 1` against `total_rungs == 2`. Every
    gate answers "cheap rungs remain" to that, and the frontier branch was
    unreachable code. Five arms whose names promised an Opus escalation shipped
    a report in which Opus was never called once, and nothing said so.

    Answered against the most escalation-friendly evidence a gate could see, so
    a `False` means structurally impossible rather than merely unlikely.
    """
    best = Difficulty(level="broad", failing=BROAD_FAILURE_ITEMS, typed=True)
    return any(gate_fn(best, attempt, n_tiers)[0]
               for attempt in range(1, max(int(max_oracle_calls), 1)))


GATES = {
    "never": gate_never,
    "after_ladder": gate_after_ladder,
    "after_1": gate_after_attempts(1),
    "after_2": gate_after_attempts(2),
    "evidence": gate_on_evidence(1),
    "evidence_immediate": gate_on_evidence(0),
}
