# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Contract tests for evidence-gated escalation (`src/routing.py`).

The routing study's whole claim is that a frontier model can be spent only
where it is needed. That claim is only worth measuring if the gate actually
gates — so these pin:

  * the difficulty signal reads the harness's TYPED extraction, not a regex
    over the digest prose;
  * a shallow failure (syntax/import) is not escalated;
  * a broad failure is;
  * a stall — the same failing identities surviving a repair turn — is;
  * each gate escalates when it says it does, and the frontier budget holds.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import straitjacket as sj  # noqa: E402

pytestmark = pytest.mark.skipif(
    not sj.available(),
    reason="ctx-harness not installed (pip install ctx-harness)",
)

# The typed fact tier needs the in-process manifest, so the difficulty signal
# only exists on the library backend. That is not a silent limitation: the
# router warns and marks the run `degraded` when an evidence gate has no typed
# evidence to read (see test_evidence_gate_degrades_loudly).
needs_facts = pytest.mark.skipif(
    sj.status().get("backend") != "library",
    reason="typed evidence requires the library backend",
)


def _evidence(solution, test_body):
    """Run a real candidate through the harness and return its Evidence."""
    from src.evaluator import run_bigcodebench
    problem = {"entry_point": "task_func", "test": test_body}
    _, err = run_bigcodebench(problem, solution)
    return err


_ONE_FAILING = ("import unittest\n"
                "class TestCases(unittest.TestCase):\n"
                "    def test_a(self): self.assertEqual(task_func(1), 2)\n")

_FOUR_FAILING = ("import unittest\n"
                 "class TestCases(unittest.TestCase):\n"
                 + "".join(f"    def test_{i}(self): "
                           f"self.assertEqual(task_func({i}), {i + 1})\n"
                           for i in range(4)))


# ------------------------------------------------------------------ signal

@needs_facts
def test_signal_reads_the_typed_fact_tier():
    """`evidence_graph()` is the profile's own extract(), so the router never
    re-derives what the harness already typed."""
    from src.routing import classify

    ev = _evidence("def task_func(x):\n    return x + 99\n", _FOUR_FAILING)
    graph = ev.run.evidence_graph()
    assert graph is not None
    assert graph.family == "unittest"
    assert {i.kind for i in graph.items} == {"failing_test"}

    d = classify(ev)
    assert d.typed is True
    assert d.failing == len(graph.items)
    assert d.failure_classes == ("AssertionError",)


@needs_facts
def test_broad_failure_is_hard():
    from src.routing import BROAD_FAILURE_ITEMS, classify

    d = classify(_evidence("def task_func(x):\n    return x + 99\n", _FOUR_FAILING))
    assert d.failing >= BROAD_FAILURE_ITEMS
    assert d.level == "broad"
    assert d.is_hard


@needs_facts
def test_local_failure_is_not_hard():
    from src.routing import classify

    d = classify(_evidence("def task_func(x):\n    return x + 99\n", _ONE_FAILING))
    assert d.level == "local"
    assert not d.is_hard


@needs_facts
def test_shallow_failure_is_not_escalated():
    """A candidate that never ran is cheap to fix. Spending a frontier model
    on a NameError is exactly the waste the gate exists to prevent."""
    from src.routing import classify

    d = classify(_evidence("def task_func(x):\n    return undefined_symbol\n",
                           _FOUR_FAILING))
    assert d.level == "shallow", d.as_dict()
    assert not d.is_hard


@needs_facts
def test_stall_is_hard_even_when_narrow():
    """One failing test that survives a repair turn unchanged means the model
    is not converging — narrower than 'broad', but a stronger signal."""
    from src.routing import classify

    first = classify(_evidence("def task_func(x):\n    return x + 99\n", _ONE_FAILING))
    assert not first.is_hard
    again = classify(_evidence("def task_func(x):\n    return x + 99\n", _ONE_FAILING),
                     previous=first)
    assert again.level == "stalled"
    assert again.is_hard


def test_missing_fact_tier_degrades_to_shallow_not_hard():
    """No typed evidence must not be read as 'hard' — that would escalate
    every unrecognised output straight to the most expensive model."""
    from src.routing import classify

    class _NoGraph:
        profile = "text/v1"

        def evidence_graph(self):
            return None

    class _Ev:
        run = _NoGraph()

    d = classify(_Ev())
    assert d.typed is False
    assert not d.is_hard


# ------------------------------------------------------------------- gates

@pytest.mark.parametrize("name,attempt,total,hard,expected", [
    ("never",              9, 3, True,  False),
    ("after_ladder",       1, 3, True,  False),
    ("after_ladder",       3, 3, False, True),
    ("after_1",            1, 3, False, True),
    ("after_2",            1, 3, True,  False),
    ("after_2",            2, 3, False, True),
    ("evidence",           1, 3, True,  True),   # escalates early on hard evidence
    ("evidence",           1, 3, False, False),  # stays cheap otherwise
    ("evidence",           3, 3, False, True),   # ...but still exhausts the ladder
    ("evidence_immediate", 0, 3, True,  True),
])
def test_gate_behaviour(name, attempt, total, hard, expected):
    from src.routing import GATES, Difficulty

    d = Difficulty(level="broad" if hard else "local")
    escalate, why = GATES[name](d, attempt, total)
    assert escalate is expected, f"{name}: {why}"
    assert why


# ------------------------------------------------------------------- router

@pytest.fixture
def stub(monkeypatch):
    """Gemini always fails, Opus always succeeds — so every escalation path is
    exercised and the frontier tier is observably the thing that solved it."""
    import src.architectures as arch
    import src.client as client
    import src.evaluator as ev

    seen = []

    def fake(model_id, prompt, max_tokens=2048, thinking_level=None, problem=None):
        seen.append((model_id, thinking_level))
        body = "return x + 1" if "opus" in model_id else "return x + 99"
        return (f"```python\ndef task_func(x):\n    {body}\n```",
                {"as_run_usd": 0.02 if "opus" in model_id else 0.002,
                 "input": 10, "output": 10, "total_tokens": 20}, 0.1)

    for mod in (client, ev, arch):
        monkeypatch.setattr(mod, "dispatch_model", fake, raising=False)
    return seen


_ROUTER_PROBLEM = {
    "task_id": "router/1",
    "entry_point": "task_func",
    "complete_prompt": "def task_func(x):\n    return x + 1\n",
    "test": _FOUR_FAILING,
}


def test_never_gate_never_calls_the_frontier(stub):
    from src.architectures import run_tiered_router
    from src.config import GEMINI_37_FLASH_ID

    r = run_tiered_router(dict(_ROUTER_PROBLEM),
                          tiers=[(GEMINI_37_FLASH_ID, "low")] * 3, gate="never")
    assert r["routing"]["frontier_used"] is False
    assert not any("opus" in m for m, _ in stub)


def test_after_ladder_escalates_only_at_the_end(stub):
    from src.architectures import run_tiered_router
    from src.config import GEMINI_35_FLASH_LITE_ID, GEMINI_37_FLASH_ID

    r = run_tiered_router(
        dict(_ROUTER_PROBLEM),
        tiers=[(GEMINI_35_FLASH_LITE_ID, None), (GEMINI_37_FLASH_ID, "low")],
        gate="after_ladder")
    rt = r["routing"]
    assert rt["frontier_used"] is True
    assert rt["frontier_rung"] == 2, rt["rungs"]
    assert r["passed"] is True


@needs_facts
def test_evidence_gate_escalates_before_the_ladder_is_exhausted(stub):
    """The point of the evidence gate: a broad failure does not have to wait
    its turn."""
    from src.architectures import run_tiered_router
    from src.config import GEMINI_35_FLASH_LITE_ID, GEMINI_37_FLASH_ID

    r = run_tiered_router(
        dict(_ROUTER_PROBLEM),
        tiers=[(GEMINI_35_FLASH_LITE_ID, None), (GEMINI_37_FLASH_ID, "low"),
               (GEMINI_37_FLASH_ID, "medium")],
        gate="evidence")
    rt = r["routing"]
    assert rt["frontier_used"] is True
    assert rt["frontier_rung"] == 1, rt["rungs"]        # skipped two cheap rungs
    assert any(d["escalate"] for d in rt["decisions"])


def test_frontier_budget_is_respected(stub):
    from src.architectures import run_tiered_router
    from src.config import GEMINI_37_FLASH_ID

    # Opus is stubbed to succeed, so force failure by asking for a task whose
    # test no stub can satisfy, and check the budget caps the calls.
    problem = dict(_ROUTER_PROBLEM)
    problem["test"] = ("import unittest\n"
                       "class TestCases(unittest.TestCase):\n"
                       "    def test_a(self): self.assertEqual(task_func(1), 12345)\n"
                       "    def test_b(self): self.assertEqual(task_func(2), 12346)\n"
                       "    def test_c(self): self.assertEqual(task_func(3), 12347)\n")
    r = run_tiered_router(problem, tiers=[(GEMINI_37_FLASH_ID, "low")] * 4,
                          gate="after_1", frontier_max_calls=1)
    opus_calls = sum(1 for m, _ in stub if "opus" in m)
    assert opus_calls <= 1, f"frontier called {opus_calls} times despite a budget of 1"
    assert r["passed"] is False


def test_routing_trace_is_auditable(stub):
    """Every gate evaluation is recorded with what it saw, so a sweep can be
    re-examined without re-running it."""
    from src.architectures import run_tiered_router
    from src.config import GEMINI_35_FLASH_LITE_ID, GEMINI_37_FLASH_ID

    r = run_tiered_router(
        dict(_ROUTER_PROBLEM),
        tiers=[(GEMINI_35_FLASH_LITE_ID, None), (GEMINI_37_FLASH_ID, "low")],
        gate="after_ladder")
    rt = r["routing"]
    assert rt["rungs"] and rt["solved_at"]
    for d in rt["decisions"]:
        assert "attempt" in d and "escalate" in d and d["why"]
        assert d["difficulty"]["level"] in ("shallow", "local", "broad", "stalled")


def test_router_still_records_the_containment_receipt(stub):
    """Router arms are straitjacket arms; the receipt must not go blank."""
    from src.architectures import run_tiered_router
    from src.config import GEMINI_37_FLASH_ID

    r = run_tiered_router(dict(_ROUTER_PROBLEM),
                          tiers=[(GEMINI_37_FLASH_ID, "low")] * 2, gate="never")
    c = r["containment"]
    assert c["treatment_events"] > 0
    assert c["treatments"] == ["straitjacket"]
    assert r.get("containment_instrumentation") != "MISSING"


def test_evidence_gate_degrades_loudly(stub, monkeypatch, capsys):
    """An evidence gate with no typed evidence is a counter gate wearing the
    wrong label. It must announce that, not quietly produce a row.

    This is the same failure class as the blank containment receipt: the arm
    still runs and still reports numbers, but it is not measuring what its
    name claims.
    """
    import src.architectures as arch
    from src.config import GEMINI_35_FLASH_LITE_ID, GEMINI_37_FLASH_ID

    monkeypatch.setattr(arch, "_warned_arms", set(), raising=False)
    monkeypatch.setattr("src.straitjacket.ContainedRun.evidence_graph",
                        lambda self: None)

    r = arch.run_tiered_router(
        dict(_ROUTER_PROBLEM),
        tiers=[(GEMINI_35_FLASH_LITE_ID, None), (GEMINI_37_FLASH_ID, "low")],
        gate="evidence")

    assert r["routing"]["degraded"] is True
    assert "NOT testing the evidence gate" in capsys.readouterr().err


def test_counter_gates_are_not_marked_degraded(stub):
    """Only evidence gates depend on the fact tier; a counter gate is complete
    on its own and must not be flagged."""
    import src.architectures as arch
    from src.config import GEMINI_37_FLASH_ID

    r = arch.run_tiered_router(dict(_ROUTER_PROBLEM),
                               tiers=[(GEMINI_37_FLASH_ID, "low")] * 2,
                               gate="after_ladder")
    assert r["routing"]["degraded"] is False
