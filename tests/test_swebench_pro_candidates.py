# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Contract tests for the 3 SWE-bench Pro candidate architectures.

Verifies:
1. Registration in SWEBENCH_PRO_VARIANTS and get_configurations group resolution.
2. Micro-contract locator role prompting & execution.
3. Patch-health aware routing logic & escalation conditions.
4. Sonnet -> Opus sweetspot execution flow and oracle-call bounds.
"""

import types
import pytest

from src import swebench_pro as sbp
from src.architectures import get_configurations
from src.routing import Difficulty
from src.config import GEMINI_37_FLASH_ID, SONNET_ID, OPUS_5_ID

PROBLEM = {
    "instance_id": "instance_demo__repo-abc123-vnan",
    "repo": "demo/repo",
    "repo_language": "python",
    "base_commit": "0" * 40,
    "problem_statement": "Issue statement",
    "requirements": "Requirements block",
    "interface": "Interface block",
    "fail_to_pass": ["tests/t.py | Key methods should return null"],
    "pass_to_pass": ["tests/t.py | Key methods should set a key"],
    "selected_test_files_to_run": ["tests/t.py"],
    "before_repo_set_cmd": "git checkout abc123 -- tests/t.py",
    "dockerhub_tag": "demo.repo-demo__repo-abc123",
    "image_name": "jefzda/sweap-images:demo.repo-demo__repo-abc123",
    "repo_workdir": "/app",
}

GOOD_PATCH = ("```diff\n"
              "diff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n"
              "@@ -1 +1 @@\n-old\n+new\n```")


def _fake_dispatch(responses):
    """Cycle through canned (text, usage) pairs."""
    it = iter(responses)
    calls = []

    def dispatch(model, prompt, **kwargs):
        calls.append({"model": model, "prompt": prompt, "kwargs": kwargs})
        text, usage = next(it)
        u = {"as_run_usd": usage.get("as_run_usd", 0.001),
             "input": usage.get("input", 100),
             "output": usage.get("output", 50),
             "total_tokens": usage.get("total_tokens", 150)}
        return text, u, 0.05

    return dispatch, calls


@pytest.fixture(autouse=True)
def _harness_stub(monkeypatch):
    """The candidate arms are `sj_required`; the harness is not installed in CI.

    Without this every test below raised `SJUnavailable` inside the `_arm`
    decorator before reaching an assertion, so three tests that never actually
    executed their arm read as a green suite on the author's machine only.
    """
    monkeypatch.setattr("src.architectures.sj.require", lambda: None)
    monkeypatch.setattr(sbp, "_treat_error",
                        lambda err, t, problem=None: ("DIGEST", {
                            "as_run_usd": 0.0, "output": 0, "total_tokens": 0}, 0.0))


class _MockEnv:
    """Stands in for the container. Also stands in for the repository: the
    grounding pass reads through the same two methods `SWEBenchProEnv` exposes."""

    sources = {"w.py": "old\n"}

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.last_ratio = None
        self.last_report = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def score(self, patch):
        self.calls.append(patch)
        passed, evidence, ratio = self.outcomes.pop(0)
        self.last_ratio = ratio
        self.last_report = {"resolved": passed, "test_pass_ratio": ratio}
        return passed, evidence

    def read_source(self, paths, budget=None, per_file=None, max_files=None):
        read = [p for p in paths if p in self.sources]
        return ([f"--- FILE: {p} ---\n{self.sources[p]}\n" for p in read],
                read, [p for p in paths if p not in self.sources])

    def grep_paths(self, terms, limit=40):
        return []


# ==============================================================================
# --- REGISTRATION & DISCOVERY TESTS ---
# ==============================================================================

def test_candidate_variants_are_registered():
    reg = sbp.SWEBENCH_PRO_VARIANTS
    for vid in ("sbp_grounded_contract", "sbp_patch_health_router", "sbp_sonnet_opus_sweetspot"):
        assert vid in reg
        assert callable(reg[vid]["fn"])
        assert "9c. SWE-bench Pro" in reg[vid]["category"]


def test_get_configurations_resolves_sbp_candidates_group():
    configs = get_configurations(dataset="swebench_pro", group="sbp_candidates")
    ids = {c["id"] for c in configs}
    assert ids == {"sbp_grounded_contract", "sbp_patch_health_router", "sbp_sonnet_opus_sweetspot"}


def test_get_configurations_includes_candidates_in_sbp_group():
    configs = get_configurations(dataset="swebench_pro", group="sbp")
    ids = {c["id"] for c in configs}
    assert "sbp_grounded_contract" in ids
    assert "sbp_patch_health_router" in ids
    assert "sbp_sonnet_opus_sweetspot" in ids
    assert "sbp_single_flash" in ids


# ==============================================================================
# --- PATCH HEALTH GATE TESTS ---
# ==============================================================================

def test_gate_patch_health_logic():
    # 1. No difficulty
    esc, why = sbp.gate_patch_health(None, 1, 2)
    assert not esc

    # 2. Shallow difficulty -> does not escalate on attempt 1
    d_shallow = Difficulty(level="shallow", reasons=("syntax error",), failing=0)
    esc, why = sbp.gate_patch_health(d_shallow, 1, 2)
    assert not esc
    assert "standard repair" in why

    # 3. Local difficulty -> does not escalate on attempt 1
    d_local = Difficulty(level="local", reasons=("1 failing identity",), failing=1)
    esc, why = sbp.gate_patch_health(d_local, 1, 2)
    assert not esc
    assert "standard repair" in why

    # 4. Hard/Broad difficulty -> escalates immediately!
    d_broad = Difficulty(level="broad", reasons=("3 failing identities",), failing=3)
    esc, why = sbp.gate_patch_health(d_broad, 1, 2)
    assert esc
    assert "patch health & evidence" in why

    # 5. Stalled difficulty -> escalates immediately!
    d_stalled = Difficulty(level="stalled", reasons=("identical failure survived",), failing=1)
    esc, why = sbp.gate_patch_health(d_stalled, 1, 2)
    assert esc

    # 6. Cheap rungs exhausted -> escalates
    esc, why = sbp.gate_patch_health(d_local, 2, 2)
    assert esc
    assert "cheap rungs exhausted" in why

    # 7. An environment failure never escalates, not even with the ladder
    #    exhausted: no model fixes a container that would not start, and the
    #    frontier tier is the most expensive thing the study can waste.
    d_env = Difficulty(level="environment", reasons=("container unavailable",),
                       guard="container_unavailable")
    for attempt in (1, 2, 5):
        esc, why = sbp.gate_patch_health(d_env, attempt, 2)
        assert not esc
        assert "environment failure" in why


def test_gate_patch_health_declares_that_it_reads_typed_evidence():
    """Undeclared, `_ladder` never sets `routing.degraded` for this gate, and a
    row routed with no fact tier reads as an evidence-routed result."""
    assert sbp.gate_patch_health.requires_typed_evidence is True


# ==============================================================================
# --- CANDIDATE 1: GROUNDED CONTRACT TESTS ---
# ==============================================================================

def test_run_sbp_grounded_contract_flow(monkeypatch):
    contract_text = "TARGET_FILES: [w.py]\nMODIFICATIONS: [fix new]\nINVARIANTS: [pass]"
    disp, calls = _fake_dispatch([
        (contract_text, {"as_run_usd": 0.0015, "output": 50}),  # contract locator
        (GOOD_PATCH, {"as_run_usd": 0.002, "output": 120}),     # flash solver
    ])
    monkeypatch.setattr(sbp, "dispatch_model", disp)
    env = _MockEnv([(True, "", 1.0)])
    monkeypatch.setattr(sbp, "SWEBenchProEnv", lambda p: env)

    out = sbp.run_sbp_grounded_contract(PROBLEM)
    assert out["passed"] is True
    assert out["repair_loops"] == 0
    assert out["test_pass_ratio"] == 1.0
    # First call must be locator with CONTRACT_LOCATOR_ROLE
    assert calls[0]["model"] == SONNET_ID
    assert "CONCISE IMPLEMENTATION CONTRACT" in calls[0]["prompt"]
    # Second call must be executor with contract included
    assert calls[1]["model"] == GEMINI_37_FLASH_ID
    assert "Architect's implementation plan" in calls[1]["prompt"] or contract_text in calls[1]["prompt"]


# ==============================================================================
# --- CANDIDATE 2: PATCH HEALTH ROUTER TESTS ---
# ==============================================================================

def test_run_sbp_patch_health_router_escalates_on_broad(monkeypatch):
    disp, calls = _fake_dispatch([
        (GOOD_PATCH, {"as_run_usd": 0.002, "output": 100}),   # flash attempt 1
        (GOOD_PATCH, {"as_run_usd": 0.015, "output": 150}),   # opus repair
    ])
    monkeypatch.setattr(sbp, "dispatch_model", disp)
    # Attempt 1 fails with broad failure
    d_broad = Difficulty(level="broad", reasons=("3 failing tests",), failing=3, typed=True)
    monkeypatch.setattr(sbp, "classify", lambda ev, previous=None: d_broad)

    env = _MockEnv([
        (False, "AssertionError: 3 tests failed", 0.2),
        (True, "", 1.0)
    ])
    monkeypatch.setattr(sbp, "SWEBenchProEnv", lambda p: env)

    out = sbp.run_sbp_patch_health_router(PROBLEM)
    assert out["passed"] is True
    assert out["repair_loops"] == 1
    # Check that model escalated to Opus-5 on repair turn
    assert calls[0]["model"] == GEMINI_37_FLASH_ID
    assert calls[1]["model"] == OPUS_5_ID


# ==============================================================================
# --- CANDIDATE 3: SONNET OPUS SWEETSPOT TESTS ---
# ==============================================================================

def test_run_sbp_sonnet_opus_sweetspot_one_shot_pass(monkeypatch):
    disp, calls = _fake_dispatch([
        (GOOD_PATCH, {"as_run_usd": 0.005, "output": 150}),   # sonnet attempt 1
    ])
    monkeypatch.setattr(sbp, "dispatch_model", disp)
    env = _MockEnv([(True, "", 1.0)])
    monkeypatch.setattr(sbp, "SWEBenchProEnv", lambda p: env)

    out = sbp.run_sbp_sonnet_opus_sweetspot(PROBLEM)
    assert out["passed"] is True
    assert out["repair_loops"] == 0
    assert calls[0]["model"] == SONNET_ID
