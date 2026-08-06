# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Production Multi-LLM Architecture Recipes with Straitjacket Zero-Cost Context Containment.
Executable examples demonstrating sweet-spot Pareto optimal pipelines.
"""

from typing import Dict, Any, Tuple

# Example Model Dispatcher Interface
def call_model(model_id: str, prompt: str, thinking_level: str = None) -> Tuple[str, float]:
    """Stub dispatch for demonstration."""
    return "def solution(): pass", 0.0015

def local_zero_cost_triage(raw_test_stderr: str) -> str:
    """Straitjacket UnittestProfile deterministic extraction ($0.00)."""
    # Deterministic assertion line & traceback extraction
    lines = [l.strip() for l in raw_test_stderr.splitlines() if any(k in l for k in ["FAIL:", "AssertionError", "Traceback", "File "])]
    return "\n".join(lines[:15]) or raw_test_stderr[-500:]

# ==============================================================================
# RECIPE 1: Pure Google Cloud Smart Repair Pipeline (Class A / Class E)
# ==============================================================================
def run_pure_gemini_smart_repair(task_prompt: str, test_runner_fn) -> Dict[str, Any]:
    """
    Tier 1: Gemini 3.6-Flash (Low Thinking) -> Local SJ Triage ->
    Tier 2: Gemini 3.5-Flash-Lite (Sub-Cent Fast Repair) -> Local SJ Triage ->
    Tier 3: Gemini 3.6-Flash (Medium Thinking Escalation).
    """
    tot_cost = 0.0

    # 1. Initial Draft with Flash Low-Thinking
    code, cost = call_model("gemini-3.6-flash", task_prompt, thinking_level="low")
    tot_cost += cost
    passed, err = test_runner_fn(code)
    if passed:
        return {"code": code, "passed": True, "cost_usd": tot_cost, "tier_resolved": 1}

    # 2. Local Zero-Cost Triage ($0.00)
    digest = local_zero_cost_triage(err)

    # 3. Sub-Cent Fast Repair Attempt
    r1_prompt = f"{task_prompt}\n\nCurrent Code:\n{code}\n\nTriaged Test Digest:\n{digest}\n\nFix the bug."
    code, cost = call_model("gemini-3.5-flash-lite", r1_prompt)
    tot_cost += cost
    passed, err = test_runner_fn(code)
    if passed:
        return {"code": code, "passed": True, "cost_usd": tot_cost, "tier_resolved": 2}

    # 4. Final Medium-Thinking Escalation
    digest = local_zero_cost_triage(err)
    r2_prompt = f"{task_prompt}\n\nCurrent Code:\n{code}\n\nTriaged Test Digest:\n{digest}\n\nPerform deep reasoning repair."
    code, cost = call_model("gemini-3.6-flash", r2_prompt, thinking_level="medium")
    tot_cost += cost
    passed, err = test_runner_fn(code)
    return {"code": code, "passed": passed, "cost_usd": tot_cost, "tier_resolved": 3 if passed else 0}

# ==============================================================================
# RECIPE 2: Cross-Provider Ultra-Sweet Hybrid (Class B - Enterprise Repositories)
# ==============================================================================
def run_ultra_sweet_hybrid(repo_context: str, issue_description: str, patch_test_runner_fn) -> Dict[str, Any]:
    """
    Step 1: Claude Sonnet-5 Architect Contract (<250 words)
    Step 2: Gemini 3.5-Flash-Lite Git Patch Generation
    Step 3: Straitjacket Zero-Cost Patch Diff Profiling
    Step 4: Claude Opus-5 Final Escalation (if regression occurs)
    """
    tot_cost = 0.0

    # 1. Contract Planning
    adv_prompt = f"Repo Context:\n{repo_context}\n\nIssue:\n{issue_description}\n\nProduce strict implementation contract under 200 words."
    contract, cost = call_model("claude-sonnet-5", adv_prompt)
    tot_cost += cost

    # 2. Patch Execution
    exec_prompt = f"Repo Context:\n{repo_context}\n\nIssue:\n{issue_description}\n\nArchitect Contract:\n{contract}\n\nOutput unified git diff."
    patch, cost = call_model("gemini-3.5-flash-lite", exec_prompt)
    tot_cost += cost

    passed, err = patch_test_runner_fn(patch)
    if passed:
        return {"patch": patch, "passed": True, "cost_usd": tot_cost, "loops": 0}

    # 3. Local Triage ($0.00)
    digest = local_zero_cost_triage(err)

    # 4. Opus-5 Escalation Repair
    repair_prompt = f"Issue:\n{issue_description}\n\nContract:\n{contract}\n\nCurrent Patch:\n{patch}\n\nRegression Digest:\n{digest}\n\nOutput corrected unified git diff."
    patch, cost = call_model("claude-opus-5", repair_prompt)
    tot_cost += cost
    passed, err = patch_test_runner_fn(patch)

    return {"patch": patch, "passed": passed, "cost_usd": tot_cost, "loops": 1}
