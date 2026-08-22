# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Production Multi-LLM Architecture Recipes with Straitjacket Context Containment (ctx-harness).
Executable examples demonstrating sweet-spot Pareto optimal pipelines.
"""

from typing import Dict, Any, Tuple

# Example Model Dispatcher Interface
def call_model(model_id: str, prompt: str, thinking_level: str = None) -> Tuple[str, float]:
    """Stub dispatch for demonstration."""
    return "def solution(): pass", 0.0015

def contained_test_run(test_command, cwd=".") -> Tuple[bool, str, str]:
    """Run a test suite under the straitjacket harness ($0.00 containment).

    The harness captures stdout/stderr at the BIRTH gate — before the bytes
    can reach the model — stores them whole, and returns a bounded,
    coverage-attested digest with retrieval addresses for what it omitted.

        ctx run -- pytest -q
        ctx get run:<id>#stdout --lines 1280:1300     # only if needed

    Returns ``(passed, digest, handle)``.

    Anti-pattern, do NOT do this:

        lines = [l for l in stderr.splitlines()
                 if any(k in l for k in ["FAIL:", "AssertionError"])]

    Keyword and head/tail selection produce a shorter string, not a digest:
    no coverage receipt, no address for the dropped regions, and the quiet
    one-line anomaly in the middle of a repetitive log is gone with no record
    that it ever existed. Position is not relevance.
    """
    import subprocess
    proc = subprocess.run(["ctx", "run", "--cwd", cwd, "--", *test_command],
                          capture_output=True, text=True)
    digest = proc.stdout.strip()
    handle = digest.split()[1].rstrip("]") if digest.startswith("[ctx ") else ""
    return proc.returncode == 0, digest, handle


def retrieve_region(handle: str, stream: str, a: int, b: int) -> str:
    """Spend one bounded, local, $0 lookup instead of re-running the suite.

        ctx get run:<id>#stderr --lines 40:52

    Retrieval must stay bounded — otherwise it is just the flood, one turn
    later.
    """
    import subprocess
    return subprocess.run(
        ["ctx", "get", f"{handle}#{stream}", "--lines", f"{a}:{b}"],
        capture_output=True, text=True).stdout

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

    # 2. Contained digest of the failing run ($0.00, already produced by the
    #    harness when the suite ran — nothing is re-summarised here).
    digest = err

    # 3. Sub-Cent Fast Repair Attempt
    r1_prompt = f"{task_prompt}\n\nCurrent Code:\n{code}\n\nTriaged Test Digest:\n{digest}\n\nFix the bug."
    code, cost = call_model("gemini-3.5-flash-lite", r1_prompt)
    tot_cost += cost
    passed, err = test_runner_fn(code)
    if passed:
        return {"code": code, "passed": True, "cost_usd": tot_cost, "tier_resolved": 2}

    # 4. Final Medium-Thinking Escalation
    digest = err
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
    Step 3: Straitjacket contained digest of the failing run (gitdiff/pytest profile)
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

    # 3. Contained digest of the failing run ($0.00)
    digest = err

    # 4. Opus-5 Escalation Repair
    repair_prompt = f"Issue:\n{issue_description}\n\nContract:\n{contract}\n\nCurrent Patch:\n{patch}\n\nRegression Digest:\n{digest}\n\nOutput corrected unified git diff."
    patch, cost = call_model("claude-opus-5", repair_prompt)
    tot_cost += cost
    passed, err = patch_test_runner_fn(patch)

    return {"patch": patch, "passed": passed, "cost_usd": tot_cost, "loops": 1}
