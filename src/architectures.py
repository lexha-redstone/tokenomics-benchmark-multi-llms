# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Comprehensive Benchmark Architectures & Variant Registry for Multi-LLM Tokenomics.
Provides unified implementations of single-model baselines, multi-model cascades,
advisor-executor splits, and zero-cost Straitjacket context containment variants.
"""

from .config import (
    GEMINI_36_FLASH_ID, GEMINI_35_FLASH_LITE_ID,
    GEMINI_FLASH_ID, GEMINI_FLASH_LITE_ID,
    SONNET_ID, OPUS_5_ID, OPUS_48_ID, OPUS_ID,
    SOLVER_ROLE, ADVISOR_ROLE, EXECUTOR_ROLE, REPAIR_ROLE,
    SWEBENCH_SOLVER_ROLE, SWEBENCH_ADVISOR_ROLE, SWEBENCH_EXECUTOR_ROLE, SWEBENCH_REPAIR_ROLE,
    WEBDEV_SOLVER_ROLE, WEBDEV_ADVISOR_ROLE
)
from .client import dispatch_model
from .evaluator import (
    extract_code, run_bigcodebench, missing_code_error,
    extract_patch, run_swebench_pro_task, missing_patch_error,
    triage_error, triage_error_straitjacket
)

# ==============================================================================
# --- PROBLEM EVALUATION DISPATCHER HELPER ---
# ==============================================================================

def _is_swebench_problem(problem):
    return problem.get("dataset_type") == "swebench" or "instance_id" in problem or "base_commit" in problem

def _build_initial_prompt(problem, role_type="solver"):
    is_swe = _is_swebench_problem(problem)
    if is_swe:
        if role_type == "advisor":
            return (
                SWEBENCH_ADVISOR_ROLE +
                f"Repository: {problem.get('repo', 'enterprise/repo')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Code Context:\n{problem.get('code_context', '')}"
            )
        else:
            return (
                SWEBENCH_SOLVER_ROLE +
                f"Repository: {problem.get('repo', 'enterprise/repo')}\n"
                f"Base Commit: {problem.get('base_commit', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Code Context:\n{problem.get('code_context', '')}\n\n"
                "Generate the COMPLETE unified git patch/diff."
            )
    else:
        # Python function completion (BCB / WebDev)
        prompt_text = problem.get("complete_prompt", "")
        if role_type == "advisor":
            return ADVISOR_ROLE + f"Problem:\n```python\n{prompt_text}\n```"
        else:
            return SOLVER_ROLE + f"Problem:\n```python\n{prompt_text}\n```\n\nWrite the complete solution."

def _eval_solution(problem, text):
    is_swe = _is_swebench_problem(problem)
    if is_swe:
        patch = extract_patch(text)
        passed, err = run_swebench_pro_task(problem, patch)
        return passed, err, patch
    else:
        code = extract_code(text)
        guard = missing_code_error(code, problem.get("entry_point", "task_func"))
        if guard:
            return False, guard, code
        passed, err = run_bigcodebench(problem, code)
        return passed, err, code

# ==============================================================================
# --- CATEGORY 1: SINGLE MODEL BASELINES ---
# ==============================================================================

def run_single(problem, model_id=GEMINI_36_FLASH_ID, thinking_level=None, max_loops=1):
    """Direct single-shot code/patch generation with optional multi-turn self-repair."""
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, dt = dispatch_model(model_id, prompt, thinking_level=thinking_level, problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    tot_dt = dt
    loop = 0

    while not passed and loop < max_loops:
        loop += 1
        digest, tr_usage, _ = triage_error_straitjacket(err, problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        if is_swe:
            repair_prompt = (
                SWEBENCH_REPAIR_ROLE +
                f"Repository: {problem.get('repo', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Current candidate patch:\n```diff\n{sol}\n```\n\n"
                f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
                "Output COMPLETE corrected unified git patch/diff."
            )
        else:
            repair_prompt = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
                f"Current solution:\n```python\n{sol}\n```\n\n"
                f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
                "Write the complete corrected solution."
            )

        r_text, r_usage, r_dt = dispatch_model(model_id, repair_prompt, thinking_level=thinking_level, problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        tot_dt += r_dt
        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(tot_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

# ==============================================================================
# --- CATEGORY 2: ADVISOR-EXECUTOR & MULTI-TIER CASCADES ---
# ==============================================================================

def run_read_write(problem, planner_model=GEMINI_36_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID):
    """Read-Heavy Advisor (Planner) + Write-Heavy Executor Split."""
    adv_prompt = _build_initial_prompt(problem, role_type="advisor")
    guidance, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024, problem=problem)

    is_swe = _is_swebench_problem(problem)
    if is_swe:
        exec_prompt = (
            SWEBENCH_EXECUTOR_ROLE +
            f"Repository: {problem.get('repo', '')}\n"
            f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
            f"Code Context:\n{problem.get('code_context', '')}\n\n"
            f"Software Architect Contract Guidance:\n{guidance}\n\n"
            "Generate the COMPLETE unified git patch/diff."
        )
    else:
        exec_prompt = (
            EXECUTOR_ROLE +
            f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
            f"Advisor guidance:\n{guidance}\n\nWrite the complete solution."
        )

    sol_text, exec_usage, exec_dt = dispatch_model(executor_model, exec_prompt, max_tokens=2560, problem=problem)
    passed, err, sol = _eval_solution(problem, sol_text)

    tot_usd = round(adv_usage["as_run_usd"] + exec_usage["as_run_usd"], 6)
    tot_out = adv_usage["output"] + exec_usage["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage["total_tokens"]
    tot_dt = adv_dt + exec_dt
    loop = 0

    if not passed:
        loop = 1
        digest, tr_usage, _ = triage_error_straitjacket(err, problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        if is_swe:
            repair_prompt = (
                SWEBENCH_REPAIR_ROLE +
                f"Repository: {problem.get('repo', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Contract:\n{guidance}\n\n"
                f"Current candidate patch:\n```diff\n{sol}\n```\n\n"
                f"Straitjacket Triaged Digest:\n```\n{digest}\n```\n\n"
                "Output COMPLETE corrected unified git patch/diff."
            )
        else:
            repair_prompt = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
                f"Advisor Guidance:\n{guidance}\n\n"
                f"Current solution:\n```python\n{sol}\n```\n\n"
                f"Unit test error:\n```\n{digest}\n```\n\n"
                "Write the complete corrected solution."
            )

        r_text, r_usage, r_dt = dispatch_model(planner_model, repair_prompt, thinking_level="low", problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        tot_dt += r_dt
        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(tot_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

def run_cascade(problem, gen_model=GEMINI_35_FLASH_LITE_ID, esc_model=GEMINI_36_FLASH_ID, max_repairs=2, escalate_after=1):
    """Two-tier generation cascade with raw stderr passing."""
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, dt = dispatch_model(gen_model, prompt, problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    tot_dt = dt
    loop = 0

    while not passed and loop < max_repairs:
        loop += 1
        escalated = loop >= escalate_after
        target_model = esc_model if escalated else gen_model
        think_level = "low" if (escalated and target_model in (GEMINI_36_FLASH_ID, GEMINI_FLASH_ID)) else None

        is_swe = _is_swebench_problem(problem)
        if is_swe:
            repair_prompt = (
                SWEBENCH_REPAIR_ROLE +
                f"Repository: {problem.get('repo', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Current candidate patch:\n```diff\n{sol}\n```\n\n"
                f"Raw Test Output:\n```\n{err[-2500:]}\n```\n\n"
                "Output COMPLETE corrected unified git patch/diff."
            )
        else:
            repair_prompt = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
                f"Current solution:\n```python\n{sol}\n```\n\n"
                f"Unit test error:\n```\n{err[-2500:]}\n```\n\n"
                "Write the complete corrected solution."
            )

        r_text, r_usage, r_dt = dispatch_model(target_model, repair_prompt, thinking_level=think_level, problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        tot_dt += r_dt
        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(tot_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

def run_hybrid(problem, planner_model=GEMINI_36_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID,
               escalate_model=GEMINI_36_FLASH_ID, triage_model=GEMINI_35_FLASH_LITE_ID):
    """Sweet Spot Hybrid with standard LLM Triage."""
    adv_prompt = _build_initial_prompt(problem, role_type="advisor")
    guidance, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024, problem=problem)

    is_swe = _is_swebench_problem(problem)
    if is_swe:
        exec_prompt = (
            SWEBENCH_EXECUTOR_ROLE +
            f"Repository: {problem.get('repo', '')}\n"
            f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
            f"Contract Guidance:\n{guidance}\n\nGenerate COMPLETE unified git patch/diff."
        )
    else:
        exec_prompt = (
            EXECUTOR_ROLE +
            f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
            f"Advisor guidance:\n{guidance}\n\nWrite the complete solution."
        )

    sol_text, exec_usage, exec_dt = dispatch_model(executor_model, exec_prompt, max_tokens=2560, problem=problem)
    passed, err, sol = _eval_solution(problem, sol_text)

    tot_usd = adv_usage["as_run_usd"] + exec_usage["as_run_usd"]
    tot_out = adv_usage["output"] + exec_usage["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage["total_tokens"]
    triage_usd = 0.0
    loop = 0

    if not passed:
        loop = 1
        digest, tr_usage, _ = triage_error(err, model_id=triage_model, is_swe=is_swe)
        tot_usd += tr_usage["as_run_usd"]
        triage_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        if is_swe:
            repair_prompt = (
                SWEBENCH_REPAIR_ROLE +
                f"Repository: {problem.get('repo', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Contract:\n{guidance}\n\n"
                f"Current candidate patch:\n```diff\n{sol}\n```\n\n"
                f"LLM Triaged Error Digest:\n```\n{digest}\n```\n\n"
                "Output COMPLETE corrected unified git patch/diff."
            )
        else:
            repair_prompt = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
                f"Current solution:\n```python\n{sol}\n```\n\n"
                f"LLM Triaged Error Digest:\n```\n{digest}\n```\n\n"
                "Write the complete corrected solution."
            )

        esc_text, esc_usage, _ = dispatch_model(escalate_model, repair_prompt, thinking_level="low", problem=problem)
        tot_usd += esc_usage["as_run_usd"]
        tot_out += esc_usage["output"]
        tot_tok += esc_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, esc_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(adv_dt + exec_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": round(triage_usd, 6),
        "patch": sol,
    }

# ==============================================================================
# --- CATEGORY 3: STRAITJACKET ZERO-COST LOCAL TRIAGE VARIANTS ---
# ==============================================================================

def run_hybrid_straitjacket(problem, planner_model=GEMINI_36_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID,
                            escalate_model=GEMINI_36_FLASH_ID):
    """Architecture E-SJ: Sweet Spot Hybrid with Straitjacket Zero-Cost Local Triage ($0.00)."""
    adv_prompt = _build_initial_prompt(problem, role_type="advisor")
    guidance, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024, problem=problem)

    is_swe = _is_swebench_problem(problem)
    if is_swe:
        exec_prompt = (
            SWEBENCH_EXECUTOR_ROLE +
            f"Repository: {problem.get('repo', '')}\n"
            f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
            f"Contract Guidance:\n{guidance}\n\nGenerate COMPLETE unified git patch/diff."
        )
    else:
        exec_prompt = (
            EXECUTOR_ROLE +
            f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
            f"Advisor guidance:\n{guidance}\n\nWrite the complete solution."
        )

    sol_text, exec_usage, exec_dt = dispatch_model(executor_model, exec_prompt, max_tokens=2560, problem=problem)
    passed, err, sol = _eval_solution(problem, sol_text)

    tot_usd = adv_usage["as_run_usd"] + exec_usage["as_run_usd"]
    tot_out = adv_usage["output"] + exec_usage["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage["total_tokens"]
    loop = 0

    if not passed:
        loop = 1
        digest, tr_usage, _ = triage_error_straitjacket(err, problem=problem)
        tot_usd += tr_usage["as_run_usd"]  # $0.000000
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        if is_swe:
            repair_prompt = (
                SWEBENCH_REPAIR_ROLE +
                f"Repository: {problem.get('repo', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Contract:\n{guidance}\n\n"
                f"Current candidate patch:\n```diff\n{sol}\n```\n\n"
                f"Straitjacket Zero-Cost Triaged Digest:\n```\n{digest}\n```\n\n"
                "Output COMPLETE corrected unified git patch/diff."
            )
        else:
            repair_prompt = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
                f"Current solution:\n```python\n{sol}\n```\n\n"
                f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
                "Write the complete corrected solution."
            )

        esc_text, esc_usage, _ = dispatch_model(escalate_model, repair_prompt, thinking_level="low", problem=problem)
        tot_usd += esc_usage["as_run_usd"]
        tot_out += esc_usage["output"]
        tot_tok += esc_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, esc_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(adv_dt + exec_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

def run_cascade_straitjacket(problem, gen_model=GEMINI_35_FLASH_LITE_ID, esc_model=GEMINI_36_FLASH_ID, max_repairs=2, escalate_after=1):
    """Architecture C-SJ: Generation Cascade with Straitjacket Local Triage & Cache Warming."""
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, dt = dispatch_model(gen_model, prompt, problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    tot_dt = dt
    loop = 0

    while not passed and loop < max_repairs:
        loop += 1
        escalated = loop >= escalate_after
        target_model = esc_model if escalated else gen_model
        think_level = "low" if (escalated and target_model in (GEMINI_36_FLASH_ID, GEMINI_FLASH_ID)) else None

        digest, tr_usage, _ = triage_error_straitjacket(err, problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        if is_swe:
            repair_prompt = (
                SWEBENCH_REPAIR_ROLE +
                f"Repository: {problem.get('repo', '')}\n"
                f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
                f"Current candidate patch:\n```diff\n{sol}\n```\n\n"
                f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
                "Output COMPLETE corrected unified git patch/diff."
            )
        else:
            repair_prompt = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
                f"Current solution:\n```python\n{sol}\n```\n\n"
                f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
                "Write the complete corrected solution."
            )

        r_text, r_usage, r_dt = dispatch_model(target_model, repair_prompt, thinking_level=think_level, problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        tot_dt += r_dt
        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(tot_dt, 1),
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

def run_escalation_shield_straitjacket(problem, lite_model=GEMINI_35_FLASH_LITE_ID, flash_model=GEMINI_36_FLASH_ID, sonnet_model=SONNET_ID):
    """
    Straitjacket Escalation Shield (Gemini 3.5-Lite -> Gemini 3.6-Flash -> Claude Sonnet-5).
    Uses zero-cost UnittestProfile triage ($0.00) and prompt cache warming.
    """
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, _ = dispatch_model(lite_model, prompt, problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    loop = 0

    if not passed:
        loop = 1
        digest, tr_usage, _ = triage_error_straitjacket(err, problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        r1_prompt = (
            role + f"Problem:\n```\n{problem.get('problem_statement', problem.get('complete_prompt', ''))}\n```\n\n"
            f"Current candidate:\n```\n{sol}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
            "Write the complete corrected solution."
        )
        r1_text, r1_usage, _ = dispatch_model(flash_model, r1_prompt, thinking_level="low", problem=problem)
        tot_usd += r1_usage["as_run_usd"]
        tot_out += r1_usage["output"]
        tot_tok += r1_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, r1_text)

    if not passed:
        loop = 2
        digest, tr_usage, _ = triage_error_straitjacket(err, problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        r2_prompt = (
            role + f"Problem:\n```\n{problem.get('problem_statement', problem.get('complete_prompt', ''))}\n```\n\n"
            f"Current candidate:\n```\n{sol}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
            "Write the complete corrected solution."
        )
        r2_text, r2_usage, _ = dispatch_model(sonnet_model, r2_prompt, problem=problem)
        tot_usd += r2_usage["as_run_usd"]
        tot_out += r2_usage["output"]
        tot_tok += r2_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, r2_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": 5.0,
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

def run_smart_repair_straitjacket(problem, flash_model=GEMINI_36_FLASH_ID, lite_model=GEMINI_35_FLASH_LITE_ID):
    """
    Straitjacket Smart Repair (Pure Gemini: 3.6-Flash -> 3.5-Lite -> 3.6-Flash Medium).
    Zero-cost local triage ($0.00) in a native Google Cloud stack.
    """
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, _ = dispatch_model(flash_model, prompt, thinking_level="low", problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    loop = 0

    if not passed:
        loop = 1
        digest, tr_usage, _ = triage_error_straitjacket(err, problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        r1_prompt = (
            role + f"Problem:\n```\n{problem.get('problem_statement', problem.get('complete_prompt', ''))}\n```\n\n"
            f"Current candidate:\n```\n{sol}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
            "Write the complete corrected solution."
        )
        r1_text, r1_usage, _ = dispatch_model(lite_model, r1_prompt, problem=problem)
        tot_usd += r1_usage["as_run_usd"]
        tot_out += r1_usage["output"]
        tot_tok += r1_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, r1_text)

    if not passed:
        loop = 2
        digest, tr_usage, _ = triage_error_straitjacket(err, problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        r2_prompt = (
            role + f"Problem:\n```\n{problem.get('problem_statement', problem.get('complete_prompt', ''))}\n```\n\n"
            f"Current candidate:\n```\n{sol}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
            "Write the complete corrected solution."
        )
        r2_text, r2_usage, _ = dispatch_model(flash_model, r2_prompt, thinking_level="medium", problem=problem)
        tot_usd += r2_usage["as_run_usd"]
        tot_out += r2_usage["output"]
        tot_tok += r2_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, r2_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": 5.0,
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

def run_ultra_sweet_straitjacket(problem, sonnet_model=SONNET_ID, lite_model=GEMINI_35_FLASH_LITE_ID, opus_model=OPUS_ID):
    """
    Straitjacket Ultra-Sweet Hybrid (Claude Sonnet-5 Advisor -> Gemini Lite Executor -> Claude Opus-5 Repair).
    Contract-guided cross-vendor synthesis with zero-cost local triage.
    """
    adv_prompt = _build_initial_prompt(problem, role_type="advisor")
    guidance, adv_usage, _ = dispatch_model(sonnet_model, adv_prompt, max_tokens=1024, problem=problem)
    tot_usd = adv_usage["as_run_usd"]
    tot_out = adv_usage["output"]
    tot_tok = adv_usage["total_tokens"]

    is_swe = _is_swebench_problem(problem)
    if is_swe:
        exec_prompt = (
            SWEBENCH_EXECUTOR_ROLE +
            f"Repository: {problem.get('repo', '')}\n"
            f"Problem Statement:\n{problem.get('problem_statement', '')}\n\n"
            f"Contract Specification:\n{guidance}\n\nGenerate COMPLETE unified git patch/diff."
        )
    else:
        exec_prompt = (
            EXECUTOR_ROLE +
            f"Problem:\n```python\n{problem.get('complete_prompt', '')}\n```\n\n"
            f"Contract Specification:\n{guidance}\n\nWrite complete solution."
        )

    sol_text, exec_usage, _ = dispatch_model(lite_model, exec_prompt, problem=problem)
    tot_usd += exec_usage["as_run_usd"]
    tot_out += exec_usage["output"]
    tot_tok += exec_usage["total_tokens"]
    passed, err, sol = _eval_solution(problem, sol_text)

    loop = 0
    if not passed:
        loop = 1
        digest, tr_usage, _ = triage_error_straitjacket(err, problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        repair_prompt = (
            role + f"Problem:\n```\n{problem.get('problem_statement', problem.get('complete_prompt', ''))}\n```\n\n"
            f"Contract:\n{guidance}\n\n"
            f"Current candidate:\n```\n{sol}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
            "Write the complete corrected solution."
        )
        r_text, r_usage, _ = dispatch_model(opus_model, repair_prompt, problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": 6.0,
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

# ==============================================================================
# --- CATEGORY 4: NEXT-GEN MULTI-PROVIDER ARCHITECTURES + STRAITJACKET ---
# ==============================================================================

def run_dual_verifier_cascade_straitjacket(problem, lite_model=GEMINI_35_FLASH_LITE_ID, flash_model=GEMINI_36_FLASH_ID,
                                          sonnet_model=SONNET_ID, opus_model=OPUS_ID):
    """4-Tier Multi-Provider Dual-Verifier Cascade with Zero-Cost Local Triage."""
    prompt = _build_initial_prompt(problem, role_type="solver")
    text, usage, _ = dispatch_model(lite_model, prompt, problem=problem)
    passed, err, sol = _eval_solution(problem, text)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    loop = 0

    tiers = [
        (flash_model, "low"),
        (sonnet_model, None),
        (opus_model, None)
    ]

    for target_model, think_level in tiers:
        if passed:
            break
        loop += 1
        digest, tr_usage, _ = triage_error_straitjacket(err, problem=problem)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        is_swe = _is_swebench_problem(problem)
        role = SWEBENCH_REPAIR_ROLE if is_swe else REPAIR_ROLE
        r_prompt = (
            role + f"Problem:\n```\n{problem.get('problem_statement', problem.get('complete_prompt', ''))}\n```\n\n"
            f"Current candidate:\n```\n{sol}\n```\n\n"
            f"Straitjacket Triaged Error Digest:\n```\n{digest}\n```\n\n"
            "Write the complete corrected solution."
        )
        r_text, r_usage, _ = dispatch_model(target_model, r_prompt, thinking_level=think_level, problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        passed, err, sol = _eval_solution(problem, r_text)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": 7.0,
        "error": "" if passed else err,
        "repair_loops": loop,
        "triage_usd": 0.0,
        "patch": sol,
    }

# ==============================================================================
# --- VARIANT REGISTRY ---
# ==============================================================================

VARIANT_REGISTRY = {
    # --- SINGLE MODELS ---
    "single_flash_lite": {
        "id": "single_flash_lite",
        "name": "1. Single: gemini-3.5-flash-lite",
        "category": "1. Single models",
        "triage_mode": "None / Direct",
        "models": "Gemini 3.5 Flash-Lite",
        "fn": lambda p: run_single(p, model_id=GEMINI_35_FLASH_LITE_ID),
    },
    "single_flash36": {
        "id": "single_flash36",
        "name": "2. Single: gemini-3.6-flash",
        "category": "1. Single models",
        "triage_mode": "None / Direct",
        "models": "Gemini 3.6 Flash",
        "fn": lambda p: run_single(p, model_id=GEMINI_36_FLASH_ID, thinking_level="low"),
    },
    "single_sonnet5": {
        "id": "single_sonnet5",
        "name": "3. Single: claude-sonnet-5",
        "category": "1. Single models",
        "triage_mode": "None / Direct",
        "models": "Claude Sonnet-5",
        "fn": lambda p: run_single(p, model_id=SONNET_ID),
    },
    "single_opus5": {
        "id": "single_opus5",
        "name": "4. Single: claude-opus-5",
        "category": "1. Single models",
        "triage_mode": "None / Direct",
        "models": "Claude Opus-5",
        "fn": lambda p: run_single(p, model_id=OPUS_5_ID),
    },
    
    # --- COMBINATIONS (NO STRAITJACKET) ---
    "combo_read_write": {
        "id": "combo_read_write",
        "name": "5. Read/Write: 3.6-Flash Plan + 3.5-Lite Exec",
        "category": "2. Combination of models",
        "triage_mode": "None / Direct",
        "models": "Gemini 3.6 Flash Plan + 3.5 Lite Exec",
        "fn": lambda p: run_read_write(p, planner_model=GEMINI_36_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID),
    },
    "combo_cascade_llm": {
        "id": "combo_cascade_llm",
        "name": "6. Cascade Baseline (Gemini 3-Tier Raw Stderr)",
        "category": "2. Combination of models",
        "triage_mode": "Raw Stderr ($0.00)",
        "models": "Gemini 3.5 Lite -> 3.6 Flash",
        "fn": lambda p: run_cascade(p, gen_model=GEMINI_35_FLASH_LITE_ID, esc_model=GEMINI_36_FLASH_ID),
    },
    "combo_hybrid_llm": {
        "id": "combo_hybrid_llm",
        "name": "7. Escalation Shield LLM Triage (Gemini -> Claude)",
        "category": "2. Combination of models",
        "triage_mode": "LLM triage_error (~$0.0018/rep)",
        "models": "Gemini Lite -> Flash -> Claude Sonnet-5",
        "fn": lambda p: run_hybrid(p, planner_model=GEMINI_36_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID, escalate_model=SONNET_ID),
    },
    
    # --- COMBINATIONS + STRAITJACKET ZERO-COST TRIAGE ---
    "sj_cascade": {
        "id": "sj_cascade",
        "name": "8. Straitjacket Cascade (3.5-Lite -> 3.6-Flash)",
        "category": "3. Combination of models + straitjacket",
        "triage_mode": "Straitjacket UnittestProfile ($0.00)",
        "models": "Gemini 3.5 Lite -> 3.6 Flash",
        "fn": lambda p: run_cascade_straitjacket(p, gen_model=GEMINI_35_FLASH_LITE_ID, esc_model=GEMINI_36_FLASH_ID),
    },
    "sj_hybrid": {
        "id": "sj_hybrid",
        "name": "9. Straitjacket Hybrid (Flash Plan + Lite Exec + Flash Repair)",
        "category": "3. Combination of models + straitjacket",
        "triage_mode": "Straitjacket UnittestProfile ($0.00)",
        "models": "Gemini 3.6 Flash + 3.5 Lite",
        "fn": lambda p: run_hybrid_straitjacket(p, planner_model=GEMINI_36_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID, escalate_model=GEMINI_36_FLASH_ID),
    },
    "sj_escalation_shield": {
        "id": "sj_escalation_shield",
        "name": "10. Straitjacket Escalation Shield (Gemini -> Claude)",
        "category": "3. Combination of models + straitjacket",
        "triage_mode": "Straitjacket UnittestProfile ($0.00)",
        "models": "Gemini Lite -> Flash -> Claude Sonnet-5",
        "fn": lambda p: run_escalation_shield_straitjacket(p, lite_model=GEMINI_35_FLASH_LITE_ID, flash_model=GEMINI_36_FLASH_ID, sonnet_model=SONNET_ID),
    },
    "sj_smart_repair": {
        "id": "sj_smart_repair",
        "name": "11. Straitjacket Smart Repair (Pure Gemini)",
        "category": "3. Combination of models + straitjacket",
        "triage_mode": "Straitjacket UnittestProfile ($0.00)",
        "models": "Gemini 3.6 Flash -> 3.5 Lite -> Flash (Med)",
        "fn": lambda p: run_smart_repair_straitjacket(p, flash_model=GEMINI_36_FLASH_ID, lite_model=GEMINI_35_FLASH_LITE_ID),
    },
    "sj_ultra_sweet": {
        "id": "sj_ultra_sweet",
        "name": "12. Straitjacket Ultra-Sweet Hybrid (Claude + Gemini)",
        "category": "3. Combination of models + straitjacket",
        "triage_mode": "Straitjacket UnittestProfile ($0.00)",
        "models": "Claude Sonnet-5 -> Gemini Lite -> Claude Opus-5",
        "fn": lambda p: run_ultra_sweet_straitjacket(p, sonnet_model=SONNET_ID, lite_model=GEMINI_35_FLASH_LITE_ID, opus_model=OPUS_5_ID),
    },
    "sj_dual_verifier": {
        "id": "sj_dual_verifier",
        "name": "13. Straitjacket Dual-Verifier Cascade (4-Tier Synergy)",
        "category": "4. Next-Gen Multi-Provider + straitjacket",
        "triage_mode": "Straitjacket UnittestProfile ($0.00)",
        "models": "Gemini Lite -> Flash -> Sonnet-5 -> Opus-5",
        "fn": lambda p: run_dual_verifier_cascade_straitjacket(p, lite_model=GEMINI_35_FLASH_LITE_ID, flash_model=GEMINI_36_FLASH_ID, sonnet_model=SONNET_ID, opus_model=OPUS_5_ID),
    },
}

def get_configurations(dataset="swebench", group="all", variant_keys=None):
    """
    Retrieve list of benchmark configurations matching dataset, group filter, or specific variant keys.
    """
    if variant_keys:
        keys = [k.strip() for k in variant_keys if k.strip()]
        selected = []
        for k in keys:
            if k in VARIANT_REGISTRY:
                selected.append(VARIANT_REGISTRY[k])
            else:
                # Find matching by partial ID or name
                matches = [v for v in VARIANT_REGISTRY.values() if k.lower() in v["id"].lower() or k.lower() in v["name"].lower()]
                if matches:
                    selected.extend(matches)
        return selected

    all_configs = list(VARIANT_REGISTRY.values())
    if group == "single":
        return [c for c in all_configs if "1. Single" in c["category"]]
    elif group == "combo":
        return [c for c in all_configs if "2. Combination" in c["category"]]
    elif group in ("straitjacket", "sj"):
        return [c for c in all_configs if "straitjacket" in c["category"].lower()]
    elif group == "nextgen":
        return [c for c in all_configs if "4. Next-Gen" in c["category"]]
    
    return all_configs
