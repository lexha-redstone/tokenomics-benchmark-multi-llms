from .config import (
    GEMINI_FLASH_ID, GEMINI_FLASH_LITE_ID,
    SOLVER_ROLE, ADVISOR_ROLE, EXECUTOR_ROLE, REPAIR_ROLE
)
from .client import dispatch_model
from .evaluator import extract_code, run_bigcodebench, missing_code_error, triage_error

def run_single(problem, model_id=GEMINI_FLASH_ID, solver_role=None):
    """Architecture A: Single-Model Direct Completion."""
    role = solver_role or SOLVER_ROLE
    prompt = role + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\nWrite the complete solution."
    text, usage, dt = dispatch_model(model_id, prompt)
    code = extract_code(text)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)
    return {
        "passed": passed,
        "as_run_usd": usage["as_run_usd"],
        "output_tokens": usage["output"],
        "total_tokens": usage["total_tokens"],
        "seconds": round(dt, 1),
        "error": "" if passed else err,
    }

def run_read_write(problem, planner_model=GEMINI_FLASH_ID, executor_model=GEMINI_FLASH_LITE_ID,
                   advisor_role=None, executor_role=None):
    """Architecture B: Read-Heavy / Write-Heavy Split (Advisor-Executor)."""
    adv_role = advisor_role or ADVISOR_ROLE
    exec_role = executor_role or EXECUTOR_ROLE
    
    adv_prompt = adv_role + f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    guidance, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024)

    exec_prompt = (exec_role + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                   f"Advisor guidance:\n{guidance}\n\nWrite the complete solution.")
    sol, exec_usage, exec_dt = dispatch_model(executor_model, exec_prompt, max_tokens=2560)
    code = extract_code(sol)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    tot_usd = round(adv_usage["as_run_usd"] + exec_usage["as_run_usd"], 6)
    tot_out = adv_usage["output"] + exec_usage["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage["total_tokens"]

    return {
        "passed": passed,
        "as_run_usd": tot_usd,
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(adv_dt + exec_dt, 1),
        "error": "" if passed else err,
    }

def run_cascade(problem, gen_model=GEMINI_FLASH_LITE_ID, esc_model=GEMINI_FLASH_ID, max_repairs=3, escalate_after=1,
                solver_role=None, repair_role=None):
    """Architecture C: Generation Offload Cascade."""
    sol_role = solver_role or SOLVER_ROLE
    rep_role = repair_role or REPAIR_ROLE
    
    prompt = sol_role + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\nWrite the complete solution."
    text, usage, dt = dispatch_model(gen_model, prompt)
    code = extract_code(text)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]

    loop = 0
    while not passed and loop < max_repairs:
        loop += 1
        escalated = loop > escalate_after
        target_model = esc_model if escalated else gen_model
        think_level = "low" if (escalated and target_model == GEMINI_FLASH_ID) else None

        repair_prompt = (rep_role + "Problem:\n```python\n" + problem["complete_prompt"] + "\n```\n\n"
                         f"Current solution:\n```python\n{code}\n```\n\n"
                         f"Unit test error:\n```\n{err[-2500:]}\n```\n\n"
                         "Write the complete corrected solution.")
        r_text, r_usage, r_dt = dispatch_model(target_model, repair_prompt, thinking_level=think_level)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]

        new_code = extract_code(r_text)
        guard = missing_code_error(new_code, problem["entry_point"])
        if guard:
            passed, err = False, guard
        else:
            code = new_code
            passed, err = run_bigcodebench(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "repair_loops": loop,
        "error": "" if passed else err,
    }

def run_hybrid(problem, planner_model=GEMINI_FLASH_ID, executor_model=GEMINI_FLASH_LITE_ID,
               escalate_model=GEMINI_FLASH_ID, triage_model=GEMINI_FLASH_LITE_ID,
               advisor_role=None, executor_role=None, repair_role=None):
    """Architecture E: Sweet Spot Hybrid (Read/Write Split + Triage + Thinking Escalation)."""
    adv_role = advisor_role or ADVISOR_ROLE
    exec_role = executor_role or EXECUTOR_ROLE
    rep_role = repair_role or REPAIR_ROLE
    
    # 1. Read-heavy planning
    adv_prompt = adv_role + f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    guidance, adv_usage, adv_dt = dispatch_model(planner_model, adv_prompt, max_tokens=1024)

    # 2. Write-heavy initial code generation
    exec_prompt = (exec_role + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                   f"Advisor guidance:\n{guidance}\n\nWrite the complete solution.")
    sol, exec_usage, exec_dt = dispatch_model(executor_model, exec_prompt, max_tokens=2560)
    code = extract_code(sol)
    guard = missing_code_error(code, problem["entry_point"])
    passed, err = (False, guard) if guard else run_bigcodebench(problem, code)

    tot_usd = adv_usage["as_run_usd"] + exec_usage["as_run_usd"]
    tot_out = adv_usage["output"] + exec_usage["output"]
    tot_tok = adv_usage["total_tokens"] + exec_usage["total_tokens"]

    # 3. Repair & Escalation
    if not passed:
        # Step 3a: Cheap repair by executor
        r1_prompt = (rep_role + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                     f"Current solution:\n```python\n{code}\n```\n\n"
                     f"Unit test error:\n```\n{err[-2500:]}\n```\n\n"
                     "Write the complete corrected solution.")
        r1_text, r1_usage, _ = dispatch_model(executor_model, r1_prompt)
        tot_usd += r1_usage["as_run_usd"]
        tot_out += r1_usage["output"]
        tot_tok += r1_usage["total_tokens"]

        new_code = extract_code(r1_text)
        guard = missing_code_error(new_code, problem["entry_point"])
        if not guard:
            code = new_code
            passed, err = run_bigcodebench(problem, code)

    if not passed:
        # Step 3b: Log triage + Escalated Thinking repair
        digest, tr_usage, _ = triage_error(err, model_id=triage_model)
        tot_usd += tr_usage["as_run_usd"]
        tot_out += tr_usage["output"]
        tot_tok += tr_usage["total_tokens"]

        esc_prompt = (rep_role + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                      f"Current solution:\n```python\n{code}\n```\n\n"
                      f"Triaged Error Digest:\n```\n{digest}\n```\n\n"
                      "Write the complete corrected solution.")
        esc_text, esc_usage, _ = dispatch_model(escalate_model, esc_prompt, thinking_level="low")
        tot_usd += esc_usage["as_run_usd"]
        tot_out += esc_usage["output"]
        tot_tok += esc_usage["total_tokens"]

        new_code = extract_code(esc_text)
        guard = missing_code_error(new_code, problem["entry_point"])
        if not guard:
            code = new_code
            passed, err = run_bigcodebench(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "error": "" if passed else err,
    }
