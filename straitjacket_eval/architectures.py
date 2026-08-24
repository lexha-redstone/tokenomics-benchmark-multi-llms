# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Multi-LLM Architectures and Harness Dispatch for Straitjacket Benchmark.
Implements:
  1. Control Baselines:
     - Single Gemini 3.7-Flash (Single-model baseline with thinking)
     - Single Claude Sonnet-5 (Single-model baseline)
  2. Core Architectures:
     - Smart Tiered Cascade Sequence Flow (2-Tiered cascade: 3.5-Lite -> 3.7-Flash)
     - Straitjacket Smart Repair (Advisor & Executor: 3.7-Flash Advisor -> 3.5-Lite Executor -> 3.7-Flash Repair)
     - Straitjacket Escalation Shield (3-Tiered Cascade: 3.5-Lite -> 3.5-Lite Repair -> 3.7-Flash Reasoning Escalation)
  3. Advanced Architectures:
     - Straitjacket DAG Wave Orchestrator (Topological waves + CAS Checkpoint Handoff + 1-Tier Escalation)
     - Straitjacket Dual-Candidate Consensus Repair (Parallel 3.5-Lite -> Failure Diff -> 3.7-Flash Synthesis)
"""

import time
from .config import (
    GEMINI_37_FLASH_ID, GEMINI_35_FLASH_LITE_ID, GEMINI_36_FLASH_ID,
    SONNET_ID, OPUS_5_ID,
    SOLVER_ROLE, ADVISOR_ROLE, EXECUTOR_ROLE, REPAIR_ROLE, SYNTHESIZER_ROLE
)
from .client import dispatch_model
from .harness import (
    extract_code, run_sandboxed_test, render_straitjacket_digest,
    StraitjacketCASStore, compute_run_diff
)

# ==============================================================================
# --- 1. CONTROL BASELINES ---
# ==============================================================================

def run_single_gemini_37(problem, max_loops=1):
    """Control Baseline: Gemini 3.7-Flash Direct Completion + Self-Repair."""
    prompt = SOLVER_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nWrite the complete solution."
    text, usage, dt = dispatch_model(GEMINI_37_FLASH_ID, prompt, thinking_level="low", problem=problem)
    code = extract_code(text)
    passed, err, code_exit = run_sandboxed_test(problem, code)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    tot_dt = dt
    loops = 0
    tokens_saved = 0

    while not passed and loops < max_loops:
        loops += 1
        digest, tr_meta = render_straitjacket_digest(err, exit_code=code_exit)
        tokens_saved += tr_meta["tokens_saved"]

        repair_prompt = (
            REPAIR_ROLE +
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Current candidate code:\n```python\n{code}\n```\n\n"
            f"Straitjacket Deterministic Error Digest:\n{digest}\n\n"
            "Analyze the failure and output the COMPLETE corrected solution."
        )
        r_text, r_usage, r_dt = dispatch_model(GEMINI_37_FLASH_ID, repair_prompt, thinking_level="low", problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        tot_dt += r_dt
        code = extract_code(r_text)
        passed, err, code_exit = run_sandboxed_test(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(tot_dt, 2),
        "repair_loops": loops,
        "tokens_saved": tokens_saved,
        "error": "" if passed else str(err)[:400]
    }

def run_single_claude_sonnet5(problem, max_loops=1):
    """Control Baseline: Claude Sonnet-5 Direct Completion + Self-Repair."""
    prompt = SOLVER_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nWrite the complete solution."
    text, usage, dt = dispatch_model(SONNET_ID, prompt, problem=problem)
    code = extract_code(text)
    passed, err, code_exit = run_sandboxed_test(problem, code)

    tot_usd = usage["as_run_usd"]
    tot_out = usage["output"]
    tot_tok = usage["total_tokens"]
    tot_dt = dt
    loops = 0
    tokens_saved = 0

    while not passed and loops < max_loops:
        loops += 1
        digest, tr_meta = render_straitjacket_digest(err, exit_code=code_exit)
        tokens_saved += tr_meta["tokens_saved"]

        repair_prompt = (
            REPAIR_ROLE +
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Current candidate code:\n```python\n{code}\n```\n\n"
            f"Straitjacket Deterministic Error Digest:\n{digest}\n\n"
            "Analyze the failure and output the COMPLETE corrected solution."
        )
        r_text, r_usage, r_dt = dispatch_model(SONNET_ID, repair_prompt, problem=problem)
        tot_usd += r_usage["as_run_usd"]
        tot_out += r_usage["output"]
        tot_tok += r_usage["total_tokens"]
        tot_dt += r_dt
        code = extract_code(r_text)
        passed, err, code_exit = run_sandboxed_test(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(tot_dt, 2),
        "repair_loops": loops,
        "tokens_saved": tokens_saved,
        "error": "" if passed else str(err)[:400]
    }

# ==============================================================================
# --- 2. CORE ARCHITECTURES ---
# ==============================================================================

def run_smart_tiered_cascade(problem):
    """
    Architecture 1: Smart Tiered Cascade Sequence Flow (2-Tiered Cascade).
    - Tier 1: Gemini 3.5-Flash-Lite initial draft ($0.30/$2.50 economy)
    - Straitjacket Context Containment & $0.00 Zero-Cost Local Triage
    - Tier 2: Gemini 3.7-Flash Thinking Repair on failure
    """
    t0 = time.time()
    prompt1 = SOLVER_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nWrite the complete solution."
    text1, u1, dt1 = dispatch_model(GEMINI_35_FLASH_LITE_ID, prompt1, problem=problem)
    code = extract_code(text1)
    passed, err, code_exit = run_sandboxed_test(problem, code)

    tot_usd = u1["as_run_usd"]
    tot_out = u1["output"]
    tot_tok = u1["total_tokens"]
    loops = 0
    tokens_saved = 0

    if not passed:
        loops = 1
        digest, tr_meta = render_straitjacket_digest(err, exit_code=code_exit)
        tokens_saved += tr_meta["tokens_saved"]

        repair_prompt = (
            REPAIR_ROLE +
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Current candidate solution:\n```python\n{code}\n```\n\n"
            f"Straitjacket Deterministic Error Digest:\n{digest}\n\n"
            "Apply precise step-by-step reasoning to fix the bug and output the COMPLETE corrected solution."
        )
        r_text, r_u, r_dt = dispatch_model(GEMINI_37_FLASH_ID, repair_prompt, thinking_level="low", problem=problem)
        tot_usd += r_u["as_run_usd"]
        tot_out += r_u["output"]
        tot_tok += r_u["total_tokens"]
        code = extract_code(r_text)
        passed, err, _ = run_sandboxed_test(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(time.time() - t0, 2),
        "repair_loops": loops,
        "tokens_saved": tokens_saved,
        "error": "" if passed else str(err)[:400]
    }

def run_straitjacket_smart_repair(problem):
    """
    Architecture 2: Straitjacket Smart Repair (Advisor & Executor).
    - Stage 1 (Architect Contract): Gemini 3.7-Flash produces concise contract guidance (<200 words, no code).
    - Stage 2 (Execution): Gemini 3.5-Flash-Lite implements full code following the contract.
    - Stage 3 (Repair): Straitjacket Zero-Cost Local Triage -> Gemini 3.7-Flash thinking repair.
    """
    t0 = time.time()
    adv_prompt = ADVISOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    guidance, adv_u, adv_dt = dispatch_model(GEMINI_37_FLASH_ID, adv_prompt, max_tokens=400, thinking_level="off", problem=problem)

    tot_usd = adv_u["as_run_usd"]
    tot_out = adv_u["output"]
    tot_tok = adv_u["total_tokens"]

    exec_prompt = (
        EXECUTOR_ROLE +
        f"Software Architect Implementation Contract Guidance:\n{guidance}\n\n"
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        "Generate the complete Python implementation honoring all specifications."
    )
    exec_text, exec_u, exec_dt = dispatch_model(GEMINI_35_FLASH_LITE_ID, exec_prompt, problem=problem)
    tot_usd += exec_u["as_run_usd"]
    tot_out += exec_u["output"]
    tot_tok += exec_u["total_tokens"]

    code = extract_code(exec_text)
    passed, err, code_exit = run_sandboxed_test(problem, code)
    loops = 0
    tokens_saved = 0

    if not passed:
        loops = 1
        digest, tr_meta = render_straitjacket_digest(err, exit_code=code_exit)
        tokens_saved += tr_meta["tokens_saved"]

        repair_prompt = (
            REPAIR_ROLE +
            f"Software Architect Contract Guidance:\n{guidance}\n\n"
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Current candidate code:\n```python\n{code}\n```\n\n"
            f"Straitjacket Deterministic Error Digest:\n{digest}\n\n"
            "Output the COMPLETE corrected Python solution."
        )
        r_text, r_u, r_dt = dispatch_model(GEMINI_37_FLASH_ID, repair_prompt, thinking_level="low", problem=problem)
        tot_usd += r_u["as_run_usd"]
        tot_out += r_u["output"]
        tot_tok += r_u["total_tokens"]
        code = extract_code(r_text)
        passed, err, _ = run_sandboxed_test(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(time.time() - t0, 2),
        "repair_loops": loops,
        "tokens_saved": tokens_saved,
        "error": "" if passed else str(err)[:400]
    }

def run_straitjacket_escalation_shield(problem):
    """
    Architecture 3: Straitjacket Escalation Shield (3-Tiered Cascade).
    - Tier 1: Gemini 3.5-Flash-Lite initial draft ($0.30/1M)
    - Tier 2: Straitjacket Zero-Cost Triage -> Gemini 3.5-Flash-Lite cheap self-repair (Economy repair)
    - Tier 3: If still failing -> Straitjacket CAS Checkpoint Escalation Shield -> Gemini 3.7-Flash medium thinking repair
    """
    t0 = time.time()
    store = StraitjacketCASStore()

    prompt1 = SOLVER_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nWrite the complete solution."
    text1, u1, dt1 = dispatch_model(GEMINI_35_FLASH_LITE_ID, prompt1, problem=problem)
    code = extract_code(text1)
    passed, err, code_exit = run_sandboxed_test(problem, code)

    tot_usd = u1["as_run_usd"]
    tot_out = u1["output"]
    tot_tok = u1["total_tokens"]
    loops = 0
    tokens_saved = 0

    if not passed:
        # Tier 2: Cheap Economy Self-Repair
        loops = 1
        digest1, tr_meta1 = render_straitjacket_digest(err, exit_code=code_exit)
        tokens_saved += tr_meta1["tokens_saved"]
        ckpt1 = store.create_checkpoint("node_t1", code, digest1, passed=False)

        prompt2 = (
            REPAIR_ROLE +
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Current candidate code:\n```python\n{code}\n```\n\n"
            f"Straitjacket Deterministic Error Digest ({ckpt1}):\n{digest1}\n\n"
            "Fix the bug and output the COMPLETE corrected solution."
        )
        text2, u2, dt2 = dispatch_model(GEMINI_35_FLASH_LITE_ID, prompt2, problem=problem)
        tot_usd += u2["as_run_usd"]
        tot_out += u2["output"]
        tot_tok += u2["total_tokens"]
        code2 = extract_code(text2)
        passed2, err2, code_exit2 = run_sandboxed_test(problem, code2)

        if passed2:
            code, passed, err = code2, True, ""
        else:
            # Tier 3: Escalation Shield -> Gemini 3.7-Flash with Medium Thinking
            loops = 2
            digest2, tr_meta2 = render_straitjacket_digest(err2, exit_code=code_exit2)
            tokens_saved += tr_meta2["tokens_saved"]
            ckpt2 = store.create_checkpoint("node_t2", code2, digest2, passed=False)

            prompt3 = (
                REPAIR_ROLE +
                f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
                f"Candidate code after cheap repair:\n```python\n{code2}\n```\n\n"
                f"Straitjacket Escalation Shield CAS Checkpoint ({ckpt2}):\n{digest2}\n\n"
                "Apply rigorous architectural reasoning and edge-case validation. Output COMPLETE corrected solution."
            )
            text3, u3, dt3 = dispatch_model(GEMINI_37_FLASH_ID, prompt3, thinking_level="medium", problem=problem)
            tot_usd += u3["as_run_usd"]
            tot_out += u3["output"]
            tot_tok += u3["total_tokens"]
            code3 = extract_code(text3)
            passed3, err3, _ = run_sandboxed_test(problem, code3)
            if passed3:
                code, passed, err = code3, True, ""
            else:
                code, passed, err = code3, False, err3

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(time.time() - t0, 2),
        "repair_loops": loops,
        "tokens_saved": tokens_saved,
        "error": "" if passed else str(err)[:400]
    }

# ==============================================================================
# --- 3. ADVANCED ARCHITECTURES ---
# ==============================================================================

def run_straitjacket_dag_wave(problem):
    """
    Architecture 4 (Bonus): Straitjacket DAG Route Wave Orchestrator (ctx.route/v1).
    - Wave 0: Gemini 3.7-Flash Architect Contract -> CAS Blob
    - Wave 1: Gemini 3.5-Flash-Lite Parallel Execution behind Straitjacket harness
    - Wave 2: Zero-cost test evaluation and CAS Checkpoint handoff
    - Wave 3: Single-Tier Escalation to Gemini 3.7-Flash (Low Thinking) if needed
    """
    t0 = time.time()
    store = StraitjacketCASStore()

    # Wave 0: Contract Node
    contract_prompt = ADVISOR_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```"
    contract_text, u_c, _ = dispatch_model(GEMINI_37_FLASH_ID, contract_prompt, max_tokens=300, thinking_level="off", problem=problem)
    contract_ref = store.put_blob(contract_text)

    tot_usd = u_c["as_run_usd"]
    tot_out = u_c["output"]
    tot_tok = u_c["total_tokens"]

    # Wave 1: Execution Node (Economy Tier)
    exec_prompt = (
        EXECUTOR_ROLE +
        f"Contract Ref: {contract_ref}\nGuidance:\n{contract_text}\n\n"
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        "Output complete code."
    )
    exec_text, u_e, _ = dispatch_model(GEMINI_35_FLASH_LITE_ID, exec_prompt, problem=problem)
    tot_usd += u_e["as_run_usd"]
    tot_out += u_e["output"]
    tot_tok += u_e["total_tokens"]
    code = extract_code(exec_text)

    # Wave 2: Zero-Cost Verification
    passed, err, code_exit = run_sandboxed_test(problem, code)
    loops = 0
    tokens_saved = 0

    if not passed:
        loops = 1
        digest, tr_meta = render_straitjacket_digest(err, exit_code=code_exit)
        tokens_saved += tr_meta["tokens_saved"]
        ckpt = store.create_checkpoint("wave1_exec", code, digest, passed=False)

        # Wave 3: Single-Tier Escalation Node
        repair_prompt = (
            REPAIR_ROLE +
            f"DAG Route Escalation Node | Upstream Checkpoint: {ckpt}\n"
            f"Contract Guidance:\n{contract_text}\n\n"
            f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
            f"Candidate Implementation:\n```python\n{code}\n```\n\n"
            f"Straitjacket Failure Digest:\n{digest}\n\n"
            "Apply reasoning escalation and output the COMPLETE corrected solution."
        )
        r_text, r_u, _ = dispatch_model(GEMINI_37_FLASH_ID, repair_prompt, thinking_level="low", problem=problem)
        tot_usd += r_u["as_run_usd"]
        tot_out += r_u["output"]
        tot_tok += r_u["total_tokens"]
        code = extract_code(r_text)
        passed, err, _ = run_sandboxed_test(problem, code)

    return {
        "passed": passed,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(time.time() - t0, 2),
        "repair_loops": loops,
        "tokens_saved": tokens_saved,
        "error": "" if passed else str(err)[:400]
    }

def run_straitjacket_dual_consensus(problem):
    """
    Architecture 5 (Bonus): Straitjacket Dual-Candidate Consensus Repair.
    - Stage 1: Dual parallel 3.5-Lite candidate generation (Candidate A: Standard, Candidate B: Edge-case robust).
    - If either passes -> returns immediately!
    - If both fail -> Straitjacket computes noise-stripped failure diff (`compute_run_diff`) -> Gemini 3.7-Flash synthesizes repair.
    """
    t0 = time.time()
    promptA = SOLVER_ROLE + f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\nWrite the complete solution."
    textA, uA, _ = dispatch_model(GEMINI_35_FLASH_LITE_ID, promptA, problem=problem)
    codeA = extract_code(textA)
    passedA, errA, exitA = run_sandboxed_test(problem, codeA)

    tot_usd = uA["as_run_usd"]
    tot_out = uA["output"]
    tot_tok = uA["total_tokens"]

    if passedA:
        return {
            "passed": True,
            "as_run_usd": round(tot_usd, 6),
            "output_tokens": tot_out,
            "total_tokens": tot_tok,
            "seconds": round(time.time() - t0, 2),
            "repair_loops": 0,
            "tokens_saved": 0,
            "error": ""
        }

    promptB = "You are an expert programmer focusing on robustness, strict typing, and boundary conditions. Output ONLY one python code block.\n\n" + problem["complete_prompt"]
    textB, uB, _ = dispatch_model(GEMINI_35_FLASH_LITE_ID, promptB, problem=problem)
    codeB = extract_code(textB)
    passedB, errB, exitB = run_sandboxed_test(problem, codeB)

    tot_usd += uB["as_run_usd"]
    tot_out += uB["output"]
    tot_tok += uB["total_tokens"]

    if passedB:
        return {
            "passed": True,
            "as_run_usd": round(tot_usd, 6),
            "output_tokens": tot_out,
            "total_tokens": tot_tok,
            "seconds": round(time.time() - t0, 2),
            "repair_loops": 1,
            "tokens_saved": 0,
            "error": ""
        }

    # Both candidates failed -> Straitjacket Zero-Cost Local Triage + Noise-Stripped Diff
    digestA, metaA = render_straitjacket_digest(errA, exit_code=exitA)
    digestB, metaB = render_straitjacket_digest(errB, exit_code=exitB)
    diff_signal = compute_run_diff(digestA, digestB)
    tokens_saved = metaA["tokens_saved"] + metaB["tokens_saved"]

    synth_prompt = (
        SYNTHESIZER_ROLE +
        f"Problem:\n```python\n{problem['complete_prompt']}\n```\n\n"
        f"Candidate A Implementation:\n```python\n{codeA}\n```\n"
        f"Candidate A Failure Profile:\n{digestA}\n\n"
        f"Candidate B Implementation:\n```python\n{codeB}\n```\n"
        f"Candidate B Failure Profile:\n{digestB}\n\n"
        f"Straitjacket Consensus Failure Diff:\n{diff_signal}\n\n"
        "Synthesize a COMPLETE, robust, and correct Python solution."
    )
    text_synth, u_s, _ = dispatch_model(GEMINI_37_FLASH_ID, synth_prompt, thinking_level="low", problem=problem)
    tot_usd += u_s["as_run_usd"]
    tot_out += u_s["output"]
    tot_tok += u_s["total_tokens"]
    code_synth = extract_code(text_synth)
    passed_synth, err_synth, _ = run_sandboxed_test(problem, code_synth)

    return {
        "passed": passed_synth,
        "as_run_usd": round(tot_usd, 6),
        "output_tokens": tot_out,
        "total_tokens": tot_tok,
        "seconds": round(time.time() - t0, 2),
        "repair_loops": 2,
        "tokens_saved": tokens_saved,
        "error": "" if passed_synth else str(err_synth)[:400]
    }
