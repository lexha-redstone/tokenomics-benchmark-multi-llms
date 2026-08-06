# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Central Configuration, Model IDs, Pricing Rates, and Role Prompts for Multi-LLM Benchmarks.
"""

import os

# ==============================================================================
# --- MODEL IDENTIFIERS ---
# ==============================================================================

# Next-Gen Google Gemini Models
GEMINI_36_FLASH_ID = "gemini-3.6-flash"
GEMINI_35_FLASH_LITE_ID = "gemini-3.5-flash-lite"
GEMINI_PRO_ID = "gemini-3.1-pro-preview"

# Previous-Gen / Baseline Gemini Models
GEMINI_FLASH_ID = "gemini-3.5-flash"
GEMINI_FLASH_LITE_ID = "gemini-3.1-flash-lite"

# Anthropic Claude Models (via Vertex AI rawPredict)
SONNET_ID = "claude-sonnet-5"
OPUS_5_ID = "claude-opus-5"
OPUS_48_ID = "claude-opus-4-8"
OPUS_ID = "claude-opus-5"  # Default Opus pointer for next-gen combinations

# ==============================================================================
# --- GOOGLE CLOUD CONFIGURATION ---
# ==============================================================================

GCP_PROJECT = os.environ.get("GCP_PROJECT", "my-argolis-prj")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")

# ==============================================================================
# --- PRICING TABLE (USD per 1,000,000 tokens) ---
# ==============================================================================

PRICING = {
    # Anthropic Models
    OPUS_5_ID:               {"input": 5.00, "output": 25.00, "cache_read": 0.50,  "cache_write": 6.25},
    OPUS_48_ID:              {"input": 5.00, "output": 25.00, "cache_read": 0.50,  "cache_write": 6.25},
    SONNET_ID:               {"input": 2.00, "output": 10.00, "cache_read": 0.20,  "cache_write": 2.50},
    
    # Next-Gen Google Gemini Models
    GEMINI_36_FLASH_ID:      {"input": 1.50, "output": 7.50,  "cache_read": 0.15,  "cache_write": 0.00},
    GEMINI_35_FLASH_LITE_ID: {"input": 0.30, "output": 2.50,  "cache_read": 0.030, "cache_write": 0.00},
    
    # Baseline / Earlier Gen Models
    GEMINI_FLASH_ID:         {"input": 1.50, "output": 9.00,  "cache_read": 0.15,  "cache_write": 0.00},
    GEMINI_FLASH_LITE_ID:    {"input": 0.25, "output": 1.50,  "cache_read": 0.025, "cache_write": 0.00},
    GEMINI_PRO_ID:           {"input": 2.00, "output": 12.00, "cache_read": 0.20,  "cache_write": 0.00},
}

# ==============================================================================
# --- PYTHON FUNCTION COMPLETION PROMPTS (BigCodeBench & General) ---
# ==============================================================================

SOLVER_ROLE = (
    "You are an expert Python programmer. Complete the function below. You are given its imports, "
    "signature, and docstring; several real libraries must be used correctly. Output the COMPLETE "
    "solution: all needed imports and the full function definition, handling edge cases and the "
    "documented return/exception behavior exactly. Output ONLY one ```python code block, no "
    "explanation.\n\n"
)

ADVISOR_ROLE = (
    "You are a senior ADVISOR in an advisor-executor coding system. You do NOT write code. Given "
    "the Python coding problem below (imports + function signature + docstring), which requires "
    "correctly using several real libraries, produce concise, precise implementation GUIDANCE for "
    "a separate executor model: which libraries/APIs to use and in what order, the intended "
    "algorithm, edge cases, and the EXACT documented return values and exceptions to honor. "
    "Under 200 words. Do NOT output any code.\n\n"
)

EXECUTOR_ROLE = (
    "You are an EXECUTOR in an advisor-executor coding system. Given a Python problem and an "
    "advisor's guidance, output the COMPLETE Python solution: all needed imports and the full "
    "function definition, honoring the documented return/exception behavior exactly. Output ONLY "
    "one ```python code block, no explanation.\n\n"
)

REPAIR_ROLE = (
    "You are an expert Python programmer. A candidate solution to the problem below FAILED its "
    "unit tests. Analyze the test error output, find the bug, and fix the code. Output the "
    "COMPLETE corrected solution: all needed imports and the full function definition. Do not "
    "output a diff or a fragment. Output ONLY one ```python code block, no explanation.\n\n"
)

TRIAGE_ROLE = (
    "You are a test-failure triage tool. Compress the Python unittest stderr below into a SHORT "
    "digest (max 12 lines) preserving EXACTLY: (1) each failing test method name, (2) the "
    "exception type and message, (3) assertion diffs with expected vs actual values (truncate "
    "values longer than ~120 chars), and (4) traceback lines inside the candidate solution "
    "(file \"prog.py\") with their line numbers. DROP unittest boilerplate, separators, and "
    "library-internal frames. Copy identifiers and values VERBATIM -- never paraphrase numbers. "
    "Output plain text only, no code fences.\n\nStderr:\n"
)

# ==============================================================================
# --- SWE-BENCH PRO PATCH GENERATION PROMPTS ---
# ==============================================================================

SWEBENCH_SOLVER_ROLE = (
    "You are an expert software engineer resolving a complex long-horizon issue in an enterprise "
    "repository. Analyze the problem statement and the repository context below. Provide the "
    "COMPLETE git patch/diff that fixes the issue without introducing regressions. "
    "Output ONLY a single ```diff code block containing the unified diff, with no extra explanation.\n\n"
)

SWEBENCH_ADVISOR_ROLE = (
    "You are a senior SOFTWARE ARCHITECT in an advisor-executor coding system for SWE-bench Pro. "
    "You do NOT write full diffs. Given the enterprise problem statement and codebase context below, "
    "produce concise, precise implementation CONTRACT GUIDANCE for an executor model: identify the "
    "exact files, functions, and line ranges to modify, the algorithmic invariants, root cause, "
    "and edge cases to respect. Keep guidance under 250 words. Do NOT output any code or diffs.\n\n"
)

SWEBENCH_EXECUTOR_ROLE = (
    "You are a skilled EXECUTOR engineer in an advisor-executor system for SWE-bench Pro. "
    "Given the problem statement, codebase context, and the software architect's contract guidance, "
    "output the COMPLETE git patch/diff that implements the fix. Ensure strict adherence to the "
    "architect's instructions. Output ONLY a single ```diff code block, no extra commentary.\n\n"
)

SWEBENCH_REPAIR_ROLE = (
    "You are an expert Python software engineer specializing in debugging and test repair for SWE-bench Pro. "
    "A candidate patch FAILED unit/regression tests. Analyze the problem statement, candidate patch, "
    "and the test error log or triaged digest below. Identify the root cause of test failure and "
    "output the COMPLETE corrected unified diff/patch. Output ONLY a single ```diff code block.\n\n"
)

SWEBENCH_TRIAGE_ROLE = (
    "You are an automated test-failure triage tool for SWE-bench Pro. Compress the test runner "
    "stderr/stdout below into a SHORT digest (max 15 lines) preserving EXACTLY: "
    "(1) each failing test name (FAIL_TO_PASS or PASS_TO_PASS regression), "
    "(2) exception type and assertion diff message, and (3) innermost traceback line numbers in the "
    "target repository files. DROP test runner boilerplate and library-internal frames. "
    "Copy identifiers and numbers VERBATIM. Output plain text only.\n\nTest Output:\n"
)

# ==============================================================================
# --- WEB-DEV SPECIFIC PROMPTS ---
# ==============================================================================

WEBDEV_SOLVER_ROLE = (
    "You are an expert Python programmer. Complete the function below. You are given its imports, "
    "signature, and docstring; several real web/networking libraries must be used correctly. Output the COMPLETE "
    "solution: all needed imports and the full function definition, handling edge cases and the "
    "documented return/exception behavior exactly. Output ONLY one ```python code block, no "
    "explanation.\n\n"
)

WEBDEV_ADVISOR_ROLE = (
    "You are a senior ADVISOR in an advisor-executor coding system. You do NOT write code. Given "
    "the Python coding problem below (imports + function signature + docstring), which requires "
    "correctly using web/networking libraries, produce concise, precise implementation GUIDANCE for "
    "a separate executor model: which libraries/APIs to use and in what order, the intended "
    "algorithm, edge cases, and the EXACT documented return values and exceptions to honor. "
    "Under 200 words. Do NOT output any code.\n\n"
)
