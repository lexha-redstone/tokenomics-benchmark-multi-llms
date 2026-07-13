import os

# --- Model IDs ---
OPUS_ID = "claude-opus-4-8"
SONNET_ID = "claude-sonnet-5"
GEMINI_FLASH_ID = "gemini-3.5-flash"
GEMINI_FLASH_LITE_ID = "gemini-3.1-flash-lite"
GEMINI_PRO_ID = "gemini-3.1-pro-preview"

# --- GCP Configuration ---
GCP_PROJECT = os.environ.get("GCP_PROJECT", "my-argolis-prj")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")

# --- Pricing (USD per 1,000,000 tokens) ---
PRICING = {
    OPUS_ID:              {"input": 5.00, "output": 25.00, "cache_read": 0.50,  "cache_write": 6.25},
    SONNET_ID:            {"input": 2.00, "output": 10.00, "cache_read": 0.20,  "cache_write": 2.50},
    GEMINI_FLASH_ID:      {"input": 1.50, "output": 9.00,  "cache_read": 0.15,  "cache_write": 1.50},
    GEMINI_FLASH_LITE_ID: {"input": 0.25, "output": 1.50,  "cache_read": 0.025, "cache_write": 0.25},
    GEMINI_PRO_ID:        {"input": 2.00, "output": 12.00, "cache_read": 0.20,  "cache_write": 2.00},
}

# --- Default Prompts ---
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
