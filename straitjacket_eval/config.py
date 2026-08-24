# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Central Configuration for Straitjacket Multi-LLM Benchmark Evaluation Suite.
Coordinates Google Gemini (3.7-Flash, 3.5-Flash-Lite) and Anthropic Claude models
with official Vertex AI pricing and Straitjacket Context Containment parameters.
"""

import os

# ==============================================================================
# --- ASSET & WORKSPACE PATHS ---
# ==============================================================================

STRAITJACKET_ASSET_DIR = "/usr/local/google/home/lexha/Desktop/work/prj/99-assets/straitjacket"
STRAITJACKET_SRC = os.path.join(STRAITJACKET_ASSET_DIR, "src")

# ==============================================================================
# --- MODEL IDENTIFIERS ---
# ==============================================================================

GEMINI_37_FLASH_ID = "gemini-3.7-flash"
GEMINI_35_FLASH_LITE_ID = "gemini-3.5-flash-lite"
GEMINI_36_FLASH_ID = "gemini-3.6-flash"
SONNET_ID = "claude-sonnet-5"
OPUS_5_ID = "claude-opus-5"

# ==============================================================================
# --- GOOGLE CLOUD CONFIGURATION ---
# ==============================================================================

GCP_PROJECT = os.environ.get("GCP_PROJECT", "my-argolis-prj")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")

# ==============================================================================
# --- PRICING TABLE (USD per 1,000,000 tokens) ---
# ==============================================================================

PRICING = {
    # Google Gemini Models (Vertex AI)
    GEMINI_35_FLASH_LITE_ID: {"input": 0.30, "output": 2.50, "cache_read": 0.030, "cache_write": 0.00},
    GEMINI_37_FLASH_ID:      {"input": 1.50, "output": 7.50, "cache_read": 0.150, "cache_write": 0.00},
    GEMINI_36_FLASH_ID:      {"input": 1.50, "output": 7.50, "cache_read": 0.150, "cache_write": 0.00},

    # Anthropic Claude Models (Vertex AI rawPredict)
    SONNET_ID:               {"input": 2.00, "output": 10.00, "cache_read": 0.200, "cache_write": 2.50},
    OPUS_5_ID:               {"input": 5.00, "output": 25.00, "cache_read": 0.500, "cache_write": 6.25},
}

# ==============================================================================
# --- PROMPTS AND CONTRACT GUIDELINES ---
# ==============================================================================

SOLVER_ROLE = (
    "You are an expert Python programmer. Complete the function below. You are given its imports, "
    "signature, and docstring; several real libraries must be used correctly. Output the COMPLETE "
    "solution: all needed imports and the full function definition, handling edge cases and the "
    "documented return/exception behavior exactly. Output ONLY one ```python code block, no "
    "explanation.\n\n"
)

ADVISOR_ROLE = (
    "You are a senior ARCHITECT & ADVISOR in a high-efficiency multi-model coding system. "
    "You do NOT write full code. Given the Python problem below, produce a concise, precise "
    "implementation SPECIFICATION CONTRACT for the executor model: identify required libraries, "
    "core algorithmic flow, critical edge cases, and exact return/exception behaviors. "
    "Keep under 200 words. Do NOT output any code blocks.\n\n"
)

EXECUTOR_ROLE = (
    "You are an EXECUTOR in a high-efficiency multi-model coding system. Given a Python problem and "
    "an architect's contract guidance, output the COMPLETE Python solution: all needed imports and the "
    "full function definition, honoring the documented return/exception behavior exactly. "
    "Output ONLY one ```python code block, no explanation.\n\n"
)

REPAIR_ROLE = (
    "You are an expert Python programmer. A candidate solution FAILED its unit tests. "
    "Analyze the Straitjacket Deterministic Test Error Digest, find the root cause, and output the "
    "COMPLETE corrected solution: all needed imports and the full function definition. "
    "Do not output diffs or partial fragments. Output ONLY one ```python code block, no explanation.\n\n"
)

SYNTHESIZER_ROLE = (
    "You are a master Python software engineer synthesizing insights from multiple parallel candidate runs. "
    "Analyze the problem and the failure digests from two distinct candidate implementations. "
    "Diagnose the underlying root causes, resolve conflicts, and produce a COMPLETE, robust, and correct "
    "solution. Output ONLY one ```python code block, no explanation.\n\n"
)
