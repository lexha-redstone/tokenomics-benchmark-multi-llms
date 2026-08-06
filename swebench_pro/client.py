# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
SWE-bench Pro Model Dispatch Client (Re-exports from unified src.client).
"""

from src.client import (
    dispatch_model, gemini_call, claude_api_call, _fallback_dispatch
)
