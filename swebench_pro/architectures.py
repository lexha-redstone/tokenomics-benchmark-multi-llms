# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
SWE-bench Pro Architectures (Re-exports from unified src.architectures).
"""

from src.architectures import (
    run_single,
    run_read_write,
    run_cascade,
    run_hybrid,
    run_hybrid_straitjacket,
    run_cascade_straitjacket,
    run_escalation_shield_straitjacket,
    run_smart_repair_straitjacket,
    run_ultra_sweet_straitjacket,
    run_dual_verifier_cascade_straitjacket,
    VARIANT_REGISTRY,
    get_configurations
)
