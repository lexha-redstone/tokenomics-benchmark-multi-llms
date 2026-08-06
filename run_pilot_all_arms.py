#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Run All 4 Arms on N=10 Pilot Subset for Straitjacket TCO Evaluation on BigCodeBench-Hard.
Delegates to the unified benchmark engine and generates comprehensive reports.
"""

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_benchmark.py")

def main():
    cmd = [
        sys.executable, RUNNER,
        "--dataset", "bcb",
        "--variants", "combo_cascade_llm,combo_hybrid_llm,sj_hybrid,sj_cascade",
        "--n", "10",
        "--report"
    ]
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
