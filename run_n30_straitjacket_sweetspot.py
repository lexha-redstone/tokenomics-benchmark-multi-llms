#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
N=30 Straitjacket Sweet-Spot Evaluation on BigCodeBench-Hard.
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
        "--group", "straitjacket",
        "--n", "30",
        "--report"
    ]
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
