#!/usr/bin/env python3
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""
Update all JSON result files and regenerate audited Markdown reports
with the updated Vertex AI pricing table:
  - gemini-3.6-flash: $1.50 input / $7.50 output per 1M tokens
  - gemini-3.5-flash-lite: $0.30 input / $2.50 output per 1M tokens
"""

import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import PRICING

RESULTS_DIR = os.path.join(ROOT, "bigCodeBench-hard", "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def main():
    print(f"Updating all benchmark results pricing in {RESULTS_DIR} and {REPORTS_DIR}...")
    import subprocess
    n30_script = os.path.join(HERE, "generate_n30_report.py")
    n50_script = os.path.join(HERE, "generate_n50_report.py")
    if os.path.exists(n30_script):
        subprocess.run([sys.executable, n30_script], check=True)
    if os.path.exists(n50_script):
        subprocess.run([sys.executable, n50_script], check=True)
    print("All reports pricing regenerated successfully!")

if __name__ == "__main__":
    main()
