# Multi-LLM Benchmark Suite for Tokenomics & Straitjacket

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Contribution Guide](https://img.shields.io/badge/Guide-Contribution%20Standards-emerald.svg)](straitjacket_benchmark_contribution_guide.md)
[![Comprehensive Report](https://img.shields.io/badge/Report-Comprehensive%20TCO%20Synthesis-purple.svg)](reports/comprehensive_multi_llm_benchmark_report_20260806.md)

This repository evaluates **Multi-LLM collaboration architectures, cascading strategies, and context containment harnesses** across realistic software engineering benchmarks. It provides empirical evidence on balancing benchmark performance (code correctness / task resolution) with API cost using Google Cloud Vertex AI (Gemini) and Anthropic (Claude) models.

---

## 📁 Repository Structure

```
.
├── README.md                                    # Repository overview and quick start guide
├── MODELS.md                                    # Model IDs, pricing rates, and architecture specifications
├── requirements.txt                             # Pinned Python package dependencies
├── straitjacket_benchmark_contribution_guide.md   # Benchmark charter, standards, and PR contribution guide
│
├── run_benchmark.py                             # 🚀 MASTER UNIFIED CLI RUNNER (End-to-End)
│
├── src/                                         # Shared Core Benchmark Library
│   ├── __init__.py
│   ├── config.py                                # Centralized model IDs, pricing table, and prompt roles
│   ├── client.py                                # Vertex AI Gemini & Claude client with retry & fallback
│   ├── evaluator.py                             # Python unittest & git patch evaluators + SJ triage
│   ├── datasets.py                              # Unified dataset loaders (BCB, SWE-bench Pro, WebDev)
│   ├── architectures.py                         # Modular multi-LLM architecture pipelines & registry
│   └── reporter.py                              # Markdown TCO report & interactive HTML dashboard generator
│
├── reports/                                     # 📊 ALL GENERATED REPORTS & DASHBOARDS (.md, .html)
│   ├── comprehensive_multi_llm_benchmark_report_20260806.md  # 🌟 Master Cross-Dataset Synthesis Report
│   ├── straitjacket_n30_comparative_tco_report.md            # BigCodeBench-Hard N=30 Audited Report
│   ├── n50_gemini_vs_claude_tco_report.md                    # BigCodeBench-Hard N=50 Comprehensive Report
│   ├── swe_bench_pro_straitjacket_report.md                  # SWE-bench Pro Comparative Report
│   ├── swe_bench_pro_dashboard.html                          # SWE-bench Pro Interactive HTML Dashboard
│   ├── bigcodebench_hard_dashboard.html                      # BigCodeBench Interactive HTML Dashboard
│   └── webdev_dashboard.html                                 # WebDev Interactive HTML Dashboard
│
├── tools/                                       # 🛠️ Post-Processing, Auditing & Pricing Scripts
│   ├── generate_n30_report.py                   # Audits N=30 BCB raw vs effective pass rates
│   ├── generate_n50_report.py                   # Audits N=50 BCB Gemini vs Claude comparison
│   └── update_all_reports_pricing.py            # Recalculates metrics with active Vertex AI pricing
│
├── bigCodeBench-hard/                           # Dataset 1: BigCodeBench-Hard (Python function completion)
│   ├── data/                                    # Downloaded/cached HF dataset (.jsonl)
│   ├── results/                                 # Raw JSON metrics & run caches
│   └── bench_runner.py                          # BCB runner adapter
│
├── swebench_pro/                                # Dataset 2: SWE-bench Pro (Enterprise git patch resolution)
│   ├── data/                                    # Cached SWE-bench Pro public tasks (.jsonl)
│   ├── results/                                 # Raw JSON metrics & run caches
│   ├── bench_runner.py                          # SWE-bench Pro runner adapter
│   └── run_swebench_pro_sweetspot.py            # Master SWE-bench Pro evaluation script
│
└── webdev/                                      # Dataset 3: Web-Dev (Web & networking tasks)
    ├── data/                                    # Local WebDev dataset (.jsonl)
    ├── results/                                 # Raw JSON metrics & run caches
    └── bench_runner.py                          # WebDev runner adapter
```

---

## ⚡ Quick Start

### 1. Setup Virtual Environment

```bash
# Create and activate virtual environment
python3 -m venv tokenomics-bench-env
source tokenomics-bench-env/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Google Cloud Credentials

```bash
gcloud auth application-default login

export GCP_PROJECT="your-gcp-project-id"
export GCP_LOCATION="global" # or us-central1
```

---

## 🚀 Running Benchmarks (Unified CLI)

The master runner [`run_benchmark.py`](run_benchmark.py) executes benchmarks and automatically places JSON metrics in `<dataset>/results/` and Markdown & HTML reports in `reports/`:

```bash
# 1. Evaluate SWE-bench Pro (30 tasks, all Straitjacket zero-cost triage variants)
python3 run_benchmark.py --dataset swebench --group straitjacket --n 30 --report

# 2. Evaluate BigCodeBench-Hard (10 tasks, specific variants)
python3 run_benchmark.py --dataset bcb --variants single_flash36,sj_hybrid,sj_smart_repair --n 10 --report

# 3. Evaluate WebDev benchmark (single-model baselines)
python3 run_benchmark.py --dataset webdev --group single --n 10 --report

# 4. Compare all variants on SWE-bench Pro without cache (fresh API execution)
python3 run_benchmark.py --dataset swebench --group all --n 30 --no-cache --report
```

---

## 📊 Benchmark Datasets & Findings

1. **BigCodeBench-Hard**:
   - Complex multi-library algorithmic and data engineering tasks.
   - **Sweet-Spot Champion**: `Straitjacket Smart Repair` (`gemini-3.6-flash` -> `3.5-flash-lite` -> `3.6-flash`) achieved **81.5% effective pass rate at $0.0076 / solved task** (35x cheaper than Claude Opus-5).
2. **SWE-bench Pro**:
   - Long-horizon enterprise repository git patch generation.
   - **Sweet-Spot Champion**: `Straitjacket Ultra-Sweet Hybrid` and `Straitjacket Escalation Shield` achieved **76.7%–80.0% pass rate at $0.00388 / solved task** (7.4x cheaper than Claude Opus-5).
3. **WebDev**:
   - Real-world web framework, REST API, parsing, and networking tasks.
   - **Sweet-Spot Champion**: `Straitjacket Hybrid` achieved **80.0% pass rate at $0.0041 / solved task** (87% cheaper than Claude Opus-5).

👉 Read the full analysis in [**Comprehensive Multi-LLM Benchmark & Tokenomics Synthesis Report (`reports/comprehensive_multi_llm_benchmark_report_20260806.md`)**](reports/comprehensive_multi_llm_benchmark_report_20260806.md).

---

## 📖 Contribution Standards

👉 [**Straitjacket Benchmark Contribution Guide (`straitjacket_benchmark_contribution_guide.md`)**](straitjacket_benchmark_contribution_guide.md)
