# Straitjacket Benchmark Contribution Guide

> **Standard Operating Protocol & Architecture Guide for Contributing to the Multi-LLM Tokenomics Benchmark Suite**

---

## 1. Overview & Core Philosophy

This repository evaluates **Multi-LLM collaboration architectures, cascading strategies, and context containment harnesses** across real-world software engineering benchmarks ([BigCodeBench-Hard](bigCodeBench-hard), [SWE-bench Pro](swebench_pro), and [WebDev](webdev)).

The benchmark follows the **Straitjacket Benchmark Charter** and **Tokenomics Initiative** principles:

1. **Receipts Before Doctrine**: Performance and cost claims must be backed by reproducible empirical receipts (JSON run logs + Markdown/HTML reports), not adjectives.
2. **Total Cost of Ownership (TCO) & Cost per Solved Task ($/solved)**: Raw accuracy is meaningless without cost accounting. The primary metric is:
   $$\text{Cost per Solved Task} = \frac{\sum \text{as\_run\_usd}}{\text{Total Solved Tasks}}$$
3. **Zero-Cost Triage Elimination**: In multi-turn repair loops, raw stack traces or LLM-based triage (`triage_error`) inflate token cost and latency. `straitjacket`'s deterministic `UnittestProfile` compresses test failures into bounded ~60–80 token digests locally at **$0.0000 cost and 0ms API latency**.
4. **Prompt Cache Prefix Warming**: Deterministic stripping of ephemeral paths, timestamps, and ANSI escape codes keeps prompt prefixes identical across attempts, maintaining high prompt cache hit rates.
5. **External Corpora as Teachers, Never Referees**: Benchmark tasks provide real hostile outputs and failure structures to evaluate architectural resilience.

---

## 2. Repository Architecture & Layout

The repository is organized into a shared core library (`src/`), centralized reports (`reports/`), analysis utilities (`tools/`), and dataset-specific benchmarks:

```
benchmark-using-multi-LLMs/
├── README.md                                    # Repository overview and quick start guide
├── MODELS.md                                    # Supported models, IDs, and pricing table
├── requirements.txt                             # Pinned Python package dependencies
├── straitjacket_benchmark_contribution_guide.md   # This guide
│
├── run_benchmark.py                             # 🚀 MASTER UNIFIED CLI RUNNER (End-to-End)
│
├── src/                                         # Shared Core Benchmark Library
│   ├── __init__.py
│   ├── config.py                                # Centralized model IDs, pricing, and prompt roles
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
│   └── bench_runner.py                          # Dataset runner adapter
│
├── swebench_pro/                                # Dataset 2: SWE-bench Pro (Enterprise repo patch resolution)
│   ├── data/                                    # Downloaded/cached SWE-bench Pro dataset (.jsonl)
│   ├── results/                                 # Raw JSON metrics & run caches
│   ├── bench_runner.py                          # SWE-bench Pro runner adapter
│   └── run_swebench_pro_sweetspot.py            # Master SWE-bench Pro runner adapter
│
└── webdev/                                      # Dataset 3: WebDev (Web & networking tasks)
    ├── data/                                    # Local WebDev dataset (.jsonl)
    ├── results/                                 # Raw JSON metrics & run caches
    └── bench_runner.py                          # WebDev runner adapter
```

---

## 3. Unified CLI Runner (`run_benchmark.py`)

The primary entry point is `run_benchmark.py`, which allows running any dataset, filtering variants, and generating reports in one shot.

### Basic Usage

```bash
# 1. Run SWE-bench Pro on 30 tasks with all Straitjacket variants and auto-generate reports
python3 run_benchmark.py --dataset swebench --group straitjacket --n 30 --report

# 2. Run BigCodeBench-Hard on 10 tasks with specific variants
python3 run_benchmark.py --dataset bcb --variants cascade_sj,hybrid_sj,smart_repair_sj --n 10 --report

# 3. Run WebDev evaluation across single models
python3 run_benchmark.py --dataset webdev --group single --n 10 --report

# 4. Run all variants on SWE-bench Pro with fresh execution (no cache)
python3 run_benchmark.py --dataset swebench --group all --n 30 --no-cache --report
```

### Supported CLI Arguments

| Argument | Shorthand | Description | Default |
|---|---|---|---|
| `--dataset` | `-d` | Target dataset: `bcb` (BigCodeBench-Hard), `swebench` (SWE-bench Pro), `webdev` | `swebench` |
| `--group` | `-g` | Preset variant group: `all`, `single`, `combo`, `straitjacket`, `nextgen` | `all` |
| `--variants` | `-v` | Comma-separated list of specific variant IDs or architecture keys | `None` (uses group) |
| `--n` | `-n` | Number of tasks to evaluate | `30` (or dataset length) |
| `--split` | `-s` | Dataset split: `test`, `v0.1.4`, `default` | Dataset default |
| `--no-cache` | | Force fresh API execution, ignoring existing task cache | `False` |
| `--out` | `-o` | Custom output path for JSON result metrics | `<dataset>/results/<group>_results.json` |
| `--report` | `-r` | Automatically generate Markdown TCO report and HTML dashboard in `reports/` | `True` |

---

## 4. Step-by-Step Contribution Workflows

### A. Adding a New Model

1. **Register the Model ID in `src/config.py`**:
   ```python
   NEW_MODEL_ID = "gemini-3.7-flash"
   ```
2. **Add Pricing Rates to `PRICING` table in `src/config.py`** (USD per 1M tokens):
   ```python
   PRICING[NEW_MODEL_ID] = {
       "input": 1.50,
       "output": 7.50,
       "cache_read": 0.15,
       "cache_write": 1.50,
   }
   ```
3. **Verify API Routing in `src/client.py`**:
   - Ensure the model prefix (`gemini-*` vs `claude-*`) dispatches to the correct SDK or endpoint.
4. **Update `MODELS.md`** with the new model's specifications.

---

### B. Adding a New Benchmark Architecture / Variant

1. **Implement the Pipeline in `src/architectures.py`**:
   - Structure the function to accept a `problem` dict.
   - Use `dispatch_model(model_id, prompt, thinking_level=..., problem=...)` for LLM calls.
   - Use `triage_error_straitjacket(err, problem=...)` for zero-cost test failure triage.
   - Return standard metrics dict:
   ```python
   def run_my_custom_pipeline(problem, planner_model=GEMINI_36_FLASH_ID, executor_model=GEMINI_35_FLASH_LITE_ID):
       """Custom multi-model collaboration architecture."""
       # 1. Planning phase
       # 2. Execution phase
       # 3. Verification & Straitjacket repair phase
       return {
           "passed": passed,
           "as_run_usd": round(tot_usd, 6),
           "output_tokens": tot_out,
           "total_tokens": tot_tok,
           "seconds": round(elapsed, 1),
           "error": "" if passed else err,
           "repair_loops": loop_count,
           "triage_usd": 0.0,
           "patch": candidate_code_or_patch,
       }
   ```
2. **Register the Variant in `VARIANT_REGISTRY` in `src/architectures.py`**:
   ```python
   register_variant(
       id="custom_pipeline_sj",
       name="Custom Pipeline (Flash Plan -> Lite Exec + SJ Triage)",
       category="3. Combination + straitjacket",
       dataset_compatibility=["bcb", "swebench", "webdev"],
       fn=run_my_custom_pipeline,
   )
   ```

---

### C. Adding a New Dataset

1. **Place Raw Data in `<dataset_name>/data/`**:
   - Use standard JSON Lines (`.jsonl`) format.
2. **Implement Dataset Loader in `src/datasets.py`**:
   - Provide a clean mapping to standard task fields (`task_id`, `prompt`, `test`, `entry_point` or `repo`, `problem_statement`, `canonical_patch`).
3. **Register Evaluation Runner in `src/evaluator.py`**.

---

## 5. Result Schema & Reporting Contract

### JSON Result Metric Schema (`<dataset>/results/*.json`)

Every benchmark run produces a JSON record matching this structure:

```json
{
  "dataset": "swebench",
  "group": "straitjacket",
  "n": 30,
  "passed": 23,
  "pass_rate": 0.767,
  "total_as_run_usd": 0.0892,
  "total_triage_usd": 0.0000,
  "cost_per_solved_usd": 0.00388,
  "avg_output_tokens": 420.5,
  "seconds": 182.4,
  "results": [
    {
      "task_id": "sympy__sympy-13480",
      "passed": true,
      "as_run_usd": 0.00284,
      "output_tokens": 380,
      "total_tokens": 2840,
      "repair_loops": 1,
      "error": ""
    }
  ]
}
```

### Auto-Generated Reports in `reports/`

- **Markdown Report (`reports/<dataset>_straitjacket_report.md`)**: Formatted comparative TCO table, $/solved rankings, error breakdown (distinguishing environment/quota errors from algorithmic bugs).
- **Interactive HTML Dashboard (`reports/<dataset>_dashboard.html`)**: Mobile-responsive dashboard with KPI scorecards, pass rate bar charts, and detailed architecture specifications.

---

## 6. Pre-Commit Verification Checklist

Before submitting a Pull Request or committing changes:

- [ ] **No Duplicate Logic**: Client calls, pricing definitions, and triage harnesses reside strictly in `src/`.
- [ ] **Deterministic Fallback**: Running without active GCP credentials falls back gracefully to deterministic simulation without raising unhandled exceptions.
- [ ] **Zero-Cost Triage Integrity**: All `*_straitjacket` variants must declare `$0.0000` triage cost and 0 additional triage tokens.
- [ ] **No Hardcoded Absolute Paths**: All file lookups must use relative paths or `os.path.dirname(os.path.abspath(__file__))`.
- [ ] **Requirements Up to Date**: Any newly introduced package is pinned in `requirements.txt`.
- [ ] **Clean Git Workspace**: No stray `.DS_Store`, `__pycache__`, or scratch scripts left untracked.
