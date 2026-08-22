# Straitjacket Benchmark Contribution Guide

> **Standard Operating Protocol & Architecture Guide for Contributing to the Multi-LLM Tokenomics Benchmark Suite**

---

## 1. Overview & Core Philosophy

This repository evaluates **Multi-LLM collaboration architectures, cascading strategies, and context containment harnesses** across real-world software engineering benchmarks ([BigCodeBench-Hard](bigCodeBench-hard), [SWE-bench Pro](swebench_pro), and [WebDev](webdev)).

The benchmark follows the **Straitjacket Benchmark Charter** and **Tokenomics Initiative** principles:

1. **Receipts Before Doctrine**: Performance and cost claims must be backed by reproducible empirical receipts (JSON run logs + Markdown/HTML reports), not adjectives.
2. **Total Cost of Ownership (TCO) & Cost per Solved Task ($/solved)**: Raw accuracy is meaningless without cost accounting. The primary metric is:
   $$\text{Cost per Solved Task} = \frac{\sum \text{as\_run\_usd}}{\text{Total Solved Tasks}}$$
3. **Context Containment**: In multi-turn repair loops, raw stack traces inflate every later turn's prompt, and LLM-based triage (`triage_error`) buys brevity with a paid round trip. `straitjacket` captures the run at its birth gate and hands the repair turn a bounded, coverage-attested digest from the upstream profile registry (`unittest/v1`, `pytest/v2`, …) at **$0.0000 and no API call** — with `ctx get` / `ctx search` addresses for everything it omitted, so shorter does not mean lost.
4. **Prompt Cache Prefix Warming**: Deterministic stripping of ephemeral paths, timestamps, and ANSI escape codes keeps prompt prefixes identical across attempts, maintaining high prompt cache hit rates.
5. **External Corpora as Teachers, Never Referees**: Benchmark tasks provide real hostile outputs and failure structures to evaluate architectural resilience.

---

## 2. Repository Architecture & Layout

The repository separates **explanations** (`docs/`) from **results** (`reports/`),
with a shared core library (`src/`), analysis utilities (`tools/`), and
dataset-specific benchmark directories:

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
│   ├── straitjacket.py                          # Bridge to the real ctx-harness (capture, digest, retrieval)
│   ├── evaluator.py                             # Python unittest & git patch evaluators + contained evidence
│   ├── datasets.py                              # Unified dataset loaders (BCB, SWE-bench Pro, WebDev)
│   ├── architectures.py                         # Modular multi-LLM architecture pipelines & registry
│   └── reporter.py                              # Markdown TCO report & interactive HTML dashboard generator
│
├── docs/                                        # 📚 Methodology & design docs (not results)
│   ├── straitjacket-implementation.md           # The harness bridge, and how it differs from upstream
│   ├── pipeline-architecture.md                 # How one task flows through the system
│   └── *-sweetspot-methodology.md               # Per-dataset experiment design
│
├── reports/                                     # 📊 RUN RESULTS ONLY, indexed by execution order
│   ├── README.md                                # The index — start here
│   └── NN_<dataset>_<tag>_n<N>.{md,html}        # Append-only; a sweep never overwrites an earlier one
│
├── tests/                                       # Contract tests for the straitjacket bridge
│
├── tools/                                       # 🛠️ Post-Processing, Auditing & Pricing Scripts
│   ├── index_reports.py                         # Assigns report indices & regenerates reports/README.md
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
   - Name the evidence treatment rather than hard-coding it: take an
     `error_treatment` parameter and call
     `_treat_error(err, error_treatment, problem=problem, is_swe=is_swe)`.
     It dispatches to `native` (raw tail), `llm` (paid triage model), or
     `straitjacket` (the harness's own digest). The registry's `triage_mode`
     label **must** name the treatment the arm actually applies.
   - Decorate the arm with `@_arm()`, or `@_arm(sj_required=True)` if it claims
     containment — the decorator resets the per-task containment ledger and,
     for straitjacket arms, refuses to start when `ctx-harness` is missing.
   - Never re-summarise, keyword-filter, or tail-truncate a failure yourself.
     `run_bigcodebench` already returns `Evidence` carrying the real digest and
     an addressable handle; producing a second "digest" on top of it is the
     anti-pattern this suite exists to measure against.
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
           "triage_usd": round(triage_usd, 6),   # what the treatment actually spent
           "patch": candidate_code_or_patch,
       }
       # `@_arm()` appends "containment": the harness's own accounting of raw
       # tokens captured, digest tokens rendered, and evidence tokens sent.
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

Reports are an append-only log indexed by execution order, so a new sweep never
overwrites an earlier one's evidence. After a run, adopt the new files and
refresh the index with `python3 tools/index_reports.py --apply`.


- **Markdown Report (`reports/NN_<dataset>_<tag>_n<N>.md`)**: Formatted comparative TCO table, $/solved rankings, error breakdown (distinguishing environment/quota errors from algorithmic bugs).
- **Interactive HTML Dashboard (`reports/NN_<dataset>_<tag>_n<N>.html`)**: Mobile-responsive dashboard with KPI scorecards, pass rate bar charts, and detailed architecture specifications.

---

## 6. Pre-Commit Verification Checklist

Before submitting a Pull Request or committing changes:

- [ ] **No Duplicate Logic**: Client calls, pricing definitions, and triage harnesses reside strictly in `src/`.
- [ ] **Deterministic Fallback**: Running without active GCP credentials falls back gracefully to deterministic simulation without raising unhandled exceptions.
- [ ] **Containment Integrity**: Every digest in a `*_straitjacket` variant comes from the upstream harness via `src/straitjacket.py`. No local re-implementation, no keyword or head/tail selection, no arm that silently degrades when `ctx-harness` is absent.
- [ ] **Truth in Labelling**: A variant's `triage_mode` names the treatment it actually applies (`native` / `llm` / `straitjacket`), and its reported `triage_usd` is what it actually spent.
- [ ] **Fair Baseline**: The uncontained arm gets the failing stream, truncated once by `SJ_RAW_CAP` and nowhere else. Never re-truncate inside an arm, and never hand the native path stdout chatter it would not otherwise have forwarded — degrading the baseline biases every comparison toward straitjacket.
- [ ] **No Re-flooding the Harness**: Nothing on the per-task path may materialise a whole captured stream. Use `ContainedRun.raw_tail(stream, nbytes)`; `raw_stdout` / `raw_stderr` are unbounded and exist for tests and debugging only.
- [ ] **Reproducible Evidence**: Identical failing code produces an identical run handle across processes. If you add a runner or an evaluator, pin whatever it prints that is not evidence (elapsed times, PIDs, temp paths, hash-ordered output) — otherwise every attempt mints a fresh artifact for the same failure.
- [ ] **Self-Consistent Receipt**: An arm whose treatment *is* the baseline must report `delta_vs_native_tokens == 0`. The A/B is `native_baseline − sent`, counted over the same events; `captured − sent` is a larger number and is not the A/B.
- [ ] **Contract Tests Pass**: `pytest tests/test_straitjacket_integration.py -q` is green on the `library` backend.
- [ ] **No Hardcoded Absolute Paths**: All file lookups must use relative paths or `os.path.dirname(os.path.abspath(__file__))`.
- [ ] **Requirements Up to Date**: Any newly introduced package is pinned in `requirements.txt`.
- [ ] **Clean Git Workspace**: No stray `.DS_Store`, `__pycache__`, or scratch scripts left untracked.
