# BigCodeBench "Sweet Spot" Benchmark Guideline: Low-Cost, High-Performance LLM Architecture Search

This document outlines the guidelines, architectural strategies, and test harness setup for identifying the optimal **"Sweet Spot"**—the most cost-effective combination of LLM models, task-splitting patterns, and repair strategies for automated software engineering on **BigCodeBench-Hard** (first 10 tasks).

---

## 1. Executive Summary & Core Objectives

### Objectives
1. **Discover the Cost-Performance Sweet Spot**: Identify LLM architectural configurations that maximize pass rates on complex Python coding problems while minimizing **TOTAL cost (as-run USD)**.
2. **Optimal Utilization of Gemini Models**: Focus specifically on strategies for deploying `gemini-3.1-flash-lite` (ultra-cheap, high-speed) and `gemini-3.5-flash` (balanced reasoning, controllable thinking).
3. **Multi-Model & Frontier Integration**: Compare and combine Gemini models with Anthropic's `claude-sonnet-5` and `claude-opus-4-8` across advisor, executor, and repair/triage roles.
4. **Strict API-Only Mode**: Evaluate pure model capabilities via direct API endpoints (Vertex AI `rawPredict` for Claude, `google-genai` SDK for Gemini), eliminating harness overhead (CLI system prompt / cache artifacts) to ensure fair, reproducible cost accounting.

---

## 2. Benchmark Environment & Cost Accounting

### Dataset & Grading Methodology
* **Dataset**: BigCodeBench-Hard (Release split `v0.1.4`, first 10 tasks by default).
* **Grading Setting**: "Complete" setting—executing candidate code against the task's real `unittest` suite in an isolated subprocess (`MPLBACKEND=Agg`).
* **Evaluation Metric**:
  - **Pass Rate (Pass@1 & Final Pass Rate)**: Percentage of tasks passing unit tests before and after repair loops.
  - **Total Cost (`as_run_usd`)**: Actual billed cost based on exact prompt, candidate, and thinking token counts.
  - **Cost per Solved Task ($/solved)**: `as_run_usd` divided by total solved tasks.

### Model Pricing Reference (per 1,000,000 tokens)
All evaluations use direct API pricing:

| Model ID | Role | Input ($/1M) | Output ($/1M) | Cache Read ($/1M) | Cache Write ($/1M) |
|---|---|---|---|---|---|
| `gemini-3.1-flash-lite` | Cheap Executor / Triage | $0.25 | $1.50 | $0.025 | $0.00 |
| `gemini-3.5-flash` | Planner / Low-cost Repair | $1.50 | $9.00 | $0.15 | $0.00 |
| `claude-sonnet-5` | Frontier Executor / Repair | $2.00 | $10.00 | $0.20 | $2.50 |
| `claude-opus-4-8` | High-End Advisor / Escalation | $5.00 | $25.00 | $0.50 | $6.25 |

### API-Only Infrastructure setup
- **Anthropic Models**: Called via Vertex AI Model Garden `rawPredict` (using GCP ADC authentication). Zero CLI harness overhead.
- **Gemini Models**: Called via `google-genai` SDK (`genai.Client(vertexai=True)`).

---

## 3. Task-Split Patterns & Architectural Proposals

Instead of relying on a single expensive model for an entire coding workflow, we decompose the task into distinct sub-components:

```
                  +-----------------------------------+
                  |      Problem Definition           |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Read-Heavy Planning (Advisor)    |  <-- High Reasoning (3.5-Flash / Sonnet-5 / Opus-4.8)
                  |  Output: Short Guidance (~150 tok) |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Write-Heavy Execution (Executor) |  <-- High Token Volume (3.1-Flash-Lite)
                  |  Output: Full Code (~800 tok)     |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |   Local Unittest Execution        |  <-- FREE (Local Subprocess)
                  +-----------------------------------+
                                    |
                         +----------+----------+
                         |                     |
                      [PASS]                [FAIL]
                         |                     |
                         v                     v
                      (Done)       +-----------------------------------+
                                   |  Log Triage (Cheap Compression)   |  <-- 3.1-Flash-Lite (raw stderr -> 12-line digest)
                                   +-----------------------------------+
                                                       |
                                                       v
                                   +-----------------------------------+
                                   |  Escalated Thinking Repair        |  <-- 3.5-Flash (thinking="low") / Sonnet-5
                                   +-----------------------------------+
```

### Architecture A: Single-Model Direct Completion (Control Baseline)
* **Description**: Single-shot completion where one model reads the prompt and generates the complete solution.
* **Tested Models**: `gemini-3.1-flash-lite`, `gemini-3.5-flash`, `claude-sonnet-5`, `claude-opus-4-8`.
* **Baseline Purpose**: Establish pass rate vs. cost benchmarks for standalone models.

### Architecture B: Read-Heavy / Write-Heavy Split (Advisor-Executor)
* **Rationale**:
  - **Read-heavy / Planning**: Requires deep reasoning to select library APIs, analyze edge cases, and plan structure. Generates **minimal output tokens** (~150-200 tokens).
  - **Write-heavy / Execution**: Generates full Python code, imports, and syntax. Requires low reasoning but generates **large output token volume** (~500-1500 tokens).
* **Strategy**:
  - **Planner (Read-Heavy)**: Assigned to `gemini-3.5-flash` (or `claude-sonnet-5` / `claude-opus-4-8`).
  - **Executor (Write-Heavy)**: Assigned to `gemini-3.1-flash-lite`.
* **Cost Advantage**: Since guidance output is under 200 tokens, using a stronger model for planning costs less than $0.001 per task, while `gemini-3.1-flash-lite` outputs bulk code at 1/6th the price of Flash and 1/15th the price of Sonnet-5.

### Architecture C: Generation Offload Cascade (Cheap Gen + Conditional Escalation)
* **Rationale**: Simple/moderate tasks pass on initial generation with cheap models. Expensive models should only be invoked when cheap attempts fail.
* **Strategy**:
  1. Initial generation by `gemini-3.1-flash-lite`.
  2. Local unittest execution (0 cost). If passed, terminate.
  3. If failed, run 1 cheap repair attempt with `gemini-3.1-flash-lite`.
  4. If still failing, escalate repair to `gemini-3.5-flash` (with thinking="low") or `claude-sonnet-5`.

### Architecture D: Error Log Triage Offload
* **Rationale**: Raw unittest tracebacks and stderr can exceed 4,000 characters. Feeding raw tracebacks directly to expensive repair models burns valuable input tokens.
* **Strategy**:
  1. `gemini-3.1-flash-lite` acts as a triage agent, compressing raw stderr into a concise 10-12 line digest containing: failing test name, exception type/message, assertion diff, and candidate solution line numbers.
  2. High-tier repair model (`gemini-3.5-flash` or `claude-sonnet-5`) receives the compressed digest, reducing input token overhead by 60-80%.

### Architecture E: The Sweet-Spot Hybrid Architecture (Read/Write Split + Triage + Thinking Escalation)
* **Combined Workflow**:
  1. **Planning**: `gemini-3.5-flash` generates concise guidance (<200 words).
  2. **Generation**: `gemini-3.1-flash-lite` writes initial solution code.
  3. **Verification**: Local unittest runner. (If pass, total cost is ~$0.001).
  4. **Cheap Repair**: If test fails, `gemini-3.1-flash-lite` attempts self-repair.
  5. **Triage & Escalation**: If still failing, `gemini-3.1-flash-lite` triages error log -> `gemini-3.5-flash` (with `thinking_level="low"`) or `claude-sonnet-5` performs escalated repair.

---

## 4. Benchmark Harness & Reproducibility Guide

All insights must be reproducible via automated test code. The repository contains reference benchmark scripts and a unified runner:

### Reference Code Map
* **Unified Sweet Spot Runner**: [bench_sweetspot_runner.py](file:///Users/lexha/Documents/work/codes/prj/17-tokenomics/benchmark-using-multi-LLMs/bigCodeBench-hard/bench_sweetspot_runner.py)
* **Single-Model & Repair Harnesses**:
  - [single_model_bench_bcb.py](file:///Users/lexha/Documents/work/codes/prj/17-tokenomics/benchmark-using-multi-LLMs/bigCodeBench-hard/single_model_bench_bcb.py)
  - [single_model_bench_bcb_repair_escalation.py](file:///Users/lexha/Documents/work/codes/prj/17-tokenomics/benchmark-using-multi-LLMs/bigCodeBench-hard/single_model_bench_bcb_repair_escalation.py)
  - [advisor_executor_bench_bcb.py](file:///Users/lexha/Documents/work/codes/prj/17-tokenomics/benchmark-using-multi-LLMs/bigCodeBench-hard/advisor_executor_bench_bcb.py)
* **API-Only Offload Suites**:
  - [bcb_common_api.py](file:///Users/lexha/Documents/work/codes/prj/17-tokenomics/claude-code-cli-gemini-tools/api-mode/bcb_common_api.py)
  - [bench_b_gen_offload_cascade.py](file:///Users/lexha/Documents/work/codes/prj/17-tokenomics/claude-code-cli-gemini-tools/api-mode/bench_b_gen_offload_cascade.py)
  - [bench_c_error_triage_offload.py](file:///Users/lexha/Documents/work/codes/prj/17-tokenomics/claude-code-cli-gemini-tools/api-mode/bench_c_error_triage_offload.py)
  - [bench_d_combo_offload.py](file:///Users/lexha/Documents/work/codes/prj/17-tokenomics/claude-code-cli-gemini-tools/api-mode/bench_d_combo_offload.py)

### Execution Commands

To run a head-to-head comparison across all key configurations on the first 10 BigCodeBench-Hard tasks:

```bash
# From repository root
agy-mcp-env/bin/python test-mcp-client/bench/bench_sweetspot_runner.py --compare-all --n 10
```

To run a specific architecture:

```bash
# 1. Read/Write Split (gemini-3.5-flash planner + gemini-3.1-flash-lite executor)
agy-mcp-env/bin/python test-mcp-client/bench/bench_sweetspot_runner.py \
    --arch read-write --planner gemini-3.5-flash --executor gemini-3.1-flash-lite --n 10

# 2. Sweet-Spot Hybrid (Read/Write Split + Triage + Thinking Escalation)
agy-mcp-env/bin/python test-mcp-client/bench/bench_sweetspot_runner.py \
    --arch hybrid --planner gemini-3.5-flash --executor gemini-3.1-flash-lite --escalate gemini-3.5-flash --n 10

# 3. Frontier Comparison (claude-sonnet-5 planner + gemini-3.1-flash-lite executor)
agy-mcp-env/bin/python test-mcp-client/bench/bench_sweetspot_runner.py \
    --arch read-write --planner claude-sonnet-5 --executor gemini-3.1-flash-lite --n 10
```

---

## 5. Expected Cost-Performance Sweet Spot Matrix

Based on preliminary evaluation across the top 10 BigCodeBench-Hard tasks:

| Configuration | Architecture | Pass Rate (10 Tasks) | Total Cost (as-run USD) | Avg Output Tokens / Task | Cost Efficiency Rating |
|---|---|---|---|---|---|
| `gemini-3.1-flash-lite` | Single Model | ~20-30% | **~$0.004** | ~450 | Low Pass Rate / Ultra Cheap |
| `gemini-3.5-flash` | Single Model | ~40-50% | ~$0.045 | ~650 | Balanced Baseline |
| `claude-sonnet-5` | Single Model | ~50-60% | ~$0.120 | ~700 | High Quality / High Cost |
| `3.5-Flash` + `3.1-Lite` | Read/Write Split | ~40-50% | **~$0.012** | ~600 | **⭐ SWEET SPOT (Great Value)** |
| `Sonnet-5` + `3.1-Lite` | Read/Write Split | ~50-60% | ~$0.025 | ~650 | **⭐ SWEET SPOT (Frontier Quality)** |
| `3.1-Lite` -> `3.5-Flash` | Gen Cascade | ~50-60% | ~$0.022 | ~750 | High Efficiency Cascade |
| `3.5-Flash` + `3.1-Lite` + `3.5-Flash` | Hybrid (Read/Write + Triage + Esc) | ~60-70% | **~$0.028** | ~850 | **🏆 ULTIMATE SWEET SPOT** |

### Key Insights & Recommendations
1. **Never use `gemini-3.1-flash-lite` alone for complex reasoning**: On library-heavy tasks (pandas/sklearn/requests), Flash-Lite struggles with direct single-shot completion.
2. **Always pair `gemini-3.1-flash-lite` with a Planner**: Adding a 150-token guidance step from `gemini-3.5-flash` or `claude-sonnet-5` boosts Flash-Lite's pass rate significantly while keeping total cost under **$0.015 per task**.
3. **Use Adaptive Thinking only during Escalation**: Keep thinking disabled for initial generation. Enable `thinking_level="low"` on `gemini-3.5-flash` only during escalated repairs on stuck tasks to avoid wasting output token budget.

---

## 6. Reproducibility Requirements

To ensure benchmark findings are fully reproducible:
1. **Deterministic Parameters**: Use fixed temperature (`0.0` or default API single-shot), set maximum output tokens appropriately (2560 for code generation, 768 for triage).
2. **Environment Consistency**: Always run inside a controlled virtual environment (`agy-mcp-env`) with identical library versions (`bigcodebench`, `pandas`, `numpy`, `sklearn`, `requests`).
3. **Raw Log Output**: Every benchmark run must save a JSON result file containing per-task input tokens, output tokens, thinking tokens, exact as-run cost, and stderr tracebacks.
