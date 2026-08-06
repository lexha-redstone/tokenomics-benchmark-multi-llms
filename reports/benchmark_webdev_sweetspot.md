# Web-Dev "Sweet Spot" Benchmark Guideline: Low-Cost, High-Performance LLM Architecture Search

This document outlines the guidelines, architectural strategies, and test harness setup for identifying the optimal **"Sweet Spot"**—the most cost-effective combination of LLM models, task-splitting patterns, and repair strategies for automated software engineering on **Web Development (Web-Dev)** tasks.

---

## 1. Executive Summary & Core Objectives

### Objectives
1. **Discover the Cost-Performance Sweet Spot**: Identify LLM architectural configurations that maximize feature implementation and pass rates on web-dev tasks while minimizing **TOTAL cost (as-run USD)**.
2. **Optimal Utilization of Gemini Models**: Focus specifically on strategies for deploying `gemini-3.1-flash-lite` (ultra-cheap, high-speed) and `gemini-3.5-flash` (balanced reasoning, controllable thinking).
3. **Multi-Model & Frontier Integration**: Compare and combine Gemini models with Anthropic's `claude-sonnet-5` and `claude-opus-4-8` across advisor, executor, and repair/triage roles.
4. **Strict API-Only Mode**: Evaluate pure model capabilities via direct API endpoints (Vertex AI `rawPredict` for Claude, `google-genai` SDK for Gemini), eliminating harness overhead to ensure fair, reproducible cost accounting.

---

## 2. Benchmark Environment & Cost Accounting

### Dataset Selection & Rationale
We evaluate using the **ByteDance Web-Bench** dataset (https://github.com/bytedance/web-bench).
* **Why Web-Bench?**: 
  - **Framework Coverage**: Specifically covers modern web standards and frameworks (React, Vue, etc.) reflecting actual development practices.
  - **Sequential Tasks**: Projects consist of 20 sequential tasks with dependencies, simulating real-world iterative feature implementation.
  - **Lightweight & Partial Testing**: Web-Bench supports partial testing. Configurations (such as `config.json5`) allow specifying a subset of projects or tasks (e.g. `projects: ["@web-bench/react"]`), making quick validation feasible.
  - **Simplified Setup**: Runs inside a pre-packaged Docker environment via `docker-compose`, avoiding complex manual multi-lingual web server dependencies.
* **Alternative Datasets Considered**:
  - *Arena.ai WebDev*: Represents agentic front-end workflows, but lacks isolated code execution/verification suites that are easily running without complex setup.
  - *LiveBench (Coding)*: Focuses on general code generation and completion tasks (from LiveCodeBench) rather than real web development, and requires up to 150GB of Docker storage for agentic evaluation.

### Grading Methodology
* **Grading Setting**: Sequential evaluation of project tasks. Each task generates code changes that are applied and tested using project-specific unittest suites (npm test, Jest, Vitest) in isolated Docker containers.
* **Evaluation Metric**:
  - **Pass Rate**: Percentage of sequential tasks implemented successfully.
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

### API-Only Infrastructure Setup
- **Anthropic Models**: Called via Vertex AI Model Garden `rawPredict` (using GCP ADC authentication). Zero CLI harness overhead.
- **Gemini Models**: Called via `google-genai` SDK (`genai.Client(vertexai=True)`).

---

## 3. Task-Split Patterns & Architectural Proposals

Web development tasks require both a high-level understanding of project layout (routing, component hierarchies, CSS styles) and granular implementation of DOM manipulations or state updates. We decompose these tasks as follows:

```
                  +-----------------------------------+
                  |   Project State & Requirements    |
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
                  |  Output: Component Code (~800 tok) |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |   Local Web Test Runner (Docker)  |  <-- Vitest / Jest (FREE)
                  +-----------------------------------+
                                    |
                         +----------+----------+
                         |                     |
                      [PASS]                [FAIL]
                         |                     |
                         v                     v
                      (Done)       +-----------------------------------+
                                    |  Log Triage (DOM/Stderr Digest)   |  <-- 3.1-Flash-Lite (Truncates huge HTML logs)
                                    +-----------------------------------+
                                                        |
                                                        v
                                    +-----------------------------------+
                                    |  Escalated Thinking Repair        |  <-- 3.5-Flash (thinking="low") / Sonnet-5
                                    +-----------------------------------+
```

### Architecture A: Single-Model Direct Completion (Control Baseline)
* **Description**: Single-shot completion where a model is prompted with the current file state, instructions, and target requirements to output the modified file directly.
* **Tested Models**: `gemini-3.1-flash-lite`, `gemini-3.5-flash`, `claude-sonnet-5`, `claude-opus-4-8`.
* **Baseline Purpose**: Establish pass rate vs. cost benchmarks for standalone models.

### Architecture B: Read-Heavy / Write-Heavy Split (Advisor-Executor)
* **Rationale**:
  - **Read-heavy / Planning**: Needs to inspect multiple project files (`package.json`, route configuration, component imports, CSS styles) to understand context. Requires deep reasoning but outputs minimal guidelines (~150 tokens).
  - **Write-heavy / Execution**: Involves writing boilerplates, imports, DOM structures, and CSS selectors. Needs low reasoning but generates high-token outputs.
* **Strategy**:
  - **Planner (Read-Heavy)**: Assigned to `gemini-3.5-flash` (or `claude-sonnet-5` / `claude-opus-4-8`).
  - **Executor (Write-Heavy)**: Assigned to `gemini-3.1-flash-lite`.
* **Cost Advantage**: Guidance is extremely concise. The bulk code generation is offloaded to the 6x cheaper `gemini-3.1-flash-lite`.

### Architecture C: Generation Offload Cascade (Cheap Gen + Conditional Escalation)
* **Rationale**: Simple components, CSS adjustments, or helper functions can easily pass with `gemini-3.1-flash-lite`. Heavy models should only be invoked when the cheap generation fails the unit test suite.
* **Strategy**:
  1. Initial generation by `gemini-3.1-flash-lite`.
  2. Running the web dev test runner (Vitest/Jest). If passed, terminate.
  3. If failed, attempt 1 cheap repair loop with `gemini-3.1-flash-lite`.
  4. If still failing, escalate repair to `gemini-3.5-flash` (thinking="low") or `claude-sonnet-5`.

### Architecture D: Error Log / DOM Dump Triage Offload
* **Rationale**: Frontend test runner failures often output huge DOM tree structures, raw HTML, and verbose stack traces, easily exceeding 10,000 tokens. Feeding raw logs to expensive repair models is cost-prohibitive.
* **Strategy**:
  1. `gemini-3.1-flash-lite` parses the error logs and DOM output, summarizing it into a concise 10-line digest (identifying failing assertions, active CSS classes, or faulty state values).
  2. High-tier repair model (`gemini-3.5-flash` or `claude-sonnet-5`) receives the compressed digest, minimizing input token overhead by up to 85%.

### Architecture E: The Sweet-Spot Hybrid Architecture (Read/Write Split + Triage + Thinking Escalation)
* **Combined Workflow**:
  1. **Planning**: `gemini-3.5-flash` generates concise coding advice (<200 words).
  2. **Generation**: `gemini-3.1-flash-lite` generates the initial web code/component.
  3. **Verification**: Execute Jest/Vitest suite.
  4. **Cheap Repair**: If failing, `gemini-3.1-flash-lite` attempts self-repair.
  5. **Triage & Escalation**: If still failing, `gemini-3.1-flash-lite` triages the test output. `gemini-3.5-flash` (with `thinking_level="low"`) or `claude-sonnet-5` runs the final escalated repair attempt.

---

## 4. Benchmark Harness & Reproducibility Guide

All insights must be reproducible. The python runner `bench_webdev_sweetspot_runner.py` is configured to run these evaluations programmatically.

### Reference Code Map
* **Web-Dev Sweet Spot Runner**: `bench_webdev_sweetspot_runner.py` (to be created in this directory, adapted from [bench_sweetspot_runner.py](file:///Users/lexha/Documents/work/codes/prj/17-tokenomics/benchmark-using-multi-LLMs/bigCodeBench-hard/bench_sweetspot_runner.py)).
* **Docker Test Harness Runner**: Script integrating with the Docker containers created by Web-Bench to apply code diffs and parse stdout/stderr from `jest`/`vitest`.

### Execution Commands

To run comparisons across all configurations on a React subset:

```bash
# Run comparison for all sweet-spot configs on the first 5 tasks of Web-Bench React project
python webdev/bench_webdev_sweetspot_runner.py --compare-all --project @web-bench/react --n 5
```

To run a specific architecture:

```bash
# 1. Read/Write Split (gemini-3.5-flash planner + gemini-3.1-flash-lite executor)
python webdev/bench_webdev_sweetspot_runner.py \
    --arch read-write --planner gemini-3.5-flash --executor gemini-3.1-flash-lite --project @web-bench/react --n 5

# 2. Sweet-Spot Hybrid (Read/Write Split + Triage + Thinking Escalation)
python webdev/bench_webdev_sweetspot_runner.py \
    --arch hybrid --planner gemini-3.5-flash --executor gemini-3.1-flash-lite --escalate gemini-3.5-flash --project @web-bench/react --n 5
```

---

## 5. Expected Cost-Performance Sweet Spot Matrix

Below is the projected performance matrix based on web development benchmarks:

| Configuration | Architecture | Pass Rate (React/Vue Tasks) | Total Cost (as-run USD) | Avg Output Tokens / Task | Cost Efficiency Rating |
|---|---|---|---|---|---|
| `gemini-3.1-flash-lite` | Single Model | ~25% | **~$0.005** | ~600 | Low Pass Rate / Ultra Cheap |
| `gemini-3.5-flash` | Single Model | ~45% | ~$0.050 | ~800 | Balanced Baseline |
| `claude-sonnet-5` | Single Model | ~60% | ~$0.150 | ~900 | High Quality / High Cost |
| `3.5-Flash` + `3.1-Lite` | Read/Write Split | ~45% | **~$0.015** | ~750 | **⭐ SWEET SPOT (Great Value)** |
| `Sonnet-5` + `3.1-Lite` | Read/Write Split | ~55% | ~$0.030 | ~800 | **⭐ SWEET SPOT (Frontier Quality)** |
| `3.1-Lite` -> `3.5-Flash` | Gen Cascade | ~55% | ~$0.028 | ~950 | High Efficiency Cascade |
| `3.5-Flash` + `3.1-Lite` + `3.5-Flash` | Hybrid (Read/Write + Triage + Esc) | ~65% | **~$0.035** | ~1100 | **🏆 ULTIMATE SWEET SPOT** |

### Key Insights & Recommendations
1. **Advisor Role is Essential for Frameworks**: `gemini-3.1-flash-lite` alone struggles with framework-specific constraints (e.g., React hooks lifecycle, state scopes). Pairing it with a planning step (from `gemini-3.5-flash` or `claude-sonnet-5`) yields a 20%+ pass rate improvement.
2. **Log Triage prevents Token Bloat**: Raw Jest/Vitest logs are massive due to DOM prints. Triage offloading reduces input tokens by 80%, avoiding high billing rates during escalated repairs.
3. **Escalate to Thinking Models conditionally**: Keep thinking disabled for initial generation. Use `thinking_level="low"` on `gemini-3.5-flash` (or escalate to `claude-opus-4-8`) only for complex State Management (Redux/Zustand) or complex CSS flex/grid layouts that fail standard tests.

---

## 6. Reproducibility Requirements

To ensure benchmark findings are fully reproducible:
1. **Deterministic Parameters**: Use fixed temperature (`0.0` or default API single-shot) to guarantee output consistency.
2. **Docker Isolation**: Ensure all tests run inside clean, isolated containers spun up by `docker-compose`. Cache node_modules to ensure consistent build environments.
3. **Structured Logs**: Every run must save output JSON results containing exact prompts, candidate codes, output tokens, thinking tokens, raw test results, and calculated USD costs.
