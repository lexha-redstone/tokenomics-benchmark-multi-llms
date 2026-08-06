# Supported Models & Architecture Pricing Table

This document details the Large Language Model (LLM) configurations, active Vertex AI pricing rates, and multi-model collaboration architectures evaluated across the benchmark suites.

---

## 1. Supported Models & Vertex AI Pricing Table

Costs are calculated in USD per **1,000,000 tokens**. Pricing is synchronized with [`src/config.py`](src/config.py):

| Model ID | Provider | Generation / Model Tier | Input ($/1M) | Output ($/1M) | Cache Read ($/1M) | Cache Write ($/1M) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `gemini-3.5-flash-lite` | Google | Next-Gen Flash-Lite | $0.30 | $2.50 | $0.030 | $0.00 |
| `gemini-3.6-flash` | Google | Next-Gen Flash | $1.50 | $7.50 | $0.150 | $0.00 |
| `gemini-3.1-flash-lite` | Google | Baseline Flash-Lite | $0.25 | $1.50 | $0.025 | $0.00 |
| `gemini-3.5-flash` | Google | Baseline Flash | $1.50 | $9.00 | $0.150 | $0.00 |
| `gemini-3.1-pro-preview` | Google | Pro Intelligence | $2.00 | $12.00 | $0.200 | $0.00 |
| `claude-sonnet-5` | Anthropic | Claude 3.5 Sonnet | $2.00 | $10.00 | $0.200 | $2.50 |
| `claude-opus-5` | Anthropic | Next-Gen Frontier Opus | $5.00 | $25.00 | $0.500 | $6.25 |
| `claude-opus-4-8` | Anthropic | Claude 3.0 Opus | $5.00 | $25.00 | $0.500 | $6.25 |

> **API Dispatch Notes**:
> - **Google Gemini models**: Driven via Google Cloud Vertex AI using `google-genai` SDK with adaptive `ThinkingConfig` headroom.
> - **Anthropic Claude models**: Driven via Vertex AI Anthropic `rawPredict` endpoint with cache token accounting.

---

## 2. Multi-Model Collaboration Architectures

### A. Single Model Direct Completion (`single`)
Direct single-model generation with optional multi-turn self-repair loops.
- `gemini-3.5-flash-lite`: Ultra-budget generation.
- `gemini-3.6-flash`: High-speed generation with adaptive reasoning.
- `claude-sonnet-5` / `claude-opus-5`: Frontier baseline capability.

### B. Read-Heavy / Write-Heavy Split (Advisor-Executor) (`read-write`)
Decouples specification/planning from code implementation.
- **Advisor (Planner)**: Generates a concise implementation contract (< 200 words) using a reasoning model (`gemini-3.6-flash` or `claude-sonnet-5`).
- **Executor**: Generates code adhering strictly to the contract using a high-throughput model (`gemini-3.5-flash-lite`).

### C. Generation Offload Cascade (`cascade`)
Offloads initial code generation to a sub-cent economy model. On test failure, escalates code repair to a thinking model.
- **Level 1**: `gemini-3.5-flash-lite` initial draft.
- **Level 2+**: Escalates to `gemini-3.6-flash` (thinking enabled) or `claude-sonnet-5`.

### D. Sweet-Spot Hybrid (`hybrid`)
Combines planning, execution, cheap self-repair, test failure triage, and thinking escalation.
1. **Plan**: `gemini-3.6-flash` Advisor.
2. **Execute**: `gemini-3.5-flash-lite` Executor.
3. **Cheap Repair**: `gemini-3.5-flash-lite` first attempt.
4. **Triage**: Compresses error log into a structured digest.
5. **Escalated Repair**: `gemini-3.6-flash` / `claude-sonnet-5` final repair.

---

## 3. Straitjacket Context Containment & Zero-Cost Triage

Standard multi-turn evaluation loops pass verbose stderr/stdout stack traces into repair model prompts, causing:
1. **Token Bloat**: Verbose unittests flood input tokens, multiplying repair costs.
2. **LLM Triage Overhead**: LLM-based summarization (`triage_error`) adds ~$0.0018 per repair, increases latency, and risks hallucinating line numbers.
3. **Cache Busting**: Ephemeral directory paths (`/tmp/bcb_xyz/`) mutate prompt prefixes, dropping prompt cache hit rates.

**Straitjacket Zero-Cost Local Triage (`*_straitjacket`)** replaces probabilistic triage with deterministic local parsing:
- **$0.00 Triage Spend**: Local `UnittestProfile` parses stderr streams in sub-milliseconds at zero API cost.
- **Normalized Prefix Stability**: Strips ephemeral paths and timestamps, ensuring stable prompt prefixes for **96–98% cache hit rates**.
- **Top-Tier $/Solved Efficiency**: Achieves equal or higher task resolution at **19–35% lower Total Cost of Ownership**.
